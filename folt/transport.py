# Extracted from CerT-MCMC-v2 experiments by quantile_core_diagnostic.py
"""
RealNVP transport map for FolT-MCMC.

A spectral-normalised RealNVP normalising flow T: z -> theta that pushes a
standard Gaussian base distribution q0 = N(0, I) forward to an approximation
q of the target pi. The flow exposes:

    forward(z)   -> (theta, log_det)   pushforward z ~ q0  into theta-space
    inverse(th)  -> (z,     log_det)   pull theta back to the latent space
    log_prob(th) -> log q(theta)       density of the pushforward

Each coupling layer uses spectral-normalised s/t networks with a clipped
log-scale (scale_clip * tanh) so the map stays bi-Lipschitz, which is what
makes the oscillation / gradient certificates non-vacuous.

Training (`train_flow`) is the FolT-OG-Anneal recipe: maximum-likelihood NLL
on target samples, plus oscillation and gradient-norm regularisers that are
linearly ramped in over the second half of training.
"""

import math
import time

import torch
import torch.nn as nn

from .oscillation_reg import smooth_oscillation, gradient_norm_mean

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
# Spectral-normalised RealNVP
# ══════════════════════════════════════════════════════════════

def sn_linear(in_f, out_f):
    """Spectral-normalised linear layer (bounds the layer Lipschitz constant)."""
    return nn.utils.spectral_norm(nn.Linear(in_f, out_f))


class CouplingLayer(nn.Module):
    """Affine coupling layer with spectral-normalised s/t networks.

    The binary `mask` selects which coordinates are passed through unchanged
    and used to condition the scale/translation of the complementary block.
    The log-scale is squashed by `scale_clip * tanh(.)` so |s| <= scale_clip,
    keeping the Jacobian (and its inverse) bounded.
    """

    def __init__(self, dim, hidden_dim, mask, scale_clip=0.7):
        super().__init__()
        self.register_buffer('mask', mask)
        self.scale_clip = scale_clip
        self.s_net = nn.Sequential(
            sn_linear(dim, hidden_dim), nn.Tanh(),
            sn_linear(hidden_dim, hidden_dim), nn.Tanh(),
            sn_linear(hidden_dim, dim))
        self.t_net = nn.Sequential(
            sn_linear(dim, hidden_dim), nn.Tanh(),
            sn_linear(hidden_dim, hidden_dim), nn.Tanh(),
            sn_linear(hidden_dim, dim))

    def forward(self, z):
        z_m = z * self.mask
        s = self.s_net(z_m) * (1 - self.mask)
        s = self.scale_clip * torch.tanh(s)
        t = self.t_net(z_m) * (1 - self.mask)
        return z * torch.exp(s) + t, s.sum(dim=-1)

    def inverse(self, theta):
        th_m = theta * self.mask
        s = self.s_net(th_m) * (1 - self.mask)
        s = self.scale_clip * torch.tanh(s)
        t = self.t_net(th_m) * (1 - self.mask)
        return (theta - t) * torch.exp(-s), -s.sum(dim=-1)


class SNRealNVP(nn.Module):
    """Spectral-normalised RealNVP flow.

    Args:
        dim:        dimensionality of theta / z.
        n_layers:   number of coupling layers (masks cycle through coordinates).
        hidden_dim: width of the s/t MLPs.
        scale_clip: bound on the per-layer log-scale.
    """

    def __init__(self, dim=2, n_layers=8, hidden_dim=64, scale_clip=0.7):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % dim] = 1.0
            self.layers.append(CouplingLayer(dim, hidden_dim, mask, scale_clip))

    def forward(self, z):
        """z ~ q0  ->  theta = T(z), with log|det dtheta/dz|."""
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in self.layers:
            x, ld = layer(x)
            log_det += ld
        return x, log_det

    def inverse(self, theta):
        """theta -> z = T^{-1}(theta), with log|det dz/dtheta|."""
        log_det = torch.zeros(theta.shape[0], device=theta.device)
        z = theta
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            log_det += ld
        return z, log_det

    def log_prob(self, theta):
        """log q(theta) of the pushforward density."""
        z, log_det = self.inverse(theta)
        log_q0 = -0.5 * (z**2).sum(dim=-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return log_q0 + log_det


# ══════════════════════════════════════════════════════════════
# Residual r(z) = log pi(theta) - log q(theta)  (up to an additive const)
# ══════════════════════════════════════════════════════════════

def compute_residual(flow, target, z):
    """Importance-weight log-residual r(z) for z ~ q0 = N(0, I).

    With theta = T(z) and log_det = log|det dtheta/dz|,

        r(z) = log pi(theta) + log_det + 0.5 ||z||^2
             = log( pi(theta) / q(theta) )  +  const

    The additive constant (0.5 * dim * log 2pi, plus the unknown log-normaliser
    of pi) cancels in any oscillation / quantile-spread computation, so the
    oscillation of r equals the oscillation of the true log importance weight.

    `target` only needs a `.log_prob(theta)` method (the original CerT banana
    exposed `-U(theta)`, which equals `log_prob`).
    """
    theta, log_det = flow(z)
    return target.log_prob(theta) + log_det + 0.5 * (z**2).sum(dim=-1)


# ══════════════════════════════════════════════════════════════
# Training: FolT-OG-Anneal
# ══════════════════════════════════════════════════════════════

def train_flow(target, D, hp, train_samples=None, verbose=True, device=DEVICE):
    """Train an SNRealNVP flow with the FolT-OG-Anneal objective.

    Loss = NLL  +  lambda_o(ramp) * smooth_osc(r)  +  lambda_g(ramp) * grad_norm

    The oscillation/gradient regularisers are ramped in linearly over the
    final 60% of training (off for the first 40%), so the flow first fits the
    target by maximum likelihood and only then tightens its certificate.

    Args:
        target:        object with `.log_prob(theta)` and (optionally)
                       `.generate_samples(n)`.
        D:             dimension.
        hp:            dict with keys n_layers, hidden_dim, n_train, n_epochs.
        train_samples: optional pre-generated target samples (N, D). If None,
                       `target.generate_samples(hp['n_train'])` is used.
        device:        torch device.

    Returns:
        (flow, history) where history is a dict with the final NLL and the
        per-checkpoint NLL trace.
    """
    if train_samples is None:
        train_samples = target.generate_samples(hp['n_train'])
    train_samples = train_samples.to(device)
    N = train_samples.shape[0]

    flow = SNRealNVP(dim=D, n_layers=hp['n_layers'],
                     hidden_dim=hp['hidden_dim']).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=hp.get('lr', 1e-3))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, hp['n_epochs'])

    lambda_o, lambda_g, tau = 0.1, 0.05, 0.1
    batch_size = hp.get('batch_size', 512)
    t0 = time.time()
    nll_trace = []

    for epoch in range(hp['n_epochs']):
        flow.train()
        ramp = max(0.0, (epoch / hp['n_epochs'] - 0.4)) / 0.6
        eff_lo, eff_lg = lambda_o * ramp, lambda_g * ramp

        idx = torch.randint(0, N, (batch_size,), device=device)
        nll = -flow.log_prob(train_samples[idx]).mean()

        osc_loss = grad_loss = torch.tensor(0.0, device=device)
        if eff_lo > 0 or eff_lg > 0:
            z_f = torch.randn(batch_size, D, device=device)
            r = compute_residual(flow, target, z_f)
            if eff_lo > 0:
                osc_loss = smooth_oscillation(r, tau=tau)
            if eff_lg > 0:
                z_g = torch.randn(64, D, device=device)
                gm, _ = gradient_norm_mean(flow, target, z_g)
                grad_loss = gm

        loss = nll + eff_lo * osc_loss + eff_lg * grad_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % max(hp['n_epochs'] // 5, 1) == 0 or epoch == hp['n_epochs'] - 1:
            nll_trace.append((epoch + 1, float(nll.item())))
            if verbose:
                print(f"    [{epoch+1:5d}/{hp['n_epochs']}] NLL={nll.item():.3f} | "
                      f"{time.time()-t0:.0f}s")

    flow.eval()
    history = {'nll_final': float(nll.item()), 'nll_trace': nll_trace}
    return flow, history
