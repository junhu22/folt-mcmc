"""
FolT-MCMC V2 — Engineering Verification
=========================================
Three critical checks before upgrading to dual-certificate paper:

1. Sailboat (D=6): Must show small Q99-Q1 and non-vacuous core certificate
2. Shear building (D=8): Must show LARGE Q99-Q1 (transport failure remains detected)
3. Empirical target-mass diagnostic: pi_hat(G_rho) should be close to 1-2*rho

Decision logic:
  - Sailboat Q99-Q1 < 2 AND shear Q99-Q1 >> sailboat  =>  V2 is valid
  - Sailboat Q99-Q1 large  =>  V2 has no advantage over V1
  - Shear Q99-Q1 small  =>  V2 is too lenient (false pass)

Usage:
  conda activate lcnf
  cd C:\\FolT-MCMC\\experiments
  python verify_engineering.py --quick          # fast sanity check
  python verify_engineering.py                  # full verification
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import argparse
import math
from scipy.stats import chi2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ══════════════════════════════════════════════════════════════
# Flow Architecture (from V1)
# ══════════════════════════════════════════════════════════════

def sn_linear(in_f, out_f):
    return nn.utils.spectral_norm(nn.Linear(in_f, out_f))

class CouplingLayer(nn.Module):
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
    def __init__(self, dim=2, n_layers=8, hidden_dim=64, scale_clip=0.7):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % dim] = 1.0
            self.layers.append(CouplingLayer(dim, hidden_dim, mask, scale_clip))

    def forward(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in self.layers:
            x, ld = layer(x)
            log_det += ld
        return x, log_det

    def inverse(self, theta):
        log_det = torch.zeros(theta.shape[0], device=theta.device)
        z = theta
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            log_det += ld
        return z, log_det

    def log_prob(self, theta):
        z, log_det = self.inverse(theta)
        log_q0 = -0.5 * (z**2).sum(dim=-1) - 0.5 * self.dim * math.log(2*math.pi)
        return log_q0 + log_det


# ══════════════════════════════════════════════════════════════
# Sailboat Target (D=6, from V1)
# ══════════════════════════════════════════════════════════════

W_CALIBRATED = np.array([
    [0.03089318, 0.25500751, 0.31534939, 0.14729265, 0.15159267, 0.09986460],
    [0.05547624, 0.24682654, 0.29048217, 0.14312768, 0.16960660, 0.09448077],
    [0.07054341, 0.17353892, 0.28472797, 0.20401001, 0.18305932, 0.08412037],
    [0.30070137, 0.24373602, 0.19286535, 0.10665094, 0.10360360, 0.05244271],
    [0.38535323, 0.24641020, 0.14570898, 0.08935119, 0.07023028, 0.06294613],
    [0.48516824, 0.24917388, 0.09948227, 0.08017624, 0.05144794, 0.03455144],
], dtype=np.float64)

FREQ_OBS_SAIL = np.array([0.91, 1.08, 1.47, 2.55, 3.16, 3.81], dtype=np.float64)
NOISE_STD_SAIL = np.array([0.025, 0.030, 0.040, 0.060, 0.080, 0.100], dtype=np.float64)


class SailboatTarget:
    def __init__(self):
        self.D = 6
        self.W = torch.tensor(W_CALIBRATED, dtype=torch.float32)
        self.freq_obs = torch.tensor(FREQ_OBS_SAIL, dtype=torch.float32)
        self.noise_std = torch.tensor(NOISE_STD_SAIL, dtype=torch.float32)
        self.freq_nominal = self.freq_obs.clone()
        self.prior_mean = 1.0
        self.prior_std = 0.35
        self.theta_min = 0.5
        self.theta_max = 2.0
        self.theta_range = self.theta_max - self.theta_min

    def _eta_to_theta(self, eta):
        return self.theta_min + self.theta_range * torch.sigmoid(eta)

    def _forward_model(self, theta):
        s = self.W.to(theta.device) @ theta
        s = torch.clamp(s, min=1e-6)
        return self.freq_nominal.to(theta.device) * torch.sqrt(s)

    def U(self, eta):
        batch = eta.shape[0]
        potentials = torch.zeros(batch, dtype=eta.dtype, device=eta.device)
        for i in range(batch):
            theta_i = self._eta_to_theta(eta[i])
            freq_pred = self._forward_model(theta_i)
            residual = (freq_pred - self.freq_obs.to(eta.device)) / self.noise_std.to(eta.device)
            log_lik = -0.5 * (residual ** 2).sum()
            log_prior = -0.5 * (((theta_i - self.prior_mean) / self.prior_std) ** 2).sum()
            sig = torch.sigmoid(eta[i])
            log_jac = (torch.log(sig.clamp(min=1e-10))
                      + torch.log((1 - sig).clamp(min=1e-10))
                      + math.log(self.theta_range)).sum()
            potentials[i] = -(log_lik + log_prior + log_jac)
        return potentials

    def log_prob(self, eta):
        return -self.U(eta)

    def sample_mala(self, n, n_warmup=10000, step_size=0.05, thin=5):
        theta_init = torch.ones(self.D) * self.prior_mean
        eta = torch.log((theta_init - self.theta_min) / (self.theta_max - theta_init))
        samples = []
        for i in range(n * thin + n_warmup):
            eta_r = eta.detach().requires_grad_(True)
            lp = self.log_prob(eta_r.unsqueeze(0)).squeeze()
            lp.backward()
            grad = eta_r.grad.detach()
            noise = torch.randn(self.D)
            proposal = eta + step_size * grad + math.sqrt(2 * step_size) * noise
            lp_prop = self.log_prob(proposal.unsqueeze(0)).squeeze().item()
            lp_curr = self.log_prob(eta.unsqueeze(0)).squeeze().item()
            prop_r = proposal.detach().requires_grad_(True)
            lp_p = self.log_prob(prop_r.unsqueeze(0)).squeeze()
            lp_p.backward()
            grad_prop = prop_r.grad.detach()
            log_q_fwd = -0.25/step_size * ((proposal - eta - step_size*grad)**2).sum().item()
            log_q_bwd = -0.25/step_size * ((eta - proposal - step_size*grad_prop)**2).sum().item()
            log_alpha = (lp_prop - lp_curr) + (log_q_bwd - log_q_fwd)
            if math.log(np.random.random() + 1e-30) < min(0, log_alpha):
                eta = proposal.detach()
            if i >= n_warmup and (i - n_warmup) % thin == 0:
                samples.append(eta.clone())
            if i % 20000 == 0 and i > 0:
                print(f"      MALA step {i}...")
        return torch.stack(samples[:n])


# ══════════════════════════════════════════════════════════════
# Shear Building Target (D=8, from V1)
# ══════════════════════════════════════════════════════════════

class ShearBuildingTarget:
    def __init__(self):
        self.D = 8
        self.n_modes = 5
        self.masses = torch.tensor([1.0] * 8, dtype=torch.float32)
        self.k_nominal = torch.tensor([1.0] * 8, dtype=torch.float32)
        self.theta_true = torch.tensor(
            [0.95, 0.90, 0.85, 0.80, 0.95, 1.00, 0.98, 0.92],
            dtype=torch.float32)
        self.freq_true = self._compute_frequencies(self.theta_true)
        self.noise_std = torch.tensor([0.015, 0.020, 0.025, 0.035, 0.055],
                                       dtype=torch.float32)
        torch.manual_seed(42)
        self.freq_obs = self.freq_true + self.noise_std * torch.randn(self.n_modes)
        self.prior_mean = 0.95
        self.prior_std = 0.20
        self.theta_min = 0.3
        self.theta_max = 1.5

    def _assemble_stiffness(self, theta):
        k = theta * self.k_nominal
        n = len(k)
        K = torch.zeros(n, n, dtype=theta.dtype)
        for i in range(n):
            K[i, i] = k[i] + (k[i+1] if i < n-1 else 0)
            if i > 0:
                K[i, i-1] = -k[i]
                K[i-1, i] = -k[i]
        return K

    def _compute_frequencies(self, theta):
        K = self._assemble_stiffness(theta)
        M = torch.diag(self.masses)
        M_inv = torch.diag(1.0 / self.masses)
        A = M_inv @ K
        eigenvalues = torch.linalg.eigvalsh(A)
        eigenvalues = torch.sort(eigenvalues.abs())[0]
        omega = torch.sqrt(eigenvalues[:self.n_modes].clamp(min=1e-10))
        return omega / (2 * math.pi)

    def _eta_to_theta(self, eta):
        return self.theta_min + (self.theta_max - self.theta_min) * torch.sigmoid(eta)

    def U(self, eta):
        batch = eta.shape[0]
        potentials = torch.zeros(batch, dtype=eta.dtype, device=eta.device)
        for i in range(batch):
            theta_i = self._eta_to_theta(eta[i])
            try:
                freq_pred = self._compute_frequencies(theta_i)
            except Exception:
                potentials[i] = 1e6
                continue
            residual = (freq_pred - self.freq_obs) / self.noise_std
            log_lik = -0.5 * (residual ** 2).sum()
            log_prior = -0.5 * (((theta_i - self.prior_mean) / self.prior_std) ** 2).sum()
            sig = torch.sigmoid(eta[i])
            log_jac = (torch.log(sig.clamp(min=1e-10))
                      + torch.log((1 - sig).clamp(min=1e-10))
                      + math.log(self.theta_max - self.theta_min)).sum()
            potentials[i] = -(log_lik + log_prior + log_jac)
        return potentials

    def log_prob(self, eta):
        return -self.U(eta)

    def sample_mala(self, n, n_warmup=10000, step_size=0.01, thin=10):
        eta = torch.zeros(self.D)
        samples = []
        for i in range(n * thin + n_warmup):
            eta_r = eta.detach().requires_grad_(True)
            lp = self.log_prob(eta_r.unsqueeze(0)).squeeze()
            lp.backward()
            grad = eta_r.grad.detach()
            noise = torch.randn(self.D)
            proposal = eta + step_size * grad + math.sqrt(2 * step_size) * noise
            lp_prop = self.log_prob(proposal.unsqueeze(0)).squeeze().item()
            lp_curr = self.log_prob(eta.unsqueeze(0)).squeeze().item()
            prop_r = proposal.detach().requires_grad_(True)
            lp_p = self.log_prob(prop_r.unsqueeze(0)).squeeze()
            lp_p.backward()
            grad_prop = prop_r.grad.detach()
            log_q_fwd = -0.25/step_size * ((proposal - eta - step_size*grad)**2).sum().item()
            log_q_bwd = -0.25/step_size * ((eta - proposal - step_size*grad_prop)**2).sum().item()
            log_alpha = (lp_prop - lp_curr) + (log_q_bwd - log_q_fwd)
            if math.log(np.random.random() + 1e-30) < min(0, log_alpha):
                eta = proposal.detach()
            if i >= n_warmup and (i - n_warmup) % thin == 0:
                samples.append(eta.clone())
            if i % 10000 == 0 and i > 0:
                print(f"      MALA step {i}...")
        return torch.stack(samples[:n])


# ══════════════════════════════════════════════════════════════
# Residual & Training
# ══════════════════════════════════════════════════════════════

def compute_residual(flow, target, z):
    theta, log_det = flow(z)
    return -target.U(theta) + log_det + 0.5 * (z**2).sum(dim=-1)

def smooth_oscillation(r, tau=0.1):
    return tau * torch.logsumexp(r / tau, dim=0) + tau * torch.logsumexp(-r / tau, dim=0)

def gradient_norm_mean(flow, target, z):
    z = z.detach().requires_grad_(True)
    r = compute_residual(flow, target, z)
    grad_r = torch.autograd.grad(r.sum(), z, create_graph=True)[0]
    norms = grad_r.norm(dim=-1)
    return norms.mean(), norms.max()

def train_flow(target, D, hp, verbose=True):
    print(f"    Generating {hp['n_train']} training samples...")
    t0 = time.time()
    train_samples = target.sample_mala(
        hp['n_train'],
        n_warmup=hp.get('n_warmup', 10000),
        step_size=hp.get('step_size', 0.05),
        thin=hp.get('thin', 5)
    ).to(DEVICE)
    print(f"    Samples done in {time.time()-t0:.0f}s")

    N = train_samples.shape[0]
    flow = SNRealNVP(dim=D, n_layers=hp['n_layers'],
                     hidden_dim=hp['hidden_dim']).to(DEVICE)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, hp['n_epochs'])

    lambda_o, lambda_g, tau = 0.1, 0.05, 0.1
    t0 = time.time()

    for epoch in range(hp['n_epochs']):
        flow.train()
        ramp = max(0.0, (epoch / hp['n_epochs'] - 0.4)) / 0.6
        eff_lo, eff_lg = lambda_o * ramp, lambda_g * ramp

        idx = torch.randint(0, N, (512,), device=DEVICE)
        nll = -flow.log_prob(train_samples[idx]).mean()

        osc_loss = grad_loss = torch.tensor(0.0, device=DEVICE)
        if eff_lo > 0 or eff_lg > 0:
            z_f = torch.randn(512, D, device=DEVICE)
            r = compute_residual(flow, target, z_f)
            if eff_lo > 0:
                osc_loss = smooth_oscillation(r, tau=tau)
            if eff_lg > 0:
                z_g = torch.randn(64, D, device=DEVICE)
                gm, _ = gradient_norm_mean(flow, target, z_g)
                grad_loss = gm

        loss = nll + eff_lo * osc_loss + eff_lg * grad_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if verbose and ((epoch + 1) % max(hp['n_epochs'] // 5, 200) == 0
                        or epoch == hp['n_epochs'] - 1):
            print(f"    [{epoch+1:5d}/{hp['n_epochs']}] NLL={nll.item():.3f} | "
                  f"{time.time()-t0:.0f}s")

    flow.eval()
    return flow


# ══════════════════════════════════════════════════════════════
# V2 Quantile Core Diagnostic (with formal certificate + target mass)
# ══════════════════════════════════════════════════════════════

def full_quantile_diagnostic(flow, target, D, n_cert=100000, zeta=0.05):
    """
    Complete V2 diagnostic including:
    1. Quantile statistics
    2. Formal Ĉ_ρ using data-defined widened core G̃_ρ
    3. V1 covering certificate for comparison
    4. Empirical target-mass estimate π̂(G̃_ρ)
    """
    flow.eval()

    # ── Sample and compute residuals ──
    batch_size = 20000
    r_list = []
    with torch.no_grad():
        for i in range(0, n_cert, batch_size):
            nb = min(batch_size, n_cert - i)
            z = torch.randn(nb, D, device=DEVICE)
            r = compute_residual(flow, target, z)
            r_list.append(r.cpu())
    r_all = torch.cat(r_list).numpy()
    n = len(r_all)

    # ── DKW bandwidth ──
    eps_n = math.sqrt(math.log(2.0 / zeta) / (2 * n))

    # ── Basic stats ──
    stats = {
        'n': n, 'D': D, 'zeta': zeta, 'eps_n': eps_n,
        'mean': float(np.mean(r_all)),
        'std': float(np.std(r_all)),
        'var': float(np.var(r_all)),
        'min': float(np.min(r_all)),
        'max': float(np.max(r_all)),
        'full_osc': float(np.ptp(r_all)),
    }

    # ── Quantile certificates: formal Ĉ_ρ using G̃_ρ ──
    rho_levels = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25]
    quantile_results = {}

    for rho in rho_levels:
        if rho <= eps_n:
            # Not enough samples for this rho level
            quantile_results[rho] = {
                'feasible': False,
                'reason': f'rho={rho} <= eps_n={eps_n:.4f}'
            }
            continue

        # Raw quantiles
        q_lo = float(np.quantile(r_all, rho))
        q_hi = float(np.quantile(r_all, 1 - rho))
        c_rho_raw = q_hi - q_lo

        # Formal Ĉ_ρ: data-defined widened core G̃_ρ
        # G̃_ρ = {z : Q̂_{ρ-ε_n} ≤ r(z) ≤ Q̂_{1-ρ+ε_n}}
        q_lo_wide = float(np.quantile(r_all, rho - eps_n))
        q_hi_wide = float(np.quantile(r_all, 1 - rho + eps_n))
        c_rho_formal = q_hi_wide - q_lo_wide

        # Core spectral gap (formal)
        if c_rho_formal > 500:
            gamma_core = 0.0
        else:
            gamma_core = 2.0 / (1.0 + math.exp(c_rho_formal))

        # ── Empirical target-mass estimate ──
        # π̂(G̃_ρ) = Σ_i exp(r_i) 1_{G̃}(z_i) / Σ_i exp(r_i)
        in_core = (r_all >= q_lo_wide) & (r_all <= q_hi_wide)
        n_in_core = int(in_core.sum())

        # Compute exp(r) with numerical stability (shift by max)
        r_shifted = r_all - r_all.max()
        w = np.exp(r_shifted)
        w_core = w[in_core].sum()
        w_total = w.sum()
        pi_hat_core = float(w_core / w_total)

        # Proposal mass of core (should be close to 1-2*rho by construction)
        mu_core = float(n_in_core / n)

        quantile_results[rho] = {
            'feasible': True,
            'q_lo': q_lo, 'q_hi': q_hi,
            'c_rho_raw': c_rho_raw,
            'q_lo_wide': q_lo_wide, 'q_hi_wide': q_hi_wide,
            'c_rho_formal': c_rho_formal,
            'gamma_core': gamma_core,
            'mu_core': mu_core,
            'pi_hat_core': pi_hat_core,
            'n_in_core': n_in_core,
        }

    stats['quantiles'] = quantile_results

    # ── V1 covering certificate ──
    R_alpha = math.sqrt(chi2.ppf(0.95, df=D))
    log_term = math.log(2.0 / zeta) / (2 * n)
    eps_star = R_alpha * (log_term + math.sqrt(log_term)) ** (1.0 / D)

    n_grad = min(2000, n)
    z_g = torch.randn(n_grad, D, device=DEVICE).requires_grad_(True)
    r_g = compute_residual(flow, target, z_g)
    grad_r = torch.autograd.grad(r_g.sum(), z_g)[0]
    grad_sup = grad_r.norm(dim=-1).max().item()

    v1_cert = stats['full_osc'] + 2 * grad_sup * eps_star
    v1_gamma = 2.0 / (1.0 + math.exp(min(v1_cert, 500)))

    stats['v1'] = {
        'cert': v1_cert,
        'gamma': v1_gamma,
        'eps_star': eps_star,
        'grad_sup': grad_sup,
        'correction': 2 * grad_sup * eps_star,
    }

    return stats


def print_engineering_diagnostic(name, stats):
    """Pretty-print diagnostic for one engineering target."""
    D = stats['D']
    print(f"\n  {'═'*75}")
    print(f"  {name} (D={D}, n={stats['n']:,})")
    print(f"  {'═'*75}")

    print(f"\n  Residual r(z): mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
          f"var={stats['var']:.6f}")
    print(f"  Full osc = {stats['full_osc']:.4f}")

    v1 = stats['v1']
    print(f"\n  V1 Covering Certificate:")
    print(f"    C_v1 = {v1['cert']:.4f} (osc={stats['full_osc']:.4f} + "
          f"corr={v1['correction']:.4f})")
    print(f"    gamma_v1 = {v1['gamma']:.6f}")
    print(f"    eps* = {v1['eps_star']:.4f}, grad_sup = {v1['grad_sup']:.4f}")

    print(f"\n  V2 Quantile Core Certificates (formal, DKW-corrected):")
    print(f"  eps_n = {stats['eps_n']:.6f}")
    print(f"  {'rho':<8} {'C_raw':>8} {'Ĉ_formal':>10} {'gamma':>10} "
          f"{'mu(G̃)':>8} {'π̂(G̃)':>8} {'π̂≈mu?':>8}")
    print(f"  {'─'*64}")

    qr = stats['quantiles']
    for rho in sorted(qr.keys()):
        res = qr[rho]
        if not res['feasible']:
            print(f"  {rho:<8.3f} {'—':>8} {'infeasible':>10}")
            continue
        c_raw = res['c_rho_raw']
        c_form = res['c_rho_formal']
        g = res['gamma_core']
        mu = res['mu_core']
        pi_hat = res['pi_hat_core']
        close = 'YES' if abs(pi_hat - mu) < 0.05 else 'no'
        print(f"  {rho:<8.3f} {c_raw:>8.4f} {c_form:>10.4f} {g:>10.6f} "
              f"{mu:>8.4f} {pi_hat:>8.4f} {close:>8}")

    # Key metrics
    q01 = qr.get(0.01, {})
    q05 = qr.get(0.05, {})
    print(f"\n  ★ KEY METRICS:")
    if q05.get('feasible'):
        print(f"    Q95-Q5 (formal) = {q05['c_rho_formal']:.4f}")
    if q01.get('feasible'):
        print(f"    Q99-Q1 (formal) = {q01['c_rho_formal']:.4f}")
    print(f"    Full osc        = {stats['full_osc']:.4f}")
    if q01.get('feasible'):
        print(f"    Ratio           = {q01['c_rho_formal']/stats['full_osc']:.3f}")
        print(f"    π̂(G̃_0.01)      = {q01['pi_hat_core']:.4f}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

HPARAMS = {
    'sailboat': dict(
        n_layers=12, hidden_dim=192, n_train=25000, n_epochs=8000,
        n_cert=100000, n_warmup=15000, step_size=0.05, thin=5,
    ),
    'shear': dict(
        n_layers=12, hidden_dim=192, n_train=25000, n_epochs=8000,
        n_cert=100000, n_warmup=20000, step_size=0.01, thin=10,
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--targets', nargs='+', default=['sailboat', 'shear'],
                        choices=['sailboat', 'shear'])
    parser.add_argument('--outdir', type=str, default='results_v2')
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    for tgt_name in args.targets:
        hp = HPARAMS[tgt_name].copy()
        if args.quick:
            hp['n_train'] = 5000
            hp['n_epochs'] = 2000
            hp['n_cert'] = 20000
            hp['n_warmup'] = 3000

        if tgt_name == 'sailboat':
            target = SailboatTarget()
            D = 6
        else:
            target = ShearBuildingTarget()
            D = 8

        print(f"\n{'='*75}")
        print(f"  ENGINEERING VERIFICATION: {tgt_name.upper()} (D={D})")
        print(f"  Training: {hp['n_epochs']} epochs, {hp['n_train']} samples")
        print(f"  Certification: {hp['n_cert']:,} samples")
        print(f"{'='*75}")

        # Train
        t0 = time.time()
        flow = train_flow(target, D, hp)
        train_time = time.time() - t0
        print(f"  Total training time: {train_time:.0f}s")

        # Save flow
        torch.save(flow.state_dict(), out_dir / f'flow_{tgt_name}.pt')

        # Full diagnostic
        print(f"\n  Running V2 diagnostic...")
        t0 = time.time()
        stats = full_quantile_diagnostic(flow, target, D, n_cert=hp['n_cert'])
        print(f"  Diagnostic done in {time.time()-t0:.1f}s")

        print_engineering_diagnostic(tgt_name.upper(), stats)
        all_stats[tgt_name] = stats

        # Save
        # Convert for JSON serialization
        save_stats = {k: v for k, v in stats.items() if k != 'quantiles'}
        save_stats['quantiles'] = {}
        for rho, res in stats['quantiles'].items():
            save_stats['quantiles'][str(rho)] = res
        with open(out_dir / f'verify_{tgt_name}.json', 'w') as f:
            json.dump(save_stats, f, indent=2)

    # ── Final verdict ──
    if len(all_stats) == 2 and 'sailboat' in all_stats and 'shear' in all_stats:
        print(f"\n{'='*75}")
        print(f"  FINAL VERDICT")
        print(f"{'='*75}")

        s_sail = all_stats['sailboat']
        s_shear = all_stats['shear']

        q01_sail = s_sail['quantiles'].get(0.01, {})
        q01_shear = s_shear['quantiles'].get(0.01, {})

        sail_c = q01_sail.get('c_rho_formal', float('inf'))
        shear_c = q01_shear.get('c_rho_formal', float('inf'))

        print(f"\n  Sailboat (D=6):  Ĉ_0.01 = {sail_c:.4f}, "
              f"gamma_core = {q01_sail.get('gamma_core', 0):.6f}")
        print(f"  Shear (D=8):     Ĉ_0.01 = {shear_c:.4f}, "
              f"gamma_core = {q01_shear.get('gamma_core', 0):.6f}")

        print(f"\n  Discrimination ratio: shear/sailboat = {shear_c/max(sail_c, 1e-10):.1f}x")

        if sail_c < 2 and shear_c > sail_c * 3:
            print(f"\n  ✅ V2 VALIDATED:")
            print(f"     Sailboat passes core certificate (good transport)")
            print(f"     Shear fails or degrades (transport failure detected)")
            print(f"     Quantile-core certificate discriminates correctly!")
        elif sail_c < 2 and shear_c <= sail_c * 3:
            print(f"\n  ⚠️  V2 PARTIAL: Sailboat good, but shear not clearly worse")
            print(f"     May indicate shear transport is actually OK at D=8")
        elif sail_c >= 2:
            print(f"\n  ❌ V2 WEAK: Sailboat core certificate also large")
            print(f"     Flow quality for sailboat may need improvement")

        # Target mass check
        sail_pi = q01_sail.get('pi_hat_core', 0)
        shear_pi = q01_shear.get('pi_hat_core', 0)
        print(f"\n  Target mass check:")
        print(f"    Sailboat π̂(G̃_0.01) = {sail_pi:.4f} "
              f"(expect ≈ {1-2*0.01:.2f} if weights uniform)")
        print(f"    Shear    π̂(G̃_0.01) = {shear_pi:.4f}")

    print(f"\n  Results saved to {out_dir}/")
    print(f"\n{'='*75}")
    print("  VERIFICATION COMPLETE")
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
