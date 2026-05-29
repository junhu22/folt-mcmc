"""
FolT-MCMC V2 — Bayesian Logistic Regression + ESS Calibration
================================================================
Two purposes:
  1. Mainstream statistical benchmark (D=20, D=50) that JRSS-B reviewers
     will recognize, demonstrating V2 on non-engineering targets.
  2. ESS calibration: show that core certificate predicts actual
     independence-MH mixing performance.

Usage:
  conda activate lcnf
  cd C:\\FolT-MCMC\\experiments
  python benchmark_logreg.py --dim 20 --quick       # fast test
  python benchmark_logreg.py --dim 20 50            # full benchmark
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
# 1. Bayesian Logistic Regression Target
# ══════════════════════════════════════════════════════════════

class LogisticRegressionTarget:
    """
    Bayesian logistic regression posterior.

    Model:
      y_i | x_i, beta ~ Bernoulli(sigmoid(x_i' beta))
      beta ~ N(0, tau^2 I_D)

    Posterior is log-concave => flow should learn well.
    This is a standard benchmark familiar to all stats reviewers.

    Args:
        D: number of covariates (= dimension of beta)
        n_obs: number of observations
        tau: prior std
        seed: random seed for reproducible synthetic data
    """
    def __init__(self, D=20, n_obs=500, tau=2.0, seed=42):
        self.D = D
        self.n_obs = n_obs
        self.tau = tau

        # Generate synthetic data
        rng = np.random.RandomState(seed)

        # True coefficients: sparse, a few large, most small
        beta_true = np.zeros(D)
        n_active = min(5, D)
        active_idx = rng.choice(D, n_active, replace=False)
        beta_true[active_idx] = rng.randn(n_active) * 1.5
        self.beta_true = torch.tensor(beta_true, dtype=torch.float32)

        # Design matrix: standardised Gaussian
        X = rng.randn(n_obs, D).astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        self.X = torch.tensor(X, dtype=torch.float32)

        # Generate responses
        logits = X @ beta_true.astype(np.float32)
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = rng.binomial(1, probs).astype(np.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        print(f"    LogReg target: D={D}, n_obs={n_obs}, tau={tau}")
        print(f"    True beta sparsity: {n_active}/{D} active")
        print(f"    Class balance: {y.mean():.2f}")

    def U(self, beta):
        """
        Potential energy: -log p(beta | y, X)
        beta: (batch, D)
        """
        X_dev = self.X.to(beta.device)
        y_dev = self.y.to(beta.device)

        # Logits: (batch, n_obs)
        logits = beta @ X_dev.T  # (batch, n_obs)

        # Log-likelihood: sum of Bernoulli log-probs
        # log p(y|beta) = sum_i [y_i * logit_i - log(1 + exp(logit_i))]
        # Use numerically stable logsigmoid
        log_lik = (y_dev * logits - torch.nn.functional.softplus(logits)).sum(dim=-1)

        # Log-prior: N(0, tau^2 I)
        log_prior = -0.5 * (beta ** 2).sum(dim=-1) / (self.tau ** 2)

        return -(log_lik + log_prior)

    def log_prob(self, beta):
        return -self.U(beta)

    def grad_log_prob(self, beta):
        """Analytical gradient for efficient MALA."""
        X_dev = self.X.to(beta.device)
        y_dev = self.y.to(beta.device)

        logits = beta @ X_dev.T  # (batch, n_obs)
        probs = torch.sigmoid(logits)  # (batch, n_obs)
        residuals = y_dev.unsqueeze(0) - probs  # (batch, n_obs)

        grad_lik = residuals @ X_dev  # (batch, D)
        grad_prior = -beta / (self.tau ** 2)

        return grad_lik + grad_prior

    def sample_mala(self, n, n_warmup=5000, step_size=0.001, thin=5):
        """MALA with analytical gradients (fast for log-concave)."""
        print(f"    MALA: n={n}, warmup={n_warmup}, step={step_size}, thin={thin}")
        beta = torch.zeros(1, self.D)
        samples = []
        n_accept = 0
        t0 = time.time()

        for i in range(n * thin + n_warmup):
            grad = self.grad_log_prob(beta).squeeze(0)  # (D,)
            noise = torch.randn(self.D)
            proposal = beta.squeeze(0) + step_size * grad + math.sqrt(2*step_size) * noise

            # Log acceptance ratio
            lp_prop = self.log_prob(proposal.unsqueeze(0)).item()
            lp_curr = self.log_prob(beta).item()

            grad_prop = self.grad_log_prob(proposal.unsqueeze(0)).squeeze(0)

            log_q_fwd = -0.25/step_size * ((proposal - beta.squeeze(0) - step_size*grad)**2).sum().item()
            log_q_bwd = -0.25/step_size * ((beta.squeeze(0) - proposal - step_size*grad_prop)**2).sum().item()

            log_alpha = (lp_prop - lp_curr) + (log_q_bwd - log_q_fwd)
            if math.log(np.random.random() + 1e-30) < min(0, log_alpha):
                beta = proposal.unsqueeze(0)
                if i >= n_warmup:
                    n_accept += 1

            if i >= n_warmup and (i - n_warmup) % thin == 0:
                samples.append(beta.squeeze(0).clone())

            if (i+1) % 50000 == 0:
                elapsed = time.time() - t0
                rate = n_accept / max(1, i - n_warmup) if i > n_warmup else 0
                print(f"      Step {i+1}, accept={rate:.3f}, {elapsed:.0f}s")

        samples = torch.stack(samples[:n])
        rate = n_accept / (n * thin)
        print(f"    Done. Accept rate={rate:.3f}, shape={samples.shape}, "
              f"{time.time()-t0:.0f}s")
        return samples


# ══════════════════════════════════════════════════════════════
# 2. Flow Architecture (from V1)
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
# 3. Residual & Training
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
    print(f"    Training flow: {hp['n_epochs']} epochs, "
          f"{hp['n_layers']} layers, hidden={hp['hidden_dim']}")
    train_samples = target.sample_mala(
        hp['n_train'],
        n_warmup=hp.get('n_warmup', 5000),
        step_size=hp.get('step_size', 0.001),
        thin=hp.get('thin', 5)
    ).to(DEVICE)

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
                z_g = torch.randn(min(32, 512), D, device=DEVICE)
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
# 4. Independence MH Sampler + ESS
# ══════════════════════════════════════════════════════════════

def run_independence_mh(flow, target, D, n_samples=5000, n_warmup=500):
    """
    Run actual independence MH chain and compute empirical diagnostics.
    Returns samples, acceptance rate, and per-sample log weights.
    """
    flow.eval()

    # Initialise from flow
    with torch.no_grad():
        z_init = torch.randn(1, D, device=DEVICE)
        theta_init, _ = flow(z_init)
    theta = theta_init.squeeze(0)
    log_w = (target.log_prob(theta.unsqueeze(0)) - flow.log_prob(theta.unsqueeze(0))).item()

    samples = []
    log_weights = []
    n_accept = 0

    for i in range(n_samples + n_warmup):
        # Propose from flow
        with torch.no_grad():
            z_prop = torch.randn(1, D, device=DEVICE)
            theta_prop, _ = flow(z_prop)
        theta_prop = theta_prop.squeeze(0)
        log_w_prop = (target.log_prob(theta_prop.unsqueeze(0))
                     - flow.log_prob(theta_prop.unsqueeze(0))).item()

        # Accept/reject
        log_alpha = log_w_prop - log_w
        if math.log(np.random.random() + 1e-30) < min(0, log_alpha):
            theta = theta_prop
            log_w = log_w_prop
            if i >= n_warmup:
                n_accept += 1

        if i >= n_warmup:
            samples.append(theta.clone())
            log_weights.append(log_w)

    samples = torch.stack(samples)
    accept_rate = n_accept / n_samples
    return samples, accept_rate, np.array(log_weights)


def compute_ess_from_samples(samples):
    """
    Compute multivariate ESS using batch means.
    """
    n = samples.shape[0]
    if n < 100:
        return float(n)

    # Use first component's autocorrelation as proxy
    x = samples[:, 0].cpu().numpy()
    x = x - x.mean()

    # Batch means ESS
    batch_size = max(10, n // 20)
    n_batches = n // batch_size
    if n_batches < 2:
        return float(n)

    batches = x[:n_batches * batch_size].reshape(n_batches, batch_size)
    batch_means = batches.mean(axis=1)
    var_bm = batch_means.var() * batch_size
    var_total = x.var()

    if var_bm < 1e-15:
        return float(n)

    ess = n * var_total / var_bm
    return float(min(ess, n))


def compute_ess_from_acceptance(n_samples, accept_rate):
    """
    For independence MH, ESS ≈ n * accept_rate (approximate).
    More precise: if spectral gap = γ, then asymptotic efficiency = γ/(2-γ).
    """
    return n_samples * accept_rate


# ══════════════════════════════════════════════════════════════
# 5. Quantile Core Diagnostic
# ══════════════════════════════════════════════════════════════

def quantile_diagnostic(flow, target, D, n_cert=200000, zeta=0.05):
    flow.eval()
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

    eps_n = math.sqrt(math.log(2.0 / zeta) / (2 * n))

    stats = {
        'n': n, 'D': D, 'eps_n': eps_n,
        'mean': float(np.mean(r_all)),
        'std': float(np.std(r_all)),
        'var': float(np.var(r_all)),
        'full_osc': float(np.ptp(r_all)),
    }

    rho_levels = [0.005, 0.01, 0.025, 0.05, 0.10, 0.25]
    quantiles = {}

    for rho in rho_levels:
        if rho <= eps_n:
            quantiles[rho] = {'feasible': False}
            continue

        q_lo = float(np.quantile(r_all, rho))
        q_hi = float(np.quantile(r_all, 1 - rho))
        c_raw = q_hi - q_lo

        q_lo_w = float(np.quantile(r_all, rho - eps_n))
        q_hi_w = float(np.quantile(r_all, 1 - rho + eps_n))
        c_formal = q_hi_w - q_lo_w

        gamma_core = 2.0 / (1.0 + math.exp(min(c_formal, 500)))

        in_core = (r_all >= q_lo_w) & (r_all <= q_hi_w)
        r_shifted = r_all - r_all.max()
        w = np.exp(r_shifted)
        pi_hat = float(w[in_core].sum() / w.sum())
        mu_core = float(in_core.sum() / n)

        quantiles[rho] = {
            'feasible': True,
            'c_raw': c_raw,
            'c_formal': c_formal,
            'gamma_core': gamma_core,
            'mu_core': mu_core,
            'pi_hat': pi_hat,
        }

    stats['quantiles'] = quantiles

    # V1 certificate
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
        'cert': v1_cert, 'gamma': v1_gamma,
        'eps_star': eps_star, 'grad_sup': grad_sup,
    }

    return stats


# ══════════════════════════════════════════════════════════════
# 6. ESS Calibration
# ══════════════════════════════════════════════════════════════

def ess_calibration(flow, target, D, n_mh=5000, n_warmup=500):
    """
    Run actual independence MH and compare with certificate predictions.
    
    For independence MH with spectral gap γ:
      - Predicted ESS ratio = γ/(2-γ)
      - Predicted acceptance rate ≈ 2γ/(1+γ) (approx for small osc)
      
    Returns dict with empirical and predicted metrics.
    """
    print(f"    Running independence MH ({n_mh} samples)...")
    t0 = time.time()
    samples, accept_rate, log_weights = run_independence_mh(
        flow, target, D, n_samples=n_mh, n_warmup=n_warmup)
    print(f"    MH done in {time.time()-t0:.1f}s, accept={accept_rate:.3f}")

    # Empirical ESS
    ess_bm = compute_ess_from_samples(samples)
    ess_accept = compute_ess_from_acceptance(n_mh, accept_rate)

    # Log-weight statistics (empirical version of residual)
    lw = log_weights
    lw_osc = lw.max() - lw.min()
    lw_std = lw.std()
    lw_q95_q5 = float(np.quantile(lw, 0.95) - np.quantile(lw, 0.05))
    lw_q99_q1 = float(np.quantile(lw, 0.99) - np.quantile(lw, 0.01))

    return {
        'n_mh': n_mh,
        'accept_rate': float(accept_rate),
        'ess_batch_means': float(ess_bm),
        'ess_from_accept': float(ess_accept),
        'ess_ratio_bm': float(ess_bm / n_mh),
        'ess_ratio_accept': float(accept_rate),
        'lw_osc': float(lw_osc),
        'lw_std': float(lw_std),
        'lw_q95_q5': float(lw_q95_q5),
        'lw_q99_q1': float(lw_q99_q1),
    }


# ══════════════════════════════════════════════════════════════
# 7. Experiment Runner
# ══════════════════════════════════════════════════════════════

HPARAMS = {
    20: dict(n_layers=12, hidden_dim=128, n_train=30000, n_epochs=6000,
             n_cert=200000, n_warmup=5000, step_size=0.001, thin=3,
             n_obs=500),
    50: dict(n_layers=16, hidden_dim=256, n_train=40000, n_epochs=8000,
             n_cert=200000, n_warmup=8000, step_size=0.0005, thin=5,
             n_obs=1000),
}


def run_benchmark(D, quick=False, out_dir=None):
    hp = HPARAMS.get(D, HPARAMS[20]).copy()
    if quick:
        hp['n_train'] = 5000
        hp['n_epochs'] = 1500
        hp['n_cert'] = 50000
        hp['n_warmup'] = 2000

    if out_dir is None:
        out_dir = Path('results_v2')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"  BAYESIAN LOGISTIC REGRESSION — D={D}")
    print(f"  n_obs={hp['n_obs']}, epochs={hp['n_epochs']}, n_cert={hp['n_cert']:,}")
    print(f"{'='*75}")

    # ── Target ──
    target = LogisticRegressionTarget(D=D, n_obs=hp['n_obs'])

    # ── Train flow ──
    print(f"\n  Training flow...")
    t0 = time.time()
    flow = train_flow(target, D, hp)
    print(f"  Training done in {time.time()-t0:.0f}s")

    torch.save(flow.state_dict(), out_dir / f'flow_logreg_D{D}.pt')

    # ── Quantile diagnostic ──
    print(f"\n  V2 Quantile diagnostic ({hp['n_cert']:,} samples)...")
    t0 = time.time()
    qstats = quantile_diagnostic(flow, target, D, n_cert=hp['n_cert'])
    print(f"  Diagnostic done in {time.time()-t0:.1f}s")

    # ── ESS calibration ──
    n_mh = 2000 if quick else 5000
    print(f"\n  ESS calibration ({n_mh} MH samples)...")
    ess_stats = ess_calibration(flow, target, D, n_mh=n_mh)

    # ── Print results ──
    print(f"\n  {'═'*75}")
    print(f"  RESULTS — Bayesian LogReg D={D}")
    print(f"  {'═'*75}")

    print(f"\n  Residual: mean={qstats['mean']:.4f}, std={qstats['std']:.4f}, "
          f"var={qstats['var']:.6f}")
    print(f"  Full osc = {qstats['full_osc']:.4f}")

    v1 = qstats['v1']
    print(f"\n  V1 Certificate: C={v1['cert']:.4f}, gamma={v1['gamma']:.6f}")
    print(f"    eps*={v1['eps_star']:.4f}, grad_sup={v1['grad_sup']:.4f}")

    print(f"\n  V2 Quantile Core Certificates:")
    print(f"  {'rho':<8} {'Ĉ_raw':>8} {'Ĉ_formal':>10} {'γ_core':>10} "
          f"{'μ(G̃)':>8} {'π̂(G̃)':>8}")
    print(f"  {'─'*56}")

    for rho in sorted(qstats['quantiles'].keys()):
        res = qstats['quantiles'][rho]
        if not res['feasible']:
            print(f"  {rho:<8.3f} {'infeasible':>30}")
            continue
        print(f"  {rho:<8.3f} {res['c_raw']:>8.4f} {res['c_formal']:>10.4f} "
              f"{res['gamma_core']:>10.6f} {res['mu_core']:>8.4f} {res['pi_hat']:>8.4f}")

    # ── ESS calibration table ──
    print(f"\n  ESS CALIBRATION:")
    print(f"    Empirical acceptance rate: {ess_stats['accept_rate']:.4f}")
    print(f"    Empirical ESS (batch means): {ess_stats['ess_batch_means']:.1f} / {n_mh}")
    print(f"    Empirical ESS ratio: {ess_stats['ess_ratio_bm']:.4f}")
    print(f"    MH log-weight Q99-Q1: {ess_stats['lw_q99_q1']:.4f}")
    print(f"    MH log-weight Q95-Q5: {ess_stats['lw_q95_q5']:.4f}")

    # Compare predicted vs actual
    print(f"\n  CERTIFICATE vs ACTUAL COMPARISON:")
    print(f"  {'Metric':<28} {'Certificate':>12} {'Actual MH':>12} {'Match?':>8}")
    print(f"  {'─'*64}")

    for rho in [0.01, 0.05, 0.10]:
        res = qstats['quantiles'].get(rho, {})
        if not res.get('feasible'):
            continue
        g = res['gamma_core']
        predicted_ess_ratio = g / (2 - g) if g < 2 else 1.0
        actual_ess_ratio = ess_stats['ess_ratio_bm']

        # Certificate predicted acceptance ≈ min acceptance on core
        predicted_accept_min = math.exp(-res['c_formal'])

        # Quantile oscillation from cert vs from MH chain
        cert_q = res['c_formal']
        mh_q = ess_stats['lw_q99_q1'] if rho == 0.01 else ess_stats['lw_q95_q5']

        match_ess = 'YES' if abs(predicted_ess_ratio - actual_ess_ratio) / max(actual_ess_ratio, 0.01) < 1.0 else 'no'

        rp = f'{rho*100:g}%'
        print(f"  γ_core (ρ={rp:<4}) → ESS ratio  {predicted_ess_ratio:>12.4f} "
              f"{actual_ess_ratio:>12.4f} {match_ess:>8}")

    # Acceptance rate comparison
    print(f"  {'Accept rate':<28} {'—':>12} {ess_stats['accept_rate']:>12.4f}")

    # ── Summary assessment ──
    q01 = qstats['quantiles'].get(0.01, {})
    q05 = qstats['quantiles'].get(0.05, {})

    print(f"\n  ★ SUMMARY:")
    v1_str = 'VACUOUS' if v1['gamma'] < 0.001 else f"gamma={v1['gamma']:.4f}"
    print(f"    V1: {v1_str}")
    if q01.get('feasible'):
        print(f"    V2 (ρ=1%): Ĉ={q01['c_formal']:.4f}, γ_core={q01['gamma_core']:.4f}")
    if q05.get('feasible'):
        print(f"    V2 (ρ=5%): Ĉ={q05['c_formal']:.4f}, γ_core={q05['gamma_core']:.4f}")
    print(f"    Actual ESS ratio: {ess_stats['ess_ratio_bm']:.4f}")
    print(f"    Actual accept rate: {ess_stats['accept_rate']:.4f}")

    # ── Save ──
    save_data = {
        'D': D,
        'hparams': {k: v for k, v in hp.items()},
        'quantile_stats': {
            k: v for k, v in qstats.items() if k != 'quantiles'
        },
        'quantiles': {str(k): v for k, v in qstats['quantiles'].items()},
        'v1': qstats['v1'],
        'ess_calibration': ess_stats,
    }
    with open(out_dir / f'benchmark_logreg_D{D}.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    return qstats, ess_stats


def make_calibration_plot(all_results, out_dir):
    """Plot ESS calibration across dimensions."""
    if len(all_results) < 1:
        return

    dims = sorted(all_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('FolT-MCMC V2: Bayesian LogReg Benchmark + ESS Calibration', fontsize=13)

    # Panel 1: Certificate C across methods/dimensions
    v1_c = [all_results[d][0]['v1']['cert'] for d in dims]
    v2_c01 = [all_results[d][0]['quantiles'].get(0.01, {}).get('c_formal', 0) for d in dims]
    v2_c05 = [all_results[d][0]['quantiles'].get(0.05, {}).get('c_formal', 0) for d in dims]

    x = np.arange(len(dims))
    w = 0.25
    axes[0].bar(x - w, v1_c, w, label='V1 covering', color='#CC4444')
    axes[0].bar(x, v2_c01, w, label='V2 core ρ=1%', color='#4488CC')
    axes[0].bar(x + w, v2_c05, w, label='V2 core ρ=5%', color='#44AA66')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'D={d}' for d in dims])
    axes[0].set_title('Certificate Value C')
    axes[0].legend(fontsize=8)

    # Panel 2: Predicted vs actual ESS ratio
    predicted_01 = []
    predicted_05 = []
    actual_ess = []
    for d in dims:
        qs, es = all_results[d]
        g01 = qs['quantiles'].get(0.01, {}).get('gamma_core', 0)
        g05 = qs['quantiles'].get(0.05, {}).get('gamma_core', 0)
        predicted_01.append(g01 / (2 - g01) if g01 > 0 else 0)
        predicted_05.append(g05 / (2 - g05) if g05 > 0 else 0)
        actual_ess.append(es['ess_ratio_bm'])

    axes[1].plot(dims, actual_ess, 'ko-', label='Actual ESS ratio', linewidth=2, markersize=8)
    axes[1].plot(dims, predicted_01, 's--', color='#4488CC', label='Predicted (ρ=1%)', linewidth=1.5)
    axes[1].plot(dims, predicted_05, '^--', color='#44AA66', label='Predicted (ρ=5%)', linewidth=1.5)
    axes[1].set_xlabel('Dimension D')
    axes[1].set_title('Predicted vs Actual ESS Ratio')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Acceptance rate
    accept = [all_results[d][1]['accept_rate'] for d in dims]
    axes[2].bar(x, accept, color='#4488CC')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'D={d}' for d in dims])
    axes[2].set_title('Actual Acceptance Rate')
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_dir / 'benchmark_logreg_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Calibration plot: {out_dir / 'benchmark_logreg_calibration.png'}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, nargs='+', default=[20, 50])
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--outdir', type=str, default='results_v2')
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    all_results = {}

    for D in args.dim:
        qstats, ess_stats = run_benchmark(D, quick=args.quick, out_dir=out_dir)
        all_results[D] = (qstats, ess_stats)

    # Cross-dimension summary
    if len(all_results) > 0:
        print(f"\n{'='*75}")
        print(f"  CROSS-DIMENSION SUMMARY — Bayesian Logistic Regression")
        print(f"{'='*75}")
        print(f"  {'D':>4} {'V1_γ':>10} {'Core_γ_1%':>12} {'Core_γ_5%':>12} "
              f"{'Accept':>8} {'ESS_ratio':>10}")
        print(f"  {'─'*60}")

        for D in sorted(all_results.keys()):
            qs, es = all_results[D]
            v1g = qs['v1']['gamma']
            g01 = qs['quantiles'].get(0.01, {}).get('gamma_core', 0)
            g05 = qs['quantiles'].get(0.05, {}).get('gamma_core', 0)
            print(f"  {D:>4} {v1g:>10.6f} {g01:>12.6f} {g05:>12.6f} "
                  f"{es['accept_rate']:>8.4f} {es['ess_ratio_bm']:>10.4f}")

    make_calibration_plot(all_results, out_dir)

    print(f"\n{'='*75}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
