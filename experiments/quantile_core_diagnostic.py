"""
FolT-MCMC V2 — Quantile Core Feasibility Diagnostic
=====================================================
Day 1 sprint: determine whether mass-aware quantile certificate
can break the D>=6 covering barrier.

For each dimension, this script:
  1. Trains FolT-OG-Anneal flow (reuses V1/exp02 training)
  2. Generates large certification sample
  3. Computes quantile statistics of r(z)
  4. Computes proxy core spectral gaps
  5. Compares with V1 covering certificate

Decision criteria:
  - D=10 banana: Q99-Q1 < 3  => STRONG GO
  - D=10 banana: Q99-Q1 < 5  => MARGINAL GO
  - D=10 banana: Q99-Q1 >= 5 => NO GO
  
Usage:
  conda activate lcnf
  cd C:\\FolT-MCMC\\experiments
  python quantile_core_diagnostic.py --dim 2 5 6 8 10 20
  python quantile_core_diagnostic.py --dim 2 5 10 --quick
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
# Target & Flow (identical to V1/exp02)
# ══════════════════════════════════════════════════════════════

class BananaTarget:
    def __init__(self, D=2, sigma1=2.0, sigma2=1.0, b=0.1):
        self.D = D
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.b = b

    def U(self, theta):
        t1, t2 = theta[:, 0], theta[:, 1]
        u = t1**2 / (2 * self.sigma1**2) + (t2 - self.b * t1**2)**2 / (2 * self.sigma2**2)
        if self.D > 2:
            u = u + 0.5 * (theta[:, 2:]**2).sum(dim=-1)
        return u

    def log_prob(self, theta):
        return -self.U(theta)

    def sample_mala(self, n, n_warmup=5000, step_size=0.05, thin=5):
        theta = torch.randn(self.D)
        samples = []
        for i in range(n * thin + n_warmup):
            theta_r = theta.detach().requires_grad_(True)
            lp = self.log_prob(theta_r.unsqueeze(0)).squeeze()
            lp.backward()
            grad = theta_r.grad.detach()
            noise = torch.randn(self.D)
            proposal = theta + step_size * grad + math.sqrt(2 * step_size) * noise
            lp_prop = self.log_prob(proposal.unsqueeze(0)).squeeze().item()
            lp_curr = self.log_prob(theta.unsqueeze(0)).squeeze().item()
            prop_r = proposal.detach().requires_grad_(True)
            lp_p = self.log_prob(prop_r.unsqueeze(0)).squeeze()
            lp_p.backward()
            grad_prop = prop_r.grad.detach()
            log_q_fwd = -0.25/step_size * ((proposal - theta - step_size*grad)**2).sum().item()
            log_q_bwd = -0.25/step_size * ((theta - proposal - step_size*grad_prop)**2).sum().item()
            log_alpha = (lp_prop - lp_curr) + (log_q_bwd - log_q_fwd)
            if math.log(np.random.random() + 1e-30) < min(0, log_alpha):
                theta = proposal.detach()
            if i >= n_warmup and (i - n_warmup) % thin == 0:
                samples.append(theta.clone())
        return torch.stack(samples[:n])

    def sample_rejection(self, n):
        assert self.D == 2
        samples = []
        while sum(len(s) for s in samples) < n:
            proposal = torch.randn(n * 5, 2)
            proposal[:, 0] *= self.sigma1 * 1.5
            proposal[:, 1] *= 3.0
            log_p = self.log_prob(proposal)
            log_q = -0.5 * (proposal[:, 0]**2 / (self.sigma1*1.5)**2
                          + proposal[:, 1]**2 / 3.0**2)
            log_ratio = log_p - log_q
            log_M = log_ratio.max()
            accept = torch.rand(len(proposal)).log() < (log_ratio - log_M)
            samples.append(proposal[accept])
        return torch.cat(samples)[:n]

    def generate_samples(self, n):
        if self.D == 2:
            return self.sample_rejection(n)
        else:
            return self.sample_mala(n, n_warmup=max(5000, n//2))


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


# ══════════════════════════════════════════════════════════════
# Training (FolT-OG-Anneal, compact version)
# ══════════════════════════════════════════════════════════════

def train_flow(target, D, hp, verbose=True):
    """Train FolT-OG-Anneal and return trained flow."""
    train_samples = target.generate_samples(hp['n_train']).to(DEVICE)
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
# Quantile Core Diagnostic
# ══════════════════════════════════════════════════════════════

def quantile_diagnostic(flow, target, D, n_cert=200000):
    """
    Core diagnostic: compute quantile statistics of r(z).
    
    Returns dict with all quantile info needed for V2 feasibility decision.
    """
    flow.eval()
    
    # Sample in batches to handle large n_cert
    batch_size = 50000
    r_all = []
    with torch.no_grad():
        for i in range(0, n_cert, batch_size):
            n_batch = min(batch_size, n_cert - i)
            z = torch.randn(n_batch, D, device=DEVICE)
            r = compute_residual(flow, target, z)
            r_all.append(r.cpu())
    
    r = torch.cat(r_all).numpy()
    n = len(r)
    
    # ── Basic statistics ──
    stats = {
        'n_cert': n,
        'D': D,
        'mean': float(np.mean(r)),
        'std': float(np.std(r)),
        'var': float(np.var(r)),
        'min': float(np.min(r)),
        'max': float(np.max(r)),
        'full_osc': float(np.max(r) - np.min(r)),
    }
    
    # ── Quantiles ──
    rho_levels = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25]
    
    for rho in rho_levels:
        q_lo = np.quantile(r, rho)
        q_hi = np.quantile(r, 1 - rho)
        c_rho = q_hi - q_lo
        
        # Proxy core spectral gap (NOT rigorous — diagnostic only)
        if c_rho > 500:
            gamma_proxy = 0.0
        else:
            gamma_proxy = 2.0 / (1.0 + math.exp(c_rho))
        
        # DKW correction for finite-sample quantile estimation
        # Massart's tight DKW: P(sup|F_n - F| > t) <= 2*exp(-2*n*t^2)
        # For quantile at level rho with confidence 1-zeta:
        # |Q_hat(rho) - Q(rho)| <= eps_dkw where eps_dkw from order stats
        zeta = 0.05
        dkw_eps = math.sqrt(math.log(2.0 / zeta) / (2 * n))
        # Quantile confidence: rho +/- dkw_eps
        rho_lo_adj = max(rho - dkw_eps, 0.0001)
        rho_hi_adj = min(1 - rho + dkw_eps, 0.9999)
        q_lo_adj = np.quantile(r, rho_lo_adj)
        q_hi_adj = np.quantile(r, rho_hi_adj)
        c_rho_adj = q_hi_adj - q_lo_adj
        
        if c_rho_adj > 500:
            gamma_adj = 0.0
        else:
            gamma_adj = 2.0 / (1.0 + math.exp(c_rho_adj))
        
        rho_pct = f'{rho*100:g}'
        stats[f'Q{rho_pct}'] = float(q_lo)
        stats[f'Q{100-rho*100:g}'] = float(q_hi)
        stats[f'C_rho_{rho_pct}'] = float(c_rho)
        stats[f'gamma_core_{rho_pct}'] = float(gamma_proxy)
        stats[f'C_rho_{rho_pct}_adj'] = float(c_rho_adj)
        stats[f'gamma_core_{rho_pct}_adj'] = float(gamma_adj)
    
    # ── V1 covering certificate for comparison ──
    R_alpha = math.sqrt(chi2.ppf(0.95, df=D))
    log_term = math.log(2.0 / 0.05) / (2 * n)
    eps_star = R_alpha * (log_term + math.sqrt(log_term)) ** (1.0 / D)
    
    # Quick gradient supremum estimate
    n_grad = min(2000, n)
    z_g = torch.randn(n_grad, D, device=DEVICE).requires_grad_(True)
    r_g = compute_residual(flow, target, z_g)
    grad_r = torch.autograd.grad(r_g.sum(), z_g)[0]
    grad_sup = grad_r.norm(dim=-1).max().item()
    
    sample_osc = stats['full_osc']
    v1_cert = sample_osc + 2 * grad_sup * eps_star
    if v1_cert > 500:
        v1_gamma = 0.0
    else:
        v1_gamma = 2.0 / (1.0 + math.exp(v1_cert))
    
    stats['v1_cert'] = float(v1_cert)
    stats['v1_gamma'] = float(v1_gamma)
    stats['v1_eps_star'] = float(eps_star)
    stats['v1_grad_sup'] = float(grad_sup)
    stats['v1_correction'] = float(2 * grad_sup * eps_star)
    
    return stats


def print_diagnostic(stats):
    """Pretty-print quantile diagnostic results."""
    D = stats['D']
    
    print(f"\n  {'─'*75}")
    print(f"  QUANTILE CORE DIAGNOSTIC — D={D} (n={stats['n_cert']:,})")
    print(f"  {'─'*75}")
    
    print(f"\n  Basic: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
          f"var={stats['var']:.4f}")
    print(f"  Full oscillation: {stats['full_osc']:.4f} "
          f"(min={stats['min']:.4f}, max={stats['max']:.4f})")
    
    print(f"\n  V1 covering certificate:")
    print(f"    C_v1 = {stats['v1_cert']:.4f} "
          f"(osc={stats['full_osc']:.4f} + corr={stats['v1_correction']:.4f})")
    print(f"    gamma_v1 = {stats['v1_gamma']:.6f}")
    print(f"    eps* = {stats['v1_eps_star']:.4f}, grad_sup = {stats['v1_grad_sup']:.4f}")
    
    print(f"\n  Quantile core certificates (proxy, not yet rigorous):")
    print(f"  {'rho':<8} {'Q_lo':>8} {'Q_hi':>8} {'C_rho':>8} "
          f"{'gamma_core':>12} {'C_adj':>8} {'gamma_adj':>12} {'vs V1':>8}")
    print(f"  {'─'*76}")
    
    rho_levels = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25]
    for rho in rho_levels:
        rp = f'{rho*100:g}'
        c_rho = stats[f'C_rho_{rp}']
        g_core = stats[f'gamma_core_{rp}']
        c_adj = stats[f'C_rho_{rp}_adj']
        g_adj = stats[f'gamma_core_{rp}_adj']
        q_lo = stats[f'Q{rp}']
        q_hi = stats[f'Q{100-rho*100:g}']
        
        v1_g = stats['v1_gamma']
        if v1_g > 0:
            ratio = g_adj / v1_g
            ratio_str = f'{ratio:.1f}x'
        else:
            ratio_str = 'inf' if g_adj > 0 else '—'
        
        print(f"  {rho:<8.3f} {q_lo:>8.4f} {q_hi:>8.4f} {c_rho:>8.4f} "
              f"{g_core:>12.6f} {c_adj:>8.4f} {g_adj:>12.6f} {ratio_str:>8}")
    
    # Decision signal
    c99_1 = stats.get('C_rho_1', None)
    c95_5 = stats.get('C_rho_5', None)
    
    print(f"\n  ★ KEY METRICS:")
    print(f"    Q95-Q5  = {c95_5:.4f}" if c95_5 else "    Q95-Q5  = N/A")
    print(f"    Q99-Q1  = {c99_1:.4f}" if c99_1 else "    Q99-Q1  = N/A")
    print(f"    Full osc = {stats['full_osc']:.4f}")
    print(f"    Ratio (Q99-Q1)/(full osc) = {c99_1/stats['full_osc']:.2f}" 
          if c99_1 and stats['full_osc'] > 0 else "")
    
    if D >= 10:
        if c99_1 is not None:
            if c99_1 < 3:
                print(f"\n  ✅ STRONG GO: Q99-Q1 = {c99_1:.2f} < 3")
            elif c99_1 < 5:
                print(f"\n  ⚠️  MARGINAL GO: Q99-Q1 = {c99_1:.2f} < 5")
            else:
                print(f"\n  ❌ NO GO: Q99-Q1 = {c99_1:.2f} >= 5")


def make_diagnostic_plot(all_stats, out_dir):
    """Summary plot across all dimensions."""
    dims = sorted(all_stats.keys())
    
    if len(dims) < 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FolT-MCMC V2: Quantile Core Feasibility Diagnostic', fontsize=14)
    
    # Panel 1: Oscillation vs dimension
    full_osc = [all_stats[d]['full_osc'] for d in dims]
    c99_1 = [all_stats[d].get('C_rho_1', 0) for d in dims]
    c95_5 = [all_stats[d].get('C_rho_5', 0) for d in dims]
    c90_10 = [all_stats[d].get('C_rho_10', 0) for d in dims]
    
    x = np.arange(len(dims))
    w = 0.2
    axes[0,0].bar(x - 1.5*w, full_osc, w, label='Full osc', color='#CC4444')
    axes[0,0].bar(x - 0.5*w, c99_1, w, label='Q99-Q1', color='#4488CC')
    axes[0,0].bar(x + 0.5*w, c95_5, w, label='Q95-Q5', color='#44AA66')
    axes[0,0].bar(x + 1.5*w, c90_10, w, label='Q90-Q10', color='#CCAA44')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels([f'D={d}' for d in dims])
    axes[0,0].set_title('Oscillation: Full vs Quantile Core')
    axes[0,0].legend(fontsize=8)
    axes[0,0].set_ylabel('Oscillation')
    
    # Panel 2: Proxy gamma vs dimension
    v1_gammas = [all_stats[d]['v1_gamma'] for d in dims]
    g99_1 = [all_stats[d].get('gamma_core_1_adj', 0) for d in dims]
    g95_5 = [all_stats[d].get('gamma_core_5_adj', 0) for d in dims]
    
    axes[0,1].semilogy(dims, v1_gammas, 'o-', color='#CC4444', label='V1 covering', linewidth=2)
    axes[0,1].semilogy(dims, g99_1, 's-', color='#4488CC', label='Core rho=1%', linewidth=2)
    axes[0,1].semilogy(dims, g95_5, '^-', color='#44AA66', label='Core rho=5%', linewidth=2)
    axes[0,1].set_xlabel('Dimension D')
    axes[0,1].set_title('Spectral Gap: V1 vs Quantile Core')
    axes[0,1].legend(fontsize=8)
    axes[0,1].set_ylabel('gamma')
    axes[0,1].grid(True, alpha=0.3)
    
    # Panel 3: Ratio (Q99-Q1)/(full osc)
    ratios = [c99_1[i] / max(full_osc[i], 1e-10) for i in range(len(dims))]
    axes[1,0].bar(x, ratios, color='#4488CC')
    axes[1,0].axhline(y=1.0, color='#CC4444', linestyle='--', alpha=0.5, label='full osc')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels([f'D={d}' for d in dims])
    axes[1,0].set_title('Ratio: (Q99-Q1) / (Full Oscillation)')
    axes[1,0].set_ylabel('Ratio')
    axes[1,0].legend(fontsize=8)
    
    # Panel 4: Residual distribution for largest D
    d_max = dims[-1]
    # Re-sample for histogram (use stored stats for annotation)
    axes[1,1].text(0.5, 0.5, 
                   f'D={d_max}\n'
                   f'std(r) = {all_stats[d_max]["std"]:.3f}\n'
                   f'Full osc = {all_stats[d_max]["full_osc"]:.3f}\n'
                   f'Q99-Q1 = {all_stats[d_max].get("C_rho_1", 0):.3f}\n'
                   f'Q95-Q5 = {all_stats[d_max].get("C_rho_5", 0):.3f}\n'
                   f'V1 C = {all_stats[d_max]["v1_cert"]:.3f}\n'
                   f'V1 gamma = {all_stats[d_max]["v1_gamma"]:.6f}\n'
                   f'Core gamma (1%) = {all_stats[d_max].get("gamma_core_1_adj", 0):.6f}',
                   transform=axes[1,1].transAxes,
                   fontsize=11, verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1,1].set_title(f'Summary for D={d_max}')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'quantile_feasibility.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Summary plot: {out_dir / 'quantile_feasibility.png'}")


# ══════════════════════════════════════════════════════════════
# Experiment Runner
# ══════════════════════════════════════════════════════════════

HPARAMS = {
    2:  dict(n_layers=8,  hidden_dim=64,  n_train=20000, n_epochs=5000,  n_cert=200000),
    5:  dict(n_layers=10, hidden_dim=128, n_train=30000, n_epochs=5000,  n_cert=200000),
    6:  dict(n_layers=12, hidden_dim=128, n_train=40000, n_epochs=6000,  n_cert=200000),
    8:  dict(n_layers=12, hidden_dim=192, n_train=50000, n_epochs=8000,  n_cert=200000),
    10: dict(n_layers=12, hidden_dim=128, n_train=50000, n_epochs=8000,  n_cert=200000),
    20: dict(n_layers=16, hidden_dim=256, n_train=80000, n_epochs=10000, n_cert=200000),
}


def run_diagnostic(D, quick=False, out_dir=None):
    hp = HPARAMS.get(D, HPARAMS[10]).copy()
    if quick:
        hp['n_epochs'] = min(hp['n_epochs'], 1500)
        hp['n_train'] = min(hp['n_train'], 8000)
        hp['n_cert'] = 50000

    if out_dir is None:
        out_dir = Path('results_v2')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"  QUANTILE DIAGNOSTIC — D={D}")
    print(f"  Training: {hp['n_epochs']} epochs, {hp['n_train']} samples")
    print(f"  Certification: {hp['n_cert']:,} samples")
    print(f"{'='*75}")

    target = BananaTarget(D=D)

    # Train
    print(f"\n  Training flow...")
    t0 = time.time()
    flow = train_flow(target, D, hp)
    print(f"  Training done in {time.time()-t0:.0f}s")

    # Save flow for potential reuse
    torch.save(flow.state_dict(), out_dir / f'flow_D{D}.pt')

    # Diagnostic
    print(f"\n  Running quantile diagnostic ({hp['n_cert']:,} samples)...")
    t0 = time.time()
    stats = quantile_diagnostic(flow, target, D, n_cert=hp['n_cert'])
    print(f"  Diagnostic done in {time.time()-t0:.1f}s")

    # Print
    print_diagnostic(stats)

    # Save
    with open(out_dir / f'quantile_D{D}.json', 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description='FolT-MCMC V2 Quantile Feasibility')
    parser.add_argument('--dim', type=int, nargs='+', default=[2, 5, 6, 8, 10])
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer epochs, smaller cert sample')
    parser.add_argument('--outdir', type=str, default='results_v2')
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    all_stats = {}

    for D in args.dim:
        stats = run_diagnostic(D, quick=args.quick, out_dir=out_dir)
        all_stats[D] = stats

    # Summary table
    print(f"\n{'='*75}")
    print(f"  FEASIBILITY SUMMARY")
    print(f"{'='*75}")
    print(f"  {'D':>4} {'full_osc':>10} {'Q99-Q1':>10} {'Q95-Q5':>10} "
          f"{'V1_gamma':>12} {'core_1%':>12} {'core_5%':>12} {'verdict':>10}")
    print(f"  {'─'*82}")
    
    for D in sorted(all_stats.keys()):
        s = all_stats[D]
        c99 = s.get('C_rho_1', 0)
        c95 = s.get('C_rho_5', 0)
        g_v1 = s['v1_gamma']
        g_c1 = s.get('gamma_core_1_adj', 0)
        g_c5 = s.get('gamma_core_5_adj', 0)
        
        if D >= 10:
            if c99 < 3:
                verdict = 'STRONG GO'
            elif c99 < 5:
                verdict = 'MARGINAL'
            else:
                verdict = 'NO GO'
        elif D >= 6:
            if c99 < 3:
                verdict = 'STRONG'
            else:
                verdict = 'OK'
        else:
            verdict = 'baseline'
        
        print(f"  {D:>4} {s['full_osc']:>10.4f} {c99:>10.4f} {c95:>10.4f} "
              f"{g_v1:>12.6f} {g_c1:>12.6f} {g_c5:>12.6f} {verdict:>10}")

    # Plot
    make_diagnostic_plot(all_stats, out_dir)

    print(f"\n  All results saved to {out_dir}/")
    print(f"  Flow checkpoints: flow_D*.pt")
    print(f"  Quantile data: quantile_D*.json")

    print(f"\n{'='*75}")
    print("  DIAGNOSTIC COMPLETE")
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
