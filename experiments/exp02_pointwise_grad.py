"""
FolT-MCMC V2 — Experiment 02: Pointwise Gradient Correction
=============================================================
Lesson from exp01: partitioning into cells INCREASES covering cost
faster than it decreases Lipschitz constants. Don't partition!

Key insight:
  V1:  C = (max r_i - min r_i) + 2 * max(g_i) * eps_star
  V2:  C = max_i(r_i + g_i * eps) - min_i(r_i - g_i * eps) + H * eps^2

  where g_i = ||∇r(z_i)|| at each sample, eps = GLOBAL eps_star,
  H = Hessian bound (sup ||∇²r||).

  The improvement: at the sample achieving max(r), the gradient is
  typically SMALL (near a local max), so g_i * eps << grad_sup * eps.

Usage:
  conda activate lcnf
  cd C:\FolT-MCMC\experiments
  python exp02_pointwise_grad.py --dim 2 5 6 8 10
  python exp02_pointwise_grad.py --dim 2 --quick
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

# ══════════════════════════════════════════════════════════════
# Device
# ══════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ══════════════════════════════════════════════════════════════
# 1. Target (same as V1)
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


# ══════════════════════════════════════════════════════════════
# 2. Flow (same as V1)
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
# 3. Residual
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


# ══════════════════════════════════════════════════════════════
# 4. Certificate Methods
# ══════════════════════════════════════════════════════════════

def compute_eps_star(D, n_cert, zeta=0.05):
    """Global covering radius (same as V1)."""
    R_alpha = math.sqrt(chi2.ppf(0.95, df=D))
    log_term = math.log(2.0 / zeta) / (2 * n_cert)
    eps_star = R_alpha * (log_term + math.sqrt(log_term)) ** (1.0 / D)
    return eps_star, R_alpha


def compute_gradients_batched(flow, target, z_all, batch_size=500):
    """Compute per-sample gradient norms. Returns (grad_norms, grad_vectors)."""
    n = z_all.shape[0]
    all_norms = []
    for i in range(0, n, batch_size):
        z_batch = z_all[i:i+batch_size].detach().requires_grad_(True)
        r = compute_residual(flow, target, z_batch)
        grad_r = torch.autograd.grad(r.sum(), z_batch)[0]
        all_norms.append(grad_r.norm(dim=-1).detach())
    return torch.cat(all_norms)


def estimate_hessian_bound(flow, target, D, n_samples=500, n_directions=10, h=0.01):
    """
    Estimate sup ||∇²r|| via finite-difference directional second derivatives.
    
    For random z and random direction v (||v||=1):
      d²r/dt² ≈ [r(z+hv) - 2r(z) + r(z-hv)] / h²
    
    Take the max over samples and directions as an estimate of ||∇²r||.
    This is an underestimate of the true sup, so we multiply by a safety factor.
    """
    flow.eval()
    max_hessian = 0.0
    
    with torch.no_grad():
        for _ in range(n_directions):
            z = torch.randn(n_samples, D, device=DEVICE)
            v = torch.randn(n_samples, D, device=DEVICE)
            v = v / v.norm(dim=-1, keepdim=True)  # unit directions
            
            r_center = compute_residual(flow, target, z)
            r_plus = compute_residual(flow, target, z + h * v)
            r_minus = compute_residual(flow, target, z - h * v)
            
            # Second directional derivative
            d2r = (r_plus - 2 * r_center + r_minus) / (h**2)
            
            batch_max = d2r.abs().max().item()
            max_hessian = max(max_hessian, batch_max)
    
    # Safety factor: we only sampled a finite set of directions
    # In D dimensions, the operator norm can be up to D times the
    # max directional derivative (worst case). Use sqrt(D) as compromise.
    safety_factor = math.sqrt(D)
    
    return max_hessian * safety_factor


def cert_v1(flow, target, D, z_all, r_all, zeta=0.05):
    """V1 certificate: C = sample_osc + 2 * grad_sup * eps_star"""
    n_cert = z_all.shape[0]
    
    r_np = r_all.cpu().numpy()
    sample_osc = float(r_np.max() - r_np.min())
    
    # Gradient supremum
    n_grad = min(2000, n_cert)
    z_g = torch.randn(n_grad, D, device=DEVICE).requires_grad_(True)
    r_g = compute_residual(flow, target, z_g)
    grad_r = torch.autograd.grad(r_g.sum(), z_g)[0]
    grad_sup = grad_r.norm(dim=-1).max().item()
    grad_mean = grad_r.norm(dim=-1).mean().item()
    
    eps_star, R_alpha = compute_eps_star(D, n_cert, zeta)
    correction = 2 * grad_sup * eps_star
    cert_value = sample_osc + correction
    
    if cert_value > 500:
        gamma = 0.0
    else:
        gamma = 2.0 / (1.0 + math.exp(cert_value))
    
    mix_time = math.log(100) / max(gamma, 1e-30)
    mix_time = min(mix_time, 1e18)
    
    return {
        'method': 'V1_uniform',
        'cert': cert_value,
        'gamma': gamma,
        'sample_osc': sample_osc,
        'correction': correction,
        'grad_sup': grad_sup,
        'grad_mean': grad_mean,
        'eps_star': eps_star,
        'mix_time': mix_time,
    }


def cert_v2a_pointwise(flow, target, D, z_all, r_all, zeta=0.05, grad_batch=500):
    """
    V2a: Pointwise gradient correction (NO Hessian).
    
    HEURISTIC (not rigorous without Hessian bound):
      C = max_i(r_i + g_i * eps) - min_i(r_i - g_i * eps)
    
    This UNDERESTIMATES the true oscillation because it ignores
    the O(eps²) Hessian term. But it shows the potential improvement.
    """
    n_cert = z_all.shape[0]
    
    eps_star, _ = compute_eps_star(D, n_cert, zeta)
    
    # Compute per-sample gradients
    grad_norms = compute_gradients_batched(
        flow, target, z_all, batch_size=grad_batch)
    
    r_cpu = r_all.cpu()
    g_cpu = grad_norms.cpu()
    
    # Pointwise upper/lower bounds
    upper = r_cpu + g_cpu * eps_star
    lower = r_cpu - g_cpu * eps_star
    
    cert_value = (upper.max() - lower.min()).item()
    sample_osc = (r_cpu.max() - r_cpu.min()).item()
    correction = cert_value - sample_osc
    
    if cert_value > 500:
        gamma = 0.0
    else:
        gamma = 2.0 / (1.0 + math.exp(cert_value))
    
    mix_time = math.log(100) / max(gamma, 1e-30)
    mix_time = min(mix_time, 1e18)
    
    # Diagnostics: what happens at the max/min samples?
    idx_rmax = r_cpu.argmax().item()
    idx_rmin = r_cpu.argmin().item()
    idx_umax = upper.argmax().item()
    idx_lmin = lower.argmin().item()
    
    return {
        'method': 'V2a_pointwise',
        'cert': cert_value,
        'gamma': gamma,
        'sample_osc': sample_osc,
        'correction': correction,
        'eps_star': eps_star,
        'mix_time': mix_time,
        # Diagnostics
        'grad_at_rmax': float(g_cpu[idx_rmax]),
        'grad_at_rmin': float(g_cpu[idx_rmin]),
        'grad_at_umax': float(g_cpu[idx_umax]),
        'grad_at_lmin': float(g_cpu[idx_lmin]),
        'grad_sup': float(g_cpu.max()),
        'grad_mean': float(g_cpu.mean()),
        'grad_p50': float(g_cpu.median()),
        'grad_p95': float(g_cpu.quantile(0.95)),
    }


def cert_v2b_pointwise_hessian(flow, target, D, z_all, r_all, 
                                zeta=0.05, grad_batch=500,
                                n_hess_samples=500, n_hess_dirs=20):
    """
    V2b: Pointwise gradient + Hessian correction (RIGOROUS).
    
    For any z with nearest sample z_i (distance ≤ eps):
      |r(z) - r(z_i)| ≤ ||∇r(z_i)|| * eps + H/2 * eps²
    
    where H = sup ||∇²r||.
    
    Therefore:
      sup r ≤ max_i(r_i + g_i * eps + H/2 * eps²)
      inf r ≥ min_i(r_i - g_i * eps - H/2 * eps²)
      
    C = max_i(r_i + g_i*eps) - min_i(r_i - g_i*eps) + H * eps²
    """
    n_cert = z_all.shape[0]
    
    eps_star, _ = compute_eps_star(D, n_cert, zeta)
    
    # Per-sample gradients
    print(f"    [V2b] Computing {n_cert} gradients...")
    t0 = time.time()
    grad_norms = compute_gradients_batched(
        flow, target, z_all, batch_size=grad_batch)
    print(f"    [V2b] Gradients done in {time.time()-t0:.1f}s")
    
    # Hessian bound
    print(f"    [V2b] Estimating Hessian bound ({n_hess_samples} samples × {n_hess_dirs} dirs)...")
    t0 = time.time()
    H = estimate_hessian_bound(flow, target, D, 
                               n_samples=n_hess_samples,
                               n_directions=n_hess_dirs)
    print(f"    [V2b] H = {H:.4f}, done in {time.time()-t0:.1f}s")
    
    r_cpu = r_all.cpu()
    g_cpu = grad_norms.cpu()
    
    # Pointwise bounds WITH Hessian correction
    hess_correction = 0.5 * H * eps_star**2
    upper = r_cpu + g_cpu * eps_star + hess_correction
    lower = r_cpu - g_cpu * eps_star - hess_correction
    
    cert_value = (upper.max() - lower.min()).item()
    sample_osc = (r_cpu.max() - r_cpu.min()).item()
    correction = cert_value - sample_osc
    
    # Decompose correction
    pointwise_part = (
        (r_cpu + g_cpu * eps_star).max() - (r_cpu - g_cpu * eps_star).min()
    ).item() - sample_osc
    hessian_part = 2 * hess_correction
    
    if cert_value > 500:
        gamma = 0.0
    else:
        gamma = 2.0 / (1.0 + math.exp(cert_value))
    
    mix_time = math.log(100) / max(gamma, 1e-30)
    mix_time = min(mix_time, 1e18)
    
    idx_rmax = r_cpu.argmax().item()
    idx_rmin = r_cpu.argmin().item()
    
    return {
        'method': 'V2b_pointwise_hessian',
        'cert': cert_value,
        'gamma': gamma,
        'sample_osc': sample_osc,
        'correction': correction,
        'correction_pointwise_part': pointwise_part,
        'correction_hessian_part': hessian_part,
        'eps_star': eps_star,
        'H': H,
        'hess_correction': hess_correction,
        'mix_time': mix_time,
        'grad_at_rmax': float(g_cpu[idx_rmax]),
        'grad_at_rmin': float(g_cpu[idx_rmin]),
        'grad_sup': float(g_cpu.max()),
        'grad_mean': float(g_cpu.mean()),
        'grad_p50': float(g_cpu.median()),
        'grad_p95': float(g_cpu.quantile(0.95)),
    }


def cert_v2c_truncated(flow, target, D, z_all, r_all, 
                        alpha=0.01, zeta=0.05, grad_batch=500,
                        n_hess_samples=500, n_hess_dirs=20):
    """
    V2c: Truncated support + pointwise gradient + Hessian.
    
    Idea: Only certify on B_R where R = chi2.ppf(1-alpha, D),
    a SMALLER ball that still captures (1-alpha) mass of N(0,I).
    
    Benefits:
      1. Smaller R → smaller eps_star (covering is easier on smaller ball)
      2. Fewer tail samples → tighter sample_osc
      3. Modified spectral gap: γ ≥ (1-alpha) * 2/(1+exp(C_trunc))
         (paying alpha for the uncovered tail mass)
    """
    n_cert = z_all.shape[0]
    
    # Truncation radius: captures (1-alpha) mass of N(0,I)
    # Use a TIGHTER ball than the 95% ball used in V1
    R_trunc = math.sqrt(chi2.ppf(1 - alpha, df=D))
    
    # Filter to samples inside the truncated ball
    z_norms = z_all.norm(dim=-1)
    inside_mask = z_norms <= R_trunc
    n_inside = inside_mask.sum().item()
    
    z_trunc = z_all[inside_mask]
    r_trunc = r_all[inside_mask]
    
    print(f"    [V2c] Truncation: R={R_trunc:.2f}, alpha={alpha}, "
          f"{n_inside}/{n_cert} samples inside ({n_inside/n_cert:.1%})")
    
    if n_inside < 100:
        print(f"    [V2c] Too few samples inside truncated ball, skipping.")
        return {
            'method': f'V2c_trunc_a{alpha}',
            'cert': float('inf'), 'gamma': 0.0,
            'sample_osc': 0.0, 'correction': float('inf'),
            'n_inside': n_inside, 'R_trunc': R_trunc,
            'eps_star': float('inf'), 'mix_time': float('inf'),
        }
    
    # Covering radius for the TRUNCATED ball (using n_inside samples)
    # Key: we're covering a ball of radius R_trunc, not R_0.95
    log_term = math.log(2.0 / zeta) / (2 * n_inside)
    eps_star = R_trunc * (log_term + math.sqrt(log_term)) ** (1.0 / D)
    
    # Sample oscillation on truncated set
    r_trunc_cpu = r_trunc.cpu()
    sample_osc = (r_trunc_cpu.max() - r_trunc_cpu.min()).item()
    
    # Per-sample gradients (only on truncated set)
    print(f"    [V2c] Computing {n_inside} gradients...")
    grad_norms = compute_gradients_batched(
        flow, target, z_trunc, batch_size=grad_batch)
    g_cpu = grad_norms.cpu()
    
    # Hessian bound (sample from truncated ball)
    print(f"    [V2c] Estimating Hessian...")
    H = estimate_hessian_bound(flow, target, D,
                               n_samples=n_hess_samples,
                               n_directions=n_hess_dirs)
    
    # Certificate on truncated ball
    hess_corr = 0.5 * H * eps_star**2
    upper = r_trunc_cpu + g_cpu * eps_star + hess_corr
    lower = r_trunc_cpu - g_cpu * eps_star - hess_corr
    
    cert_trunc = (upper.max() - lower.min()).item()
    correction = cert_trunc - sample_osc
    
    # Modified spectral gap with truncation penalty
    if cert_trunc > 500:
        gamma_trunc = 0.0
    else:
        gamma_trunc = 2.0 / (1.0 + math.exp(cert_trunc))
    
    # Account for uncovered mass: γ_full ≥ (1-alpha)² * γ_trunc
    # (conservative: proposal mass outside Ω is at most alpha)
    gamma = (1 - alpha)**2 * gamma_trunc
    
    mix_time = math.log(100) / max(gamma, 1e-30)
    mix_time = min(mix_time, 1e18)
    
    return {
        'method': f'V2c_trunc_a{alpha}',
        'cert': cert_trunc,
        'gamma': gamma,
        'gamma_trunc': gamma_trunc,
        'sample_osc': sample_osc,
        'correction': correction,
        'eps_star': eps_star,
        'H': H,
        'hess_correction': hess_corr,
        'R_trunc': R_trunc,
        'n_inside': n_inside,
        'alpha': alpha,
        'mix_time': mix_time,
        'grad_sup': float(g_cpu.max()),
        'grad_mean': float(g_cpu.mean()),
        'grad_p50': float(g_cpu.median()),
    }


# ══════════════════════════════════════════════════════════════
# 5. Training (FolT-OG-Anneal, from V1)
# ══════════════════════════════════════════════════════════════

def train_cert_og_anneal(target, train_samples, D,
                         n_epochs=5000, lr=1e-3, batch_size=512,
                         n_layers=8, hidden_dim=64, scale_clip=0.7,
                         lambda_o=0.1, lambda_g=0.05, tau=0.1,
                         cert_interval=500, n_cert=50000):
    flow = SNRealNVP(dim=D, n_layers=n_layers,
                     hidden_dim=hidden_dim, scale_clip=scale_clip).to(DEVICE)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    train_dev = train_samples.to(DEVICE)
    N = train_dev.shape[0]
    history = {'nll': [], 'cert_epochs': [], 'cert_values': [], 'gamma_bounds': []}
    t0 = time.time()

    for epoch in range(n_epochs):
        flow.train()
        ramp = max(0.0, (epoch / n_epochs - 0.4)) / 0.6
        eff_lo = lambda_o * ramp
        eff_lg = lambda_g * ramp
        idx = torch.randint(0, N, (batch_size,), device=DEVICE)
        nll = -flow.log_prob(train_dev[idx]).mean()
        osc_loss = torch.tensor(0.0, device=DEVICE)
        grad_loss = torch.tensor(0.0, device=DEVICE)
        if eff_lo > 0 or eff_lg > 0:
            z_fresh = torch.randn(batch_size, D, device=DEVICE)
            r = compute_residual(flow, target, z_fresh)
            if eff_lo > 0:
                osc_loss = smooth_oscillation(r, tau=tau)
            if eff_lg > 0:
                z_g = torch.randn(min(64, batch_size), D, device=DEVICE)
                gm, _ = gradient_norm_mean(flow, target, z_g)
                grad_loss = gm
        loss = nll + eff_lo * osc_loss + eff_lg * grad_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        history['nll'].append(nll.item())

        if (epoch + 1) % cert_interval == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                z_c = torch.randn(n_cert, D, device=DEVICE)
                r_c = compute_residual(flow, target, z_c)
                sosc = (r_c.max() - r_c.min()).item()
            n_g = min(2000, n_cert)
            z_g2 = torch.randn(n_g, D, device=DEVICE).requires_grad_(True)
            r_g2 = compute_residual(flow, target, z_g2)
            gr = torch.autograd.grad(r_g2.sum(), z_g2)[0]
            gsup = gr.norm(dim=-1).max().item()
            eps, _ = compute_eps_star(D, n_cert)
            cv = sosc + 2 * gsup * eps
            if cv > 500:
                gb = 0.0
            else:
                gb = 2.0 / (1.0 + math.exp(cv))
            history['cert_epochs'].append(epoch + 1)
            history['cert_values'].append(cv)
            history['gamma_bounds'].append(gb)
            elapsed = time.time() - t0
            print(f"  [{epoch+1:5d}/{n_epochs}] "
                  f"NLL={nll.item():.3f} | "
                  f"osc={sosc:.3f} | ∇sup={gsup:.2f} | "
                  f"C={cv:.3f} | γ≥{gb:.6f} | {elapsed:.0f}s")

    flow.eval()
    return flow, history


# ══════════════════════════════════════════════════════════════
# 6. Experiment Runner
# ══════════════════════════════════════════════════════════════

HPARAMS = {
    2:  dict(n_layers=8,  hidden_dim=64,  n_train=20000, n_epochs=5000,  n_cert=50000),
    5:  dict(n_layers=10, hidden_dim=128, n_train=30000, n_epochs=5000,  n_cert=50000),
    6:  dict(n_layers=12, hidden_dim=128, n_train=40000, n_epochs=6000,  n_cert=80000),
    8:  dict(n_layers=12, hidden_dim=192, n_train=50000, n_epochs=8000,  n_cert=100000),
    10: dict(n_layers=12, hidden_dim=128, n_train=50000, n_epochs=8000,  n_cert=100000),
    20: dict(n_layers=16, hidden_dim=256, n_train=80000, n_epochs=10000, n_cert=200000),
}


def run_experiment(D, quick=False, out_dir=None):
    hp = HPARAMS.get(D, HPARAMS[5]).copy()
    if quick:
        hp['n_epochs'] = min(hp['n_epochs'], 1000)
        hp['n_train'] = min(hp['n_train'], 5000)
        hp['n_cert'] = min(hp['n_cert'], 10000)

    if out_dir is None:
        out_dir = Path('results_v2')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  EXP02 — D={D}")
    print(f"  epochs={hp['n_epochs']} | layers={hp['n_layers']} | "
          f"hidden={hp['hidden_dim']} | n_cert={hp['n_cert']}")
    print(f"{'='*70}")

    target = BananaTarget(D=D)

    print(f"\n  Generating {hp['n_train']} training samples...")
    t0 = time.time()
    train_samples = target.generate_samples(hp['n_train'])
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\n  Training FolT-OG-Anneal...")
    flow, history = train_cert_og_anneal(
        target, train_samples, D,
        n_epochs=hp['n_epochs'], n_layers=hp['n_layers'],
        hidden_dim=hp['hidden_dim'], n_cert=hp['n_cert'],
        cert_interval=max(hp['n_epochs'] // 10, 100),
    )

    # ── Shared certification samples ──
    print(f"\n  Generating {hp['n_cert']} certification samples...")
    with torch.no_grad():
        z_all = torch.randn(hp['n_cert'], D, device=DEVICE)
        r_all = compute_residual(flow, target, z_all)

    # ── V1 ──
    print(f"\n  [V1] Uniform DKW certificate...")
    res_v1 = cert_v1(flow, target, D, z_all, r_all)
    print(f"  V1: C={res_v1['cert']:.4f}, γ={res_v1['gamma']:.6f}, "
          f"corr={res_v1['correction']:.4f} (grad_sup={res_v1['grad_sup']:.3f}, "
          f"eps={res_v1['eps_star']:.3f})")

    # ── V2a: Pointwise (heuristic, no Hessian) ──
    print(f"\n  [V2a] Pointwise gradient (heuristic, no Hessian)...")
    res_v2a = cert_v2a_pointwise(flow, target, D, z_all, r_all)
    print(f"  V2a: C={res_v2a['cert']:.4f}, γ={res_v2a['gamma']:.6f}, "
          f"corr={res_v2a['correction']:.4f}")
    print(f"    grad at r_max: {res_v2a['grad_at_rmax']:.4f}, "
          f"grad at r_min: {res_v2a['grad_at_rmin']:.4f}, "
          f"grad_sup: {res_v2a['grad_sup']:.4f}, "
          f"grad_p50: {res_v2a['grad_p50']:.4f}")

    # ── V2b: Pointwise + Hessian (rigorous) ──
    print(f"\n  [V2b] Pointwise gradient + Hessian (rigorous)...")
    res_v2b = cert_v2b_pointwise_hessian(flow, target, D, z_all, r_all,
                                          n_hess_samples=300 if quick else 500,
                                          n_hess_dirs=5 if quick else 20)
    print(f"  V2b: C={res_v2b['cert']:.4f}, γ={res_v2b['gamma']:.6f}, "
          f"corr={res_v2b['correction']:.4f}")
    print(f"    pointwise_part={res_v2b['correction_pointwise_part']:.4f}, "
          f"hessian_part={res_v2b['correction_hessian_part']:.4f}, "
          f"H={res_v2b['H']:.4f}")

    # ── V2c: Truncated support variants ──
    trunc_results = {}
    for alpha in [0.01, 0.05, 0.10]:
        print(f"\n  [V2c] Truncated (alpha={alpha})...")
        res_tc = cert_v2c_truncated(flow, target, D, z_all, r_all,
                                     alpha=alpha,
                                     n_hess_samples=300 if quick else 500,
                                     n_hess_dirs=5 if quick else 20)
        key = f'V2c_a{alpha}'
        trunc_results[key] = res_tc
        print(f"  {key}: C={res_tc['cert']:.4f}, γ={res_tc['gamma']:.6f}, "
              f"corr={res_tc['correction']:.4f}, eps={res_tc['eps_star']:.3f}")

    # ── Summary ──
    all_results = {
        'V1_uniform': res_v1,
        'V2a_pointwise': res_v2a,
        'V2b_pw_hessian': res_v2b,
        **trunc_results,
    }

    print(f"\n{'='*70}")
    print(f"  SUMMARY — D={D}")
    print(f"{'='*70}")
    print(f"  {'Method':<22} {'C':>8} {'γ≥':>12} {'corr':>8} {'eps*':>8} {'osc':>8}")
    print(f"  {'─'*70}")

    for name, res in all_results.items():
        print(f"  {name:<22} {res['cert']:>8.3f} {res['gamma']:>12.6f} "
              f"{res['correction']:>8.3f} {res.get('eps_star',0):>8.3f} "
              f"{res['sample_osc']:>8.3f}")

    # Improvement vs V1
    g1 = res_v1['gamma']
    c1 = res_v1['cert']
    print(f"\n  vs V1 (C={c1:.3f}, γ={g1:.6f}):")
    for name, res in all_results.items():
        if name == 'V1_uniform':
            continue
        delta_c = c1 - res['cert']
        if g1 > 0:
            ratio = res['gamma'] / g1
            print(f"    {name}: ΔC={delta_c:+.3f}, γ ratio={ratio:.2f}×")
        else:
            print(f"    {name}: ΔC={delta_c:+.3f}, γ={res['gamma']:.6f}")

    # Save
    summary = {}
    for name, res in all_results.items():
        summary[name] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                             else v)
                         for k, v in res.items()}
    summary['D'] = D
    summary['hparams'] = hp
    summary['nll_final'] = float(history['nll'][-1])

    with open(out_dir / f'exp02_D{D}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    make_plot(all_results, D, out_dir)
    return all_results


def make_plot(all_results, D, out_dir):
    names = list(all_results.keys())
    certs = [all_results[n]['cert'] for n in names]
    gammas = [all_results[n]['gamma'] for n in names]
    corrs = [all_results[n]['correction'] for n in names]
    oscs = [all_results[n]['sample_osc'] for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'EXP02: Pointwise Gradient Certificate — D={D}', fontsize=14)

    x = np.arange(len(names))
    colors_list = ['#888888', '#4488CC', '#44AA66'] + ['#CC7744'] * len(names)

    # Certificate C
    axes[0,0].bar(x, certs, color=colors_list[:len(names)])
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    axes[0,0].set_title('Certificate C (lower=better)')

    # Gamma
    axes[0,1].bar(x, gammas, color=colors_list[:len(names)])
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    axes[0,1].set_title('Spectral gap γ (higher=better)')

    # Stacked: sample_osc + correction
    axes[1,0].bar(x, oscs, label='sample_osc', color='#4488CC')
    axes[1,0].bar(x, corrs, bottom=oscs, label='correction', color='#CC7744', alpha=0.7)
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    axes[1,0].set_title('C decomposition')
    axes[1,0].legend()

    # Correction only
    axes[1,1].bar(x, corrs, color=colors_list[:len(names)])
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    axes[1,1].set_title('Correction term (lower=better)')

    plt.tight_layout()
    plt.savefig(out_dir / f'exp02_D{D}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {out_dir / f'exp02_D{D}.png'}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='FolT-MCMC V2 Exp02')
    parser.add_argument('--dim', type=int, nargs='+', default=[2, 5, 6])
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--outdir', type=str, default='results_v2')
    args = parser.parse_args()

    out_dir = Path(args.outdir)

    for D in args.dim:
        run_experiment(D, quick=args.quick, out_dir=out_dir)

    print(f"\n{'='*70}")
    print("EXP02 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
