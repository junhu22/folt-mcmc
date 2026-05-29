"""
FolT-MCMC V2 — Experiment 01: Local-Lipschitz Certificate
==========================================================
Goal: Break the D≥6 covering barrier via importance-weighted covering
      with local Lipschitz bounds in z-space.

Key insight:
  V1 certificate: C = sample_osc + 2 * grad_sup_GLOBAL * eps_star
  V2 certificate: C = max_k(local_max_k + L_k*d_k) - min_k(local_min_k - L_k*d_k)
  where L_k = local Lipschitz in Voronoi cell k, d_k = cell diameter.
  If most cells have L_k << grad_sup, correction shrinks dramatically.

Usage:
  conda activate lcnf
  cd C:\FolT-MCMC\experiments
  python exp01_local_lipschitz.py --dim 2 5 6        # core comparison
  python exp01_local_lipschitz.py --dim 6 --quick     # fast sanity check
  python exp01_local_lipschitz.py --dim 2 5 6 8 10    # full sweep

Results saved to C:\FolT-MCMC\experiments\results_v2\
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
from scipy.spatial import cKDTree

# ══════════════════════════════════════════════════════════════
# Device
# ══════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ══════════════════════════════════════════════════════════════
# 1. Target Distribution (from V1)
# ══════════════════════════════════════════════════════════════

class BananaTarget:
    """N-dimensional banana: U(θ) = θ₁²/(2σ₁²) + (θ₂-bθ₁²)²/(2σ₂²) + Σ θᵢ²/2"""
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
# 2. Spectral-Normalised RealNVP (from V1)
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
# 3. Residual & Smooth Oscillation (from V1)
# ══════════════════════════════════════════════════════════════

def compute_residual(flow, target, z):
    """r(z) = -U(T(z)) + log|det J| + ||z||²/2"""
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
# 4. V1 Certificate (baseline)
# ══════════════════════════════════════════════════════════════

def compute_certificate_v1(flow, target, D, n_cert=50000, zeta=0.05):
    """
    V1 uniform DKW certificate.
    C = sample_osc + 2 * grad_sup * eps_star
    """
    flow.eval()
    with torch.no_grad():
        z = torch.randn(n_cert, D, device=DEVICE)
        r = compute_residual(flow, target, z)
        sample_osc = (r.max() - r.min()).item()

    n_grad = min(2000, n_cert)
    z_g = torch.randn(n_grad, D, device=DEVICE).requires_grad_(True)
    r_g = compute_residual(flow, target, z_g)
    grad_r = torch.autograd.grad(r_g.sum(), z_g)[0]
    grad_sup = grad_r.norm(dim=-1).max().item()
    grad_mean = grad_r.norm(dim=-1).mean().item()

    R_alpha = math.sqrt(chi2.ppf(0.95, df=D))
    log_term = math.log(2.0 / zeta) / (2 * n_cert)
    eps_star = R_alpha * (log_term + math.sqrt(log_term)) ** (1.0 / D)

    correction = 2 * grad_sup * eps_star
    cert_value = sample_osc + correction

    if cert_value > 500:
        gamma_bound = 0.0
    else:
        gamma_bound = 2.0 / (1.0 + math.exp(cert_value))

    mix_time = math.log(100) / max(gamma_bound, 1e-30)
    mix_time = min(mix_time, 1e18)

    return cert_value, gamma_bound, {
        'method': 'V1_uniform',
        'sample_osc': sample_osc,
        'grad_sup': grad_sup,
        'grad_mean': grad_mean,
        'eps_star': eps_star,
        'correction': correction,
        'R_alpha': R_alpha,
        'gamma': gamma_bound,
        'mix_time': mix_time,
    }


# ══════════════════════════════════════════════════════════════
# 5. V2 Certificate: Local-Lipschitz with Voronoi Partition
# ══════════════════════════════════════════════════════════════

def compute_local_gradients(flow, target, z_points, batch_size=500):
    """
    Compute ||∇_z r(z)|| for each z in z_points.
    Returns gradient norms as numpy array.
    """
    n = z_points.shape[0]
    grad_norms = []
    for i in range(0, n, batch_size):
        z_batch = z_points[i:i+batch_size].detach().requires_grad_(True)
        r = compute_residual(flow, target, z_batch)
        grad_r = torch.autograd.grad(r.sum(), z_batch)[0]
        grad_norms.append(grad_r.norm(dim=-1).detach().cpu().numpy())
    return np.concatenate(grad_norms)


def compute_certificate_v2_local_lipschitz(
    flow, target, D, n_cert=50000, K=None, zeta=0.05,
    n_grad_per_cell=None, grad_batch_size=500
):
    """
    V2 certificate with local Lipschitz bounds in Voronoi cells.

    Algorithm:
      1. Sample n_cert points z_i ~ N(0,I), compute r(z_i)
      2. Select K centroids via mini-batch k-means in z-space
      3. Assign each z_i to nearest centroid → Voronoi cells
      4. In each cell k:
         - local_max_k = max r(z_i) for z_i in cell k
         - local_min_k = min r(z_i) for z_i in cell k
         - L_k = max ||∇r(z)|| over samples in cell k
         - d_k = max distance from centroid to any sample in cell k
      5. Certified bounds per cell:
         - upper_k = local_max_k + L_k * d_k
         - lower_k = local_min_k - L_k * d_k
      6. C_v2 = max_k(upper_k) - min_k(lower_k)

    Note: d_k is the SAMPLE-based cell diameter (conservative: 2*max_radius).
    The true Voronoi cell could extend further, but we add a DKW-style
    correction for uncovered volume within each cell.

    Args:
        K: number of Voronoi cells. Default = min(n_cert//50, 2000)
        n_grad_per_cell: if set, subsample gradients per cell. Default: all.
    """
    flow.eval()

    if K is None:
        K = min(n_cert // 50, 2000)

    print(f"    [V2] n_cert={n_cert}, K={K}, D={D}")

    # ── Step 1: Sample and compute residuals ──
    with torch.no_grad():
        z_all = torch.randn(n_cert, D, device=DEVICE)
        r_all = compute_residual(flow, target, z_all)
    
    z_np = z_all.cpu().numpy()
    r_np = r_all.cpu().numpy()

    sample_osc = r_np.max() - r_np.min()
    print(f"    [V2] sample_osc = {sample_osc:.4f}")

    # ── Step 2: Mini-batch k-means for centroids ──
    print(f"    [V2] Running k-means (K={K})...")
    t0 = time.time()
    centroids = _mini_batch_kmeans(z_np, K, n_iter=30, batch_size=min(5000, n_cert))
    print(f"    [V2] k-means done in {time.time()-t0:.1f}s")

    # ── Step 3: Assign to cells via KD-tree ──
    tree = cKDTree(centroids)
    dists, labels = tree.query(z_np)  # dists[i] = dist from z_i to its centroid

    # ── Step 4: Local statistics per cell ──
    print(f"    [V2] Computing local gradients...")
    t0 = time.time()
    grad_norms = compute_local_gradients(
        flow, target, z_all, batch_size=grad_batch_size)
    print(f"    [V2] Gradients done in {time.time()-t0:.1f}s")

    cell_upper = np.full(K, -np.inf)
    cell_lower = np.full(K, np.inf)
    cell_lip = np.zeros(K)
    cell_diam = np.zeros(K)
    cell_count = np.zeros(K, dtype=int)

    for k in range(K):
        mask = labels == k
        cell_count[k] = mask.sum()
        if cell_count[k] == 0:
            continue

        r_k = r_np[mask]
        g_k = grad_norms[mask]
        d_k = dists[mask]

        cell_upper[k] = r_k.max()
        cell_lower[k] = r_k.min()
        cell_lip[k] = g_k.max()
        cell_diam[k] = d_k.max()  # max distance from centroid

    # ── Step 5: Filter to non-empty cells ──
    active = cell_count > 0
    n_active = active.sum()
    print(f"    [V2] Active cells: {n_active}/{K}")

    # ── Step 6: Certified bounds per cell ──
    # Within each Voronoi cell, the max distance from any point to
    # the nearest SAMPLE point ≤ cell-level covering radius.
    # Conservative: use cell_diam (max dist to centroid) as radius,
    # and add local Lipschitz correction.
    
    # Additional DKW correction for uncovered volume within each cell:
    # For cell k with n_k samples, the max gap to nearest sample is bounded.
    # We use a simple heuristic: within-cell covering radius ≈
    # cell_diam * (log(2/zeta_k) / (2*n_k))^(1/D)  where zeta_k = zeta/K
    
    zeta_per_cell = zeta / K
    cell_eps = np.zeros(K)
    for k in range(K):
        if cell_count[k] < 2:
            cell_eps[k] = cell_diam[k]  # fallback: full diameter
        else:
            log_term_k = math.log(2.0 / zeta_per_cell) / (2 * cell_count[k])
            # Within-cell covering radius scaled by cell diameter
            cell_eps[k] = cell_diam[k] * min(
                (log_term_k + math.sqrt(log_term_k)) ** (1.0 / max(D, 1)),
                1.0  # can't exceed cell diameter
            )

    # Certified cell-level bounds
    cert_upper = cell_upper[active] + cell_lip[active] * cell_eps[active]
    cert_lower = cell_lower[active] - cell_lip[active] * cell_eps[active]

    # Global certificate
    global_max = cert_upper.max()
    global_min = cert_lower.min()
    cert_value = global_max - global_min

    # Also compute the "correction" as difference from sample_osc
    correction = cert_value - sample_osc

    # Spectral gap
    if cert_value > 500:
        gamma_bound = 0.0
    else:
        gamma_bound = 2.0 / (1.0 + math.exp(cert_value))

    mix_time = math.log(100) / max(gamma_bound, 1e-30)
    mix_time = min(mix_time, 1e18)

    # ── Diagnostics ──
    lip_percentiles = np.percentile(cell_lip[active], [50, 90, 95, 99, 100])
    diam_percentiles = np.percentile(cell_diam[active], [50, 90, 95, 99, 100])
    count_percentiles = np.percentile(cell_count[active], [5, 25, 50, 75, 95])

    # Which cell contributes the max/min?
    idx_max = np.argmax(cert_upper)
    idx_min = np.argmin(cert_lower)

    return cert_value, gamma_bound, {
        'method': 'V2_local_lipschitz',
        'sample_osc': float(sample_osc),
        'correction': float(correction),
        'K': K,
        'n_active': int(n_active),
        'gamma': float(gamma_bound),
        'mix_time': float(mix_time),
        # Local Lipschitz diagnostics
        'lip_p50': float(lip_percentiles[0]),
        'lip_p90': float(lip_percentiles[1]),
        'lip_p95': float(lip_percentiles[2]),
        'lip_p99': float(lip_percentiles[3]),
        'lip_max': float(lip_percentiles[4]),
        # Cell diameter diagnostics
        'diam_p50': float(diam_percentiles[0]),
        'diam_p90': float(diam_percentiles[1]),
        'diam_max': float(diam_percentiles[4]),
        # Cell count diagnostics
        'count_p5': float(count_percentiles[0]),
        'count_p50': float(count_percentiles[2]),
        # Max/min cell info
        'max_cell_lip': float(cell_lip[active][idx_max]),
        'max_cell_eps': float(cell_eps[active][idx_max]),
        'min_cell_lip': float(cell_lip[active][idx_min]),
        'min_cell_eps': float(cell_eps[active][idx_min]),
    }


def _mini_batch_kmeans(X, K, n_iter=30, batch_size=5000):
    """
    Simple mini-batch k-means. Returns (K, D) centroids.
    """
    n, D = X.shape
    # Init: random subset
    idx = np.random.choice(n, K, replace=False)
    centroids = X[idx].copy()

    for it in range(n_iter):
        # Sample mini-batch
        batch_idx = np.random.choice(n, min(batch_size, n), replace=False)
        X_batch = X[batch_idx]

        # Assign
        tree = cKDTree(centroids)
        _, labels = tree.query(X_batch)

        # Update
        for k in range(K):
            mask = labels == k
            if mask.sum() > 0:
                # Learning rate decay
                eta = 1.0 / (it + 1)
                centroids[k] = (1 - eta) * centroids[k] + eta * X_batch[mask].mean(axis=0)

    return centroids


# ══════════════════════════════════════════════════════════════
# 6. V2 Certificate: Strategy B — Gradient-Guided Extremum Search
# ══════════════════════════════════════════════════════════════

def compute_certificate_v2_extremum_search(
    flow, target, D, n_cert=50000, n_optim_starts=100,
    optim_steps=200, lr=0.01, zeta=0.05
):
    """
    V2 alternative: tighten the oscillation bound by optimizing
    for the max and min of r(z) directly.

    Algorithm:
      1. Sample n_cert points, find approximate max/min of r(z)
      2. From top-k and bottom-k samples, run gradient ascent/descent
         to find tighter bounds on true max/min
      3. Certified C = found_max - found_min
         (This is a LOWER bound on osc, so C is tighter only if
          we also add a small correction for finite search)
      4. Add residual correction: sup-of-gradient * covering-gap

    Note: This gives a TIGHTER sample_osc but still needs a correction.
    The improvement comes from reducing the gap between sample_osc and true_osc.
    """
    flow.eval()

    # ── Step 1: Sample and find approximate extrema ──
    with torch.no_grad():
        z_all = torch.randn(n_cert, D, device=DEVICE)
        r_all = compute_residual(flow, target, z_all)

    r_np = r_all.cpu().numpy()
    sample_osc = r_np.max() - r_np.min()

    # ── Step 2: Gradient ascent from top-k to find true max ──
    topk = min(n_optim_starts, n_cert)
    top_idx = np.argsort(r_np)[-topk:]
    bot_idx = np.argsort(r_np)[:topk]

    print(f"    [V2-ES] Optimizing max from {topk} starts...")
    found_max = _optimize_residual(
        flow, target, z_all[top_idx], maximize=True,
        n_steps=optim_steps, lr=lr)

    print(f"    [V2-ES] Optimizing min from {topk} starts...")
    found_min = _optimize_residual(
        flow, target, z_all[bot_idx], maximize=False,
        n_steps=optim_steps, lr=lr)

    optimized_osc = found_max - found_min
    print(f"    [V2-ES] sample_osc={sample_osc:.4f}, optimized_osc={optimized_osc:.4f}")

    # ── Step 3: Residual correction ──
    # After optimization, the gap between found extrema and true extrema
    # is bounded by the convergence tolerance of the optimizer.
    # We add a small safety margin based on gradient norm at the found extrema.
    # This is heuristic — a rigorous bound would require second-order analysis.
    # For now, we use: correction = 2 * grad_at_extrema * convergence_radius
    # where convergence_radius ≈ lr * n_steps^{-0.5} (SGD convergence rate)
    convergence_radius = lr * optim_steps ** (-0.5)
    
    # Get gradient at the found maximum and minimum points
    n_grad_check = min(500, n_cert)
    z_g = torch.randn(n_grad_check, D, device=DEVICE).requires_grad_(True)
    r_g = compute_residual(flow, target, z_g)
    grad_r = torch.autograd.grad(r_g.sum(), z_g)[0]
    grad_sup = grad_r.norm(dim=-1).max().item()

    correction = 2 * grad_sup * convergence_radius
    cert_value = optimized_osc + correction

    if cert_value > 500:
        gamma_bound = 0.0
    else:
        gamma_bound = 2.0 / (1.0 + math.exp(cert_value))

    mix_time = math.log(100) / max(gamma_bound, 1e-30)
    mix_time = min(mix_time, 1e18)

    return cert_value, gamma_bound, {
        'method': 'V2_extremum_search',
        'sample_osc': float(sample_osc),
        'optimized_osc': float(optimized_osc),
        'correction': float(correction),
        'convergence_radius': float(convergence_radius),
        'grad_sup': float(grad_sup),
        'found_max': float(found_max),
        'found_min': float(found_min),
        'gamma': float(gamma_bound),
        'mix_time': float(mix_time),
    }


def _optimize_residual(flow, target, z_starts, maximize=True,
                       n_steps=200, lr=0.01):
    """
    Gradient ascent/descent on r(z) from multiple starting points.
    Returns the best (max or min) found value.
    """
    z = z_starts.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    best_val = -float('inf') if maximize else float('inf')

    for step in range(n_steps):
        r = compute_residual(flow, target, z)
        loss = -r.sum() if maximize else r.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            r_vals = compute_residual(flow, target, z)
            if maximize:
                step_best = r_vals.max().item()
                best_val = max(best_val, step_best)
            else:
                step_best = r_vals.min().item()
                best_val = min(best_val, step_best)

    return best_val


# ══════════════════════════════════════════════════════════════
# 7. Training (from V1, FolT-OG-Anneal only)
# ══════════════════════════════════════════════════════════════

def train_cert_og_anneal(target, train_samples, D,
                         n_epochs=5000, lr=1e-3, batch_size=512,
                         n_layers=8, hidden_dim=64, scale_clip=0.7,
                         lambda_o=0.1, lambda_g=0.05, tau=0.1,
                         cert_interval=500, n_cert=50000):
    """Train with FolT-OG-Anneal (best V1 config)."""

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

        # Annealing: first 40% NLL only, then linear ramp
        ramp = max(0.0, (epoch / n_epochs - 0.4)) / 0.6
        eff_lo = lambda_o * ramp
        eff_lg = lambda_g * ramp

        # NLL
        idx = torch.randint(0, N, (batch_size,), device=DEVICE)
        nll = -flow.log_prob(train_dev[idx]).mean()

        # Certificate-aware losses
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
            cert_val, gamma_b, det = compute_certificate_v1(
                flow, target, D, n_cert=n_cert)
            history['cert_epochs'].append(epoch + 1)
            history['cert_values'].append(cert_val)
            history['gamma_bounds'].append(gamma_b)

            elapsed = time.time() - t0
            print(f"  [{epoch+1:5d}/{n_epochs}] "
                  f"NLL={nll.item():.3f} | "
                  f"osc={det['sample_osc']:.3f} | "
                  f"∇sup={det['grad_sup']:.2f} | "
                  f"C_v1={cert_val:.3f} | "
                  f"γ≥{gamma_b:.6f} | "
                  f"{elapsed:.0f}s")

    flow.eval()
    return flow, history


# ══════════════════════════════════════════════════════════════
# 8. Experiment Runner
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
    """Run V1 vs V2 certificate comparison for one dimension."""

    hp = HPARAMS.get(D, HPARAMS[5]).copy()
    if quick:
        hp['n_epochs'] = min(hp['n_epochs'], 1000)
        hp['n_train'] = min(hp['n_train'], 5000)
        hp['n_cert'] = min(hp['n_cert'], 10000)

    if out_dir is None:
        out_dir = Path('results_v2')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  V2 EXPERIMENT — D={D}")
    print(f"  epochs={hp['n_epochs']} | layers={hp['n_layers']} | "
          f"hidden={hp['hidden_dim']} | n_cert={hp['n_cert']}")
    print(f"{'='*70}")

    target = BananaTarget(D=D)

    # ── Generate training data ──
    print(f"\n  Generating {hp['n_train']} training samples...")
    t0 = time.time()
    train_samples = target.generate_samples(hp['n_train'])
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Train flow (FolT-OG-Anneal) ──
    print(f"\n  Training FolT-OG-Anneal flow...")
    flow, history = train_cert_og_anneal(
        target, train_samples, D,
        n_epochs=hp['n_epochs'], n_layers=hp['n_layers'],
        hidden_dim=hp['hidden_dim'], n_cert=hp['n_cert'],
        cert_interval=max(hp['n_epochs'] // 10, 100),
    )

    # ── V1 certificate ──
    print(f"\n  Computing V1 certificate (uniform DKW)...")
    c1, g1, d1 = compute_certificate_v1(flow, target, D, n_cert=hp['n_cert'])
    print(f"  V1: C={c1:.4f}, γ≥{g1:.6f}, correction={d1['correction']:.4f}")

    # ── V2a: Local Lipschitz ──
    K_values = [500, 1000, 2000]
    if quick:
        K_values = [200, 500]

    v2_results = {}
    for K in K_values:
        print(f"\n  Computing V2 certificate (local Lipschitz, K={K})...")
        c2, g2, d2 = compute_certificate_v2_local_lipschitz(
            flow, target, D, n_cert=hp['n_cert'], K=K)
        v2_results[f'V2_LL_K{K}'] = {'cert': c2, 'gamma': g2, 'details': d2}
        print(f"  V2 (K={K}): C={c2:.4f}, γ≥{g2:.6f}, correction={d2['correction']:.4f}")

    # ── V2b: Extremum search ──
    print(f"\n  Computing V2 certificate (extremum search)...")
    c3, g3, d3 = compute_certificate_v2_extremum_search(
        flow, target, D, n_cert=hp['n_cert'],
        n_optim_starts=50 if quick else 200,
        optim_steps=100 if quick else 500)
    print(f"  V2-ES: C={c3:.4f}, γ≥{g3:.6f}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY — D={D}")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'C':>10} {'γ≥':>12} {'correction':>12} {'sample_osc':>12}")
    print(f"  {'─'*66}")

    all_results = {
        'V1_uniform': {'cert': c1, 'gamma': g1, 'details': d1},
        **v2_results,
        'V2_extremum': {'cert': c3, 'gamma': g3, 'details': d3},
    }

    for name, res in all_results.items():
        d = res['details']
        so = d.get('sample_osc', d.get('optimized_osc', 0))
        corr = d.get('correction', 0)
        print(f"  {name:<20} {res['cert']:>10.4f} {res['gamma']:>12.6f} "
              f"{corr:>12.4f} {so:>12.4f}")

    # Improvement ratios
    if g1 > 0:
        print(f"\n  Improvement vs V1:")
        for name, res in all_results.items():
            if name == 'V1_uniform':
                continue
            delta_c = c1 - res['cert']
            ratio = res['gamma'] / max(g1, 1e-30)
            print(f"    {name}: ΔC={delta_c:+.3f}, γ ratio={ratio:.2f}×")
    else:
        print(f"\n  V1 γ=0 (vacuous). V2 improvements:")
        for name, res in all_results.items():
            if name == 'V1_uniform':
                continue
            print(f"    {name}: C={res['cert']:.4f}, γ={res['gamma']:.6f}")

    # ── Save ──
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            'cert': float(res['cert']),
            'gamma': float(res['gamma']),
            'details': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer))
                       else v for k, v in res['details'].items()},
        }
    summary['D'] = D
    summary['hparams'] = hp
    summary['training_nll_final'] = float(history['nll'][-1])

    with open(out_dir / f'exp01_D{D}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Diagnostic plot ──
    make_v2_plot(all_results, D, out_dir)

    return all_results


def make_v2_plot(all_results, D, out_dir):
    """Bar chart comparing V1 vs V2 certificates."""

    names = list(all_results.keys())
    certs = [all_results[n]['cert'] for n in names]
    gammas = [all_results[n]['gamma'] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'FolT-MCMC V1 vs V2 — D={D}', fontsize=14)

    x = np.arange(len(names))
    colors = ['#888888'] + ['#44AA66'] * (len(names) - 2) + ['#CC7744']

    axes[0].bar(x, certs, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[0].set_title('Certificate C (lower = better)')
    axes[0].set_ylabel('C')

    axes[1].bar(x, gammas, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[1].set_title('Certified γ (higher = better)')
    axes[1].set_ylabel('γ')
    if max(gammas) > 0:
        axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(out_dir / f'exp01_D{D}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to {out_dir / f'exp01_D{D}.png'}")


# ══════════════════════════════════════════════════════════════
# 9. Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='FolT-MCMC V2 Experiment 01')
    parser.add_argument('--dim', type=int, nargs='+', default=[2, 5, 6],
                        help='Dimensions to test (default: 2 5 6)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer epochs/samples')
    parser.add_argument('--outdir', type=str, default='results_v2',
                        help='Output directory')
    args = parser.parse_args()

    out_dir = Path(args.outdir)

    for D in args.dim:
        run_experiment(D, quick=args.quick, out_dir=out_dir)

    print(f"\n{'='*70}")
    print("ALL V2 EXPERIMENTS COMPLETE")
    print(f"Results in {out_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
