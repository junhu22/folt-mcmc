"""
FolT-MCMC toy experiment: asymmetric double banana.

Compares two independence-MH transport pipelines on a reflection-symmetric
target whose symmetry group is G = {id, theta1 -> -theta1}, |G| = 2:

  (A) Unfolded baseline -- train a RealNVP transport in the original
      theta-space, which must cover BOTH reflected copies of the target.

  (B) Folded -- fold the symmetry away with F(theta) = (|theta1|, theta2),
      train the transport in the z1 >= 0 half-plane (a single copy), then
      unfold samples back to theta-space by choosing a random branch (sign).

Because the folded transport only has to model half as many modes, its
importance-weight residual r = log(pi/q) oscillates less, which tightens the
certified spectral gap and improves ESS. This script measures that.

Metrics (per pipeline): full oscillation osc(r), DKW-corrected quantile-core
certified gamma, ESS per sample, ESS per model evaluation, MH acceptance, and
final training NLL. Results are written to results/toy_fold_results.json and
five diagnostic figures are saved to results/.

Run:
    conda activate lcnf
    set KMP_DUPLICATE_LIB_OK=TRUE
    cd C:\\FolT-MCMC
    python experiments/run_toy_fold.py
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Make the project root importable when run as `python experiments/run_toy_fold.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from folt.folding import ReflectionFold
from folt.targets.symmetric_banana import AsymmetricDoubleBanana
from folt.transport import SNRealNVP, train_flow
from folt.certification import (compute_oscillation_bound, compute_spectral_gap,
                                mengersen_tweedie_bound, quantile_core_certificate,
                                v1_covering_certificate)
from folt.mh_kernel import run_independence_mh_multichain, compute_ess_from_samples

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# Hyperparameters (identical for both pipelines -> apples-to-apples).
# Transport architecture matches the CerT-MCMC-v2 banana experiment
# (n_layers=8, hidden_dim=64, scale_clip=0.7). Training length is reduced to
# 2000 epochs (from 5000) since the 2D toy converges quickly; both pipelines
# use the same budget.
# ──────────────────────────────────────────────────────────────
HP = dict(n_layers=8, hidden_dim=64, lr=1e-3, n_epochs=2000,
          n_train=20000, batch_size=512)

N_CERT = 20000          # samples for the oscillation / quantile certificates
ZETA = 0.05             # DKW confidence (1 - zeta = 0.95)
HEADLINE_RHO = 0.05     # quantile-core trim level reported as certified_gamma

N_CHAINS = 4
CHAIN_LENGTH = 5000
BURNIN = 1000

SEED = 0


# ══════════════════════════════════════════════════════════════
# Target adapters
# ══════════════════════════════════════════════════════════════

class UnfoldedTarget:
    """The asymmetric double banana in the original theta-space (2 modes)."""
    def __init__(self, base):
        self.base = base
        self.dim = base.dim

    def log_prob(self, theta):
        return self.base.log_prob(theta)


class FoldedTargetSmooth:
    """Folded target pi_F on z-space, smooth everywhere (no hard support).

    Used for training / oscillation-regularisation, where a smooth log-density
    avoids -inf blow-ups. pi_F == pi (the t1^2 fold leaves the formula
    unchanged up to an irrelevant additive log|G| constant); the benefit comes
    from training only on the z1 >= 0 half-plane data.
    """
    def __init__(self, base):
        self.base = base
        self.dim = base.dim

    def log_prob(self, z):
        return self.base.log_prob_folded(z)


class FoldedTargetConstrained:
    """Folded target restricted to the fundamental domain z1 >= 0.

    Returns -inf for z1 < 0 so the independence-MH chain cannot leave the
    half-plane (such proposals always reject).
    """
    def __init__(self, base):
        self.base = base
        self.dim = base.dim

    def log_prob(self, z):
        lp = self.base.log_prob_folded(z)
        return torch.where(z[:, 0] >= 0, lp, torch.full_like(lp, float('-inf')))


def half_plane_support(theta):
    """Support indicator for the folded fundamental domain (theta1 >= 0)."""
    return theta[:, 0] >= 0


# ══════════════════════════════════════════════════════════════
# Sampling the target (rejection sampler in 2D)
# ══════════════════════════════════════════════════════════════

def sample_banana(target, n, device, s1=4.5, s2=12.0, mu2=3.0, max_iter=200):
    """Rejection sampler for the 2D asymmetric banana.

    Proposal: theta1 ~ N(0, s1^2), theta2 ~ N(mu2, s2^2). The proposal is wide
    enough to cover both curvature arms and the theta1-symmetric copies.
    """
    collected = []
    total = 0
    m = max(n * 10, 200000)
    it = 0
    while total < n and it < max_iter:
        it += 1
        prop = torch.randn(m, 2, device=device)
        prop[:, 0] = prop[:, 0] * s1
        prop[:, 1] = prop[:, 1] * s2 + mu2
        log_q = (-0.5 * prop[:, 0] ** 2 / s1 ** 2
                 - 0.5 * (prop[:, 1] - mu2) ** 2 / s2 ** 2)
        log_p = target.log_prob(prop)
        log_ratio = log_p - log_q
        log_M = log_ratio.max()
        u = torch.rand(m, device=device).log()
        accept = u < (log_ratio - log_M)
        acc = prop[accept]
        collected.append(acc)
        total += acc.shape[0]
    samples = torch.cat(collected)
    if samples.shape[0] < n:
        raise RuntimeError(f"Rejection sampler only produced {samples.shape[0]}/{n}")
    return samples[:n]


# ══════════════════════════════════════════════════════════════
# One pipeline (train -> certify -> sample)
# ══════════════════════════════════════════════════════════════

def run_pipeline(name, train_target, cert_target, mh_target,
                 train_samples, support_fn, out_dir):
    """Train a transport, certify it, and run multi-chain independence MH.

    Args:
        name:          'unfolded' or 'folded' (for logging / checkpoints).
        train_target:  target used for NLL + osc-regularisation (smooth).
        cert_target:   target used for the certificate residual (smooth).
        mh_target:     target used by the MH kernel (may enforce support).
        train_samples: (N, 2) samples to fit the flow by MLE.
        support_fn:    optional theta -> mask restricting the oscillation to
                       the support of the (folded) target; None for unfolded.

    Returns a dict of metrics.
    """
    D = train_target.dim
    print(f"\n{'='*72}\n  PIPELINE [{name}]\n{'='*72}")

    # ── Train ──
    print(f"  Training transport ({HP['n_epochs']} epochs, "
          f"{train_samples.shape[0]} samples)...")
    t0 = time.time()
    flow, history = train_flow(train_target, D, HP,
                               train_samples=train_samples, device=DEVICE)
    print(f"  Training done in {time.time()-t0:.0f}s, final NLL={history['nll_final']:.3f}")
    torch.save(flow.state_dict(), out_dir / f'flow_{name}.pt')

    # ── Certify ──
    print(f"  Certifying ({N_CERT} samples)...")
    osc = compute_oscillation_bound(flow, cert_target, N_CERT, D,
                                    support_fn=support_fn, device=DEVICE)
    full_osc = osc['full_osc']
    r = osc['r']
    gamma_full = compute_spectral_gap(full_osc)
    gamma_mt = mengersen_tweedie_bound(full_osc)

    qcore = quantile_core_certificate(r, zeta=ZETA)
    v1 = v1_covering_certificate(flow, cert_target, full_osc, D, len(r),
                                 zeta=ZETA, device=DEVICE)

    head = qcore['levels'].get(HEADLINE_RHO, {})
    if head.get('feasible'):
        certified_gamma = head['gamma_core']
        certified_osc = head['c_formal']
    else:  # fall back to full oscillation if the headline level is infeasible
        certified_gamma = gamma_full
        certified_osc = full_osc

    print(f"    full_osc={full_osc:.4f}  gamma_full={gamma_full:.4g}  "
          f"gamma_MT={gamma_mt:.4g}")
    print(f"    quantile-core (rho={HEADLINE_RHO}): C_formal={certified_osc:.4f}  "
          f"gamma_core={certified_gamma:.4g}")
    print(f"    V1 covering: C={v1['cert']:.4f}  gamma={v1['gamma']:.4g}  "
          f"(grad_sup={v1['grad_sup']:.3f}, eps*={v1['eps_star']:.4f})")

    # ── Sample (independence MH) ──
    print(f"  Running {N_CHAINS} MH chains (len={CHAIN_LENGTH}, burnin={BURNIN})...")
    mh = run_independence_mh_multichain(flow, mh_target, D,
                                        n_chains=N_CHAINS, chain_length=CHAIN_LENGTH,
                                        burnin=BURNIN, device=DEVICE)
    n_prop_total = N_CHAINS * (CHAIN_LENGTH + BURNIN)
    ess_per_gradient = mh['ess_total'] / n_prop_total
    print(f"    accept={mh['accept_rate']:.3f}  ESS_total={mh['ess_total']:.0f}  "
          f"ESS/sample={mh['ess_per_sample']:.4f}")

    metrics = {
        'osc_bound': float(full_osc),
        'certified_gamma': float(certified_gamma),
        'ess_per_sample': float(mh['ess_per_sample']),
        'ess_per_gradient': float(ess_per_gradient),
        'acceptance_rate': float(mh['accept_rate']),
        'train_loss_final': float(history['nll_final']),
        # ---- extended diagnostics ----
        'extended': {
            'gamma_full_osc': float(gamma_full),
            'gamma_mengersen_tweedie': float(gamma_mt),
            'certified_osc_core': float(certified_osc),
            'headline_rho': HEADLINE_RHO,
            'n_cert_valid': osc['n_valid'],
            'n_cert_eval': osc['n_eval'],
            'r_mean': osc['mean'], 'r_std': osc['std'],
            'ess_total': mh['ess_total'], 'ess_each': mh['ess_each'],
            'accept_each': mh['accept_each'],
            'v1_covering': v1,
            'quantile_core': {
                'eps_n': qcore['eps_n'],
                'levels': {str(k): v for k, v in qcore['levels'].items()},
            },
        },
    }
    # carry artifacts needed for plotting (not serialised)
    artifacts = {'flow': flow, 'r': r, 'samples': mh['samples_concat'],
                 'trace': mh['samples'][0], 'history': history}
    return metrics, artifacts


# ══════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════

def _grid(x_range, y_range, nx=240, ny=240, device=DEVICE):
    xs = torch.linspace(*x_range, nx)
    ys = torch.linspace(*y_range, ny)
    gx, gy = torch.meshgrid(xs, ys, indexing='xy')
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).to(device)
    return gx.numpy(), gy.numpy(), pts


def plot_targets(base, out_dir):
    """banana_target.png and folded_target.png."""
    # Original target with fold line theta1 = 0
    gx, gy, pts = _grid((-5, 5), (-3, 18))
    with torch.no_grad():
        lp = base.log_prob(pts).reshape(gx.shape).cpu().numpy()
    dens = np.exp(lp - lp.max())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(gx, gy, dens, levels=30, cmap='viridis')
    ax.axvline(0.0, color='red', ls='--', lw=1.5, label=r'fold line $\theta_1=0$')
    ax.set_xlabel(r'$\theta_1$'); ax.set_ylabel(r'$\theta_2$')
    ax.set_title(r'Original target $\pi(\theta)$ (asymmetric double banana)')
    ax.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(out_dir / 'banana_target.png', dpi=150); plt.close(fig)

    # Folded target on z1 >= 0
    gx, gy, pts = _grid((0, 5), (-3, 18))
    with torch.no_grad():
        lp = base.log_prob_folded(pts).reshape(gx.shape).cpu().numpy()
    dens = np.exp(lp - lp.max())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(gx, gy, dens, levels=30, cmap='viridis')
    ax.set_xlabel(r'$z_1 = |\theta_1|$'); ax.set_ylabel(r'$z_2$')
    ax.set_title(r'Folded target $\pi_F(z)$ on $z_1 \geq 0$')
    fig.tight_layout(); fig.savefig(out_dir / 'folded_target.png', dpi=150); plt.close(fig)


def _residual_field(flow, target, x_range, y_range):
    gx, gy, pts = _grid(x_range, y_range, nx=160, ny=160)
    with torch.no_grad():
        logpi = target.log_prob(pts)
        logq = flow.log_prob(pts)
        r = (logpi - logq).cpu().numpy().reshape(gx.shape)
        logpi = logpi.cpu().numpy().reshape(gx.shape)
    # mask low-density regions where the log-weight is meaningless
    r = np.where(logpi > logpi.max() - 8.0, r, np.nan)
    r = r - np.nanmedian(r)
    return gx, gy, r


def plot_oscillation_comparison(art_u, art_f, unfolded_t, folded_t, out_dir):
    """oscillation_comparison.png -- residual (log q/pi) field, both spaces."""
    gxu, gyu, ru = _residual_field(art_u['flow'], unfolded_t, (-5, 5), (-3, 18))
    gxf, gyf, rf = _residual_field(art_f['flow'], folded_t, (0, 5), (-3, 18))
    vmax = np.nanmax(np.abs(np.concatenate([ru[~np.isnan(ru)], rf[~np.isnan(rf)]])))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, gx, gy, rr, title, xl in [
        (axes[0], gxu, gyu, ru, 'Unfolded: residual $\\log(\\pi/q)$', r'$\theta_1$'),
        (axes[1], gxf, gyf, rf, 'Folded: residual $\\log(\\pi_F/q_F)$', r'$z_1$')]:
        pc = ax.pcolormesh(gx, gy, rr, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(r'$\theta_2$')
        fig.colorbar(pc, ax=ax, label='centered $r$')
    fig.suptitle('Oscillation of the log importance weight (lower spread = tighter certificate)')
    fig.tight_layout(); fig.savefig(out_dir / 'oscillation_comparison.png', dpi=150); plt.close(fig)


def plot_chain_trace(art_u, art_f, out_dir):
    """chain_trace.png -- first-coordinate trace for chain 0 of each pipeline."""
    tu = art_u['trace'][:, 0].cpu().numpy()
    tf = art_f['trace'][:, 0].cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(tu, lw=0.5, color='#CC4444'); axes[0].set_ylabel(r'$\theta_1$')
    axes[0].set_title('Unfolded chain (theta-space)')
    axes[1].plot(tf, lw=0.5, color='#4488CC'); axes[1].set_ylabel(r'$z_1$')
    axes[1].set_title('Folded chain (z-space, $z_1 \\geq 0$)')
    axes[1].set_xlabel('iteration')
    fig.tight_layout(); fig.savefig(out_dir / 'chain_trace.png', dpi=150); plt.close(fig)


def plot_scatter_comparison(art_u, art_f, theta_from_fold, out_dir):
    """scatter_comparison.png -- unfolded theta vs folded z vs unfolded-from-fold."""
    su = art_u['samples'].cpu().numpy()
    sf = art_f['samples'].cpu().numpy()
    tf = theta_from_fold.cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = [
        (axes[0], su, 'Unfolded MH ($\\theta$-space)', r'$\theta_1$', '#CC4444'),
        (axes[1], sf, 'Folded MH ($z$-space)', r'$z_1$', '#4488CC'),
        (axes[2], tf, 'Unfolded-from-fold ($\\theta$, random branch)', r'$\theta_1$', '#44AA66')]
    for ax, s, title, xl, c in specs:
        ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.15, color=c)
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(r'$\theta_2$')
        ax.set_xlim(-5, 5); ax.set_ylim(-3, 18)
    axes[1].set_xlim(0, 5)
    fig.tight_layout(); fig.savefig(out_dir / 'scatter_comparison.png', dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")

    out_dir = Path(__file__).resolve().parent / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Target ──
    base = AsymmetricDoubleBanana(a=1.0, b=0.5, sigma=0.5, tau=2.0, device=str(DEVICE))
    unfolded_t = UnfoldedTarget(base)
    folded_smooth = FoldedTargetSmooth(base)
    folded_constr = FoldedTargetConstrained(base)
    fold = ReflectionFold(dim=2, fold_axes=[0])

    # ── Training data ──
    print("Generating target samples (rejection)...")
    theta_samples = sample_banana(base, HP['n_train'], DEVICE)
    z_samples = fold.fold(theta_samples)          # (|theta1|, theta2), z1 >= 0
    print(f"  theta1 range [{theta_samples[:,0].min():.2f}, {theta_samples[:,0].max():.2f}], "
          f"folded z1 in [{z_samples[:,0].min():.2f}, {z_samples[:,0].max():.2f}]")

    # ── Pipeline A: unfolded ──
    m_unf, art_unf = run_pipeline('unfolded', unfolded_t, unfolded_t, unfolded_t,
                                  theta_samples, support_fn=None, out_dir=out_dir)

    # ── Pipeline B: folded ──
    m_fold, art_fold = run_pipeline('folded', folded_smooth, folded_smooth, folded_constr,
                                    z_samples, support_fn=half_plane_support, out_dir=out_dir)

    # ── Unfold folded chain back to theta-space (random branch) ──
    z_chain = art_fold['samples']
    branch = torch.randint(0, fold.n_branches(), (z_chain.shape[0],), device=z_chain.device)
    theta_from_fold = z_chain.clone()
    flip = (branch & 1).bool()                    # branch 1 flips axis 0 sign
    theta_from_fold[flip, 0] = -theta_from_fold[flip, 0]
    ess_unfolded_theta = compute_ess_from_samples(theta_from_fold)
    m_fold['extended']['ess_unfolded_theta'] = float(ess_unfolded_theta)
    m_fold['extended']['note_ess'] = (
        "ess_per_sample is measured on the native z-chain (true mixing); "
        "ess_unfolded_theta is on theta1 after random-branch unfolding, where "
        "the i.i.d. sign artificially decorrelates theta1.")

    # ── Improvement ratios ──
    def ratio(a, b):
        return float(a / b) if b > 0 else float('inf')
    improvement = {
        'osc_reduction_ratio': ratio(m_unf['osc_bound'], m_fold['osc_bound']),
        'gamma_improvement_ratio': ratio(m_fold['certified_gamma'], m_unf['certified_gamma']),
        'ess_improvement_ratio': ratio(m_fold['ess_per_sample'], m_unf['ess_per_sample']),
    }

    results = {'unfolded': m_unf, 'folded': m_fold, 'improvement': improvement,
               'config': {'hp': HP, 'n_cert': N_CERT, 'zeta': ZETA,
                          'headline_rho': HEADLINE_RHO, 'n_chains': N_CHAINS,
                          'chain_length': CHAIN_LENGTH, 'burnin': BURNIN, 'seed': SEED,
                          'target': 'AsymmetricDoubleBanana(a=1.0,b=0.5,sigma=0.5,tau=2.0)'}}

    with open(out_dir / 'toy_fold_results.json', 'w') as fp:
        json.dump(results, fp, indent=2)

    # ── Plots ──
    print("\nGenerating figures...")
    plot_targets(base, out_dir)
    plot_oscillation_comparison(art_unf, art_fold, unfolded_t, folded_smooth, out_dir)
    plot_chain_trace(art_unf, art_fold, out_dir)
    plot_scatter_comparison(art_unf, art_fold, theta_from_fold, out_dir)

    # ── Report ──
    print(f"\n{'='*72}\n  RESULTS\n{'='*72}")
    hdr = f"  {'metric':<26}{'unfolded':>16}{'folded':>16}"
    print(hdr); print("  " + "-" * 56)
    for key, label in [('osc_bound', 'osc(log pi/q)'),
                       ('certified_gamma', f'certified gamma (rho={HEADLINE_RHO})'),
                       ('ess_per_sample', 'ESS / sample'),
                       ('ess_per_gradient', 'ESS / model-eval'),
                       ('acceptance_rate', 'MH acceptance'),
                       ('train_loss_final', 'final NLL')]:
        print(f"  {label:<26}{m_unf[key]:>16.5f}{m_fold[key]:>16.5f}")
    print("  " + "-" * 56)
    print(f"  {'osc reduction (x)':<26}{improvement['osc_reduction_ratio']:>32.3f}")
    print(f"  {'gamma improvement (x)':<26}{improvement['gamma_improvement_ratio']:>32.3f}")
    print(f"  {'ESS improvement (x)':<26}{improvement['ess_improvement_ratio']:>32.3f}")
    print(f"\n  Results -> {out_dir / 'toy_fold_results.json'}")
    print(f"  Figures -> {out_dir}/*.png")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
