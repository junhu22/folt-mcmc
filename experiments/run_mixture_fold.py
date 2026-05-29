"""
FolT-MCMC experiment: well-separated symmetric Gaussian mixture.

Target:  pi(theta) = 0.5 N(+mu, I) + 0.5 N(-mu, I),  mu = (separation/2, 0, ...).
The reflection symmetry is theta_0 -> -theta_0 and the fold boundary theta_0 = 0
sits in the LOW-density valley between the modes (valley/peak ~ exp(-||mu||^2/2)).
This is the regime where folding should win cleanly on every metric: the
unfolded transport must synthesise bimodality on coordinate 0 (hard for a
RealNVP, since the other coordinates carry no information to condition on),
whereas the folded transport only has to model a single unimodal half-space.

Two experiments, both reusing the Phase-2 module stack
(folt.transport / certification / mh_kernel / oscillation_reg) unchanged:

  A. D=2 clean demonstration (mirrors the Phase-2 banana A/B comparison).
  B. Dimension scaling D = 2, 5, 10, 20.

The only differences from the banana experiment are (i) the target and (ii)
that training data come from the mixture's EXACT sampler (no MCMC pre-run).

Run:
    conda activate lcnf
    set KMP_DUPLICATE_LIB_OK=TRUE
    cd C:\\FolT-MCMC
    python experiments/run_mixture_fold.py
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from folt.folding import ReflectionFold
from folt.targets.gaussian_mixture import SymmetricGaussianMixture
from folt.transport import train_flow
from folt.certification import (compute_oscillation_bound, compute_spectral_gap,
                                mengersen_tweedie_bound, quantile_core_certificate,
                                v1_covering_certificate)
from folt.mh_kernel import run_independence_mh_multichain, compute_ess_from_samples

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# Per-dimension transport architecture -- the CerT-MCMC-v2 banana HPARAMS
# (n_layers, hidden_dim). Unfolded and folded share the SAME architecture at
# each D so the comparison is apples-to-apples. Training length is reduced
# from the CerT banana values (5000-10000) since the well-separated mixture
# converges quickly; both pipelines use the same budget per D.
# ──────────────────────────────────────────────────────────────
ARCH_BY_DIM = {
    2:  dict(n_layers=8,  hidden_dim=64),
    5:  dict(n_layers=10, hidden_dim=128),
    10: dict(n_layers=12, hidden_dim=128),
    20: dict(n_layers=16, hidden_dim=256),
}
EPOCHS_BY_DIM = {2: 1500, 5: 2000, 10: 2500, 20: 3000}
NTRAIN_BY_DIM = {2: 20000, 5: 30000, 10: 40000, 20: 50000}

# Quick mode (env FOLT_QUICK=1) for smoke tests.
QUICK = os.environ.get('FOLT_QUICK') == '1'

N_CERT = 20000
ZETA = 0.05
HEADLINE_RHO = 0.05
N_CHAINS = 4
CHAIN_LENGTH = 5000
BURNIN = 1000
SEED = 0


# ══════════════════════════════════════════════════════════════
# Target adapters
# ══════════════════════════════════════════════════════════════

class UnfoldedTarget:
    """Mixture in the original theta-space (2 modes)."""
    def __init__(self, base):
        self.base = base
        self.dim = base.dim

    def log_prob(self, theta):
        return self.base.log_prob(theta)


class FoldedTargetSmooth:
    """Folded target pi_F = 2*pi, smooth everywhere (training / certification)."""
    def __init__(self, base):
        self.base = base
        self.dim = base.dim

    def log_prob(self, z):
        return self.base.log_prob_folded(z)


class FoldedTargetConstrained:
    """Folded target restricted to z_0 >= 0 (-inf outside) for the MH kernel."""
    def __init__(self, base):
        self.base = base
        self.dim = base.dim

    def log_prob(self, z):
        return self.base.log_prob_folded_hard(z)


def half_plane_support(theta):
    return theta[:, 0] >= 0


# ══════════════════════════════════════════════════════════════
# One transport pipeline (train -> certify -> sample)
# ══════════════════════════════════════════════════════════════

def run_pipeline(name, dim, hp, train_target, cert_target, mh_target,
                 train_samples, support_fn, out_dir):
    print(f"\n  ---- pipeline [{name}] (D={dim}) ----")
    t0 = time.time()
    flow, history = train_flow(train_target, dim, hp,
                               train_samples=train_samples, device=DEVICE,
                               verbose=False)
    print(f"    trained {hp['n_epochs']} epochs in {time.time()-t0:.0f}s, "
          f"NLL={history['nll_final']:.3f}")

    osc = compute_oscillation_bound(flow, cert_target, N_CERT, dim,
                                    support_fn=support_fn, device=DEVICE)
    full_osc = osc['full_osc']
    r = osc['r']
    gamma_full = compute_spectral_gap(full_osc)
    gamma_mt = mengersen_tweedie_bound(full_osc)
    qcore = quantile_core_certificate(r, zeta=ZETA)
    v1 = v1_covering_certificate(flow, cert_target, full_osc, dim, len(r),
                                 zeta=ZETA, device=DEVICE)

    head = qcore['levels'].get(HEADLINE_RHO, {})
    if head.get('feasible'):
        certified_gamma = head['gamma_core']
        certified_osc = head['c_formal']
    else:
        certified_gamma = gamma_full
        certified_osc = full_osc

    mh = run_independence_mh_multichain(flow, mh_target, dim,
                                        n_chains=N_CHAINS, chain_length=CHAIN_LENGTH,
                                        burnin=BURNIN, device=DEVICE, verbose=False)
    n_prop_total = N_CHAINS * (CHAIN_LENGTH + BURNIN)
    print(f"    full_osc={full_osc:.3f}  gamma_core(rho={HEADLINE_RHO})={certified_gamma:.4g}  "
          f"accept={mh['accept_rate']:.3f}  ESS/sample={mh['ess_per_sample']:.4f}")

    metrics = {
        'osc_bound': float(full_osc),
        'certified_gamma': float(certified_gamma),
        'certified_osc_core': float(certified_osc),
        'ess_per_sample': float(mh['ess_per_sample']),
        'ess_per_gradient': float(mh['ess_total'] / n_prop_total),
        'acceptance_rate': float(mh['accept_rate']),
        'train_loss_final': float(history['nll_final']),
        'extended': {
            'gamma_full_osc': float(gamma_full),
            'gamma_mengersen_tweedie': float(gamma_mt),
            'headline_rho': HEADLINE_RHO,
            'n_cert_valid': osc['n_valid'], 'n_cert_eval': osc['n_eval'],
            'r_mean': osc['mean'], 'r_std': osc['std'],
            'ess_total': mh['ess_total'], 'ess_each': mh['ess_each'],
            'accept_each': mh['accept_each'],
            'v1_covering': v1,
            'quantile_core': {'eps_n': qcore['eps_n'],
                              'levels': {str(k): v for k, v in qcore['levels'].items()}},
        },
    }
    artifacts = {'flow': flow, 'r': r, 'samples': mh['samples_concat'],
                 'trace': mh['samples'][0]}
    return metrics, artifacts


# ══════════════════════════════════════════════════════════════
# 2D plots (detailed, D=2 only)
# ══════════════════════════════════════════════════════════════

def _grid(x_range, y_range, dim, nx=240, ny=240):
    xs = torch.linspace(*x_range, nx)
    ys = torch.linspace(*y_range, ny)
    gx, gy = torch.meshgrid(xs, ys, indexing='xy')
    pts = torch.zeros(nx * ny, dim)
    pts[:, 0] = gx.reshape(-1)
    pts[:, 1] = gy.reshape(-1)
    return gx.numpy(), gy.numpy(), pts.to(DEVICE)


def plot_targets(base, tag, out_dir):
    gx, gy, pts = _grid((-6, 6), (-5, 5), base.dim)
    with torch.no_grad():
        lp = base.log_prob(pts).reshape(gx.shape).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(gx, gy, np.exp(lp - lp.max()), levels=30, cmap='viridis')
    ax.axvline(0.0, color='red', ls='--', lw=1.5, label=r'fold line $\theta_0=0$')
    ax.set_xlabel(r'$\theta_0$'); ax.set_ylabel(r'$\theta_1$')
    ax.set_title(r'Mixture target $\pi(\theta)$ (valley at fold line)')
    ax.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(out_dir / f'{tag}_target.png', dpi=150); plt.close(fig)

    gx, gy, pts = _grid((0, 6), (-5, 5), base.dim)
    with torch.no_grad():
        lp = base.log_prob_folded(pts).reshape(gx.shape).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(gx, gy, np.exp(lp - lp.max()), levels=30, cmap='viridis')
    ax.set_xlabel(r'$z_0=|\theta_0|$'); ax.set_ylabel(r'$z_1$')
    ax.set_title(r'Folded target $\pi_F(z)$ on $z_0 \geq 0$ (unimodal)')
    fig.tight_layout(); fig.savefig(out_dir / f'{tag}_folded_target.png', dpi=150); plt.close(fig)


def _residual_field(flow, target, x_range, y_range, dim):
    gx, gy, pts = _grid(x_range, y_range, dim, nx=160, ny=160)
    with torch.no_grad():
        logpi = target.log_prob(pts)
        logq = flow.log_prob(pts)
        r = (logpi - logq).cpu().numpy().reshape(gx.shape)
        logpi = logpi.cpu().numpy().reshape(gx.shape)
    r = np.where(logpi > logpi.max() - 8.0, r, np.nan)
    return gx, gy, r - np.nanmedian(r)


def plot_oscillation_comparison(art_u, art_f, unf_t, fol_t, dim, tag, out_dir):
    gxu, gyu, ru = _residual_field(art_u['flow'], unf_t, (-6, 6), (-5, 5), dim)
    gxf, gyf, rf = _residual_field(art_f['flow'], fol_t, (0, 6), (-5, 5), dim)
    allr = np.concatenate([ru[~np.isnan(ru)], rf[~np.isnan(rf)]])
    vmax = np.nanmax(np.abs(allr)) if allr.size else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, gx, gy, rr, title, xl in [
        (axes[0], gxu, gyu, ru, r'Unfolded: $\log(\pi/q)$', r'$\theta_0$'),
        (axes[1], gxf, gyf, rf, r'Folded: $\log(\pi_F/q_F)$', r'$z_0$')]:
        pc = ax.pcolormesh(gx, gy, rr, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(r'$\theta_1$')
        fig.colorbar(pc, ax=ax, label='centered $r$')
    fig.suptitle('Log importance-weight residual (lower spread = tighter certificate)')
    fig.tight_layout(); fig.savefig(out_dir / f'{tag}_oscillation_comparison.png', dpi=150)
    plt.close(fig)


def plot_chain_trace(art_u, art_f, tag, out_dir):
    tu = art_u['trace'][:, 0].cpu().numpy()
    tf = art_f['trace'][:, 0].cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(tu, lw=0.5, color='#CC4444'); axes[0].set_ylabel(r'$\theta_0$')
    axes[0].set_title('Unfolded chain (theta-space, must hop between modes)')
    axes[1].plot(tf, lw=0.5, color='#4488CC'); axes[1].set_ylabel(r'$z_0$')
    axes[1].set_title('Folded chain (z-space, single mode)')
    axes[1].set_xlabel('iteration')
    fig.tight_layout(); fig.savefig(out_dir / f'{tag}_chain_trace.png', dpi=150); plt.close(fig)


def plot_scatter_comparison(art_u, art_f, theta_from_fold, tag, out_dir):
    su = art_u['samples'].cpu().numpy()
    sf = art_f['samples'].cpu().numpy()
    tf = theta_from_fold.cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = [
        (axes[0], su, r'Unfolded MH ($\theta$)', r'$\theta_0$', '#CC4444', (-6, 6)),
        (axes[1], sf, r'Folded MH ($z$)', r'$z_0$', '#4488CC', (0, 6)),
        (axes[2], tf, r'Unfolded-from-fold ($\theta$)', r'$\theta_0$', '#44AA66', (-6, 6))]
    for ax, s, title, xl, c, xlim in specs:
        ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.15, color=c)
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(r'$\theta_1$')
        ax.set_xlim(*xlim); ax.set_ylim(-5, 5)
    fig.tight_layout(); fig.savefig(out_dir / f'{tag}_scatter_comparison.png', dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# One full (dim, separation) experiment
# ══════════════════════════════════════════════════════════════

def run_single_experiment(dim, separation, tag, out_dir, detail=False):
    print(f"\n{'='*72}\n  EXPERIMENT [{tag}]  D={dim}, separation={separation}\n{'='*72}")
    arch = ARCH_BY_DIM[dim]
    n_epochs = EPOCHS_BY_DIM[dim]
    n_train = NTRAIN_BY_DIM[dim]
    if QUICK:
        n_epochs = min(n_epochs, 60)
        n_train = min(n_train, 3000)
    hp = dict(n_layers=arch['n_layers'], hidden_dim=arch['hidden_dim'],
              lr=1e-3, n_epochs=n_epochs, n_train=n_train, batch_size=512)

    base = SymmetricGaussianMixture(dim=dim, separation=separation, device=str(DEVICE))
    unf_t = UnfoldedTarget(base)
    fol_smooth = FoldedTargetSmooth(base)
    fol_constr = FoldedTargetConstrained(base)
    fold = ReflectionFold(dim=dim, fold_axes=[0])

    # Exact training data (no MCMC pre-run needed).
    theta_train = base.sample(n_train)
    z_train = base.sample_folded(n_train)

    m_unf, art_unf = run_pipeline('unfolded', dim, hp, unf_t, unf_t, unf_t,
                                  theta_train, support_fn=None, out_dir=out_dir)
    m_fold, art_fold = run_pipeline('folded', dim, hp, fol_smooth, fol_smooth, fol_constr,
                                    z_train, support_fn=half_plane_support, out_dir=out_dir)

    # Save checkpoints
    torch.save(art_unf['flow'].state_dict(), out_dir / f'{tag}_flow_unfolded.pt')
    torch.save(art_fold['flow'].state_dict(), out_dir / f'{tag}_flow_folded.pt')

    # Unfold the folded chain back to theta-space (random branch).
    z_chain = art_fold['samples']
    branch = torch.randint(0, fold.n_branches(), (z_chain.shape[0],), device=z_chain.device)
    theta_from_fold = z_chain.clone()
    flip = (branch & 1).bool()
    theta_from_fold[flip, 0] = -theta_from_fold[flip, 0]
    m_fold['extended']['ess_unfolded_theta'] = float(compute_ess_from_samples(theta_from_fold))

    def ratio(a, b):
        return float(a / b) if b > 0 else float('inf')
    improvement = {
        'osc_reduction_ratio': ratio(m_unf['osc_bound'], m_fold['osc_bound']),
        'gamma_improvement_ratio': ratio(m_fold['certified_gamma'], m_unf['certified_gamma']),
        'ess_improvement_ratio': ratio(m_fold['ess_per_sample'], m_unf['ess_per_sample']),
        'osc_core_reduction_ratio': ratio(m_unf['certified_osc_core'],
                                          m_fold['certified_osc_core']),
    }
    result = {'unfolded': m_unf, 'folded': m_fold, 'improvement': improvement,
              'config': {'dim': dim, 'separation': separation, 'arch': arch,
                         'n_epochs': hp['n_epochs'], 'n_train': hp['n_train'],
                         'n_cert': N_CERT, 'zeta': ZETA, 'headline_rho': HEADLINE_RHO,
                         'n_chains': N_CHAINS, 'chain_length': CHAIN_LENGTH, 'burnin': BURNIN}}

    if detail:
        plot_targets(base, tag, out_dir)
        plot_oscillation_comparison(art_unf, art_fold, unf_t, fol_smooth, dim, tag, out_dir)
        plot_chain_trace(art_unf, art_fold, tag, out_dir)
        plot_scatter_comparison(art_unf, art_fold, theta_from_fold, tag, out_dir)
        with open(out_dir / f'{tag}_results.json', 'w') as fp:
            json.dump(result, fp, indent=2)
        print(f"  [detail figures + {tag}_results.json saved]")

    return result


# ══════════════════════════════════════════════════════════════
# Scaling plots
# ══════════════════════════════════════════════════════════════

def plot_scaling(scaling, out_dir):
    dims = sorted(int(k.split('=')[1]) for k in scaling.keys())
    unf_g = [scaling[f'D={d}']['unfolded']['certified_gamma'] for d in dims]
    fol_g = [scaling[f'D={d}']['folded']['certified_gamma'] for d in dims]
    unf_o = [scaling[f'D={d}']['unfolded']['certified_osc_core'] for d in dims]
    fol_o = [scaling[f'D={d}']['folded']['certified_osc_core'] for d in dims]
    unf_n = [scaling[f'D={d}']['unfolded']['train_loss_final'] for d in dims]
    fol_n = [scaling[f'D={d}']['folded']['train_loss_final'] for d in dims]

    # gamma vs dim (headline)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dims, unf_g, 'o-', color='#CC4444', lw=2, label='unfolded')
    ax.plot(dims, fol_g, 's-', color='#4488CC', lw=2, label='folded')
    ax.set_xlabel('dimension D'); ax.set_ylabel(rf'certified $\gamma$ (quantile-core, $\rho$={HEADLINE_RHO})')
    ax.set_title('Certified spectral gap vs dimension'); ax.set_xticks(dims)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / 'mixture_scaling_gamma_vs_dim.png', dpi=150); plt.close(fig)

    # osc vs dim
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dims, unf_o, 'o-', color='#CC4444', lw=2, label='unfolded')
    ax.plot(dims, fol_o, 's-', color='#4488CC', lw=2, label='folded')
    ax.set_xlabel('dimension D'); ax.set_ylabel(rf'quantile-core osc $C_\rho$ ($\rho$={HEADLINE_RHO})')
    ax.set_title('Quantile-core oscillation vs dimension'); ax.set_xticks(dims)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / 'mixture_scaling_osc_vs_dim.png', dpi=150); plt.close(fig)

    # nll vs dim
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dims, unf_n, 'o-', color='#CC4444', lw=2, label='unfolded')
    ax.plot(dims, fol_n, 's-', color='#4488CC', lw=2, label='folded')
    ax.set_xlabel('dimension D'); ax.set_ylabel('final NLL'); ax.set_xticks(dims)
    ax.set_title('Transport fit (NLL) vs dimension')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / 'mixture_scaling_nll_vs_dim.png', dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Experiment runners
# ══════════════════════════════════════════════════════════════

def run_d2_demo(out_dir):
    print("\n########## EXPERIMENT A: D=2 CLEAN DEMONSTRATION ##########")
    return run_single_experiment(dim=2, separation=6.0, tag='mixture_d2',
                                 out_dir=out_dir, detail=True)


def run_dimension_scaling(out_dir, d2_result=None):
    print("\n########## EXPERIMENT B: DIMENSION SCALING ##########")
    dims = [2, 5, 10, 20]
    if QUICK:
        dims = [2, 5]
    scaling = {}
    for d in dims:
        if d == 2 and d2_result is not None:
            scaling['D=2'] = d2_result          # reuse the demo run
        else:
            scaling[f'D={d}'] = run_single_experiment(dim=d, separation=6.0,
                                                      tag=f'mixture_d{d}', out_dir=out_dir,
                                                      detail=False)
    with open(out_dir / 'mixture_scaling_results.json', 'w') as fp:
        json.dump(scaling, fp, indent=2)
    plot_scaling(scaling, out_dir)
    print("  [scaling json + 3 scaling figures saved]")
    return scaling


def print_report(d2, scaling):
    print(f"\n{'='*72}\n  SUMMARY\n{'='*72}")
    u, f = d2['unfolded'], d2['folded']
    print("\n  D=2 mixture:")
    print(f"    {'metric':<22}{'unfolded':>14}{'folded':>14}")
    for k, lbl in [('osc_bound', 'full osc'),
                   ('certified_gamma', f'qc gamma (r={HEADLINE_RHO})'),
                   ('certified_osc_core', f'qc osc (r={HEADLINE_RHO})'),
                   ('train_loss_final', 'NLL'),
                   ('ess_per_sample', 'ESS/sample'),
                   ('acceptance_rate', 'acceptance')]:
        print(f"    {lbl:<22}{u[k]:>14.4f}{f[k]:>14.4f}")

    print("\n  Dimension scaling (quantile-core, rho=0.05):")
    print(f"    {'D':>3}{'unf gamma':>13}{'fol gamma':>13}{'ratio':>9}"
          f"{'unf osc':>11}{'fol osc':>11}")
    for key in sorted(scaling.keys(), key=lambda s: int(s.split('=')[1])):
        d = int(key.split('=')[1])
        uu, ff = scaling[key]['unfolded'], scaling[key]['folded']
        rr = scaling[key]['improvement']['gamma_improvement_ratio']
        print(f"    {d:>3}{uu['certified_gamma']:>13.5f}{ff['certified_gamma']:>13.5f}"
              f"{rr:>9.2f}{uu['certified_osc_core']:>11.4f}{ff['certified_osc_core']:>11.4f}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}  (QUICK={QUICK})")
    out_dir = Path(__file__).resolve().parent / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)

    d2 = run_d2_demo(out_dir)
    scaling = run_dimension_scaling(out_dir, d2_result=d2)
    print_report(d2, scaling)
    print(f"\n  All outputs -> {out_dir}")


if __name__ == "__main__":
    main()
