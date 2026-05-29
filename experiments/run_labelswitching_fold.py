"""
FolT-MCMC experiment: PermutationFold on label-switching Gaussian mixtures.

Target:  pi(theta) = (1/k!) sum_{sigma in S_k} N(theta; mu_sigma, I), where theta
splits into k component blocks of size p and S_k permutes the blocks. This is
the canonical multi-modality of structural identification (label switching):
k!-fold permutation symmetry.

  (A) Unfolded -- train a RealNVP in the full D=k*p space, which must cover all
      k! equivalent modes.
  (B) Folded   -- PermutationFold sorts component blocks by their first
      parameter, collapsing the k! modes into a single mode in the sorted
      fundamental domain. Sampling, certification and training all happen IN
      the folded space (no unfold needed -- we only care about sorted params).

This mirrors the Phase-3 mixture experiment exactly (same module stack, same
certificate, same MH kernel); only the symmetry group (S_k instead of a single
reflection) and the target change. Four configs span k! = 2, 6, 24, 6.

Run:
    conda activate lcnf
    set KMP_DUPLICATE_LIB_OK=TRUE
    cd C:\\FolT-MCMC
    python experiments/run_labelswitching_fold.py
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

from folt.folding import PermutationFold
from folt.targets.label_switching_mixture import LabelSwitchingMixture
from folt.transport import train_flow
from folt.certification import (compute_oscillation_bound, compute_spectral_gap,
                                mengersen_tweedie_bound, quantile_core_certificate,
                                v1_covering_certificate)
from folt.mh_kernel import run_independence_mh_multichain, compute_ess_from_samples

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QUICK = os.environ.get('FOLT_QUICK') == '1'

N_CERT = 20000
ZETA = 0.05
HEADLINE_RHO = 0.05
N_CHAINS = 4
CHAIN_LENGTH = 5000
BURNIN = 1000
SEED = 0

# Component centers separated by 3 std along every parameter -> very low valley
# density between permutation modes. Architecture per config follows the
# CerT-MCMC-v2 banana scaling rule (n_layers, hidden_dim grow with D).
CONFIGS = [
    dict(tag='k2_p2', k=2, p=2,
         centers=[[2.0, 0.5], [5.0, 0.8]],
         n_layers=8,  hidden_dim=128, n_epochs=2000, n_train=20000),
    dict(tag='k3_p2', k=3, p=2,
         centers=[[2.0, 0.5], [5.0, 0.8], [8.0, 0.3]],
         n_layers=10, hidden_dim=128, n_epochs=2500, n_train=30000),
    dict(tag='k4_p2', k=4, p=2,
         centers=[[2.0, 0.5], [5.0, 0.8], [8.0, 0.3], [11.0, 0.6]],
         n_layers=12, hidden_dim=192, n_epochs=3500, n_train=40000),
    dict(tag='k3_p4', k=3, p=4,
         centers=[[2.0, 0.5, 1.0, 0.3], [5.0, 0.8, 0.5, 1.2], [8.0, 0.3, 0.8, 0.7]],
         n_layers=14, hidden_dim=256, n_epochs=3000, n_train=40000),
]


# ══════════════════════════════════════════════════════════════
# Target adapters
# ══════════════════════════════════════════════════════════════

class UnfoldedTarget:
    def __init__(self, base):
        self.base = base; self.dim = base.dim
    def log_prob(self, theta):
        return self.base.log_prob(theta)


class FoldedTargetSmooth:
    def __init__(self, base):
        self.base = base; self.dim = base.dim
    def log_prob(self, z):
        return self.base.log_prob_folded(z)


class FoldedTargetConstrained:
    def __init__(self, base):
        self.base = base; self.dim = base.dim
    def log_prob(self, z):
        return self.base.log_prob_folded_hard(z)


def make_sorted_support(k, p):
    """Support indicator for the sorted fundamental domain (blocks ordered by param 0)."""
    def support(theta):
        blocks = theta.view(-1, k, p)
        sv = blocks[:, :, 0]
        return (sv[:, :-1] <= sv[:, 1:]).all(dim=-1)
    return support


# ══════════════════════════════════════════════════════════════
# One transport pipeline (identical to Phase 3)
# ══════════════════════════════════════════════════════════════

def run_pipeline(name, dim, hp, train_target, cert_target, mh_target,
                 train_samples, support_fn):
    print(f"\n  ---- pipeline [{name}] (D={dim}) ----")
    t0 = time.time()
    flow, history = train_flow(train_target, dim, hp,
                               train_samples=train_samples, device=DEVICE,
                               verbose=False)
    print(f"    trained {hp['n_epochs']} epochs in {time.time()-t0:.0f}s, "
          f"NLL={history['nll_final']:.3f}")

    osc = compute_oscillation_bound(flow, cert_target, N_CERT, dim,
                                    support_fn=support_fn, device=DEVICE)
    full_osc, r = osc['full_osc'], osc['r']
    gamma_full = compute_spectral_gap(full_osc)
    gamma_mt = mengersen_tweedie_bound(full_osc)
    qcore = quantile_core_certificate(r, zeta=ZETA)
    v1 = v1_covering_certificate(flow, cert_target, full_osc, dim, len(r),
                                 zeta=ZETA, device=DEVICE)

    head = qcore['levels'].get(HEADLINE_RHO, {})
    if head.get('feasible'):
        certified_gamma, certified_osc = head['gamma_core'], head['c_formal']
    else:
        certified_gamma, certified_osc = gamma_full, full_osc

    mh = run_independence_mh_multichain(flow, mh_target, dim,
                                        n_chains=N_CHAINS, chain_length=CHAIN_LENGTH,
                                        burnin=BURNIN, device=DEVICE, verbose=False)
    n_prop_total = N_CHAINS * (CHAIN_LENGTH + BURNIN)
    print(f"    full_osc={full_osc:.3f}  gamma_core(rho={HEADLINE_RHO})={certified_gamma:.4g}  "
          f"accept={mh['accept_rate']:.3f}  ESS/sample={mh['ess_per_sample']:.4f}  "
          f"(n_valid={osc['n_valid']})")

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
            'accept_each': mh['accept_each'], 'v1_covering': v1,
            'quantile_core': {'eps_n': qcore['eps_n'],
                              'levels': {str(k): v for k, v in qcore['levels'].items()}},
        },
    }
    artifacts = {'samples': mh['samples_concat']}
    return metrics, artifacts


# ══════════════════════════════════════════════════════════════
# One config
# ══════════════════════════════════════════════════════════════

def run_config(cfg, out_dir):
    k, p = cfg['k'], cfg['p']
    dim = k * p
    centers = torch.tensor(cfg['centers'], dtype=torch.float32)
    base = LabelSwitchingMixture(k=k, p=p, component_centers=centers,
                                 component_std=1.0, device=str(DEVICE))
    n_perms = base.n_perms
    print(f"\n{'='*72}\n  CONFIG [{cfg['tag']}]  k={k}, p={p}, D={dim}, k!={n_perms}\n{'='*72}")

    n_epochs = cfg['n_epochs']
    n_train = cfg['n_train']
    if QUICK:
        n_epochs = min(n_epochs, 60); n_train = min(n_train, 3000)
    hp = dict(n_layers=cfg['n_layers'], hidden_dim=cfg['hidden_dim'],
              lr=1e-3, n_epochs=n_epochs, n_train=n_train, batch_size=512)

    unf_t = UnfoldedTarget(base)
    fol_smooth = FoldedTargetSmooth(base)
    fol_constr = FoldedTargetConstrained(base)
    support_fn = make_sorted_support(k, p)

    theta_train = base.sample(n_train)
    z_train = base.sample_folded(n_train)

    m_unf, art_unf = run_pipeline('unfolded', dim, hp, unf_t, unf_t, unf_t,
                                  theta_train, support_fn=None)
    m_fold, art_fold = run_pipeline('folded', dim, hp, fol_smooth, fol_smooth, fol_constr,
                                    z_train, support_fn=support_fn)

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
              'config': {'tag': cfg['tag'], 'k': k, 'p': p, 'dim': dim, 'n_perms': n_perms,
                         'n_layers': cfg['n_layers'], 'hidden_dim': cfg['hidden_dim'],
                         'n_epochs': hp['n_epochs'], 'n_train': hp['n_train']}}

    # Per-config 2D scatter (first block: param0 vs param1).
    plot_scatter(art_unf['samples'], art_fold['samples'], base, cfg['tag'], dim, out_dir)
    return result


# ══════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════

def plot_scatter(unf_samples, fol_samples, base, tag, dim, out_dir):
    su = unf_samples.cpu().numpy()
    sf = fol_samples.cpu().numpy()
    centers = base.centers.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, s, title, c in [(axes[0], su, 'Unfolded MH (all k! modes)', '#CC4444'),
                            (axes[1], sf, 'Folded MH (sorted, 1 mode)', '#4488CC')]:
        ax.scatter(s[:, 0], s[:, 1], s=3, alpha=0.15, color=c)
        ax.scatter(centers[:, 0], centers[:, 1], marker='x', s=80, color='k',
                   zorder=5, label='component centers')
        ax.set_title(title); ax.set_xlabel('block-0 param 0'); ax.set_ylabel('block-0 param 1')
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(f'{tag}: first component block projection')
    fig.tight_layout(); fig.savefig(out_dir / f'labelswitching_{tag}_scatter.png', dpi=150)
    plt.close(fig)


def plot_summary(results, out_dir):
    order = sorted(results.keys(), key=lambda t: results[t]['config']['n_perms'])
    nperms = [results[t]['config']['n_perms'] for t in order]
    dims = [results[t]['config']['dim'] for t in order]
    unf_g = [results[t]['unfolded']['certified_gamma'] for t in order]
    fol_g = [results[t]['folded']['certified_gamma'] for t in order]

    # gamma vs k! (number of modes)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(nperms, unf_g, 'o-', color='#CC4444', lw=2, label='unfolded')
    ax.plot(nperms, fol_g, 's-', color='#4488CC', lw=2, label='folded')
    for t, x, y in zip(order, nperms, fol_g):
        ax.annotate(t, (x, y), fontsize=7, xytext=(0, 6), textcoords='offset points')
    ax.set_xlabel('number of equivalent modes  k!'); ax.set_xscale('log')
    ax.set_ylabel(rf'certified $\gamma$ (quantile-core, $\rho$={HEADLINE_RHO})')
    ax.set_title('Certified spectral gap vs label-switching multiplicity')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / 'labelswitching_gamma_vs_k.png', dpi=150); plt.close(fig)

    # gamma vs D
    order_d = sorted(results.keys(), key=lambda t: results[t]['config']['dim'])
    dd = [results[t]['config']['dim'] for t in order_d]
    ug = [results[t]['unfolded']['certified_gamma'] for t in order_d]
    fg = [results[t]['folded']['certified_gamma'] for t in order_d]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dd, ug, 'o-', color='#CC4444', lw=2, label='unfolded')
    ax.plot(dd, fg, 's-', color='#4488CC', lw=2, label='folded')
    for t, x, y in zip(order_d, dd, fg):
        ax.annotate(t, (x, y), fontsize=7, xytext=(0, 6), textcoords='offset points')
    ax.set_xlabel('dimension D = k*p')
    ax.set_ylabel(rf'certified $\gamma$ (quantile-core, $\rho$={HEADLINE_RHO})')
    ax.set_title('Certified spectral gap vs dimension')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / 'labelswitching_gamma_vs_D.png', dpi=150); plt.close(fig)


def print_report(results):
    print(f"\n{'='*72}\n  SUMMARY\n{'='*72}")
    print(f"  {'config':<10}{'D':>4}{'k!':>5}{'unf gamma':>12}{'fol gamma':>12}"
          f"{'ratio':>9}{'unf osc':>10}{'fol osc':>10}")
    for t in sorted(results.keys(), key=lambda s: results[s]['config']['n_perms']):
        c, u, f = results[t]['config'], results[t]['unfolded'], results[t]['folded']
        rr = results[t]['improvement']['gamma_improvement_ratio']
        print(f"  {t:<10}{c['dim']:>4}{c['n_perms']:>5}{u['certified_gamma']:>12.5f}"
              f"{f['certified_gamma']:>12.5f}{rr:>9.2f}"
              f"{u['certified_osc_core']:>10.3f}{f['certified_osc_core']:>10.3f}")


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"Device: {DEVICE}  (QUICK={QUICK})")
    out_dir = Path(__file__).resolve().parent / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = CONFIGS[:2] if QUICK else CONFIGS
    results = {}
    for cfg in configs:
        results[cfg['tag']] = run_config(cfg, out_dir)
        # incremental save so a late crash still leaves partial results
        with open(out_dir / 'labelswitching_results.json', 'w') as fp:
            json.dump(results, fp, indent=2)

    plot_summary(results, out_dir)
    print_report(results)
    print(f"\n  All outputs -> {out_dir}")


if __name__ == "__main__":
    main()
