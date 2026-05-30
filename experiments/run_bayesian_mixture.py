"""
FolT-MCMC Phase 7: standard Bayesian Gaussian mixture benchmark.

A genuine textbook Bayesian posterior (not a synthetic target): N=500 draws from
a 3-component univariate Gaussian mixture, known weights, unknown component
means/scales. theta = (mu_1, logsig_1, ..., mu_3, logsig_3), D=6, with exact
S_3 label-switching symmetry (3! = 6 equivalent modes). Four samplers:

  (A) Unfolded IMH   -- RealNVP + IMH in the full D=6 space (label switching).
  (B) FolT-MCMC      -- PermutationFold (sort by mu), train/certify/sample in
                        the sorted fundamental domain.
  (C) RPS            -- the unfolded flow of (A) + random label permutation each
                        step (Fruhwirth-Schnatter 2001; Phase 5).
  (D) Post-hoc sort  -- (A)'s chain relabelled by sorting samples by mu (the
                        naivest relabelling baseline; same kernel/certificate as A).

Label switching collapses the *raw* unfolded per-block estimates to the symmetric
average (mu_1 ~ mu_2 ~ mu_3), so unfolded inference is reported both raw (to show
the pathology) and sorted. FolT, RPS and post-hoc sort all recover the distinct
true means; the differentiator is the certificate -- only FolT and the unfolded
kernel get a quantile-core gamma, and only FolT's is non-vacuous.

Architecture/hyperparameters follow Phase-4 k3_p2 (10 layers / 128 hidden /
2000 epochs). Block-shared, permutation-equivariant whitening preconditions the
disparate mu (~3) and logsig (~0) scales. Module stack reused unchanged.

Run:
    conda activate lcnf
    set KMP_DUPLICATE_LIB_OK=TRUE
    cd C:\\FolT-MCMC
    python experiments/run_bayesian_mixture.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from folt.targets.bayesian_mixture import BayesianGaussianMixture
from folt.folding import PermutationFold
from folt.transport import train_flow
from folt.certification import (compute_oscillation_bound, compute_spectral_gap,
                                mengersen_tweedie_bound, quantile_core_certificate,
                                v1_covering_certificate)
from folt.mh_kernel import run_independence_mh_multichain, compute_ess_from_samples
from folt.rps_baseline import RandomPermutationSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUICK = os.environ.get('FOLT_QUICK') == '1'

K, P = 3, 2
DIM = K * P
SEED = 0

# True data-generating parameters.
W_TRUE = [1.0/3, 1.0/3, 1.0/3]
MU_TRUE = [0.0, 3.0, 6.0]
SIG_TRUE = [0.8, 1.0, 0.7]
N_DATA = 500
DATA_SEED = 42

N_CERT = 20000
ZETA = 0.05
HEADLINE_RHO = 0.05
N_CHAINS = 4
CHAIN_LENGTH = 5000
BURNIN = 1000
N_TRAIN = 50000

HP = dict(n_layers=10, hidden_dim=128, lr=1e-3, n_epochs=2000,
          n_train=N_TRAIN, batch_size=512)

RESULTS_DIR = Path(__file__).resolve().parent / 'results'


# ══════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════

def generate_data():
    rng = np.random.RandomState(DATA_SEED)
    comp = rng.choice(K, size=N_DATA, p=W_TRUE)
    data = np.array([rng.normal(MU_TRUE[c], SIG_TRUE[c]) for c in comp])
    return data.astype(np.float64)


# ══════════════════════════════════════════════════════════════
# Block-shared (permutation-equivariant) whitening
# ══════════════════════════════════════════════════════════════

class Whitener:
    """Standardise coordinates with (mean, std) shared across component blocks,
    so the transform commutes with block permutation and preserves S_K symmetry
    and the sort-by-mu fundamental domain. Jacobian is constant (drops out)."""
    def __init__(self, theta_train, k, p, device):
        t = theta_train.view(-1, k, p)
        mean_blk = torch.stack([t[:, :, j].mean() for j in range(p)])
        std_blk = torch.stack([t[:, :, j].std() + 1e-8 for j in range(p)])
        self.mean = mean_blk.repeat(k).to(device)
        self.std = std_blk.repeat(k).to(device)
    def whiten(self, theta):   return (theta - self.mean) / self.std
    def unwhiten(self, u):     return self.mean + self.std * u


class WTarget:
    """Wrap a physical-space log-prob fn to operate on whitened coordinates."""
    def __init__(self, fn, whitener, dim):
        self.fn = fn; self.w = whitener; self.dim = dim
    def log_prob(self, u):
        return self.fn(self.w.unwhiten(u))


def sorted_support_phys(theta):
    mu = theta[:, 0::2]
    return (mu[:, :-1] <= mu[:, 1:]).all(dim=-1)

def wsupport(whitener, phys_support):
    return lambda u: phys_support(whitener.unwhiten(u))


# ══════════════════════════════════════════════════════════════
# Training-data generation: ensemble RW-MH on the sorted posterior
# ══════════════════════════════════════════════════════════════

def generate_training_samples(target, n_samples, n_walkers=2000, burnin=1500,
                              device=DEVICE, seed=SEED):
    """Sample the folded (sorted-by-mu) posterior by vectorised RW-MH, then
    populate all K! unfolded modes by random permutation. Sampling the single
    sorted mode is easy; relabelling gives balanced, correct samples of the
    K!-modal unfolded posterior for the unfolded flow."""
    torch.manual_seed(seed)
    fold = PermutationFold(K, P, sort_param_idx=0)
    mu0 = (torch.rand(n_walkers, K, device=device) * 8 - 1).sort(dim=1).values
    ls0 = torch.randn(n_walkers, K, device=device) * 0.2
    th = torch.stack([mu0, ls0], dim=-1).reshape(n_walkers, DIM)
    lp = target.log_prob_folded_hard(th)
    step = torch.tensor([0.06, 0.04] * K, device=device)

    collected = []
    need = math.ceil(n_samples / n_walkers)
    for it in range(burnin + need):
        prop = th + torch.randn_like(th) * step
        lpp = target.log_prob_folded_hard(prop)
        acc = torch.rand(n_walkers, device=device).log() < (lpp - lp)
        th = torch.where(acc.unsqueeze(1), prop, th)
        lp = torch.where(acc, lpp, lp)
        if it >= burnin:
            collected.append(th.clone())
    z_sorted = fold.fold(torch.cat(collected, dim=0)[:n_samples])
    branches = torch.randint(0, fold.n_branches(), (z_sorted.shape[0],))
    theta_unf = torch.empty_like(z_sorted)
    for b in range(fold.n_branches()):
        m = branches == b
        if m.any():
            theta_unf[m] = fold.unfold(z_sorted[m], branch=b)
    return theta_unf, z_sorted


# ══════════════════════════════════════════════════════════════
# Estimation helpers
# ══════════════════════════════════════════════════════════════

def sort_by_mu(samples):
    n = samples.shape[0]
    blocks = samples.view(n, K, P)
    idx = blocks[:, :, 0].argsort(dim=1).unsqueeze(-1).expand_as(blocks)
    return blocks.gather(1, idx).view(n, -1)


def estimates(samples, sort=True):
    """Posterior mean / median / 95% CI of (mu_k, sig_k). If sort, relabel by mu."""
    s = sort_by_mu(samples) if sort else samples
    s = s.cpu().numpy()
    mu = s[:, 0::2]; sig = np.exp(s[:, 1::2])
    return {
        'mu_mean': [float(x) for x in mu.mean(0)],
        'sig_mean': [float(x) for x in sig.mean(0)],
        'mu_median': [float(x) for x in np.median(mu, 0)],
        'sig_median': [float(x) for x in np.median(sig, 0)],
        'mu_ci95': [[float(np.percentile(mu[:, j], 2.5)),
                     float(np.percentile(mu[:, j], 97.5))] for j in range(K)],
        'sig_ci95': [[float(np.percentile(sig[:, j], 2.5)),
                      float(np.percentile(sig[:, j], 97.5))] for j in range(K)],
    }


def sorted_ess(stacked):
    """Per-sample ESS after sorting each chain by mu (batch-means on coord 0)."""
    n_chains, L, _ = stacked.shape
    ess_each = [compute_ess_from_samples(sort_by_mu(stacked[c])) for c in range(n_chains)]
    return float(sum(ess_each)) / (n_chains * L)


# ══════════════════════════════════════════════════════════════
# Transport pipeline (mirrors Phase 4/6)
# ══════════════════════════════════════════════════════════════

def run_pipeline(name, train_target, mh_target, train_samples, support_fn, whitener):
    print(f"\n  ---- pipeline [{name}] (D={DIM}) ----")
    t0 = time.time()
    flow, history = train_flow(train_target, DIM, HP, train_samples=train_samples,
                               device=DEVICE, verbose=False)
    print(f"    trained {HP['n_epochs']} epochs in {time.time()-t0:.0f}s, "
          f"NLL={history['nll_final']:.3f}")

    osc = compute_oscillation_bound(flow, train_target, N_CERT, DIM,
                                    support_fn=support_fn, device=DEVICE)
    full_osc, r = osc['full_osc'], osc['r']
    qcore = quantile_core_certificate(r, zeta=ZETA)
    head = qcore['levels'].get(HEADLINE_RHO, {})
    if head.get('feasible'):
        certified_gamma, certified_osc = head['gamma_core'], head['c_formal']
    else:
        certified_gamma, certified_osc = compute_spectral_gap(full_osc), full_osc

    t1 = time.time()
    mh = run_independence_mh_multichain(flow, mh_target, DIM, n_chains=N_CHAINS,
                                        chain_length=CHAIN_LENGTH, burnin=BURNIN,
                                        device=DEVICE, verbose=False)
    wall = time.time() - t1
    print(f"    full_osc={full_osc:.3f}  gamma_core={certified_gamma:.4g}  "
          f"accept={mh['accept_rate']:.3f}  (n_valid={osc['n_valid']})")

    metrics = {
        'osc_bound': float(full_osc),
        'certified_gamma': float(certified_gamma),
        'certified_osc_core': float(certified_osc),
        'acceptance_rate': float(mh['accept_rate']),
        'wall_time_seconds': float(wall),
        'train_loss_final': float(history['nll_final']),
        'extended': {
            'gamma_full_osc': float(compute_spectral_gap(full_osc)),
            'gamma_mengersen_tweedie': float(mengersen_tweedie_bound(full_osc)),
            'n_cert_valid': osc['n_valid'],
            'v1_covering': v1_covering_certificate(flow, train_target, full_osc,
                                                   DIM, len(r), zeta=ZETA, device=DEVICE),
            'quantile_core_eps_n': qcore['eps_n'],
        },
    }
    samples_concat = whitener.unwhiten(mh['samples_concat'])
    samples_stacked = whitener.unwhiten(mh['samples'])
    return metrics, flow, samples_concat, samples_stacked


# ══════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════

def plot_data(data, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(data, bins=40, density=True, alpha=0.5, color='#888', label='data (N=500)')
    xs = np.linspace(data.min() - 1, data.max() + 1, 500)
    dens = np.zeros_like(xs)
    for w, m, s in zip(W_TRUE, MU_TRUE, SIG_TRUE):
        dens += w * np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    ax.plot(xs, dens, color='#CC3333', lw=2, label='true mixture density')
    for m in MU_TRUE:
        ax.axvline(m, color='#3366CC', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel('x'); ax.set_ylabel('density')
    ax.set_title('Bayesian Gaussian mixture: data and true density')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_trace(unf_stacked, rps_stacked, fol_stacked, out):
    cols = ['#CC3333', '#33AA55', '#3366CC']
    panels = [(unf_stacked, 'Unfolded IMH (frozen: accept $\\approx$ 0)'),
              (rps_stacked, 'RPS (label switching)'),
              (fol_stacked, 'FolT-MCMC (sorted, stable)')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (st, title) in zip(axes, panels):
        dat = st[0][:, 0::2].cpu().numpy()
        n_show = min(1500, dat.shape[0])
        for j in range(3):
            ax.plot(dat[:n_show, j], color=cols[j], lw=0.6, alpha=0.85,
                    label=f'$\\mu$ block {j}')
        for m in MU_TRUE:
            ax.axhline(m, color='0.6', ls=':', lw=0.7)
        ax.set_xlabel('MH step'); ax.set_title(title); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r'$\mu$'); axes[0].legend(loc='center right', fontsize=8)
    fig.suptitle('Component-mean traces: unfolded IMH cannot move; RPS swaps labels; '
                 'FolT stays sorted and mixes')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_posterior(fol_concat, est, out):
    mu = fol_concat[:, 0::2].cpu().numpy()
    cols = ['#CC3333', '#33AA55', '#3366CC']
    fig, ax = plt.subplots(figsize=(8, 5))
    for j in range(3):
        ax.hist(mu[:, j], bins=70, density=True, alpha=0.55, color=cols[j],
                label=f'$\\mu_{j+1}$ (sorted)')
        ax.axvline(MU_TRUE[j], color=cols[j], ls='--', lw=1.5)
        lo, hi = est['mu_ci95'][j]
        ax.axvspan(lo, hi, color=cols[j], alpha=0.08)
    ax.set_xlabel(r'$\mu$'); ax.set_ylabel('posterior density')
    ax.set_title('Sorted posterior of component means (FolT-MCMC)\n'
                 'dashed = true values; shaded = 95% CI')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_pairplot(unf_post, fol_concat, out):
    # (mu of block 0, mu of block 1). The unfolded posterior is shown via the
    # reference posterior samples (sorted RW-MH + random relabelling): the
    # unfolded IMH chain itself is frozen and would show only one chamber.
    n = min(8000, unf_post.shape[0])
    idx = torch.randperm(unf_post.shape[0])[:n]
    u = unf_post[idx][:, [0, 2]].cpu().numpy()
    f = fol_concat[:, [0, 2]].cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, dat, title, c in [(axes[0], u, 'Unfolded posterior (3! = 6 chambers)', '#CC4444'),
                              (axes[1], f, r'FolT-MCMC (single sorted chamber $\mu_1 \leq \mu_2$)', '#4488CC')]:
        ax.scatter(dat[:, 0], dat[:, 1], s=3, alpha=0.12, color=c)
        ax.set_xlabel(r'$\mu$ block 0'); ax.set_title(title); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    axes[1].plot([-2, 8], [-2, 8], color='0.5', ls='--', lw=0.8)
    axes[0].set_ylabel(r'$\mu$ block 1')
    fig.suptitle('Joint posterior of two component means')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Label-switch diagnostic
# ══════════════════════════════════════════════════════════════

def label_switch_rate(stacked):
    rates = []
    for c in range(stacked.shape[0]):
        mu = stacked[c][:, 0::2].cpu().numpy()
        order = np.argsort(mu, axis=1)
        rates.append(float((order[1:] != order[:-1]).any(axis=1).mean()))
    return float(np.mean(rates))


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}  (QUICK={QUICK})")

    n_train = N_TRAIN
    if QUICK:
        HP['n_epochs'] = min(HP['n_epochs'], 400); n_train = min(n_train, 8000)

    data = generate_data()
    print(f"Data: N={len(data)} from 3-comp GMM  (true mu={MU_TRUE}, sig={SIG_TRUE}, w={W_TRUE})")
    base = BayesianGaussianMixture(data, W_TRUE, n_components=K, device=str(DEVICE))

    print(f"\nGenerating {n_train} training samples (ensemble RW-MH on sorted posterior)...")
    t0 = time.time()
    theta_train, z_train = generate_training_samples(base, n_train, device=DEVICE)
    print(f"  done in {time.time()-t0:.0f}s")

    whit = Whitener(theta_train, K, P, DEVICE)
    print(f"  whitening: mu~N({whit.mean[0]:.2f},{whit.std[0]:.2f}), "
          f"logsig~N({whit.mean[1]:.3f},{whit.std[1]:.3f})")
    u_train = whit.whiten(theta_train)
    uz_train = whit.whiten(z_train)

    unf_train_t = WTarget(base.log_prob, whit, DIM)
    unf_mh_t = WTarget(base.log_prob, whit, DIM)            # full support, no cutoff
    fol_train_t = WTarget(base.log_prob_folded, whit, DIM)
    fol_mh_t = WTarget(base.log_prob_folded_hard, whit, DIM)
    fol_sup = wsupport(whit, sorted_support_phys)

    # (A) Unfolded IMH  (full support -> support_fn=None for the certificate)
    m_unf, flow_unf, c_unf, s_unf = run_pipeline(
        'unfolded', unf_train_t, unf_mh_t, u_train, support_fn=None, whitener=whit)
    # (B) FolT-MCMC
    m_fol, flow_fol, c_fol, s_fol = run_pipeline(
        'folded', fol_train_t, fol_mh_t, uz_train, support_fn=fol_sup, whitener=whit)

    # (C) RPS: reuse the unfolded flow + random permutation (whitened space)
    print("\n  ---- pipeline [rps] (reuse unfolded flow) ----")
    def proposal_sampler(n):
        z = torch.randn(n, DIM, device=DEVICE)
        with torch.no_grad():
            th, _ = flow_unf(z)
        return th
    def proposal_log_prob(u):
        with torch.no_grad():
            return flow_unf.log_prob(u)
    rps = RandomPermutationSampler(unf_mh_t.log_prob, proposal_sampler,
                                   proposal_log_prob, K, P, device=DEVICE)
    t1 = time.time()
    mh_rps = rps.run_multichain(N_CHAINS, CHAIN_LENGTH, BURNIN)
    rps_wall = time.time() - t1
    c_rps = whit.unwhiten(mh_rps['samples_concat'])
    s_rps = whit.unwhiten(mh_rps['samples'])
    print(f"    accept={mh_rps['accept_rate']:.3f}  wall={rps_wall:.1f}s")

    # (D) Post-hoc sort: relabel (A)'s chain by mu (same kernel/certificate as A)

    # ── metrics ─────────────────────────────────────────────────────────────
    ess_unf = sorted_ess(s_unf)
    ess_fol = sorted_ess(s_fol)
    ess_rps = sorted_ess(s_rps)
    ls_unf, ls_fol, ls_rps = (label_switch_rate(s_unf), label_switch_rate(s_fol),
                              label_switch_rate(s_rps))

    methods = {
        'unfolded_imh_raw': {   # raw per-block estimates: collapse to symmetric mean
            'estimates': estimates(c_unf, sort=False),
            'sorted_ess_per_sample': float(ess_unf),
            'acceptance_rate': m_unf['acceptance_rate'],
            'qc_gamma_rho005': m_unf['certified_gamma'],
            'label_switch_rate': ls_unf, 'note': 'raw unsorted per-block'},
        'folt_mcmc': {
            'estimates': estimates(c_fol, sort=True),
            'sorted_ess_per_sample': float(ess_fol),
            'acceptance_rate': m_fol['acceptance_rate'],
            'qc_gamma_rho005': m_fol['certified_gamma'],
            'label_switch_rate': ls_fol},
        'rps': {
            'estimates': estimates(c_rps, sort=True),
            'sorted_ess_per_sample': float(ess_rps),
            'acceptance_rate': float(mh_rps['accept_rate']),
            'qc_gamma_rho005': 'N/A (non-standard kernel)',
            'wall_time_seconds': float(rps_wall), 'label_switch_rate': ls_rps},
        'posthoc_sort': {       # (A)'s samples sorted by mu
            'estimates': estimates(c_unf, sort=True),
            'sorted_ess_per_sample': float(ess_unf),
            'acceptance_rate': m_unf['acceptance_rate'],
            'qc_gamma_rho005': m_unf['certified_gamma'],
            'note': 'same kernel/certificate as Unfolded IMH'},
    }
    results = {
        'data': {'n': N_DATA, 'seed': DATA_SEED, 'w_true': W_TRUE,
                 'mu_true': MU_TRUE, 'sig_true': SIG_TRUE},
        'config': {'k': K, 'p': P, 'dim': DIM, 'n_perms': math.factorial(K),
                   'n_layers': HP['n_layers'], 'hidden_dim': HP['hidden_dim'],
                   'n_epochs': HP['n_epochs'], 'n_train': n_train,
                   'n_chains': N_CHAINS, 'chain_length': CHAIN_LENGTH, 'burnin': BURNIN},
        'methods': methods,
        'pipeline_metrics': {'unfolded': m_unf, 'folded': m_fol},
        'improvement': {
            'gamma_improvement_ratio': float(m_fol['certified_gamma'] / m_unf['certified_gamma'])
                                       if m_unf['certified_gamma'] > 0 else float('inf'),
            'osc_reduction_ratio': float(m_unf['osc_bound'] / m_fol['osc_bound']),
        },
    }
    with open(RESULTS_DIR / 'bayesian_mixture_results.json', 'w') as fp:
        json.dump(results, fp, indent=2)

    # ── plots ───────────────────────────────────────────────────────────────
    plot_data(data, RESULTS_DIR / 'bayesian_mixture_data.png')
    plot_trace(s_unf, s_rps, s_fol, RESULTS_DIR / 'bayesian_mixture_trace.png')
    plot_posterior(c_fol, methods['folt_mcmc']['estimates'],
                   RESULTS_DIR / 'bayesian_mixture_posterior.png')
    plot_pairplot(theta_train, c_fol, RESULTS_DIR / 'bayesian_mixture_pairplot.png')

    # ── report ──────────────────────────────────────────────────────────────
    def gfmt(v): return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
    print(f"\n{'='*92}\n  STANDARD BAYESIAN GMM BENCHMARK  (K=3, D=6, 3!=6; true mu=[0,3,6])\n{'='*92}")
    print(f"  {'Method':<18}{'QC gamma':>12}{'Accept':>9}{'ESS/samp':>10}"
          f"{'mu1(0.0)':>11}{'mu2(3.0)':>11}{'mu3(6.0)':>11}")
    rows = [('Unfolded IMH (raw)', 'unfolded_imh_raw'),
            ('FolT-MCMC', 'folt_mcmc'),
            ('RPS', 'rps'),
            ('Post-hoc sort', 'posthoc_sort')]
    for label, key in rows:
        m = methods[key]; mu = m['estimates']['mu_mean']
        print(f"  {label:<18}{gfmt(m['qc_gamma_rho005']):>12}{m['acceptance_rate']:>9.3f}"
              f"{m['sorted_ess_per_sample']:>10.4f}"
              f"{mu[0]:>11.3f}{mu[1]:>11.3f}{mu[2]:>11.3f}")
    print(f"\n  label-switch rate:  unfolded={ls_unf:.3f}  folded={ls_fol:.3f}  rps={ls_rps:.3f}")
    fe = methods['folt_mcmc']['estimates']
    print(f"  FolT sorted sigma est: {np.array2string(np.array(fe['sig_mean']), precision=3)} "
          f"(true {SIG_TRUE})")
    print(f"  gamma improvement (folded/unfolded): "
          f"{results['improvement']['gamma_improvement_ratio']:.1f}x")
    print(f"\n  outputs -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
