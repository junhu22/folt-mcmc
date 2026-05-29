"""
FolT-MCMC Phase 6: closely-spaced structural modal identification.

Real application of PermutationFold to operational modal analysis (OMA). The
response of a tall building during a typhoon contains three closely-spaced
modes whose modal parameters are exchangeable, so the Bayesian posterior over
theta = (f1, xi1, f2, xi2, f3, xi3) has exact S_3 (label-switching) symmetry:
3! = 6 equivalent modes. We compare

  (A) Unfolded  -- RealNVP + independence MH in the full D=6 space (must cover
                   all 6 label permutations -> label-switching in the chain).
  (B) FolT-MCMC -- PermutationFold (sort by frequency, f1<=f2<=f3), training,
                   certification and MH in the sorted fundamental domain.

Likelihood: a simplified single-output Whittle likelihood on the measured PSD
(`folt.targets.closely_spaced_modes`). This is deliberately *not* a full
AD-BOMA / multi-output OMA -- only the minimal closely-spaced-mode model needed
to exhibit the label-switching symmetry.

Data: one 30-min typhoon window from the SAT-MCMC processed dataset (total-power
spectrum = trace of the 15-channel cross-PSD matrix over 0.7-1.1 Hz, the band of
the closely-spaced TX2/TY2 cluster). The three modes lie in different physical
directions, so the trace spectrum -- not any single channel -- captures all
three. Building identity/location/height are anonymised. If the dataset is
unavailable, a synthetic PSD calibrated to the same band is generated instead.

Architecture/hyperparameters follow the Phase-4 k3_p2 config (10 layers / 128
hidden). Module stack (transport, certification, mh_kernel, PermutationFold) is
reused unchanged.

Run:
    conda activate lcnf
    set KMP_DUPLICATE_LIB_OK=TRUE
    cd C:\\FolT-MCMC
    python experiments/run_structural_id.py
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

from folt.targets.closely_spaced_modes import CloselySpacedModeLikelihood
from folt.folding import PermutationFold
from folt.transport import train_flow
from folt.certification import (compute_oscillation_bound, compute_spectral_gap,
                                mengersen_tweedie_bound, quantile_core_certificate,
                                v1_covering_certificate)
from folt.mh_kernel import run_independence_mh_multichain, compute_ess_from_samples

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUICK = os.environ.get('FOLT_QUICK') == '1'

K, P = 3, 2
DIM = K * P
C0 = 0.01                      # relative noise floor in the PSD model
F_RANGE = (0.7, 1.1)
XI_RANGE = (0.001, 0.05)

N_CERT = 20000
ZETA = 0.05
HEADLINE_RHO = 0.05
N_CHAINS = 4
CHAIN_LENGTH = 5000
BURNIN = 1000
N_TRAIN = 50000
SEED = 0

# Phase-4 k3_p2 architecture.
HP = dict(n_layers=10, hidden_dim=128, lr=1e-3, n_epochs=2500,
          n_train=N_TRAIN, batch_size=512)

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
PSD_CACHE = RESULTS_DIR / 'structural_psd_data.npz'
# Paper-4 nominal closely-spaced cluster (anonymised), for the sanity check.
MODES_REF = np.array([0.8516, 0.8906, 0.9258])


# ══════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════

def load_psd():
    """Load the cached real-data trace PSD, or synthesise a calibrated fallback."""
    if PSD_CACHE.exists():
        d = np.load(PSD_CACHE, allow_pickle=True)
        meta = f"real typhoon window (id={int(d['window_id'])}, {str(d['source'])})"
        return d['freq_hz'].astype(np.float64), d['psd'].astype(np.float64), meta, False

    # Synthetic fallback calibrated to the closely-spaced band.
    print("  [data] PSD cache not found -> synthetic fallback")
    df = 0.0078125
    freq = np.arange(0.7, 1.1 + df / 2, df)
    w = 2 * np.pi * freq
    f_true, xi_true, A_true, S0 = MODES_REF, [0.012, 0.012, 0.012], [1.0, 1.0, 1.0], 0.01
    S = np.full_like(freq, S0)
    for fj, xj, Aj in zip(f_true, xi_true, A_true):
        wj = 2 * np.pi * fj
        S += Aj / ((w**2 - wj**2)**2 + (2 * xj * wj * w)**2) / 3.0e3
    rng = np.random.default_rng(0)
    psd = S * rng.exponential(1.0, size=S.shape)   # Whittle (exponential) noise
    return freq, psd, "synthetic PSD (calibrated to closely-spaced band)", True


# ══════════════════════════════════════════════════════════════
# Training-data generation: ensemble RW-MH on the sorted posterior
# ══════════════════════════════════════════════════════════════

def generate_training_samples(target, n_samples, n_walkers=2000, burnin=1500,
                              device=DEVICE, seed=SEED):
    """Sample the *folded* (sorted) posterior by vectorised random-walk MH, then
    populate all k! unfolded modes by random permutation.

    Returns (theta_unfolded, z_sorted) each (n_samples, DIM). Sampling the single
    sorted mode is easy; random relabelling then gives balanced, correct samples
    of the k!-modal unfolded posterior to train the unfolded flow."""
    torch.manual_seed(seed)
    fold = PermutationFold(K, P, sort_param_idx=0)
    # init walkers inside the box, sorted by frequency
    f0 = torch.rand(n_walkers, K, device=device) * (F_RANGE[1] - F_RANGE[0]) + F_RANGE[0]
    f0 = f0.sort(dim=1).values
    xi0 = torch.rand(n_walkers, K, device=device) * (XI_RANGE[1] - XI_RANGE[0]) + XI_RANGE[0]
    th = torch.stack([f0, xi0], dim=-1).reshape(n_walkers, DIM)
    lp = target.log_prob_folded_hard(th)
    step = torch.tensor([0.004, 0.0025] * K, device=device)

    collected = []
    need = math.ceil(n_samples / n_walkers)
    total_steps = burnin + need
    for it in range(total_steps):
        prop = th + torch.randn_like(th) * step
        lpp = target.log_prob_folded_hard(prop)
        acc = torch.rand(n_walkers, device=device).log() < (lpp - lp)
        th = torch.where(acc.unsqueeze(1), prop, th)
        lp = torch.where(acc, lpp, lp)
        if it >= burnin:
            collected.append(th.clone())
    z_sorted = torch.cat(collected, dim=0)[:n_samples]      # (n, DIM), sorted
    z_sorted = fold.fold(z_sorted)                          # enforce exact sort
    # unfold: assign each sample a uniformly random permutation branch
    branches = torch.randint(0, fold.n_branches(), (z_sorted.shape[0],))
    theta_unf = torch.empty_like(z_sorted)
    for b in range(fold.n_branches()):
        m = branches == b
        if m.any():
            theta_unf[m] = fold.unfold(z_sorted[m], branch=b)
    return theta_unf, z_sorted


# ══════════════════════════════════════════════════════════════
# Block-shared whitening (permutation-equivariant preconditioning)
# ══════════════════════════════════════════════════════════════
#
# Frequencies (~0.8 Hz, std ~0.02) and damping ratios (~0.015, std ~0.008) live
# on tiny, very different scales -- far from the flow's N(0,I) base, so an
# un-preconditioned flow leaks essentially all proposals outside the prior box.
# We standardise each coordinate, but share the (mean, std) across the three
# component blocks (one pair for all frequencies, one for all dampings) so the
# transform commutes with block permutation: sorting by frequency in whitened
# space equals sorting by frequency in physical space, and the S_3 symmetry is
# preserved exactly. The Jacobian is constant, so it drops from the MH ratio and
# the oscillation certificate.

class Whitener:
    def __init__(self, theta_train, k, p, device):
        t = theta_train.view(-1, k, p)
        mean_blk = torch.stack([t[:, :, j].mean() for j in range(p)])   # (p,)
        std_blk = torch.stack([t[:, :, j].std() + 1e-8 for j in range(p)])
        self.mean = mean_blk.repeat(k).to(device)        # (k*p,)
        self.std = std_blk.repeat(k).to(device)
    def whiten(self, theta):    return (theta - self.mean) / self.std
    def unwhiten(self, u):      return self.mean + self.std * u


class WTarget:
    """Wrap a physical-space log-prob fn to operate on whitened coordinates."""
    def __init__(self, fn, whitener, dim):
        self.fn = fn; self.w = whitener; self.dim = dim
    def log_prob(self, u):
        return self.fn(self.w.unwhiten(u))


def sorted_support_phys(theta):
    f = theta[:, 0::2]
    return (f[:, :-1] <= f[:, 1:]).all(dim=-1)

def wsupport(whitener, phys_support):
    """Lift a physical-space support indicator to whitened space."""
    return lambda u: phys_support(whitener.unwhiten(u))


# ══════════════════════════════════════════════════════════════
# One transport pipeline (mirrors Phase 4)
# ══════════════════════════════════════════════════════════════

def run_pipeline(name, train_target, cert_target, mh_target, train_samples,
                 support_fn, whitener):
    print(f"\n  ---- pipeline [{name}] (D={DIM}) ----")
    t0 = time.time()
    flow, history = train_flow(train_target, DIM, HP, train_samples=train_samples,
                               device=DEVICE, verbose=False)
    print(f"    trained {HP['n_epochs']} epochs in {time.time()-t0:.0f}s, "
          f"NLL={history['nll_final']:.3f}")

    osc = compute_oscillation_bound(flow, cert_target, N_CERT, DIM,
                                    support_fn=support_fn, device=DEVICE)
    full_osc, r = osc['full_osc'], osc['r']
    gamma_full = compute_spectral_gap(full_osc)
    gamma_mt = mengersen_tweedie_bound(full_osc)
    qcore = quantile_core_certificate(r, zeta=ZETA)
    v1 = v1_covering_certificate(flow, cert_target, full_osc, DIM, len(r),
                                 zeta=ZETA, device=DEVICE)
    head = qcore['levels'].get(HEADLINE_RHO, {})
    if head.get('feasible'):
        certified_gamma, certified_osc = head['gamma_core'], head['c_formal']
    else:
        certified_gamma, certified_osc = gamma_full, full_osc

    t1 = time.time()
    mh = run_independence_mh_multichain(flow, mh_target, DIM, n_chains=N_CHAINS,
                                        chain_length=CHAIN_LENGTH, burnin=BURNIN,
                                        device=DEVICE, verbose=False)
    wall = time.time() - t1
    print(f"    full_osc={full_osc:.3f}  gamma_core(rho={HEADLINE_RHO})={certified_gamma:.4g}  "
          f"accept={mh['accept_rate']:.3f}  ESS/sample={mh['ess_per_sample']:.4f}  "
          f"(n_valid={osc['n_valid']})")

    metrics = {
        'osc_bound': float(full_osc),
        'certified_gamma': float(certified_gamma),
        'certified_osc_core': float(certified_osc),
        'ess_per_sample': float(mh['ess_per_sample']),
        'acceptance_rate': float(mh['accept_rate']),
        'wall_time_seconds': float(wall),
        'train_loss_final': float(history['nll_final']),
        'extended': {
            'gamma_full_osc': float(gamma_full),
            'gamma_mengersen_tweedie': float(gamma_mt),
            'n_cert_valid': osc['n_valid'], 'n_cert_eval': osc['n_eval'],
            'ess_total': mh['ess_total'], 'ess_each': mh['ess_each'],
            'accept_each': mh['accept_each'], 'v1_covering': v1,
            'quantile_core': {'eps_n': qcore['eps_n'],
                              'levels': {str(k): v for k, v in qcore['levels'].items()}},
        },
    }
    # MH ran in whitened space; map samples back to physical (Hz / damping).
    samples_concat = whitener.unwhiten(mh['samples_concat'])
    samples_stacked = whitener.unwhiten(mh['samples'])
    artifacts = {'flow': flow, 'samples_concat': samples_concat,
                 'samples_stacked': samples_stacked}
    return metrics, artifacts


# ══════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════

def plot_psd_fit(base, freq, psd, fol_samples, out):
    Sd = (psd / psd.mean())
    th = fol_samples.to(DEVICE)
    post_mean = th.mean(dim=0, keepdim=True)
    with torch.no_grad():
        S_mean = base.model_psd(post_mean)[0].cpu().numpy()
    f = post_mean[0, 0::2].cpu().numpy(); xi = post_mean[0, 1::2].cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(freq, Sd, color='0.4', lw=1.2, label='data PSD (normalised)')
    ax.semilogy(freq, S_mean, color='#CC3333', lw=2, label='model PSD (posterior mean)')
    for j, fj in enumerate(np.sort(f)):
        ax.axvline(fj, color='#3366CC', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('PSD (a.u.)')
    ax.set_title('Closely-spaced modes: PSD fit\n'
                 f'posterior-mean f = {np.array2string(np.sort(f), precision=3)} Hz')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_trace(unf_stacked, fol_stacked, out):
    # chain 0 frequency coordinates (theta[:,0::2]) vs step
    uf = unf_stacked[0][:, 0::2].cpu().numpy()
    ff = fol_stacked[0][:, 0::2].cpu().numpy()
    n_show = min(1500, uf.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cols = ['#CC3333', '#33AA55', '#3366CC']
    for ax, dat, title in [(axes[0], uf, 'Unfolded MH (label switching)'),
                           (axes[1], ff, 'FolT-MCMC (sorted, stable)')]:
        for j in range(3):
            ax.plot(dat[:n_show, j], color=cols[j], lw=0.6, alpha=0.8,
                    label=f'block-{j} freq')
        ax.set_xlabel('MH step'); ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('frequency (Hz)'); axes[0].legend(loc='upper right', fontsize=8)
    fig.suptitle('Frequency traces: unfolded labels swap between modes; '
                 'folded stays sorted')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_marginals(fol_concat, out):
    f = fol_concat[:, 0::2].cpu().numpy()   # already sorted
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = ['#CC3333', '#33AA55', '#3366CC']
    for j in range(3):
        ax.hist(f[:, j], bins=60, density=True, alpha=0.6, color=cols[j],
                label=f'$f_{j+1}$ (sorted)')
        ax.axvline(MODES_REF[j], color=cols[j], ls='--', lw=1.2)
    ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('posterior density')
    ax.set_title('Marginal frequency posteriors (folded, sorted)\n'
                 'dashed = Paper-4 nominal closely-spaced cluster')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_gamma(m_unf, m_fol, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ['Unfolded', 'FolT-MCMC']
    vals = [m_unf['certified_gamma'], m_fol['certified_gamma']]
    bars = ax.bar(labels, vals, color=['#CC4444', '#4488CC'])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}',
                ha='center', va='bottom', fontsize=11)
    ax.set_ylabel(rf'certified $\gamma$ (quantile-core, $\rho$={HEADLINE_RHO})')
    ax.set_title('Certified spectral gap: structural modal ID')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def label_switch_rate(unf_stacked):
    """Fraction of steps where the argsort of the 3 block frequencies changes
    in the unfolded chain -- a direct measure of label switching."""
    rates = []
    for c in range(unf_stacked.shape[0]):
        f = unf_stacked[c][:, 0::2].cpu().numpy()
        order = np.argsort(f, axis=1)
        switches = (order[1:] != order[:-1]).any(axis=1).mean()
        rates.append(float(switches))
    return float(np.mean(rates))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}  (QUICK={QUICK})")

    n_epochs, n_train = HP['n_epochs'], N_TRAIN
    if QUICK:
        HP['n_epochs'] = min(n_epochs, 400); n_train = min(n_train, 8000)

    freq, psd, data_desc, is_synth = load_psd()
    print(f"Data: {data_desc}")
    print(f"  {len(freq)} PSD ordinates in [{freq[0]:.3f}, {freq[-1]:.3f}] Hz, "
          f"df={freq[1]-freq[0]:.4f} Hz, peak/floor={psd.max()/np.median(psd[-8:]):.0f}")

    base = CloselySpacedModeLikelihood(freq, psd, k=K, f_range=F_RANGE,
                                       xi_range=XI_RANGE, c0=C0, device=str(DEVICE))

    print(f"\nGenerating {n_train} training samples (ensemble RW-MH on sorted posterior)...")
    t0 = time.time()
    theta_train, z_train = generate_training_samples(base, n_train, device=DEVICE)
    print(f"  done in {time.time()-t0:.0f}s")

    # Block-shared whitening, fit on the unfolded training samples.
    whit = Whitener(theta_train, K, P, DEVICE)
    print(f"  whitening: freq~N({whit.mean[0]:.3f},{whit.std[0]:.3f}), "
          f"xi~N({whit.mean[1]:.4f},{whit.std[1]:.4f})")
    u_train = whit.whiten(theta_train)        # unfolded, whitened
    uz_train = whit.whiten(z_train)           # folded (sorted), whitened

    # Whitened target adapters: smooth (training/cert) and hard (MH).
    unf_train_t = WTarget(base.log_lik, whit, DIM)
    unf_mh_t = WTarget(base.log_prob, whit, DIM)
    fol_train_t = WTarget(base.log_prob_folded, whit, DIM)
    fol_mh_t = WTarget(base.log_prob_folded_hard, whit, DIM)
    box_sup = wsupport(whit, base.in_box)
    fol_sup = wsupport(whit, lambda th: sorted_support_phys(th) & base.in_box(th))

    m_unf, a_unf = run_pipeline('unfolded', unf_train_t, unf_train_t, unf_mh_t,
                                u_train, support_fn=box_sup, whitener=whit)
    m_fol, a_fol = run_pipeline('folded', fol_train_t, fol_train_t, fol_mh_t,
                                uz_train, support_fn=fol_sup, whitener=whit)

    # ── posterior summaries (sorted folded chain) ──────────────────────────
    fol_c = a_fol['samples_concat']
    f_post = fol_c[:, 0::2].cpu().numpy(); xi_post = fol_c[:, 1::2].cpu().numpy()
    f_mean = np.sort(f_post.mean(0));
    post = {
        'freq_mean_hz': [float(x) for x in f_mean],
        'freq_std_hz': [float(x) for x in f_post.std(0)],
        'damping_mean': [float(x) for x in xi_post.mean(0)],
        'modes_ref_paper4': [float(x) for x in MODES_REF],
    }
    ls_unf = label_switch_rate(a_unf['samples_stacked'])
    ls_fol = label_switch_rate(a_fol['samples_stacked'])

    def ratio(a, b): return float(a / b) if b > 0 else float('inf')
    results = {
        'data': {'description': data_desc, 'is_synthetic': bool(is_synth),
                 'n_freq': len(freq), 'band_hz': [float(freq[0]), float(freq[-1])]},
        'config': {'k': K, 'p': P, 'dim': DIM, 'n_perms': base.n_perms, 'c0': C0,
                   'f_range': list(F_RANGE), 'xi_range': list(XI_RANGE),
                   'n_layers': HP['n_layers'], 'hidden_dim': HP['hidden_dim'],
                   'n_epochs': HP['n_epochs'], 'n_train': n_train,
                   'n_chains': N_CHAINS, 'chain_length': CHAIN_LENGTH, 'burnin': BURNIN},
        'unfolded': m_unf, 'folded': m_fol,
        'posterior': post,
        'label_switch_rate': {'unfolded': ls_unf, 'folded': ls_fol},
        'improvement': {
            'gamma_improvement_ratio': ratio(m_fol['certified_gamma'], m_unf['certified_gamma']),
            'osc_reduction_ratio': ratio(m_unf['osc_bound'], m_fol['osc_bound']),
            'ess_improvement_ratio': ratio(m_fol['ess_per_sample'], m_unf['ess_per_sample']),
        },
    }
    with open(RESULTS_DIR / 'structural_id_results.json', 'w') as fp:
        json.dump(results, fp, indent=2)

    # ── plots ───────────────────────────────────────────────────────────────
    plot_psd_fit(base, freq, psd, fol_c, RESULTS_DIR / 'structural_psd_fit.png')
    plot_trace(a_unf['samples_stacked'], a_fol['samples_stacked'],
               RESULTS_DIR / 'structural_trace.png')
    plot_marginals(fol_c, RESULTS_DIR / 'structural_marginals.png')
    plot_gamma(m_unf, m_fol, RESULTS_DIR / 'structural_gamma.png')

    # ── report ────────────────────────────────────────────────────────────
    print(f"\n{'='*74}\n  STRUCTURAL MODAL ID  (k=3, p=2, D=6, 3!=6)\n{'='*74}")
    print(f"  {'Method':<12}{'QC gamma(0.05)':>16}{'Accept':>10}{'ESS/sample':>13}{'Wall(s)':>10}")
    for nm, m in [('Unfolded', m_unf), ('FolT-MCMC', m_fol)]:
        print(f"  {nm:<12}{m['certified_gamma']:>16.4f}{m['acceptance_rate']:>10.3f}"
              f"{m['ess_per_sample']:>13.4f}{m['wall_time_seconds']:>10.2f}")
    print(f"\n  label-switch rate:  unfolded={ls_unf:.3f}   folded={ls_fol:.3f}")
    print(f"  posterior freqs (Hz, sorted): {np.array2string(f_mean, precision=4)}")
    print(f"  Paper-4 nominal cluster:      {np.array2string(MODES_REF, precision=4)}")
    print(f"  gamma improvement: {results['improvement']['gamma_improvement_ratio']:.2f}x")
    print(f"\n  outputs -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
