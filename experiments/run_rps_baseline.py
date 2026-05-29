"""
Phase 5: Random Permutation Sampler (RPS) baseline for label switching.

Compares three samplers on the Phase-4 k3_p2 label-switching Gaussian mixture
(m=3, p=2, D=6, m!=6):

  (A) Unfolded IMH  -- independence MH proposing from a RealNVP trained in the
                       full unfolded D=6 space (must cover all 6 modes).
  (B) FolT-MCMC     -- PermutationFold: IMH in the sorted fundamental domain
                       (single mode), proposing from a folded RealNVP.
  (C) RPS           -- the SAME unfolded flow + IMH as (A), plus a random label
                       permutation after every step (Frühwirth-Schnatter 2001).

Pipelines A and C share one flow; the only difference is C's permutation step.
The fair-comparison metric is the *sorted-space* ESS: every sample is mapped to
the fundamental domain (component blocks sorted by their first parameter) before
the ESS is computed, so all three methods are scored on the same marginal.

RPS is a composite (IMH + permutation) kernel without the Mengersen-Tweedie
minorisation structure, so no certified spectral gap is reported for it -- only
empirical ESS. A and B reuse the Phase-4 quantile-core certificate (rho=0.05).

Reuses the Phase-4 architecture/hyperparameters exactly. Phase 4 did not persist
label-switching flow checkpoints, so the k3_p2 flows are trained once here and
cached to disk; reruns load the cache instead of retraining.

Run:
    conda activate lcnf
    set KMP_DUPLICATE_LIB_OK=TRUE
    cd C:\\FolT-MCMC
    python experiments/run_rps_baseline.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from folt.targets.label_switching_mixture import LabelSwitchingMixture
from folt.transport import SNRealNVP, train_flow
from folt.mh_kernel import run_independence_mh_multichain, compute_ess_from_samples
from folt.rps_baseline import RandomPermutationSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUICK = os.environ.get('FOLT_QUICK') == '1'

# ── k3_p2 config, identical to Phase 4 (run_labelswitching_fold.py) ──────────
CFG = dict(tag='k3_p2', k=3, p=2,
           centers=[[2.0, 0.5], [5.0, 0.8], [8.0, 0.3]],
           n_layers=10, hidden_dim=128, n_epochs=2500, n_train=30000)

# Longer chains than Phase 4 (5000) to give RPS room to mix across all 6 modes.
N_CHAINS = 4
CHAIN_LENGTH = 10000
BURNIN = 2000
SEED = 0

RESULTS_DIR = Path(__file__).resolve().parent / 'results'
PHASE4_JSON = RESULTS_DIR / 'labelswitching_results.json'


# ══════════════════════════════════════════════════════════════
# Target adapters (match Phase 4)
# ══════════════════════════════════════════════════════════════

class UnfoldedTarget:
    def __init__(self, base):
        self.base = base; self.dim = base.dim
    def log_prob(self, theta):
        return self.base.log_prob(theta)


class FoldedTargetSmooth:
    """Smooth folded log-prob (no hard ordering cutoff) -- used for training,
    matching Phase 4. The osc regulariser evaluates the target at flow-generated
    points, so a -inf hard cutoff there would poison the loss with NaNs."""
    def __init__(self, base):
        self.base = base; self.dim = base.dim
    def log_prob(self, z):
        return self.base.log_prob_folded(z)


class FoldedTargetConstrained:
    """Hard ordering constraint (-inf outside the sorted domain) -- used for MH."""
    def __init__(self, base):
        self.base = base; self.dim = base.dim
    def log_prob(self, z):
        return self.base.log_prob_folded_hard(z)


# ══════════════════════════════════════════════════════════════
# Sorted-space ESS (the fair-comparison metric)
# ══════════════════════════════════════════════════════════════

def sort_to_fundamental(samples, k, p):
    """Map samples to the fundamental domain: sort the k blocks by param 0."""
    n = samples.shape[0]
    blocks = samples.view(n, k, p)
    keys = blocks[:, :, 0]
    idx = keys.argsort(dim=1).unsqueeze(-1).expand_as(blocks)
    return blocks.gather(1, idx).view(n, -1)


def sorted_ess(stacked, k, p):
    """Per-sample ESS after sorting each chain into the fundamental domain.

    stacked : (n_chains, L, D). Returns (ess_per_sample, ess_total, ess_each).
    Mirrors Phase 4's batch-means-on-coordinate-0 ESS, but on sorted samples.
    """
    n_chains, L, _ = stacked.shape
    ess_each = []
    for c in range(n_chains):
        s_sorted = sort_to_fundamental(stacked[c], k, p)
        ess_each.append(compute_ess_from_samples(s_sorted))
    ess_total = float(sum(ess_each))
    return ess_total / (n_chains * L), ess_total, [float(e) for e in ess_each]


# ══════════════════════════════════════════════════════════════
# Flow training / caching
# ══════════════════════════════════════════════════════════════

def get_flow(name, target, dim, hp, train_samples):
    """Load a cached k3_p2 flow, or train it once and cache."""
    ckpt = RESULTS_DIR / f"labelswitching_{CFG['tag']}_flow_{name}.pt"
    if ckpt.exists():
        blob = torch.load(ckpt, map_location=DEVICE)
        flow = SNRealNVP(dim=dim, n_layers=blob['n_layers'],
                         hidden_dim=blob['hidden_dim']).to(DEVICE)
        flow.load_state_dict(blob['state_dict'])
        flow.eval()
        print(f"    [{name}] loaded cached flow (NLL={blob.get('nll_final', float('nan')):.3f})")
        return flow, blob.get('nll_final', float('nan'))

    print(f"    [{name}] training flow ({hp['n_epochs']} epochs)...")
    t0 = time.time()
    flow, history = train_flow(target, dim, hp, train_samples=train_samples,
                               device=DEVICE, verbose=False)
    torch.save({'state_dict': flow.state_dict(), 'n_layers': hp['n_layers'],
                'hidden_dim': hp['hidden_dim'], 'nll_final': history['nll_final']}, ckpt)
    print(f"    [{name}] trained in {time.time()-t0:.0f}s, NLL={history['nll_final']:.3f} "
          f"-> cached {ckpt.name}")
    return flow, history['nll_final']


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    k, p = CFG['k'], CFG['p']
    dim = k * p
    print(f"Device: {DEVICE}  (QUICK={QUICK})")
    print(f"Config {CFG['tag']}: k={k}, p={p}, D={dim}, m!={math.factorial(k)}")
    print(f"MCMC: {N_CHAINS} chains x {CHAIN_LENGTH} steps, burn-in {BURNIN}\n")

    n_epochs, n_train = CFG['n_epochs'], CFG['n_train']
    if QUICK:
        n_epochs, n_train = min(n_epochs, 400), min(n_train, 8000)
    hp = dict(n_layers=CFG['n_layers'], hidden_dim=CFG['hidden_dim'],
              lr=1e-3, n_epochs=n_epochs, n_train=n_train, batch_size=512)

    centers = torch.tensor(CFG['centers'], dtype=torch.float32)
    base = LabelSwitchingMixture(k=k, p=p, component_centers=centers,
                                 component_std=1.0, device=str(DEVICE))
    unf_t = UnfoldedTarget(base)
    fol_smooth = FoldedTargetSmooth(base)      # training target (no -inf)
    fol_t = FoldedTargetConstrained(base)      # MH target (hard ordering)

    # ── Phase-4 quantile-core certificates (rho=0.05) ───────────────────────
    qc_unf = qc_fol = None
    if PHASE4_JSON.exists():
        p4 = json.load(open(PHASE4_JSON))[CFG['tag']]
        qc_unf = p4['unfolded']['certified_gamma']
        qc_fol = p4['folded']['certified_gamma']
        print(f"Phase-4 QC gamma (rho=0.05): unfolded={qc_unf:.4f}, folded={qc_fol:.4f}\n")

    # ── Train / load flows (A&C share the unfolded flow) ────────────────────
    print("Flows:")
    theta_train = base.sample(n_train)
    z_train = base.sample_folded(n_train)
    unf_flow, _ = get_flow('unfolded', unf_t, dim, hp, theta_train)
    fol_flow, _ = get_flow('folded', fol_smooth, dim, hp, z_train)
    print()

    # ── (A) Unfolded IMH ────────────────────────────────────────────────────
    print("(A) Unfolded IMH ...")
    t0 = time.time()
    mh_unf = run_independence_mh_multichain(unf_flow, unf_t, dim, n_chains=N_CHAINS,
                                            chain_length=CHAIN_LENGTH, burnin=BURNIN,
                                            device=DEVICE, verbose=False)
    t_unf = time.time() - t0
    ess_unf, esstot_unf, _ = sorted_ess(mh_unf['samples'], k, p)

    # ── (B) FolT-MCMC (folded IMH in sorted space) ──────────────────────────
    print("(B) FolT-MCMC ...")
    t0 = time.time()
    mh_fol = run_independence_mh_multichain(fol_flow, fol_t, dim, n_chains=N_CHAINS,
                                            chain_length=CHAIN_LENGTH, burnin=BURNIN,
                                            device=DEVICE, verbose=False)
    t_fol = time.time() - t0
    ess_fol, esstot_fol, _ = sorted_ess(mh_fol['samples'], k, p)

    # ── (C) Random Permutation Sampler (same unfolded flow as A) ────────────
    print("(C) Random Permutation Sampler ...")
    def proposal_sampler(n):
        z = torch.randn(n, dim, device=DEVICE)
        with torch.no_grad():
            theta, _ = unf_flow(z)
        return theta
    def proposal_log_prob(theta):
        with torch.no_grad():
            return unf_flow.log_prob(theta)
    rps = RandomPermutationSampler(
        target_log_prob=unf_t.log_prob, proposal_sampler=proposal_sampler,
        proposal_log_prob=proposal_log_prob, n_components=k, block_size=p,
        device=DEVICE)
    t0 = time.time()
    mh_rps = rps.run_multichain(N_CHAINS, CHAIN_LENGTH, BURNIN)
    t_rps = time.time() - t0
    ess_rps, esstot_rps, _ = sorted_ess(mh_rps['samples'], k, p)

    # ── Assemble results ────────────────────────────────────────────────────
    def per_eff(t, ess_total):
        return float(t / ess_total) if ess_total > 0 else float('inf')

    results = {
        'unfolded_imh': {
            'sorted_ess_per_sample': float(ess_unf),
            'acceptance_rate': float(mh_unf['accept_rate']),
            'wall_time_seconds': float(t_unf),
            'sec_per_effective_sample': per_eff(t_unf, esstot_unf),
            'qc_gamma_rho005': float(qc_unf) if qc_unf is not None else None,
        },
        'folt_mcmc': {
            'sorted_ess_per_sample': float(ess_fol),
            'acceptance_rate': float(mh_fol['accept_rate']),
            'wall_time_seconds': float(t_fol),
            'sec_per_effective_sample': per_eff(t_fol, esstot_fol),
            'qc_gamma_rho005': float(qc_fol) if qc_fol is not None else None,
        },
        'rps': {
            'sorted_ess_per_sample': float(ess_rps),
            'acceptance_rate': float(mh_rps['accept_rate']),
            'wall_time_seconds': float(t_rps),
            'sec_per_effective_sample': per_eff(t_rps, esstot_rps),
            'qc_gamma_rho005': 'N/A (non-standard kernel)',
        },
        'config': {**CFG, 'dim': dim, 'n_perms': int(math.factorial(k)),
                   'n_chains': N_CHAINS, 'chain_length': CHAIN_LENGTH, 'burnin': BURNIN},
    }
    out = RESULTS_DIR / 'rps_baseline_results.json'
    with open(out, 'w') as fp:
        json.dump(results, fp, indent=2)

    # ── Report ──────────────────────────────────────────────────────────────
    def gfmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
    print(f"\n{'='*78}\n  SUMMARY  (k3_p2: m=3, p=2, D=6, m!=6 | {N_CHAINS}x{CHAIN_LENGTH}, burn-in {BURNIN})\n{'='*78}")
    print(f"  {'Method':<16}{'Sorted ESS/samp':>17}{'Accept':>9}{'Wall(s)':>10}{'s/eff':>9}{'QC gamma(0.05)':>17}")
    rows = [('Unfolded IMH', results['unfolded_imh']),
            ('FolT-MCMC', results['folt_mcmc']),
            ('RPS', results['rps'])]
    for name, r in rows:
        print(f"  {name:<16}{r['sorted_ess_per_sample']:>17.4f}{r['acceptance_rate']:>9.3f}"
              f"{r['wall_time_seconds']:>10.1f}{r['sec_per_effective_sample']:>9.4f}"
              f"{gfmt(r['qc_gamma_rho005']):>17}")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
