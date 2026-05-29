# Extracted from CerT-MCMC-v2 experiments by benchmark_logreg.py
"""
Independence Metropolis-Hastings kernel for FolT-MCMC.

The proposal is drawn i.i.d. from the trained transport map (theta' = T(z'),
z' ~ N(0, I)), so the MH acceptance ratio reduces to the ratio of importance
weights w = pi / q:

    log alpha = log w(theta') - log w(theta),
    log w(theta) = log pi(theta) - log q(theta).

Because proposals are independent of the current state, the chain's mixing is
controlled entirely by osc(log w) -- exactly the quantity the certificates in
`folt.certification` bound. ESS is estimated by batch means on the first
coordinate.

Multi-chain support is provided via `run_independence_mh_multichain`. A target
that returns -inf outside its support (e.g. the folded half-plane) is handled
correctly: such proposals always reject.
"""

import math
import time

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_weight(flow, target, theta):
    """log w(theta) = log pi(theta) - log q(theta) for a single point (1, D)."""
    return (target.log_prob(theta.unsqueeze(0))
            - flow.log_prob(theta.unsqueeze(0))).item()


def run_independence_mh(flow, target, D, n_samples=5000, n_warmup=500,
                        device=DEVICE, max_init_tries=1000):
    """Run one independence-MH chain proposing from `flow`.

    Returns:
        samples:     (n_samples, D) tensor of post-warmup states.
        accept_rate: post-warmup acceptance fraction.
        log_weights: (n_samples,) numpy array of log w at each retained state.

    Because the proposals are independent of the current state, all
    M = n_samples + n_warmup proposals and their log-weights are precomputed in
    a single batched forward pass; the accept/reject loop is then pure scalar
    arithmetic. This is distributionally identical to a per-step chain but far
    faster (no per-step GPU launches). The initial state is the first proposal
    with a finite log-weight, so a folded target with -inf outside its support
    is initialised on its support.
    """
    flow.eval()
    M = n_samples + n_warmup

    with torch.no_grad():
        z = torch.randn(M, D, device=device)
        theta_prop, _ = flow(z)
        log_w_all = (target.log_prob(theta_prop)
                     - flow.log_prob(theta_prop)).cpu().numpy()
    theta_cpu = theta_prop.cpu()

    finite = np.isfinite(log_w_all)
    if not finite.any():
        raise RuntimeError("No finite-weight proposal: chain has no support.")
    state_idx = int(np.argmax(finite))   # first finite proposal -> initial state
    cur_w = float(log_w_all[state_idx])

    u = np.log(np.random.random(M) + 1e-30)
    state_indices = np.empty(n_samples, dtype=np.int64)
    log_weights = np.empty(n_samples, dtype=np.float64)
    n_accept = 0
    out_pos = 0

    for i in range(M):
        wp = log_w_all[i]
        # wp = -inf (off-support) gives log_alpha = -inf -> always rejects.
        if u[i] < min(0.0, wp - cur_w):
            state_idx = i
            cur_w = wp
            if i >= n_warmup:
                n_accept += 1
        if i >= n_warmup:
            state_indices[out_pos] = state_idx
            log_weights[out_pos] = cur_w
            out_pos += 1

    samples = theta_cpu[state_indices].to(device)
    accept_rate = n_accept / n_samples
    return samples, accept_rate, log_weights


def run_independence_mh_multichain(flow, target, D, n_chains=4,
                                   chain_length=5000, burnin=1000,
                                   device=DEVICE, verbose=True):
    """Run `n_chains` independent MH chains and aggregate diagnostics.

    Returns a dict with the stacked samples (n_chains, chain_length, D), the
    concatenated samples, per-chain and mean acceptance, total ESS (summed
    over chains, batch-means on coordinate 0) and the concatenated log-weights.
    """
    all_samples, all_logw, accepts, ess_each = [], [], [], []
    t0 = time.time()
    for c in range(n_chains):
        s, ar, lw = run_independence_mh(flow, target, D,
                                        n_samples=chain_length, n_warmup=burnin,
                                        device=device)
        all_samples.append(s)
        all_logw.append(lw)
        accepts.append(ar)
        ess_each.append(compute_ess_from_samples(s))
        if verbose:
            print(f"    chain {c+1}/{n_chains}: accept={ar:.3f}, "
                  f"ESS={ess_each[-1]:.0f}  ({time.time()-t0:.1f}s)")

    stacked = torch.stack(all_samples)                 # (n_chains, L, D)
    concat = stacked.reshape(-1, D)                    # (n_chains*L, D)
    logw_concat = np.concatenate(all_logw)

    total_n = n_chains * chain_length
    ess_total = float(sum(ess_each))
    return {
        'samples': stacked,
        'samples_concat': concat,
        'log_weights': logw_concat,
        'accept_rate': float(np.mean(accepts)),
        'accept_each': [float(a) for a in accepts],
        'ess_total': ess_total,
        'ess_each': [float(e) for e in ess_each],
        'ess_per_sample': ess_total / total_n,
        'n_total': total_n,
    }


def compute_ess_from_samples(samples):
    """Batch-means effective sample size, using coordinate 0 as the scalar QoI.

    ESS = n * Var(x) / (b * Var(batch_means)), capped at n. Falls back to n
    when the chain is too short or the batch-means variance underflows.
    """
    n = samples.shape[0]
    if n < 100:
        return float(n)

    x = samples[:, 0].cpu().numpy()
    x = x - x.mean()

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
    """Crude independence-MH ESS proxy: ESS ~ n * accept_rate.

    For an independence sampler with spectral gap gamma the asymptotic
    efficiency is gamma / (2 - gamma); the acceptance rate is a cheap,
    monotone surrogate used for sanity-checking the batch-means estimate.
    """
    return n_samples * accept_rate
