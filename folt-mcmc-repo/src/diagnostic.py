"""Oscillation-based convergence diagnostic (QC gamma).

For an independence Metropolis-Hastings sampler with target pi and proposal q,
the spectral gap is bounded below by 2 / (1 + exp(osc(h))), where
h = log pi - log q and osc is the oscillation over a high-probability set.

This module computes the quantile-core (QC) version of the diagnostic from
samples: it trims the most extreme rho-fraction of |h - median(h)| values and
evaluates the oscillation plus a covering correction on the retained core.

The diagnostic controls the oscillation on the high-posterior-mass core; it is
not a full-support spectral-gap certificate (see Remark on scope in the paper).
"""

import math
import torch


def log_ratio(target, proposal, x):
    """h(x) = log pi(x) - log q(x)."""
    return target.log_prob(x) - proposal.log_prob(x)


def quantile_core_indices(h, rho=0.05):
    """Indices of the (1 - rho) fraction closest to the median of h."""
    med = torch.median(h)
    dev = torch.abs(h - med)
    n = h.shape[0]
    keep = int(math.floor((1.0 - rho) * n))
    order = torch.argsort(dev)
    return order[:keep]


def diagnostic(target, proposal, samples, rho=0.05, covering_correction=0.0):
    """Quantile-core convergence diagnostic QC gamma.

    Parameters
    ----------
    samples : (n, d) tensor
        Certification samples (i.i.d. from the target where available, or
        thinned MCMC draws).
    rho : float
        Trimming fraction for the quantile core.
    covering_correction : float
        Optional additive term 2 * M * eps* accounting for the gap between the
        sample oscillation and the true oscillation over the credible set.

    Returns
    -------
    dict with keys: osc, qc_gamma
    """
    h = log_ratio(target, proposal, samples)
    finite = torch.isfinite(h)
    h = h[finite]
    idx = quantile_core_indices(h, rho)
    h_core = h[idx]
    osc = (h_core.max() - h_core.min()).item() + covering_correction
    qc_gamma = 2.0 / (1.0 + math.exp(osc))
    return {"osc": osc, "qc_gamma": qc_gamma}
