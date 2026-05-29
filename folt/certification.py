# Extracted from CerT-MCMC-v2 experiments by quantile_core_diagnostic.py / benchmark_logreg.py
"""
Spectral-gap certification for FolT-MCMC.

Given a trained transport flow q and a target pi, the importance-weight
residual is

    r(z) = log( pi(theta) / q(theta) ) + const,   theta = T(z),  z ~ N(0, I)

(see `folt.transport.compute_residual`). For an independence MH sampler that
proposes from q, the certified spectral gap is governed by the oscillation of
the log-weight:

    osc(r) = ess-sup r - ess-inf r           =>   gamma >= 2 / (1 + exp(osc))

This module provides three certificates, all extracted unchanged from
CerT-MCMC-v2:

  * `compute_oscillation_bound`   -- full-sample oscillation ptp(r).
  * `quantile_core_certificate`   -- DKW-corrected quantile-core oscillation
                                     C_rho = Q_{1-rho} - Q_rho, which trims the
                                     extreme tails to break the high-dimension
                                     covering barrier.
  * `v1_covering_certificate`     -- the original uniform DKW covering bound
                                     C = osc + 2 * grad_sup * eps*.

`compute_spectral_gap` and `mengersen_tweedie_bound` map an oscillation value
to a certified gap.
"""

import math

import numpy as np
import torch
from scipy.stats import chi2

from .transport import compute_residual

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
# Oscillation -> spectral gap maps
# ══════════════════════════════════════════════════════════════

def compute_spectral_gap(osc_bound):
    """Certified independence-sampler spectral gap from an oscillation bound.

        gamma = 2 / (1 + exp(C)),   C = osc(r).

    C >= ~500 is treated as vacuous (gamma = 0). C = 0 gives gamma = 1.
    """
    if osc_bound > 500:
        return 0.0
    return 2.0 / (1.0 + math.exp(osc_bound))


def mengersen_tweedie_bound(osc_bound):
    """Mengersen-Tweedie uniform-ergodicity gap bound.

    If w = pi/q is bounded with sup w / inf w = exp(osc), the independence
    sampler is uniformly ergodic with rate (1 - 1/M), giving a gap

        gamma_MT = exp(-osc).

    Reported alongside `compute_spectral_gap` as a (looser, classical)
    reference. Vacuous (0) once osc exceeds ~700 to avoid underflow.
    """
    if osc_bound > 700:
        return 0.0
    return math.exp(-osc_bound)


# ══════════════════════════════════════════════════════════════
# Oscillation bound
# ══════════════════════════════════════════════════════════════

def compute_oscillation_bound(flow, target, n_samples, dim,
                              support_fn=None, batch_size=50000, device=DEVICE):
    """Estimate osc(r) = max(r) - min(r) over n_samples latent draws.

    Args:
        flow:       trained SNRealNVP.
        target:     object with `.log_prob(theta)`.
        n_samples:  number of latent z ~ N(0, I) draws.
        dim:        dimension.
        support_fn: optional callable theta -> bool mask. Residuals are kept
                    only where the mask is True; used in the folded pipeline to
                    restrict the oscillation to the fundamental domain
                    (the support of pi_F). If None, all draws are kept.

    Returns:
        dict with `full_osc`, `r` (the kept residuals, numpy), basic moments,
        and bookkeeping counts.
    """
    flow.eval()
    r_list = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            nb = min(batch_size, n_samples - i)
            z = torch.randn(nb, dim, device=device)
            theta, log_det = flow(z)
            r = target.log_prob(theta) + log_det + 0.5 * (z**2).sum(dim=-1)
            if support_fn is not None:
                keep = support_fn(theta)
                r = r[keep]
            r_list.append(r.cpu())
    r_all = torch.cat(r_list).numpy()
    n_valid = len(r_all)

    return {
        'n_eval': n_samples,
        'n_valid': int(n_valid),
        'D': dim,
        'mean': float(np.mean(r_all)),
        'std': float(np.std(r_all)),
        'var': float(np.var(r_all)),
        'min': float(np.min(r_all)),
        'max': float(np.max(r_all)),
        'full_osc': float(np.ptp(r_all)),
        'r': r_all,
    }


# ══════════════════════════════════════════════════════════════
# Quantile-core certificate (DKW corrected)
# ══════════════════════════════════════════════════════════════

def quantile_core_certificate(r, zeta=0.05,
                              rho_levels=(0.005, 0.01, 0.025, 0.05, 0.10, 0.25)):
    """DKW-corrected quantile-core oscillation certificate.

    Instead of the full oscillation (driven by the extreme tails of r), the
    quantile core trims a mass rho from each tail:

        C_rho = Q_{1-rho}(r) - Q_rho(r).

    Finite-sample validity uses the Dvoretzky-Kiefer-Wolfowitz (Massart)
    bound: with confidence 1 - zeta the empirical CDF is within

        eps_n = sqrt( log(2 / zeta) / (2 n) )

    of the truth uniformly, so we widen the quantile levels by eps_n
    (rho -> rho - eps_n on the low side, 1 - rho -> 1 - rho + eps_n on the
    high side) to get a conservative, certified C_rho. A level rho is only
    feasible when rho > eps_n.

    Returns a dict keyed by rho, each with c_raw, c_formal, gamma_core and the
    estimated core mass; plus top-level `eps_n`.
    """
    r = np.asarray(r)
    n = len(r)
    eps_n = math.sqrt(math.log(2.0 / zeta) / (2 * n))

    out = {'n': n, 'eps_n': eps_n, 'levels': {}}
    for rho in rho_levels:
        if rho <= eps_n:
            out['levels'][rho] = {'feasible': False}
            continue

        q_lo = float(np.quantile(r, rho))
        q_hi = float(np.quantile(r, 1 - rho))
        c_raw = q_hi - q_lo

        # DKW-widened (conservative / certified) core
        q_lo_w = float(np.quantile(r, rho - eps_n))
        q_hi_w = float(np.quantile(r, 1 - rho + eps_n))
        c_formal = q_hi_w - q_lo_w

        gamma_core = compute_spectral_gap(c_formal)

        in_core = (r >= q_lo_w) & (r <= q_hi_w)
        r_shifted = r - r.max()
        w = np.exp(r_shifted)
        pi_hat = float(w[in_core].sum() / w.sum())   # target mass inside core
        mu_core = float(in_core.sum() / n)           # proposal mass inside core

        out['levels'][rho] = {
            'feasible': True,
            'c_raw': c_raw,
            'c_formal': c_formal,
            'gamma_core': gamma_core,
            'mu_core': mu_core,
            'pi_hat': pi_hat,
        }
    return out


# ══════════════════════════════════════════════════════════════
# V1 covering certificate
# ══════════════════════════════════════════════════════════════

def v1_covering_certificate(flow, target, full_osc, dim, n, zeta=0.05,
                            n_grad=2000, device=DEVICE):
    """Uniform DKW covering certificate (CerT-MCMC V1).

        C_v1 = osc_sample + 2 * grad_sup * eps*,

    where eps* is the covering radius implied by an n-sample DKW bound in
    dimension `dim`, and grad_sup is the empirical sup of ||grad_z r||. This is
    the bound that collapses (becomes vacuous) for D >= 6, motivating the
    quantile-core certificate above.
    """
    R_alpha = math.sqrt(chi2.ppf(0.95, df=dim))
    log_term = math.log(2.0 / zeta) / (2 * n)
    eps_star = R_alpha * (log_term + math.sqrt(log_term)) ** (1.0 / dim)

    z_g = torch.randn(min(n_grad, n), dim, device=device).requires_grad_(True)
    r_g = compute_residual(flow, target, z_g)
    grad_r = torch.autograd.grad(r_g.sum(), z_g)[0]
    grad_sup = grad_r.norm(dim=-1).max().item()

    v1_cert = full_osc + 2 * grad_sup * eps_star
    return {
        'cert': float(v1_cert),
        'gamma': compute_spectral_gap(v1_cert),
        'eps_star': float(eps_star),
        'grad_sup': float(grad_sup),
        'correction': float(2 * grad_sup * eps_star),
    }
