"""
Simplified Whittle-likelihood target for closely-spaced modal identification.

Posterior over the modal parameters of three closely-spaced structural modes
observed in the response of a tall building during a typhoon. Each mode j has a
natural frequency f_j (Hz) and a damping ratio xi_j, so theta = (f1, xi1, f2,
xi2, f3, xi3) and D = 6.

The single-output PSD model is a sum of three SDOF (Lorentzian) resonances on a
flat noise floor:

    S_model(w) = A * [ sum_j  1 / ((w^2 - w_j^2)^2 + (2 xi_j w_j w)^2)  +  c0 ]

with w = 2*pi*f. The overall scale A is a nuisance parameter profiled out in
closed form (the Whittle MLE A* = mean_k S_data(w_k)/h(w_k)), and c0 is a fixed
relative noise floor, so the *sampled* parameters are exactly the six modal
quantities. Because the resonance sum is invariant under permuting the three
(f_j, xi_j) blocks, the likelihood -- and hence the posterior -- has exact S_3
(label-switching) symmetry: 3! = 6 equivalent modes. This is the structural-ID
analogue of the LabelSwitchingMixture, and PermutationFold (sort by frequency)
collapses the six modes into a single sorted fundamental domain f1<=f2<=f3.

The Whittle log-likelihood of a one-sided PSD (exponentially distributed
periodogram ordinates) is, up to an additive constant,

    log L(theta) = sum_k [ -log S_model(w_k) - S_data(w_k) / S_model(w_k) ].

A uniform (box) prior f_j in [f_lo, f_hi], xi_j in [xi_lo, xi_hi] is imposed by
returning -inf outside the box; inside the box it is constant and drops out.

Interface mirrors LabelSwitchingMixture: log_prob (unfolded), log_prob_folded
(smooth, +log k! offset, for training/certification) and log_prob_folded_hard
(adds the sorted-ordering -inf cutoff, for the MH kernel).
"""

import math
from itertools import permutations

import torch


class CloselySpacedModeLikelihood:
    """
    Whittle likelihood for k closely-spaced SDOF modes (k=3, p=2, D=6).

    Parameters
    ----------
    freq_axis : array-like, shape (F,)
        Frequencies (Hz) of the measured PSD ordinates.
    psd_data : array-like, shape (F,)
        Measured one-sided PSD values at freq_axis.
    k : int
        Number of modes (default 3).
    f_range, xi_range : (float, float)
        Uniform-prior box for frequency (Hz) and damping ratio.
    c0 : float
        Fixed relative noise floor inside the bracket (sets the model's
        peak-to-floor ratio; the absolute level is absorbed by the profiled A).
    device : str
    """

    def __init__(self, freq_axis, psd_data, k=3,
                 f_range=(0.7, 1.1), xi_range=(0.001, 0.05),
                 c0=0.02, device="cuda"):
        self.k = k
        self.p = 2
        self.dim = k * self.p
        self.device = device
        self.f_lo, self.f_hi = f_range
        self.xi_lo, self.xi_hi = xi_range
        self.c0 = c0

        freq = torch.as_tensor(freq_axis, dtype=torch.float32, device=device)
        psd = torch.as_tensor(psd_data, dtype=torch.float32, device=device)
        # Normalise the PSD to unit mean for numerical conditioning. A constant
        # rescaling of S_data only shifts log L by a constant (the profiled A
        # rescales identically), so this does not change the posterior.
        psd = psd / psd.mean()
        self.omega = 2.0 * math.pi * freq          # (F,) rad/s
        self.omega2 = self.omega ** 2
        self.psd = psd                              # (F,)
        self.n_freq = freq.shape[0]

        self.n_perms = math.factorial(k)
        self.log_n_perms = math.log(self.n_perms)

    # ── model PSD shape h(w) and profiled amplitude ─────────────────────────
    def _bracket(self, theta):
        """h(w; theta) = sum_j L_j(w) + c0, shape (batch, F)."""
        f = theta[:, 0::2]                          # (B, k) frequencies
        xi = theta[:, 1::2]                         # (B, k) damping ratios
        wj = 2.0 * math.pi * f                      # (B, k)
        wj2 = wj ** 2                               # (B, k)
        # broadcast over frequency axis -> (B, F, k)
        wk2 = self.omega2.view(1, -1, 1)
        wk = self.omega.view(1, -1, 1)
        diff = wk2 - wj2.unsqueeze(1)               # (B, F, k)
        damp = 2.0 * xi.unsqueeze(1) * wj.unsqueeze(1) * wk   # (B, F, k)
        denom = diff ** 2 + damp ** 2               # (B, F, k)
        L = 1.0 / (denom + 1e-30)
        return L.sum(dim=-1) + self.c0              # (B, F)

    def model_psd(self, theta):
        """Profiled model PSD S_model(w) = A*(theta) * h(w), shape (batch, F)."""
        h = self._bracket(theta)                    # (B, F)
        A = (self.psd.view(1, -1) / h).mean(dim=-1, keepdim=True)  # (B,1) Whittle MLE
        return A * h

    # ── log-likelihood and box prior ────────────────────────────────────────
    def log_lik(self, theta):
        """Pure Whittle log-likelihood (no prior cutoff), finite for all theta.

        Used for flow training and certification: the oscillation regulariser
        evaluates the target at flow-generated points, so a -inf box cutoff
        there would poison the loss with NaNs (cf. Phase 5). The box prior is
        instead imposed via a support_fn mask on the certificate and as a -inf
        cutoff only in the MH targets below."""
        S = self.model_psd(theta)                   # (B, F)
        return (-torch.log(S) - self.psd.view(1, -1) / S).sum(dim=-1)

    def in_box(self, theta):
        """Uniform-prior support indicator (also the certificate support_fn)."""
        f = theta[:, 0::2]; xi = theta[:, 1::2]
        return (((f >= self.f_lo) & (f <= self.f_hi)).all(dim=-1)
                & ((xi >= self.xi_lo) & (xi <= self.xi_hi)).all(dim=-1))

    # ── unfolded posterior ──────────────────────────────────────────────────
    def log_prob(self, theta):
        """log posterior (up to const) in unfolded theta-space (k! modes)."""
        loglik = self.log_lik(theta)
        return torch.where(self.in_box(theta), loglik,
                           torch.full_like(loglik, -float('inf')))

    # ── folded posterior (sorted fundamental domain) ────────────────────────
    def log_prob_folded(self, z):
        """Smooth folded log-prob (no box / ordering cutoff), finite everywhere.

        pi_F = k! * pi on the fundamental domain; the +log(k!) is a constant
        that cancels in the MH ratio and the oscillation. Used for training /
        certification (with a support_fn restricting to the sorted box)."""
        return self.log_lik(z) + self.log_n_perms

    def log_prob_folded_hard(self, z):
        """Folded log-prob with hard box + f1<=f2<=f3 ordering cutoff (MH)."""
        lp = self.log_prob_folded(z)
        f = z[:, 0::2]                              # (B, k) frequencies
        sorted_ok = (f[:, :-1] <= f[:, 1:]).all(dim=-1)
        keep = sorted_ok & self.in_box(z)
        return torch.where(keep, lp, torch.full_like(lp, -float('inf')))
