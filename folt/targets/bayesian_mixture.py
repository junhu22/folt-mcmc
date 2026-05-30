"""
Standard Bayesian Gaussian mixture model posterior.

A textbook K-component univariate Gaussian mixture with known weights and
unknown component means/scales. Parameters are

    theta = (mu_1, logsig_1, mu_2, logsig_2, ..., mu_K, logsig_K),   D = 2K,

with sigma_k = exp(logsig_k) so the parameter space is unbounded (needed for
RealNVP). Because the mixture likelihood is invariant under permuting the K
component labels, the posterior has exact S_K (label-switching) symmetry: K!
equivalent modes. PermutationFold sorts the (mu_k, logsig_k) blocks by mu_k,
collapsing the K! modes into a single sorted fundamental domain.

This is the standard-statistics counterpart of the synthetic LabelSwitchingMixture:
same symmetry and module stack, but a genuine Bayesian posterior on real data.

Interface mirrors the other FolT targets: log_prob (unfolded), log_prob_folded
(smooth, +log K! offset, finite everywhere -- for training/certification) and
log_prob_folded_hard (adds the sorted-ordering -inf cutoff -- for the MH kernel).
The unfolded posterior already has full support (Gaussian priors), so no box
support_fn is needed for the unfolded pipeline.
"""

import math

import torch


class BayesianGaussianMixture:
    """
    Posterior for a K-component Gaussian mixture with known weights.

    Parameters
    ----------
    data : array-like, shape (N,)
        Observations.
    weights : array-like, shape (K,)
        Known mixture weights (sum to 1).
    mu_prior_mean, mu_prior_std : float
        Gaussian prior on each mu_k.
    logsig_prior_mean, logsig_prior_std : float
        Gaussian prior on each logsig_k = log sigma_k.
    n_components : int
    device : str
    """

    def __init__(self, data, weights, mu_prior_mean=3.0, mu_prior_std=5.0,
                 logsig_prior_mean=0.0, logsig_prior_std=0.5,
                 n_components=3, device="cuda"):
        self.data = torch.as_tensor(data, dtype=torch.float32, device=device)
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.log_weights = torch.log(self.weights)
        self.K = n_components
        self.p = 2
        self.dim = 2 * n_components
        self.device = device

        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_std = mu_prior_std
        self.logsig_prior_mean = logsig_prior_mean
        self.logsig_prior_std = logsig_prior_std

        self.log_n_perms = math.lgamma(n_components + 1)   # log(K!)

    # ── likelihood / prior / posterior ──────────────────────────────────────
    def log_likelihood(self, theta):
        """theta: (batch, 2K) -> (batch,) mixture log-likelihood."""
        mu = theta[:, 0::2]                       # (B, K)
        logsig = theta[:, 1::2]                   # (B, K)
        sig = torch.exp(logsig)                   # (B, K)

        x = self.data.view(1, 1, -1)              # (1, 1, N)
        mu_e = mu.unsqueeze(2)                    # (B, K, 1)
        sig_e = sig.unsqueeze(2)                  # (B, K, 1)
        logsig_e = logsig.unsqueeze(2)            # (B, K, 1)

        log_comp = (-0.5 * math.log(2 * math.pi) - logsig_e
                    - 0.5 * ((x - mu_e) / sig_e) ** 2)        # (B, K, N)
        log_w = self.log_weights.view(1, -1, 1)               # (1, K, 1)
        log_mix = torch.logsumexp(log_comp + log_w, dim=1)    # (B, N)
        return log_mix.sum(dim=1)                              # (B,)

    def log_prior(self, theta):
        """theta: (batch, 2K) -> (batch,) log-prior."""
        mu = theta[:, 0::2]
        logsig = theta[:, 1::2]
        lp_mu = (-0.5 * ((mu - self.mu_prior_mean) / self.mu_prior_std) ** 2).sum(dim=1)
        lp_ls = (-0.5 * ((logsig - self.logsig_prior_mean) / self.logsig_prior_std) ** 2).sum(dim=1)
        return lp_mu + lp_ls

    def log_prob(self, theta):
        """Unnormalised log-posterior in unfolded theta-space (K! modes).

        Finite for all theta (Gaussian priors -> full support), so this doubles
        as the smooth training/certification target for the unfolded pipeline."""
        return self.log_likelihood(theta) + self.log_prior(theta)

    # ── folded posterior (sorted fundamental domain) ────────────────────────
    def log_prob_folded(self, z):
        """Smooth folded log-posterior: pi_F = K! * pi on the fundamental domain.

        The +log(K!) is a constant (cancels in the MH ratio and the oscillation);
        no ordering cutoff, so this is finite everywhere -- safe for the
        oscillation regulariser. Restrict to the sorted domain via a support_fn."""
        return self.log_prob(z) + self.log_n_perms

    def log_prob_folded_hard(self, z):
        """Folded log-posterior with the hard mu_1<=mu_2<=...<=mu_K cutoff (MH)."""
        lp = self.log_prob_folded(z)
        mu = z[:, 0::2]
        violations = (mu[:, :-1] > mu[:, 1:]).any(dim=-1)
        return torch.where(violations, torch.full_like(lp, -float('inf')), lp)
