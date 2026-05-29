"""
Well-separated symmetric Gaussian mixture for FolT-MCMC experiments.

pi(theta) = 0.5 * N(theta; +mu, Sigma) + 0.5 * N(theta; -mu, Sigma)

Symmetry group: G = {id, reflection along axis 0}, |G| = 2.
Fold boundary theta_1 = 0 sits in the low-density valley between modes.
Valley density ~ exp(-||mu||^2 / 2) relative to peak.
"""

import torch
import math


class SymmetricGaussianMixture:
    """
    Symmetric 2-component Gaussian mixture in D dimensions.

    Modes at (+separation/2, 0, ..., 0) and (-separation/2, 0, ..., 0).
    Covariance = diag(variances) or identity.
    Reflection symmetry along dim 0.

    Parameters
    ----------
    dim : int
        Dimensionality.
    separation : float
        Distance between mode centers along axis 0 (each mode at ±separation/2).
        Default 6.0 means modes at ±3.0 with unit variance → valley density
        ~ exp(-9/2) ≈ 0.011 of peak.
    variances : torch.Tensor or None
        Diagonal covariance entries. If None, uses identity.
    """

    def __init__(self, dim: int = 2, separation: float = 6.0,
                 variances: torch.Tensor = None, device: str = "cuda"):
        self.dim = dim
        self.separation = separation
        self.device = device

        if variances is None:
            self.variances = torch.ones(dim, device=device)
        else:
            self.variances = variances.to(device)

        # Mode centers: mu1 = (+sep/2, 0, ..., 0), mu2 = (-sep/2, 0, ..., 0)
        self.mu1 = torch.zeros(dim, device=device)
        self.mu1[0] = separation / 2
        self.mu2 = torch.zeros(dim, device=device)
        self.mu2[0] = -separation / 2

        # Log normalizing constant of each component (for numerical stability)
        self.log_norm = -0.5 * dim * math.log(2 * math.pi) - 0.5 * self.variances.log().sum()

    def _log_component(self, theta: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Log density of a single Gaussian component (unnormalized ok, but we include norm)."""
        diff = theta - mu.unsqueeze(0)  # (batch, dim)
        return self.log_norm - 0.5 * (diff ** 2 / self.variances.unsqueeze(0)).sum(dim=-1)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Log probability in original theta-space (2 modes)."""
        log_p1 = self._log_component(theta, self.mu1) + math.log(0.5)
        log_p2 = self._log_component(theta, self.mu2) + math.log(0.5)
        return torch.logsumexp(torch.stack([log_p1, log_p2], dim=-1), dim=-1)

    def log_prob_folded(self, z: torch.Tensor) -> torch.Tensor:
        """
        Log probability in folded z-space (z_0 >= 0 half-space).

        pi_F(z) = pi(z) + pi(reflect(z))  for z_0 >= 0
                = 2 * 0.5 * N(z; mu1, Sigma) + 2 * 0.5 * N(reflect(z); mu1, Sigma)

        But since pi(reflect(z)) where reflect flips z_0:
          N(reflect(z); mu1, Sigma) = N(z; mu2, Sigma) (already captured in log_prob)
          N(reflect(z); mu2, Sigma) = N(z; mu1, Sigma) (already captured in log_prob)

        So pi_F(z) = pi(z) + pi(reflect(z)) = 2 * pi(z) for symmetric mixture.
        """
        return self.log_prob(z) + math.log(2.0)

    def log_prob_folded_hard(self, z: torch.Tensor) -> torch.Tensor:
        """Folded log-prob with hard constraint: -inf for z_0 < 0."""
        lp = self.log_prob_folded(z)
        lp[z[:, 0] < 0] = -float('inf')
        return lp

    def sample(self, n: int) -> torch.Tensor:
        """Exact samples from the mixture (for training data)."""
        # Assign each sample to component 1 or 2 with equal probability
        component = torch.randint(0, 2, (n,), device=self.device)
        samples = torch.randn(n, self.dim, device=self.device) * self.variances.sqrt().unsqueeze(0)
        samples[component == 0] += self.mu1.unsqueeze(0)
        samples[component == 1] += self.mu2.unsqueeze(0)
        return samples

    def sample_folded(self, n: int) -> torch.Tensor:
        """Exact samples from the folded distribution (z_0 >= 0)."""
        samples = self.sample(n)
        samples[:, 0] = samples[:, 0].abs()  # Reflect into half-space
        return samples
