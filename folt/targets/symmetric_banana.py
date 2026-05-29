"""
Symmetric double banana target for FolT-MCMC toy experiment.

pi(theta) propto exp(-(theta2 - theta1^2)^2 / (2*sigma^2)) * exp(-theta1^2 / (2*tau^2))
                + exp(-(theta2 - (-theta1)^2)^2 / (2*sigma^2)) * exp(-theta1^2 / (2*tau^2))

Symmetry group: G = {id, reflection theta1 -> -theta1}, |G| = 2.
Folded target (theta1 >= 0 half-plane) is unimodal.
"""

import torch
import math


class SymmetricDoubleBanana:
    """Symmetric double banana distribution in 2D."""

    def __init__(self, sigma: float = 0.5, tau: float = 2.0, device: str = "cuda"):
        self.sigma = sigma
        self.tau = tau
        self.dim = 2
        self.device = device

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Log probability (unnormalized) in original theta-space."""
        t1, t2 = theta[:, 0], theta[:, 1]
        log_mode1 = -(t2 - t1**2)**2 / (2 * self.sigma**2) - t1**2 / (2 * self.tau**2)
        log_mode2 = -(t2 - (-t1)**2)**2 / (2 * self.sigma**2) - t1**2 / (2 * self.tau**2)
        # Note: (-t1)^2 = t1^2, so for this specific banana the two modes
        # have the SAME log_prob. This is exact symmetry.
        # For a non-trivial test, use asymmetric curvature:
        return torch.logsumexp(torch.stack([log_mode1, log_mode2], dim=-1), dim=-1)

    def log_prob_folded(self, z: torch.Tensor) -> torch.Tensor:
        """Log probability in folded z-space (z1 >= 0 half-plane).
        pi_F(z) = 2 * pi(z) for z1 > 0 (two preimages contribute equally).
        """
        return self.log_prob(z) + math.log(2.0)


class AsymmetricDoubleBanana:
    """
    Two banana modes with DIFFERENT curvatures, still reflection-symmetric.
    Mode 1: theta2 = a * theta1^2   (theta1 > 0)
    Mode 2: theta2 = b * theta1^2   (theta1 < 0, equivalently theta2 = b * theta1^2)

    This breaks the trivial (-t1)^2 = t1^2 identity and creates genuinely
    distinct modes that are still related by theta1 -> -theta1.
    """

    def __init__(self, a: float = 1.0, b: float = 0.5,
                 sigma: float = 0.5, tau: float = 2.0, device: str = "cuda"):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.tau = tau
        self.dim = 2
        self.device = device

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        t1, t2 = theta[:, 0], theta[:, 1]
        # Mode 1: centered at theta1 > 0, curvature a
        log_mode1 = -(t2 - self.a * t1**2)**2 / (2 * self.sigma**2) - t1**2 / (2 * self.tau**2)
        # Mode 2: reflection of mode 1 with curvature b
        log_mode2 = -(t2 - self.b * t1**2)**2 / (2 * self.sigma**2) - t1**2 / (2 * self.tau**2)
        return torch.logsumexp(torch.stack([log_mode1, log_mode2], dim=-1), dim=-1)

    def log_prob_folded(self, z: torch.Tensor) -> torch.Tensor:
        """After folding theta1 -> |theta1|, the two modes overlap in z-space."""
        z1, z2 = z[:, 0], z[:, 1]
        # Preimage 1: theta1 = z1 > 0
        log_pre1 = -(z2 - self.a * z1**2)**2 / (2 * self.sigma**2) - z1**2 / (2 * self.tau**2)
        # Preimage 2: theta1 = -z1 < 0, but curvature b applies
        log_pre2 = -(z2 - self.b * z1**2)**2 / (2 * self.sigma**2) - z1**2 / (2 * self.tau**2)
        return torch.logsumexp(torch.stack([log_pre1, log_pre2], dim=-1), dim=-1)
