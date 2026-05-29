"""
Label-switching Gaussian mixture for PermutationFold experiments.

pi(theta) = (1/k!) * sum_{sigma in S_k} N(theta; mu_{sigma}, Sigma)

where theta = (theta_1, ..., theta_k), each theta_i in R^p represents
one "component" (e.g., frequency + damping of a structural mode),
and S_k is the symmetric group (all permutations of k components).

This has k!-fold permutation symmetry: permuting the component blocks
gives the same density. PermutationFold eliminates this by sorting
components by their first parameter.

Structural ID interpretation:
- k = number of modes to identify
- p = parameters per mode (e.g., p=2: frequency + damping ratio)
- D = k * p total parameters
- Permutation symmetry = label switching between modes
"""

import torch
import math
from itertools import permutations


class LabelSwitchingMixture:
    """
    k-component Gaussian mixture with full permutation symmetry.

    Parameters
    ----------
    k : int
        Number of components (modes to identify).
    p : int
        Parameters per component.
    component_centers : torch.Tensor, shape (k, p)
        Center of each component block. These should be well-separated
        along the sort dimension (dim 0 of each block) to make the
        permutation modes well-separated.
    component_std : float
        Isotropic std within each component block.
    """

    def __init__(self, k: int, p: int, component_centers: torch.Tensor,
                 component_std: float = 1.0, device: str = "cuda"):
        self.k = k
        self.p = p
        self.dim = k * p
        self.component_std = component_std
        self.device = device
        self.centers = component_centers.to(device)  # (k, p)
        assert self.centers.shape == (k, p)

        # Precompute all k! permutations of component indices
        self.perms = list(permutations(range(k)))
        self.n_perms = len(self.perms)  # = k!
        self.log_n_perms = math.log(self.n_perms)

        # Log normalization of single Gaussian
        self.log_norm = -0.5 * self.dim * math.log(2 * math.pi) \
                        - self.dim * math.log(component_std)

    def _make_mean(self, perm: tuple) -> torch.Tensor:
        """Construct full D-dimensional mean by permuting component centers."""
        return self.centers[list(perm), :].reshape(-1)  # (D,)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Log probability in original theta-space (k! equivalent modes).

        pi(theta) = (1/k!) sum_{sigma} N(theta; mu_sigma, sigma^2 I)
        """
        batch = theta.shape[0]
        log_components = []
        for perm in self.perms:
            mu = self._make_mean(perm)  # (D,)
            diff = theta - mu.unsqueeze(0)  # (batch, D)
            log_p = self.log_norm - 0.5 * (diff ** 2).sum(dim=-1) / (self.component_std ** 2)
            log_components.append(log_p)
        # logsumexp over k! permutations, minus log(k!)
        log_stack = torch.stack(log_components, dim=-1)  # (batch, k!)
        return torch.logsumexp(log_stack, dim=-1) - self.log_n_perms

    def log_prob_folded(self, z: torch.Tensor) -> torch.Tensor:
        """
        Log probability in folded z-space (components sorted by first param).

        pi_F(z) = k! * pi(z)  when z is in the fundamental domain
                              (z_{1,0} <= z_{2,0} <= ... <= z_{k,0})

        This is because all k! permutation preimages contribute equally.
        """
        return self.log_prob(z) + self.log_n_perms

    def log_prob_folded_hard(self, z: torch.Tensor) -> torch.Tensor:
        """Folded log-prob with hard ordering constraint."""
        lp = self.log_prob_folded(z)
        # Check ordering: z reshaped as (batch, k, p), first param of each block
        z_blocks = z.view(-1, self.k, self.p)
        sort_vals = z_blocks[:, :, 0]  # (batch, k)
        # Check if sorted: sort_vals[:, i] <= sort_vals[:, i+1] for all i
        violations = (sort_vals[:, :-1] > sort_vals[:, 1:]).any(dim=-1)  # (batch,)
        lp[violations] = -float('inf')
        return lp

    def sample(self, n: int) -> torch.Tensor:
        """Exact samples from the mixture."""
        # Randomly pick a permutation for each sample
        perm_idx = torch.randint(0, self.n_perms, (n,))
        samples = torch.randn(n, self.dim, device=self.device) * self.component_std
        for i in range(n):
            mu = self._make_mean(self.perms[perm_idx[i]])
            samples[i] += mu
        return samples

    def sample_folded(self, n: int) -> torch.Tensor:
        """Exact samples from the folded distribution (sorted component order)."""
        samples = self.sample(n)
        # Sort by first parameter of each component block
        blocks = samples.view(n, self.k, self.p)
        sort_keys = blocks[:, :, 0]  # (batch, k)
        indices = sort_keys.argsort(dim=1)
        indices_exp = indices.unsqueeze(-1).expand_as(blocks)
        sorted_blocks = blocks.gather(1, indices_exp)
        return sorted_blocks.view(n, -1)
