"""
Folding maps for FolT-MCMC.

A folding map F: Theta -> Z is a k-to-1 surjection that collapses
orbits of a symmetry group G into single points in a reduced space.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class FoldingMap(ABC):
    """Base class for folding maps."""

    @abstractmethod
    def fold(self, theta: torch.Tensor) -> torch.Tensor:
        """Map theta-space points to folded z-space. Many-to-one."""
        ...

    @abstractmethod
    def unfold(self, z: torch.Tensor, branch: int = 0) -> torch.Tensor:
        """Map z-space point back to one of k preimages in theta-space."""
        ...

    @abstractmethod
    def n_branches(self) -> int:
        """Number of preimages |G| for each z."""
        ...

    @abstractmethod
    def log_det_fold(self, theta: torch.Tensor) -> torch.Tensor:
        """Log |det dF/dtheta| evaluated at theta (for density transform)."""
        ...


class ReflectionFold(FoldingMap):
    """
    Fold by reflection symmetry along specified axes.
    F(theta) = (|theta_1|, ..., |theta_k|, theta_{k+1}, ..., theta_d)

    Symmetric double banana toy example:
        fold_axes = [0]  ->  F(theta1, theta2) = (|theta1|, theta2)
        n_branches = 2
    """

    def __init__(self, dim: int, fold_axes: list[int]):
        self.dim = dim
        self.fold_axes = fold_axes

    def fold(self, theta: torch.Tensor) -> torch.Tensor:
        z = theta.clone()
        z[:, self.fold_axes] = z[:, self.fold_axes].abs()
        return z

    def unfold(self, z: torch.Tensor, branch: int = 0) -> torch.Tensor:
        theta = z.clone()
        # branch index encodes sign pattern: 0 = all positive, 1 = first axis flipped, etc.
        for i, ax in enumerate(self.fold_axes):
            if branch & (1 << i):
                theta[:, ax] = -theta[:, ax]
        return theta

    def n_branches(self) -> int:
        return 2 ** len(self.fold_axes)

    def log_det_fold(self, theta: torch.Tensor) -> torch.Tensor:
        # |det dF/dtheta| = 1 for reflection fold (sign change has unit Jacobian)
        return torch.zeros(theta.shape[0], device=theta.device)


class PermutationFold(FoldingMap):
    """
    Fold by permutation symmetry (label switching).
    F(theta) sorts the parameter blocks so that block_1 <= block_2 <= ...

    For k components each with p parameters, fold_dim = k*p,
    sorting is by the first parameter of each block (e.g., frequency).
    """

    def __init__(self, n_components: int, block_size: int, sort_param_idx: int = 0):
        import math
        from itertools import permutations
        self.n_components = n_components
        self.block_size = block_size
        self.sort_param_idx = sort_param_idx
        # Enumerate the k! permutations once; branch j applies self.perms[j].
        self.perms = list(permutations(range(n_components)))
        self._n_branches = math.factorial(n_components)

    def fold(self, theta: torch.Tensor) -> torch.Tensor:
        batch = theta.shape[0]
        blocks = theta.view(batch, self.n_components, self.block_size)
        sort_keys = blocks[:, :, self.sort_param_idx]
        indices = sort_keys.argsort(dim=1)
        indices_exp = indices.unsqueeze(-1).expand_as(blocks)
        sorted_blocks = blocks.gather(1, indices_exp)
        return sorted_blocks.view(batch, -1)

    def unfold(self, z: torch.Tensor, branch: int = 0) -> torch.Tensor:
        """Reorder the (sorted) component blocks of z by permutation `branch`.

        branch indexes self.perms (0 = identity / sorted order). The j-th
        permutation sigma is applied to the component blocks: output block i
        is input block sigma[i]. Choosing branch uniformly in 0..k!-1 inverts
        the fold by mapping a sorted z to one of its k! preimages.
        """
        perm = self.perms[branch]
        batch = z.shape[0]
        blocks = z.view(batch, self.n_components, self.block_size)
        idx = torch.tensor(perm, device=z.device)
        permuted = blocks.index_select(1, idx)
        return permuted.reshape(batch, -1)

    def n_branches(self) -> int:
        return self._n_branches

    def log_det_fold(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.zeros(theta.shape[0], device=theta.device)
