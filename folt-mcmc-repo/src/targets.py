"""Target distributions used in the FolT-MCMC experiments.

Each target exposes:
    log_prob(x)      -> log density (unnormalised is fine; constants cancel)
    sample(n)        -> exact i.i.d. samples (where available, for certification)
    dim              -> dimension

All targets are defined analytically; see Appendix A of the paper for the
exact specifications. Synthetic data are regenerated deterministically from
these classes (fixed seeds in scripts/make_synthetic_data.py).
"""

import math
import torch


class GaussianMixture:
    """Equal-weight Gaussian mixture, optionally label-symmetric.

    For the dimension-scaling experiment we use a two-component reflection-
    symmetric mixture; for the label-switching experiment we use m exchangeable
    components whose means are permutation-symmetric.
    """

    def __init__(self, means, cov_scale=1.0):
        # means: (m, d) tensor of component means
        self.means = torch.as_tensor(means, dtype=torch.float64)
        self.m, self.d = self.means.shape
        self.cov_scale = float(cov_scale)
        self.dim = self.d

    def log_prob(self, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        if x.ndim == 1:
            x = x[None, :]
        # log-sum-exp over components
        diff = x[:, None, :] - self.means[None, :, :]          # (n, m, d)
        quad = -0.5 * (diff ** 2).sum(-1) / self.cov_scale     # (n, m)
        norm = -0.5 * self.d * math.log(2 * math.pi * self.cov_scale)
        comp = quad + norm - math.log(self.m)
        return torch.logsumexp(comp, dim=1)

    def sample(self, n, generator=None):
        idx = torch.randint(0, self.m, (n,), generator=generator)
        eps = torch.randn(n, self.d, generator=generator, dtype=torch.float64)
        return self.means[idx] + math.sqrt(self.cov_scale) * eps


class Banana:
    """Rosenbrock-style 'banana' target with curvature kappa."""

    def __init__(self, dim=2, kappa=0.1):
        self.dim = dim
        self.kappa = kappa

    def log_prob(self, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        if x.ndim == 1:
            x = x[None, :]
        lp = -0.5 * x[:, 0] ** 2
        lp = lp - 0.5 * self.kappa * (x[:, 1] - x[:, 0] ** 2) ** 2
        if self.dim > 2:
            lp = lp - 0.5 * (x[:, 2:] ** 2).sum(-1)
        return lp

    def sample(self, n, generator=None):
        z = torch.randn(n, self.dim, generator=generator, dtype=torch.float64)
        x = z.clone()
        x[:, 0] = z[:, 0]
        x[:, 1] = z[:, 1] / math.sqrt(self.kappa) + z[:, 0] ** 2
        return x


class Funnel:
    """Neal's funnel: x0 ~ N(0, 9), x_j | x0 ~ N(0, exp(x0))."""

    def __init__(self, dim=10):
        self.dim = dim

    def log_prob(self, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        if x.ndim == 1:
            x = x[None, :]
        lp = -0.5 * x[:, 0] ** 2 / 9.0 - 0.5 * math.log(2 * math.pi * 9.0)
        var = torch.exp(x[:, 0])
        d_rest = self.dim - 1
        lp = lp - 0.5 * (x[:, 1:] ** 2).sum(-1) / var
        lp = lp - 0.5 * d_rest * (torch.log(2 * math.pi * var))
        return lp

    def sample(self, n, generator=None):
        x0 = 3.0 * torch.randn(n, 1, generator=generator, dtype=torch.float64)
        std = torch.exp(0.5 * x0)
        rest = std * torch.randn(n, self.dim - 1, generator=generator,
                                 dtype=torch.float64)
        return torch.cat([x0, rest], dim=1)


def labelswitch_means(m, d, sep=4.0):
    """Permutation-symmetric component means for an m-component, d-dim mixture.

    The first coordinate of each component is placed on a regular grid; the
    components are exchangeable under permutation of their labels, giving the
    label-switching symmetry studied in the paper.
    """
    means = torch.zeros(m, d, dtype=torch.float64)
    grid = torch.linspace(-(m - 1) / 2, (m - 1) / 2, m, dtype=torch.float64)
    means[:, 0] = sep * grid
    return means
