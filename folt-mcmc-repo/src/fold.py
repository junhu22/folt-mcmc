"""Fundamental-domain folding and the symmetrised (quotient) proposal.

The symmetry group G acts on R^d by permuting blocks of coordinates (label
switching) or by reflection. The fundamental domain D is the sorted chamber
(for permutation symmetry) or a half-space (for reflection symmetry). The
folded target restricts and renormalises pi to D; the quotient proposal is the
orbit sum of the flow density. See Section 3 of the paper.
"""

import itertools
import torch


# ----------------------------------------------------------------------
# Group actions
# ----------------------------------------------------------------------

class PermutationGroup:
    """Symmetric group S_m acting by permuting m blocks of size `block`.

    A parameter vector is laid out as m consecutive blocks; permuting the
    block order is the label-switching action.
    """

    def __init__(self, m, block):
        self.m = m
        self.block = block
        self.perms = list(itertools.permutations(range(m)))
        self.order = len(self.perms)

    def apply(self, x, perm):
        # x: (..., m*block) -> permute blocks according to perm
        shape = x.shape
        x = x.reshape(*shape[:-1], self.m, self.block)
        x = x[..., list(perm), :]
        return x.reshape(*shape)

    def to_fundamental(self, x):
        """Sort blocks by their first coordinate (canonical representative)."""
        shape = x.shape
        xb = x.reshape(*shape[:-1], self.m, self.block)
        key = xb[..., 0]                       # (..., m)
        order = torch.argsort(key, dim=-1)
        xb_sorted = torch.gather(
            xb, -2, order.unsqueeze(-1).expand_as(xb))
        return xb_sorted.reshape(*shape)

    def in_fundamental(self, x):
        shape = x.shape
        xb = x.reshape(*shape[:-1], self.m, self.block)
        key = xb[..., 0]
        return torch.all(key[..., :-1] <= key[..., 1:], dim=-1)


class ReflectionGroup:
    """Z_2 acting by sign flip on the first coordinate."""

    def __init__(self):
        self.order = 2

    def apply(self, x, g):
        if g == 0:
            return x
        y = x.clone()
        y[..., 0] = -y[..., 0]
        return y

    def to_fundamental(self, x):
        y = x.clone()
        y[..., 0] = torch.abs(y[..., 0])
        return y

    def in_fundamental(self, x):
        return x[..., 0] >= 0


# ----------------------------------------------------------------------
# Folded target and quotient proposal
# ----------------------------------------------------------------------

class FoldedTarget:
    """pi restricted to the fundamental domain D and renormalised by |G|."""

    def __init__(self, base_target, group):
        self.base = base_target
        self.group = group
        self.dim = base_target.dim
        self.log_s = torch.log(torch.tensor(float(group.order),
                                            dtype=torch.float64))

    def log_prob(self, x):
        # pi_F(z) = |G| * pi(z) for z in D, -inf outside
        lp = self.base.log_prob(x) + self.log_s
        inside = self.group.in_fundamental(x)
        return torch.where(inside, lp,
                           torch.full_like(lp, float("-inf")))


class QuotientProposal:
    """Orbit-summed flow density q_F(z) = sum_g q(g . z), restricted to D."""

    def __init__(self, flow, group):
        self.flow = flow
        self.group = group
        self.dim = flow.dim

    def log_prob(self, x):
        if isinstance(self.group, PermutationGroup):
            terms = []
            for perm in self.group.perms:
                gx = self.group.apply(x, perm)
                terms.append(self.flow.log_prob(gx))
            stacked = torch.stack(terms, dim=0)         # (|G|, n)
            return torch.logsumexp(stacked, dim=0)
        else:  # reflection
            terms = [self.flow.log_prob(self.group.apply(x, g))
                     for g in range(self.group.order)]
            return torch.logsumexp(torch.stack(terms, 0), dim=0)

    def sample(self, n, generator=None):
        """Sample from the flow, then fold into the fundamental domain."""
        z = torch.randn(n, self.dim, generator=generator, dtype=torch.float64)
        x, _ = self.flow.inverse(z)
        return self.group.to_fundamental(x)
