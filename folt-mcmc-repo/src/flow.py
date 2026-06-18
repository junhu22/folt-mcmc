"""Spectrally-constrained RealNVP normalising flow.

A coupling-based flow whose scale/shift sub-networks are spectrally normalised
(spectral-norm target 1) with a soft scale clip, which bounds the forward-map
Lipschitz constant. See Section 3 of the paper.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def _mlp(d_in, d_out, hidden=(64, 64), spec_norm=True):
    layers = []
    dims = [d_in] + list(hidden) + [d_out]
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        if spec_norm:
            lin = spectral_norm(lin)
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class CouplingLayer(nn.Module):
    def __init__(self, dim, mask, hidden=(64, 64), clip=0.5, spec_norm=True):
        super().__init__()
        self.register_buffer("mask", mask.double())
        self.clip = clip
        d = dim
        self.scale_net = _mlp(d, d, hidden, spec_norm).double()
        self.shift_net = _mlp(d, d, hidden, spec_norm).double()

    def forward(self, x):
        x_a = x * self.mask
        s = self.scale_net(x_a) * (1 - self.mask)
        s = self.clip * torch.tanh(s)          # soft scale clip
        t = self.shift_net(x_a) * (1 - self.mask)
        y = x_a + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(-1)
        return y, log_det

    def inverse(self, y):
        y_a = y * self.mask
        s = self.scale_net(y_a) * (1 - self.mask)
        s = self.clip * torch.tanh(s)
        t = self.shift_net(y_a) * (1 - self.mask)
        x = y_a + (1 - self.mask) * ((y - t) * torch.exp(-s))
        log_det = -s.sum(-1)
        return x, log_det


class RealNVP(nn.Module):
    """Composition of L coupling layers with alternating masks."""

    def __init__(self, dim, n_layers=6, hidden=(64, 64), clip=0.5,
                 spec_norm=True):
        super().__init__()
        self.dim = dim
        masks = []
        for i in range(n_layers):
            m = torch.zeros(dim)
            m[i % 2::2] = 1.0           # alternating checkerboard in 1D
            masks.append(m)
        self.layers = nn.ModuleList([
            CouplingLayer(dim, masks[i], hidden, clip, spec_norm)
            for i in range(n_layers)
        ])

    def forward(self, x):
        """Map data x -> latent z; return z and sum log|det dz/dx|."""
        log_det = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer.forward(z)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, z):
        """Map latent z -> data x; return x and sum log|det dx/dz|."""
        log_det = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        x = z
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            log_det = log_det + ld
        return x, log_det

    def log_prob(self, x):
        """Flow density q(x) with standard-normal base."""
        z, log_det = self.forward(x)
        d = self.dim
        base = -0.5 * (z ** 2).sum(-1) - 0.5 * d * torch.log(
            torch.tensor(2 * torch.pi, dtype=x.dtype))
        return base + log_det
