# Extracted from CerT-MCMC-v2 experiments by quantile_core_diagnostic.py
"""
Oscillation / gradient regularisers (OscReg) for FolT-OG-Anneal training.

These two penalties shape the importance-weight residual r(z) so that the
trained flow admits a non-vacuous spectral-gap certificate:

  * `smooth_oscillation` is a differentiable surrogate for the oscillation
    osc(r) = max(r) - min(r), built from soft-max / soft-min via logsumexp at
    temperature `tau`. Driving it down tightens the certificate
    gamma = 2 / (1 + exp(osc)).

  * `gradient_norm_mean` returns the mean / max of ||grad_z r(z)||, which
    controls the local-Lipschitz term in the covering certificate.

Both operate on the residual defined in `folt.transport.compute_residual`.
They depend only on torch.
"""

import torch


def smooth_oscillation(r, tau=0.1):
    """Differentiable surrogate for osc(r) = max(r) - min(r).

    softmax(r) + softmax(-r) at temperature `tau`:

        tau * logsumexp(r / tau)  +  tau * logsumexp(-r / tau)

    As tau -> 0 this converges to max(r) - min(r). Smaller tau is tighter but
    numerically stiffer; the training recipe uses tau = 0.1.
    """
    return tau * torch.logsumexp(r / tau, dim=0) + tau * torch.logsumexp(-r / tau, dim=0)


def gradient_norm_mean(flow, target, z):
    """Mean and max of ||grad_z r(z)|| over a batch of latent points.

    Used as the gradient regulariser during training and as the local
    Lipschitz estimate for the covering certificate. Requires grad through the
    flow, so `create_graph=True`.
    """
    from .transport import compute_residual  # local import avoids circular dep
    z = z.detach().requires_grad_(True)
    r = compute_residual(flow, target, z)
    grad_r = torch.autograd.grad(r.sum(), z, create_graph=True)[0]
    norms = grad_r.norm(dim=-1)
    return norms.mean(), norms.max()
