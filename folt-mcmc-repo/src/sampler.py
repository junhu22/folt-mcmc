"""Independence Metropolis-Hastings driver.

Proposes z' ~ N(0, I), maps to x' via the (inverse) flow, and accepts with the
standard independence-MH ratio. For the folded sampler the proposal is the
QuotientProposal and the target is the FoldedTarget; the chain lives on the
fundamental domain.
"""

import torch


def independence_mh(target, proposal, n_steps, x_init=None,
                    generator=None, thin=1):
    """Run independence MH targeting `target` with `proposal`.

    proposal must expose .sample(n) and .log_prob(x); target must expose
    .log_prob(x). Returns (samples, acceptance_rate).
    """
    if x_init is None:
        x = proposal.sample(1, generator=generator)
    else:
        x = torch.as_tensor(x_init, dtype=torch.float64).reshape(1, -1)

    h_curr = target.log_prob(x) - proposal.log_prob(x)
    kept = []
    n_accept = 0

    # pre-draw proposals in batches for efficiency
    proposals = proposal.sample(n_steps, generator=generator)
    u = torch.rand(n_steps, generator=generator, dtype=torch.float64)

    for t in range(n_steps):
        x_prop = proposals[t:t + 1]
        h_prop = target.log_prob(x_prop) - proposal.log_prob(x_prop)
        # independence-MH log acceptance = h_curr - h_prop  (since
        # alpha = min(1, [pi'/q'] / [pi/q]) = min(1, exp(h_prop - h_curr))... )
        log_alpha = (h_prop - h_curr).item()
        if torch.log(u[t]).item() < log_alpha:
            x = x_prop
            h_curr = h_prop
            n_accept += 1
        if t % thin == 0:
            kept.append(x.clone())

    samples = torch.cat(kept, dim=0)
    return samples, n_accept / n_steps
