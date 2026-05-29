"""
Random Permutation Sampler (Frühwirth-Schnatter, 2001).

After each MCMC step, randomly permute the component labels. This ensures the
chain visits all k! symmetric modes uniformly, breaking the label-switching
deadlock that traps a vanilla independence sampler in whichever mode it happens
to land on.

We implement RPS on top of the *same* independence-MH kernel used by FolT-MCMC
(propose theta' = T(z'), z' ~ N(0,I); accept on the ratio of importance weights
w = pi/q) so the only difference versus the unfolded IMH baseline is the extra
permutation step. That makes the comparison clean: identical proposal, identical
acceptance rule, identical flow -- RPS just relabels the current state each step.

Validity. pi is permutation-invariant by construction, and a relabelling is a
measure-preserving bijection, so applying a random permutation to the current
state is a pi-invariant move. Composing it with the pi-invariant IMH step leaves
pi invariant. The relabelling is *not* q-invariant, though, so the importance
weight of the current state (specifically log q) changes after each permutation
and must be recomputed -- this is precisely what perturbs the chain out of a
single mode.

Unlike a standard independence sampler, the composite kernel (IMH + random
permutation) does not have the Mengersen-Tweedie minorisation structure, so the
quantile-core spectral-gap certificate does not apply -- RPS gets empirical ESS
only, no certified gamma. That methodological gap (mixing aid without a
guarantee) is exactly the contrast against FolT-MCMC.
"""

import numpy as np
import torch
from itertools import permutations


class RandomPermutationSampler:
    """
    Independence MH + random label permutation at each step.

    Parameters
    ----------
    target_log_prob : callable
        Log probability of the original (unfolded) target. Batched: (B, D) -> (B,).
    proposal_sampler : callable
        Samples from the proposal q_U (unfolded flow): n -> (n, D).
    proposal_log_prob : callable
        Log probability under q_U. Batched: (B, D) -> (B,).
    n_components : int (m)
        Number of mixture components.
    block_size : int (p)
        Parameters per component.
    """

    def __init__(self, target_log_prob, proposal_sampler,
                 proposal_log_prob, n_components, block_size, device="cuda"):
        self.target_log_prob = target_log_prob
        self.proposal_sampler = proposal_sampler
        self.proposal_log_prob = proposal_log_prob
        self.m = n_components
        self.p = block_size
        self.dim = n_components * block_size
        self.device = device

        # Precompute all m! permutations of the component blocks.
        self.perms = [list(perm) for perm in permutations(range(n_components))]
        self.n_perms = len(self.perms)  # = m!

    def random_permute(self, theta):
        """Apply an independent random block permutation to each row of theta.

        theta : (B, D) -> (B, D) with the m blocks of size p reshuffled per row.
        """
        batch = theta.shape[0]
        blocks = theta.view(batch, self.m, self.p)
        perm_ids = np.random.randint(self.n_perms, size=batch)
        out = torch.empty_like(blocks)
        for i in range(batch):
            out[i] = blocks[i][self.perms[perm_ids[i]]]
        return out.view(batch, -1)

    def run_chain(self, theta_init, n_steps, return_all=True):
        """
        Run one RPS chain: each step (1) propose via independence MH, (2)
        accept/reject, (3) randomly permute the current state's labels.

        Proposals are independent of the current state, so all `n_steps`
        proposals and their (log pi, log q) are precomputed in a single batched
        pass -- distributionally identical to a per-step chain but far faster.
        The permutation step is genuinely sequential (it changes the current
        state's log q, which feeds the next acceptance), so it runs in a loop.

        Returns
        -------
        samples : (n_steps, dim) tensor of post-step states (if return_all)
        accept_count : int  (number of accepted proposals)
        """
        with torch.no_grad():
            # --- precompute the independence-MH proposal stream -------------
            theta_prop = self.proposal_sampler(n_steps)               # (M, D)
            log_pi_prop = self.target_log_prob(theta_prop)            # (M,)
            log_q_prop = self.proposal_log_prob(theta_prop)           # (M,)
            log_w_prop = (log_pi_prop - log_q_prop).cpu().numpy()     # (M,)
            theta_prop_cpu = theta_prop.cpu()

            u = np.log(np.random.random(n_steps) + 1e-30)

            # initial state = first proposal (unfolded target has full support)
            theta_curr = theta_init.view(1, self.dim).to(self.device).clone()
            log_pi_curr = self.target_log_prob(theta_curr)
            log_q_curr = self.proposal_log_prob(theta_curr)
            log_w_curr = float((log_pi_curr - log_q_curr).item())
            log_pi_curr_val = float(log_pi_curr.item())

            samples = [] if return_all else None
            accept_count = 0

            for i in range(n_steps):
                # (1)-(2) independence-MH accept/reject
                if u[i] < min(0.0, log_w_prop[i] - log_w_curr):
                    theta_curr = theta_prop_cpu[i:i + 1].to(self.device)
                    log_pi_curr_val = float(log_pi_prop[i].item())
                    log_w_curr = float(log_w_prop[i])
                    accept_count += 1

                # (3) random label permutation. pi is permutation-invariant, so
                # log pi is unchanged; q is not, so recompute log q (one eval).
                theta_curr = self.random_permute(theta_curr)
                log_q_curr = float(self.proposal_log_prob(theta_curr).item())
                log_w_curr = log_pi_curr_val - log_q_curr

                if return_all:
                    samples.append(theta_curr.squeeze(0).clone())

            if return_all:
                samples = torch.stack(samples, dim=0)
        return samples, accept_count

    def run_multichain(self, n_chains, chain_length, burnin):
        """Run `n_chains` RPS chains; return stacked post-burn-in samples.

        Returns a dict mirroring run_independence_mh_multichain: stacked samples
        (n_chains, chain_length, D), concatenated samples, and per-chain /
        mean acceptance.
        """
        total = chain_length + burnin
        all_samples, accepts = [], []
        for _ in range(n_chains):
            theta_init = self.proposal_sampler(1)
            s, n_acc = self.run_chain(theta_init, total, return_all=True)
            all_samples.append(s[burnin:].to(self.device))
            accepts.append(n_acc / total)

        stacked = torch.stack(all_samples)            # (n_chains, L, D)
        concat = stacked.reshape(-1, self.dim)
        return {
            'samples': stacked,
            'samples_concat': concat,
            'accept_rate': float(np.mean(accepts)),
            'accept_each': [float(a) for a in accepts],
            'n_total': n_chains * chain_length,
        }
