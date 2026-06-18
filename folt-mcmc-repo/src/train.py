"""Flow training with optional oscillation regularisation.

Trains the RealNVP flow by maximising the likelihood of target samples
(equivalently minimising NLL), optionally adding a batch-oscillation penalty
lambda * osc(log pi - log q) as described in Section 6 of the paper.
"""

import torch


def train_flow(flow, target, n_train=50000, lr=1e-3, batch_size=256,
               max_epochs=2000, patience=80, osc_lambda=0.0,
               warmup_epochs=100, generator=None, verbose=False):
    train_x = target.sample(n_train, generator=generator)
    n_val = n_train // 5
    val_x = target.sample(n_val, generator=generator)

    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    bad = 0

    n_batches = max(1, n_train // batch_size)
    for epoch in range(max_epochs):
        perm = torch.randperm(n_train, generator=generator)
        flow.train()
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = train_x[idx]
            nll = -flow.log_prob(xb).mean()
            loss = nll
            if osc_lambda > 0:
                lam = osc_lambda * min(1.0, epoch / max(1, warmup_epochs))
                h = target.log_prob(xb) - flow.log_prob(xb)
                osc = h.max() - h.min()
                loss = loss + lam * osc
            opt.zero_grad()
            loss.backward()
            opt.step()

        flow.eval()
        with torch.no_grad():
            val_nll = -flow.log_prob(val_x).mean().item()
        if val_nll < best_val - 1e-4:
            best_val = val_nll
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and epoch % 50 == 0:
            print(f"epoch {epoch:4d}  val_nll {val_nll:.4f}")

    if best_state is not None:
        flow.load_state_dict(best_state)
    return flow, best_val
