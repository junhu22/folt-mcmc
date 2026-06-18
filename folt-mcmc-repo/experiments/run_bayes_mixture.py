"""Bayesian three-component mixture posterior benchmark.

A standard equal-weight three-component Gaussian-mixture posterior with exact
S_3 label symmetry. Demonstrates that post-hoc relabelling recovers correct
point estimates but leaves the unfolded diagnostic vacuous, whereas the folded
sampler yields a non-vacuous diagnostic (paper, Section 5).

For simplicity this script uses the mixture-of-means posterior with known
weights and unit variances; the means are the unknown parameters with exact
S_3 symmetry.
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from src.targets import GaussianMixture          # noqa: E402
from src.flow import RealNVP                      # noqa: E402
from src.fold import (PermutationGroup, FoldedTarget,  # noqa: E402
                      QuotientProposal)
from src.train import train_flow                  # noqa: E402
from src.sampler import independence_mh           # noqa: E402
from src.diagnostic import diagnostic             # noqa: E402


def run(cfg):
    g = torch.Generator().manual_seed(cfg["seed"])

    # 3 exchangeable components in 1D each -> parameter dim 3, group S_3
    m, block = 3, 1
    d = m * block
    true_means = torch.tensor([[-3.0], [0.0], [6.0]], dtype=torch.float64)
    # the posterior over component means under exact label symmetry is itself
    # a 3!-symmetric mixture; we model it directly as the symmetrised target
    sym_means = []
    import itertools
    for perm in itertools.permutations(range(m)):
        sym_means.append(true_means[list(perm)].reshape(-1))
    sym_means = torch.stack(sym_means)            # (6, 3)
    target = GaussianMixture(sym_means, cov_scale=0.25)
    group = PermutationGroup(m, block)

    flow = RealNVP(d, n_layers=cfg["flow"]["n_layers"],
                   hidden=tuple(cfg["flow"]["hidden"]),
                   clip=cfg["flow"]["clip"],
                   spec_norm=cfg["flow"]["spectral_norm"])
    flow, _ = train_flow(
        flow, target,
        n_train=cfg["training"]["n_train"],
        lr=cfg["training"]["lr"],
        batch_size=cfg["training"]["batch_size"],
        max_epochs=cfg["training"]["max_epochs"],
        patience=cfg["training"]["patience"],
        osc_lambda=0.1, generator=g)

    cert = target.sample(10000, generator=g)
    unf = diagnostic(target, flow, cert, rho=cfg["diagnostic"]["rho"])

    ftarget = FoldedTarget(target, group)
    fprop = QuotientProposal(flow, group)

    samples, acc = independence_mh(
        ftarget, fprop, cfg["mcmc"]["n_steps"],
        generator=g, thin=cfg["mcmc"]["thin"])
    cert_f = group.to_fundamental(cert)
    fold = diagnostic(ftarget, fprop, cert_f, rho=cfg["diagnostic"]["rho"])

    sorted_mean = group.to_fundamental(samples).mean(0)
    print("Bayesian 3-component mixture (exact S_3 symmetry)")
    print(f"  unfolded QC gamma : {unf['qc_gamma']:.4g}")
    print(f"  folded   QC gamma : {fold['qc_gamma']:.4g}")
    print(f"  folded acceptance : {acc:.3f}")
    print(f"  sorted posterior mean of means : "
          f"{sorted_mean.numpy().round(3).tolist()}")
    print("  (true sorted means: [-3.0, 0.0, 6.0])")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "config.yaml"))
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
