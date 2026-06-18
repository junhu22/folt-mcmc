"""Label-switching experiment.

Reproduces the folded vs unfolded diagnostic on m-component exchangeable
Gaussian mixtures (permutation symmetry S_m), where the posterior has m!
equivalent modes (paper, Section 5).
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from src.targets import GaussianMixture, labelswitch_means   # noqa: E402
from src.flow import RealNVP                                 # noqa: E402
from src.fold import (PermutationGroup, FoldedTarget,        # noqa: E402
                      QuotientProposal)
from src.train import train_flow                             # noqa: E402
from src.diagnostic import diagnostic                        # noqa: E402


def run(cases, cfg):
    g = torch.Generator().manual_seed(cfg["seed"])
    print(f"{'m':>3} {'block':>5} {'modes':>6} "
          f"{'QC gamma (unfolded)':>20} {'QC gamma (folded)':>20}")
    for (m, block) in cases:
        d = m * block
        means = labelswitch_means(m, d, sep=4.0)
        target = GaussianMixture(means, cov_scale=1.0)
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
            osc_lambda=0.02, generator=g)

        cert = target.sample(10000, generator=g)
        unf = diagnostic(target, flow, cert, rho=cfg["diagnostic"]["rho"])

        ftarget = FoldedTarget(target, group)
        fprop = QuotientProposal(flow, group)
        cert_f = group.to_fundamental(cert)
        fold = diagnostic(ftarget, fprop, cert_f,
                          rho=cfg["diagnostic"]["rho"])

        print(f"{m:>3} {block:>5} {group.order:>6} "
              f"{unf['qc_gamma']:>20.4f} {fold['qc_gamma']:>20.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "config.yaml"))
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # (m, block): m components, each block-dimensional
    cases = [(2, 1), (3, 2), (4, 1)]
    run(cases, cfg)
