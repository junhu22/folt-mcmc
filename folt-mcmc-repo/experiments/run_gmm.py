"""Gaussian-mixture dimension-scaling experiment.

Reproduces the comparison of the folded vs unfolded convergence diagnostic on
reflection-symmetric Gaussian mixtures across dimensions d = 2..20
(paper, Section 5). Prints QC gamma for both samplers.
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from src.targets import GaussianMixture          # noqa: E402
from src.flow import RealNVP                      # noqa: E402
from src.fold import (ReflectionGroup, FoldedTarget,  # noqa: E402
                      QuotientProposal)
from src.train import train_flow                  # noqa: E402
from src.sampler import independence_mh           # noqa: E402
from src.diagnostic import diagnostic             # noqa: E402


def reflection_gmm(d, sep=3.0):
    means = torch.zeros(2, d, dtype=torch.float64)
    means[0, 0] = sep
    means[1, 0] = -sep
    return GaussianMixture(means, cov_scale=1.0)


def run(dims, cfg):
    g = torch.Generator().manual_seed(cfg["seed"])
    print(f"{'d':>4} {'QC gamma (unfolded)':>20} {'QC gamma (folded)':>20}")
    for d in dims:
        target = reflection_gmm(d)
        group = ReflectionGroup()

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

        # unfolded: target = pi on R^d, proposal = flow
        unf = diagnostic(target, flow, cert, rho=cfg["diagnostic"]["rho"])

        # folded: target = pi_F on D, proposal = quotient proposal
        ftarget = FoldedTarget(target, group)
        fprop = QuotientProposal(flow, group)
        cert_f = group.to_fundamental(cert)
        fold = diagnostic(ftarget, fprop, cert_f,
                          rho=cfg["diagnostic"]["rho"])

        print(f"{d:>4} {unf['qc_gamma']:>20.4f} {fold['qc_gamma']:>20.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[2, 5, 10, 20])
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(args.dims, cfg)
