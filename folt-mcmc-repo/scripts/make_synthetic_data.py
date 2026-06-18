"""Regenerate all synthetic datasets used in the paper.

Synthetic targets are analytic; this script draws fixed-seed reference samples
for each target and saves them under data/synthetic/ so that the experiments
are fully reproducible without any external download.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from src.targets import (Banana, Funnel, GaussianMixture,   # noqa: E402
                         labelswitch_means)

OUT = os.path.join(os.path.dirname(__file__), os.pardir, "data", "synthetic")
SEED = 42
N = 50000


def main():
    os.makedirs(OUT, exist_ok=True)
    g = torch.Generator().manual_seed(SEED)

    targets = {}
    for d in [2, 5, 10, 20]:
        targets[f"banana_d{d}"] = Banana(dim=d, kappa=0.1)
    targets["funnel_d10"] = Funnel(dim=10)
    for d in [2, 5, 10, 20]:
        means = torch.zeros(2, d, dtype=torch.float64)
        means[0, 0], means[1, 0] = 3.0, -3.0
        targets[f"gmm_refl_d{d}"] = GaussianMixture(means, 1.0)
    for (m, block) in [(2, 1), (3, 2), (4, 1)]:
        d = m * block
        means = labelswitch_means(m, d, sep=4.0)
        targets[f"labelswitch_m{m}_b{block}"] = GaussianMixture(means, 1.0)

    for name, tgt in targets.items():
        x = tgt.sample(N, generator=g)
        path = os.path.join(OUT, f"{name}.pt")
        torch.save(x, path)
        print(f"wrote {path}  shape={tuple(x.shape)}")

    print(f"\nAll synthetic datasets written to {OUT} (seed={SEED}).")


if __name__ == "__main__":
    main()
