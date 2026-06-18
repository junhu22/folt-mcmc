# Data Availability

## Synthetic data

All synthetic target distributions used in the paper (the banana family, the
Gaussian mixtures, Neal's funnel, the Bayesian logistic-regression posterior,
and the label-switching targets) are defined analytically. Their exact
specifications are given in Appendix A of the paper, and they can be regenerated
deterministically (fixed random seeds) by running:

```bash
python scripts/make_synthetic_data.py
```

No external data download is required for any synthetic experiment.

## Real accelerometer data

The structural modal-identification example in the paper uses ambient vibration
(accelerometer) measurements recorded on a supertall building during a typhoon.
These measurements were provided by a third party under a confidentiality
agreement, and the building owner has required that the structure not be
identified. For these reasons the raw accelerometer data **cannot be shared
publicly** and are not included in this repository.

The data-processing pipeline (spectral-density estimation, the simplified
Whittle likelihood, and the three-mode model) is described in full in the paper,
so the methodology can be applied to any comparable ambient-vibration dataset.

Researchers seeking access to the specific dataset should contact the
corresponding author; access is subject to the owner's confidentiality
conditions and cannot be guaranteed.
