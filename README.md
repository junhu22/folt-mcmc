# FolT-MCMC: Self-Certifying Transport MCMC via Dual Spectral-Gap Certificates

Code for reproducing the experiments and figures reported in the manuscript. The manuscript and supplementary material are submitted separately.

## Requirements

- Python 3.10+
- PyTorch >= 2.0
- NumPy, SciPy, Matplotlib, scikit-learn

Install: `pip install -r experiments/requirements.txt`

## Directory Structure

```
code_supplement/
├── README.md
├── experiments/
│   ├── requirements.txt
│   ├── quantile_core_diagnostic.py    # Table 1: Banana D=2-20
│   ├── benchmark_logreg.py            # Table 2: Synthetic LogReg D=20
│   ├── benchmark_logreg_realdata.py   # Table 3: Heart Disease D=13
│   ├── verify_engineering.py          # Table 5: Sailboat + Shear
│   ├── negative_control.py            # Table 6: Negative control
│   ├── benchmark_logreg_real.py       # Breast Cancer stress test
│   ├── benchmark_logreg_whitened.py   # Breast Cancer + Laplace whitening
│   ├── exp01_local_lipschitz.py       # Supplement A: covering refinements
│   └── exp02_pointwise_grad.py        # Supplement A: covering refinements
└── paper/
    ├── make_hero_figure.py            # Figure 1
    ├── make_fig3_variance.py          # Figure 2
    └── remake_fig3_negctrl.py         # Figure 3
```

## Reproducing Experiments

Each script is self-contained. Example:

```bash
cd experiments
python quantile_core_diagnostic.py          # Banana D=2-20 (Table 1)
python benchmark_logreg.py                  # Synthetic LogReg D=20 (Table 2)
python benchmark_logreg_realdata.py --dataset heart  # Heart Disease (Table 3)
python negative_control.py                  # Negative control (Table 6)
```

Add `--quick` for fast validation runs. Full runs reproduce the paper's numbers.

## Reproducing Figures

```bash
cd paper
python make_hero_figure.py --dpi 300        # Figure 1
python make_fig3_variance.py                # Figure 2
python remake_fig3_negctrl.py               # Figure 3
```
