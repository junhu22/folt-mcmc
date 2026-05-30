"""
Figure 2 (Headline): QC certified lower bound vs dimension.

Gaussian mixture, ReflectionFold (s=2), d=2,5,10,20.
FolT-MCMC certificate is empirically nearly dimension-free;
unfolded certificate collapses.

Usage:
    python plot_fig2_gamma_vs_dim.py
    # Outputs: fig_gamma_vs_dim.pdf, fig_gamma_vs_dim.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
    'figure.dpi': 300,
})

# ── Data from Phase 3 (mixture_scaling_results.json) ──
D = [2, 5, 10, 20]
gamma_U = [0.402, 0.094, 0.016, 0.016]
gamma_F = [0.936, 0.941, 0.922, 0.902]

# ── Plot ──
fig, ax = plt.subplots(figsize=(5.5, 3.8))

ax.semilogy(D, gamma_U, 's--', color='#854F0B',
            label='Unfolded',
            markerfacecolor='white', markeredgewidth=1.5, markeredgecolor='#854F0B')
ax.semilogy(D, gamma_F, 'o-', color='#0F6E56',
            label='FolT-MCMC',
            markerfacecolor='#0F6E56')

ax.set_xlabel('Dimension $d$')
ax.set_ylabel('QC certified lower bound ($\\rho = 0.05$)')
ax.set_xticks(D)
ax.set_ylim(0.008, 2.0)
ax.legend(frameon=True, fancybox=False, edgecolor='#ccc', fontsize=10)
ax.grid(True, alpha=0.3, linewidth=0.5)

# Ratio annotations
for i, d in enumerate(D):
    ratio = gamma_F[i] / gamma_U[i]
    ax.annotate(f'{ratio:.0f}x', xy=(d, gamma_U[i]),
                xytext=(0, -18), textcoords='offset points',
                ha='center', fontsize=8.5, color='#854F0B')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('fig_gamma_vs_dim.pdf', bbox_inches='tight')
fig.savefig('fig_gamma_vs_dim.png', bbox_inches='tight', dpi=300)
print("Figure 2 saved: fig_gamma_vs_dim.pdf, fig_gamma_vs_dim.png")
