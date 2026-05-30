"""
Figure 3: QC certified lower bound across label-switching configurations.

PermutationFold, four configs: k2p2/k3p2/k3p4/k4p2.
Unfolded certificate collapses superlinearly with mode count s=|G|;
FolT-MCMC certificate varies much less.

Usage:
    python plot_fig3_gamma_vs_s.py
    # Outputs: fig_gamma_vs_s.pdf, fig_gamma_vs_s.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

# ── Data from Phase 4 (labelswitching_results.json) ──
configs = [
    {'label': '$m{=}2, p{=}2$\n$d{=}4, s{=}2$',   'gamma_U': 0.366, 'gamma_F': 0.820},
    {'label': '$m{=}3, p{=}2$\n$d{=}6, s{=}6$',   'gamma_U': 0.066, 'gamma_F': 0.685},
    {'label': '$m{=}3, p{=}4$\n$d{=}12, s{=}6$',  'gamma_U': 0.036, 'gamma_F': 0.683},
    {'label': '$m{=}4, p{=}2$\n$d{=}8, s{=}24$',  'gamma_U': 0.004, 'gamma_F': 0.624},
]

labels = [c['label'] for c in configs]
gamma_U = [c['gamma_U'] for c in configs]
gamma_F = [c['gamma_F'] for c in configs]

# ── Plot ──
fig, ax = plt.subplots(figsize=(5.5, 3.8))

x_pos = np.arange(len(configs))
width = 0.32

bars_U = ax.bar(x_pos - width/2, gamma_U, width,
                color='#BA7517', alpha=0.7,
                label='Unfolded',
                edgecolor='#854F0B', linewidth=0.8)
bars_F = ax.bar(x_pos + width/2, gamma_F, width,
                color='#0F6E56', alpha=0.7,
                label='FolT-MCMC',
                edgecolor='#085041', linewidth=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('QC certified lower bound ($\\rho = 0.05$)')
ax.set_xlabel('Configuration')
ax.legend(frameon=True, fancybox=False, edgecolor='#ccc', fontsize=10,
          loc='upper right')
ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)

# Ratio annotations
for i in range(len(configs)):
    ratio = gamma_F[i] / gamma_U[i]
    y_top = max(gamma_F[i], gamma_U[i])
    ax.annotate(f'{ratio:.0f}x', xy=(x_pos[i], y_top),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold', color='#3C3489')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig('fig_gamma_vs_s.pdf', bbox_inches='tight')
fig.savefig('fig_gamma_vs_s.png', bbox_inches='tight', dpi=300)
print("Figure 3 saved: fig_gamma_vs_s.pdf, fig_gamma_vs_s.png")
