"""
Figure 1: FolT-MCMC pipeline schematic.

Left: original multi-modal target -> fold -> single-mode folded target.
Right: four-stage pipeline (Train -> Proposal -> IMH -> Diagnose).

A single axes holds both halves in one coordinate system, so the panels stay
compact and a connecting arrow can run from the folded space into the pipeline.

Usage:
    python plot_fig1_schematic.py
    # Outputs: fig_schematic.pdf, fig_schematic.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── single coordinate system for the whole schematic ──
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0.0, 0.06, 1.0, 0.90])
ax.set_xlim(0, 13.2)
ax.set_ylim(0, 6.6)
ax.set_aspect('equal')
ax.axis('off')

# colours (unchanged scheme)
C_ORIG = '#534AB7'
C_FOLD = '#0F6E56'

# ══════════════════════════════════════════════════════════════
# LEFT: quotient-space reduction
# ══════════════════════════════════════════════════════════════
ax.text(2.7, 6.42, 'Quotient-space reduction', ha='center', va='center',
        fontsize=13, fontweight='medium')

# --- original space (dashed box around two symmetric modes) ---
rect_orig = FancyBboxPatch((0.2, 4.05), 5.0, 1.75, boxstyle='round,pad=0.12',
                           fill=False, edgecolor='gray', linestyle='--',
                           linewidth=0.5)
ax.add_patch(rect_orig)
ax.text(2.7, 5.92, r'Original space $\pi(\theta)$', ha='center', va='bottom',
        fontsize=10, fontweight='medium')

for cx, label in [(1.45, 'mode 1'), (3.95, 'mode 2')]:
    ax.add_patch(mpatches.Ellipse((cx, 4.85), 1.5, 0.95, alpha=0.12, color=C_ORIG,
                                  linewidth=0.5))
    ax.add_patch(mpatches.Ellipse((cx, 4.85), 0.85, 0.55, alpha=0.25, color=C_ORIG,
                                  linewidth=0))
    ax.text(cx, 4.85, label, ha='center', va='center', fontsize=10.5,
            color=C_ORIG, style='italic')

# fold boundary: dashed symmetry axis between the modes, label set to the SIDE
# (rotated, alongside the line) so it never collides with the fold arrow below.
ax.plot([2.7, 2.7], [4.25, 5.45], '--', color='gray', linewidth=0.8)
ax.text(2.86, 4.85, 'fold boundary', rotation=90, ha='left', va='center',
        fontsize=7, color='gray', style='italic')

# --- fold arrow (in the clear channel between the two boxes) ---
ax.annotate('', xy=(2.7, 3.18), xytext=(2.7, 3.92),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.6))
ax.text(2.94, 3.55, 'fold', ha='left', va='center', fontsize=9,
        style='italic', color='#333')

# --- folded space (single mode) ---
rect_fold = FancyBboxPatch((0.9, 1.15), 3.6, 1.85, boxstyle='round,pad=0.12',
                           fill=False, edgecolor='gray', linestyle='--',
                           linewidth=0.5)
ax.add_patch(rect_fold)
ax.text(2.7, 3.08, r'Folded space $\pi_F = s \cdot \pi$', ha='center', va='bottom',
        fontsize=10, fontweight='medium')
ax.add_patch(mpatches.Ellipse((2.55, 1.95), 1.8, 1.05, alpha=0.12, color=C_FOLD,
                              linewidth=0.5))
ax.add_patch(mpatches.Ellipse((2.55, 1.95), 1.0, 0.62, alpha=0.30, color=C_FOLD,
                              linewidth=0))
ax.text(2.55, 1.95, 'single mode', ha='center', va='center', fontsize=10.5,
        color=C_FOLD, style='italic')
# solid fundamental-domain edge (left side of the folded box)
ax.plot([0.9, 0.9], [1.15, 3.0], '-', color='#333', linewidth=1.5)

# ══════════════════════════════════════════════════════════════
# RIGHT: pipeline stages
# ══════════════════════════════════════════════════════════════
ax.text(9.6, 6.42, 'FolT-MCMC pipeline', ha='center', va='center',
        fontsize=13, fontweight='medium')

stages = [
    ('Stage 1: Train $T_F$ on $\\pi_F$', 'Spectrally constrained flow', C_FOLD, '#E1F5EE'),
    ('Stage 2: Quotient proposal', '$q_F(z) = \\sum_g q_{T_F}(g \\cdot z)$', C_FOLD, '#E1F5EE'),
    ('Stage 3: IMH on $D$', 'Accept/reject with $h_F$', C_ORIG, '#EEEDFE'),
    ('Stage 4: Diagnose', 'Density-ratio diagnostic', '#D85A30', '#FAECE7'),
]

box_w, box_h = 3.8, 0.78
gap = 0.34
x0 = 7.7
y_top = 5.12                          # top stage sits below the title, not clipped

stage_centers = []
for i, (title, subtitle, text_color, bg_color) in enumerate(stages):
    y = y_top - i * (box_h + gap)
    rect = FancyBboxPatch((x0, y), box_w, box_h, boxstyle='round,pad=0.10',
                          facecolor=bg_color, edgecolor=text_color, linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x0 + box_w / 2, y + box_h * 0.64, title, ha='center', va='center',
            fontsize=11, fontweight='medium', color=text_color)
    ax.text(x0 + box_w / 2, y + box_h * 0.27, subtitle, ha='center', va='center',
            fontsize=10, color=text_color, alpha=0.78)
    stage_centers.append((x0, y + box_h / 2))     # left-edge midpoint
    if i < len(stages) - 1:
        ax.annotate('', xy=(x0 + box_w / 2, y - gap),
                    xytext=(x0 + box_w / 2, y),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

# output box
y_out = y_top - len(stages) * (box_h + gap) + gap - 0.08
rect_out = FancyBboxPatch((x0 + 0.5, y_out), box_w - 1.0, 0.58,
                          boxstyle='round,pad=0.08', facecolor='#F1EFE8',
                          edgecolor='#888780', linewidth=0.8)
ax.add_patch(rect_out)
ax.text(x0 + box_w / 2, y_out + 0.29, 'Core convergence diagnostic',
        ha='center', va='center', fontsize=11, fontweight='medium', color='#444441')
ax.annotate('', xy=(x0 + box_w / 2, y_out + 0.58),
            xytext=(x0 + box_w / 2, y_out + 0.58 + gap),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

# ══════════════════════════════════════════════════════════════
# connecting arrow: folded space  ->  Stage 1 (feeds pi_F into the pipeline)
# ══════════════════════════════════════════════════════════════
s1_x, s1_y = stage_centers[0]
ax.annotate('', xy=(s1_x - 0.04, s1_y), xytext=(4.55, 2.35),
            arrowprops=dict(arrowstyle='-|>', color='#777', lw=1.4,
                            connectionstyle='arc3,rad=-0.32'))
ax.text(5.95, 4.30, r'$\pi_F$', ha='center', va='center', fontsize=11,
        color='#555', fontweight='medium')

# ── summary strip ──
fig.text(0.5, 0.03,
         'cross-mode oscillation removed  →  core convergence diagnostic improves',
         ha='center', va='bottom', fontsize=10, style='italic', color='#555')

fig.savefig('fig_schematic.pdf', bbox_inches='tight', dpi=300)
fig.savefig('fig_schematic.png', bbox_inches='tight', dpi=300)
print("Figure 1 (schematic) saved: fig_schematic.pdf, fig_schematic.png")
