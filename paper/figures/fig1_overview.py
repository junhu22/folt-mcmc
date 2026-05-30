#!/usr/bin/env python3
"""
Figure 1 — Four-panel overview v4.
A: simplified linear flow (cue → relevant → missing control → action)
B: strict alignment, all text inside boxes
C: frontier (unchanged)
D: richer content
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, Rectangle, Circle, Polygon)
import matplotlib.font_manager as fm
import numpy as np

INK="#16202c"; MUTE="#5c6b7a"; LT="#94a3b8"
BLUE="#2563eb"; BLUE_L="#dbeafe"; BLUE_D="#1e40af"
RED="#d7263d"; RED_L="#fde8eb"; RED_D="#9f1239"
GREEN="#059669"; GREEN_L="#d1fae5"
AMBER="#d97706"; AMBER_L="#fef3c7"
CARD="#f8fafc"; BORDER="#dde3ea"; WHITE="#ffffff"

def rbox(ax,x,y,w,h,fc=WHITE,ec=BORDER,lw=1.0,r=0.005,zorder=3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,zorder=zorder))
def T(ax,x,y,s,**kw):
    kw.setdefault("color",INK); kw.setdefault("fontsize",12)
    kw.setdefault("va","center"); kw.setdefault("ha","center"); kw.setdefault("zorder",8)
    return ax.text(x,y,s,**kw)
def arr(ax,x0,y0,x1,y1,color=INK,lw=1.5,ms=10,zorder=7,ls="-",**kw):
    ax.annotate("",xy=(x1,y1),xytext=(x0,y0),
        arrowprops=dict(arrowstyle="-|>",color=color,lw=lw,mutation_scale=ms,
                        linestyle=ls,**kw),zorder=zorder)
def plabel(ax,x,y,letter,color):
    rbox(ax,x,y-0.011,0.019,0.022,fc=color,ec="none",r=0.004,zorder=9)
    T(ax,x+0.0095,y,letter,color=WHITE,fontsize=14,fontweight="bold")

# ────────────────────────────────────────────────────────────────────────────
# Panel A — Simplified: linear flow diagram
# ────────────────────────────────────────────────────────────────────────────
def panel_A(ax, L, B, W, H):
    rbox(ax, L, B, W, H, fc=CARD, ec=BORDER, lw=1.2, r=0.007)
    plabel(ax, L+0.008, B+H-0.018, "A", RED)
    T(ax, L+0.034, B+H-0.018, "Relevant does not mean warranted",
      fontsize=14, fontweight="bold", ha="left")

    pad = 0.014
    iL = L + pad; iW = W - 2*pad

    # ---- Speech bubble at top ----
    bub_y = B + H - 0.055
    ax.add_patch(Circle((iL + 0.010, bub_y), 0.009, fc=LT, ec="none", zorder=8))
    T(ax, iL+0.010, bub_y, "●", fontsize=6, color="#475569")
    bub_l = iL + 0.025; bub_w = iW * 0.65
    rbox(ax, bub_l, bub_y-0.013, bub_w, 0.026, fc=WHITE, ec="#cbd5e1", lw=0.8, r=0.005, zorder=5)
    T(ax, bub_l+bub_w/2, bub_y, "Maybe we should revisit that trip idea.",
      fontsize=10.5, style="italic")

    # ---- Linear flow ----
    mid_y = B + H*0.52

    box_h = 0.058
    box_w = [0.100, 0.115, 0.095, 0.090]
    gaps = [0.020, 0.044, 0.020]  # middle gap wider for the red-line crossing
    total_w = sum(box_w) + sum(gaps)
    start_x = iL + (iW - total_w)/2

    boxes = [
        ("User cue",          WHITE,   "#cbd5e1", INK),
        ("Retrieved\nmemory / cue", BLUE_L,  BLUE,     BLUE_D),
        ("Direct\nplanner",   WHITE,   "#cbd5e1", INK),
        ("Action\n/ plan",    GREEN_L, GREEN,    GREEN),
    ]

    positions = []
    x = start_x
    for i, (label, fc, ec, tc) in enumerate(boxes):
        w = box_w[i]
        y = mid_y - box_h/2
        rbox(ax, x, y, w, box_h, fc=fc, ec=ec, lw=1.3, r=0.006, zorder=5)
        T(ax, x+w/2, mid_y, label, fontsize=9, color=tc, fontweight="bold", linespacing=1.1)
        positions.append((x, x+w/2, x+w, mid_y))
        if i < 3:
            x += w + gaps[i]

    for i in range(3):
        arr(ax, positions[i][2]+0.003, mid_y, positions[i+1][0]-0.003, mid_y,
            color=LT, lw=1.5, ms=10)

    T(ax, positions[1][1], mid_y-box_h/2-0.016, "Relevant cue",
      fontsize=8, color=BLUE, style="italic")

    # "missing control" — longer red line
    ctrl_x = (positions[1][2] + positions[2][0]) / 2
    ax.plot([ctrl_x, ctrl_x], [mid_y-box_h/2-0.018, mid_y+box_h/2+0.018],
            color=RED, lw=2.5, zorder=7)
    T(ax, ctrl_x, mid_y+box_h/2+0.035, "missing\ncontrol", fontsize=8, color=RED,
      fontweight="bold", linespacing=1.0)

    # ---- Bottom annotations (wider boxes, more spread) ----
    warn_w = 0.350; warn_h = 0.028
    warn_x = iL + (iW - warn_w)/2
    warn_y = mid_y - box_h/2 - 0.068
    rbox(ax, warn_x, warn_y, warn_w, warn_h, fc=RED_L, ec=RED, lw=1.0, r=0.005, zorder=5)
    T(ax, warn_x+warn_w/2, warn_y+warn_h/2,
      "Premature action  =  unwarranted commitment",
      fontsize=9.5, color=RED, fontweight="bold")

    thesis_w = 0.260; thesis_h = 0.026
    thesis_x = iL + (iW - thesis_w)/2
    thesis_y = warn_y - 0.042
    rbox(ax, thesis_x, thesis_y, thesis_w, thesis_h, fc=WHITE, ec=BORDER, lw=0.8, r=0.004, zorder=5)
    T(ax, thesis_x+thesis_w/2, thesis_y+thesis_h/2,
      "Relevance  ≠  commitment", fontsize=10, fontweight="bold", color=INK)


# ────────────────────────────────────────────────────────────────────────────
# Panel B — Strict alignment, all text inside boxes
# ────────────────────────────────────────────────────────────────────────────
def panel_B(ax, L, B, W, H):
    rbox(ax, L, B, W, H, fc=CARD, ec=BORDER, lw=1.2, r=0.007)
    plabel(ax, L+0.008, B+H-0.018, "B", BLUE)
    T(ax, L+0.034, B+H-0.018, "Commitment-control layer",
      fontsize=14, fontweight="bold", ha="left")

    pad = 0.012
    iL = L + pad; iR = L + W - pad; iW = iR - iL
    mid_y = B + H*0.50

    # ---- geometry: three columns (spread wider) ----
    col1_w = 0.068   # inputs
    col2_w = 0.155   # center box
    col3_w = 0.068   # outputs
    arr_gap = 0.032  # wider gaps between columns
    total = col1_w + col2_w + col3_w + 2*arr_gap
    col1_x = iL + (iW - total)/2
    col2_x = col1_x + col1_w + arr_gap
    col3_x = col2_x + col2_w + arr_gap

    # ---- LEFT: 3 input boxes ----
    ih = 0.035
    input_labels = ["User query", "Retrieved\ncues", "Tools &\nschema"]
    i_centers = []
    for k, label in enumerate(input_labels):
        iy = mid_y + (1-k)*0.050 - ih/2
        rbox(ax, col1_x, iy, col1_w, ih, fc=WHITE, ec="#cbd5e1", lw=0.9, r=0.004, zorder=5)
        T(ax, col1_x+col1_w/2, iy+ih/2, label, fontsize=8, fontweight="bold", linespacing=1.0)
        i_centers.append((col1_x+col1_w, iy+ih/2))

    # ---- CENTER: commitment-control layer (taller, title inside) ----
    ch = 0.220
    cy = mid_y - ch/2
    rbox(ax, col2_x, cy, col2_w, ch, fc=BLUE_L, ec=BLUE, lw=1.5, r=0.007, zorder=4)
    # Title well inside the box
    T(ax, col2_x+col2_w/2, cy+ch-0.022,
      "Commitment-control\nlayer", fontsize=8, fontweight="bold", color=BLUE_D, linespacing=1.05)

    sub_labels = ["Rule scorer", "Learned scorer", "Decision logic"]
    sw = col2_w - 0.016; sh = 0.034
    sy = cy + ch - 0.068  # more room below 2-line title
    for label in sub_labels:
        sx = col2_x + 0.008
        rbox(ax, sx, sy, sw, sh, fc=WHITE, ec="#93c5fd", lw=0.8, r=0.004, zorder=5)
        T(ax, sx+sw/2, sy+sh/2, label, fontsize=8.5, fontweight="bold", color=BLUE_D)
        sy -= 0.046

    # ---- RIGHT: 4 output boxes ----
    oh = 0.030
    out_items = [
        ("Commit",  GREEN),
        ("Clarify",  AMBER),
        ("Defer",    BLUE),
        ("Reject",   RED),
    ]
    o_centers = []
    for k, (label, col) in enumerate(out_items):
        oy = mid_y + (1.5-k)*0.044 - oh/2
        rbox(ax, col3_x, oy, col3_w, oh, fc=WHITE, ec=col, lw=1.3, r=0.004, zorder=5)
        ax.add_patch(Rectangle((col3_x, oy), 0.004, oh, fc=col, ec="none", zorder=6))
        T(ax, col3_x+col3_w/2, oy+oh/2, label, fontsize=8.5, fontweight="bold", color=col)
        o_centers.append((col3_x, oy+oh/2))

    # ---- arrows: inputs → center ----
    for (xr, yr) in i_centers:
        arr(ax, xr+0.003, yr, col2_x-0.003, mid_y, color=LT, lw=1.0, ms=8)

    # ---- arrows: center → outputs ----
    cx_out = col2_x + col2_w
    for (xl, yl) in o_centers:
        arr(ax, cx_out+0.003, mid_y, xl-0.003, yl, color=LT, lw=1.0, ms=8)

    # "→ Planner" annotation (small, right of Commit box, inside panel)
    commit_y = o_centers[0][1]
    T(ax, col3_x+col3_w+0.005, commit_y, "→ Planner",
      fontsize=7, color=MUTE, style="italic", ha="left")

    # Footer note inside panel
    T(ax, L+W/2, B+0.012,
      "The planner is invoked only after a commit.",
      fontsize=8, color=MUTE, style="italic")


# ────────────────────────────────────────────────────────────────────────────
# Panel C — Safety–coverage frontier (unchanged)
# ────────────────────────────────────────────────────────────────────────────
def panel_C(fig, ax_main, L, B, W, H):
    rbox(ax_main, L, B, W, H, fc=CARD, ec=BORDER, lw=1.2, r=0.007)
    plabel(ax_main, L+0.008, B+H-0.018, "C", AMBER)
    T(ax_main, L+0.034, B+H-0.018, "Safety–coverage frontier",
      fontsize=14, fontweight="bold", ha="left")

    pax = fig.add_axes([L+0.050, B+0.058, W-0.075, H-0.092])
    pax.set_xlim(-0.05, 1.08); pax.set_ylim(-0.05, 1.08)
    pax.set_xlabel("Coverage = 1 − MIR", fontsize=9, color=INK, labelpad=3)
    pax.set_ylabel("Safety = 1 − FIR", fontsize=9, color=INK, labelpad=3)
    pax.tick_params(labelsize=9, colors=MUTE)
    for sp in pax.spines.values(): sp.set_color(BORDER); sp.set_linewidth(0.8)
    pax.set_facecolor("#fafbfc")
    pax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
    pax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])

    t = np.linspace(0,1,300)
    pax.plot(t, 1-t**2.5, color=BLUE, lw=2.0, alpha=0.5, zorder=3)
    pax.fill_between(t, 1-t**2.5, 1.0, alpha=0.04, color=BLUE, zorder=2)

    pts = [
        (0.03, 0.99, "Never-call",           "#6b7280", (14,-8),  "left"),
        (0.50, 0.94, "Risk-aware\nthreshold", AMBER,    (10,10),  "left"),
        (0.68, 0.88, "Rule scorer",           BLUE,     (10,12),  "left"),
        (0.82, 0.78, "Learned scorer",        GREEN,    (10,-16), "left"),
        (0.98, 0.04, "Always-call",           RED,      (-14,14), "right"),
    ]
    for cx,cy,lab,col,off,ha in pts:
        pax.scatter(cx,cy,s=50,color=col,zorder=6,edgecolors=WHITE,linewidths=1.2)
        pax.annotate(lab,(cx,cy),xytext=off,textcoords="offset points",
            fontsize=8.5,color=col,fontweight="bold",ha=ha,va="center",linespacing=1.1,
            arrowprops=dict(arrowstyle="-",color=col,lw=0.6)
            if max(abs(off[0]),abs(off[1]))>8 else None)
    pax.scatter(0.82,0.78,s=140,marker="*",color=GREEN,zorder=7,edgecolors=WHITE,linewidths=0.7)
    pax.plot([0.82,0.98],[0.78,0.04],color=RED,ls="--",lw=1.2,alpha=0.4,zorder=3)
    pax.text(0.94,0.36,"More coverage,\nbut less safety",fontsize=7,
             color=RED,ha="center",style="italic",alpha=0.55,linespacing=1.1)


# ────────────────────────────────────────────────────────────────────────────
# Panel D — Richer: 3 hero stats + supporting details
# ────────────────────────────────────────────────────────────────────────────
def panel_D(ax, L, B, W, H):
    rbox(ax, L, B, W, H, fc=CARD, ec=BORDER, lw=1.2, r=0.007)
    plabel(ax, L+0.008, B+H-0.018, "D", GREEN)
    T(ax, L+0.034, B+H-0.018, "Empirical payoff",
      fontsize=14, fontweight="bold", ha="left")

    pad = 0.012
    iL = L + pad; iR = L + W - pad; iW = iR - iL

    # ---- Compute vertical centering ----
    # Content block: heroes + gap + detail + gap + supports
    hh = 0.100          # hero card height
    gap1 = 0.016        # hero → detail text
    detail_h = 0.010    # detail text line
    gap2 = 0.028        # detail → support cards
    sh = 0.068           # support card height
    total_content = hh + gap1 + detail_h + gap2 + sh

    usable_top = B + H - 0.042   # below title
    usable_bot = B + 0.024       # above footer
    usable_h = usable_top - usable_bot
    top_y = usable_bot + (usable_h + total_content) / 2  # top of content block

    # ---- Top row: 3 hero numbers ----
    hy = top_y
    n_heroes = 3
    hgap = 0.008
    hw = (iW - (n_heroes-1)*hgap) / n_heroes

    heroes = [
        ("3.6×",  "fewer\nfalse calls", GREEN, GREEN_L, "#6ee7b7"),
        ("50%",   "fewer\nmodel calls",  BLUE,  BLUE_L,  "#93c5fd"),
        ("1.00",  "ExecBench\nsafety",    "#7c3aed", "#ede9fe", "#c4b5fd"),
    ]
    for i, (num, label, tc, fc, ec) in enumerate(heroes):
        hx = iL + i*(hw + hgap)
        rbox(ax, hx, hy-hh, hw, hh, fc=fc, ec=ec, lw=1.2, r=0.006, zorder=5)
        T(ax, hx+hw/2, hy-hh*0.35, num, fontsize=24, fontweight="bold", color=tc)
        T(ax, hx+hw/2, hy-hh*0.78, label, fontsize=8.5, color=tc,
          fontweight="bold", linespacing=1.1)

    # ---- Middle row: detail sub-lines ----
    dy = hy - hh - gap1
    details = [
        "BFCL: 0.20 → 0.056",
        "Across tasks & benchmarks",
        "Qwen + Llama backends",
    ]
    for i, detail in enumerate(details):
        hx = iL + i*(hw + hgap)
        T(ax, hx+hw/2, dy, detail, fontsize=7.5, color=MUTE)

    # ---- Bottom row: 4 supporting cards ----
    sy_top = dy - detail_h - gap2
    n_cards = 4
    sgap = 0.006
    sw = (iW - (n_cards-1)*sgap) / n_cards

    supports = [
        ("77.5%",     "calls saved",   GREEN,  GREEN_L, "#a7f3d0"),
        ("Tunable",   "operating\npoint", AMBER, AMBER_L, "#fcd34d"),
        ("≈ quality", "no argument\ndegradation", INK, WHITE, BORDER),
        ("Transfers", "across models\n& benchmarks", BLUE, BLUE_L, "#93c5fd"),
    ]
    for i, (title, sub, tc, fc, ec) in enumerate(supports):
        sx = iL + i*(sw + sgap)
        rbox(ax, sx, sy_top-sh, sw, sh, fc=fc, ec=ec, lw=0.9, r=0.004, zorder=5)
        T(ax, sx+sw/2, sy_top-sh*0.38, title, fontsize=10, fontweight="bold", color=tc)
        T(ax, sx+sw/2, sy_top-sh*0.72, sub, fontsize=7, color=MUTE, linespacing=1.0)

    # ---- Footer ----
    T(ax, L+W/2, B+0.012,
      "Cloud or local deployment  ·  Bootstrap CIs exclude zero",
      fontsize=7, color=MUTE, style="italic")


# ── Assemble ──
def build(stem="fig1_overview"):
    fig = plt.figure(figsize=(11, 8.0), dpi=200)
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

    m=0.010; gx=0.014; gy=0.012
    pw = (1-2*m-gx)/2
    ph = (1-2*m-gy)/2

    TL=(m, m+ph+gy, pw, ph)
    TR=(m+pw+gx, m+ph+gy, pw, ph)
    BL=(m, m, pw, ph)
    BR=(m+pw+gx, m, pw, ph)

    panel_A(ax, *TL)
    panel_B(ax, *TR)
    panel_C(fig, ax, *BL)
    panel_D(ax, *BR)

    fig.savefig(f"{stem}.pdf", facecolor="white")
    fig.savefig(f"{stem}.png", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"wrote {stem}.pdf / {stem}.png")

if __name__=="__main__": build()
