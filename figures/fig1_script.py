"""Figure 1 — per-class AP versus positive counts in the report split.

Self-contained: reads only ``fig1_data.csv`` in this folder and regenerates
``fig1.pdf`` / ``fig1.png``. Requires matplotlib, numpy, pandas, scipy.

Run:  python fig1_script.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "fig1_data.csv")

df = pd.read_csv(DATA)
rows = [(r.abbr, int(r.report_pos), float(r.ap_3seed_mean)) for r in df.itertuples()]

x = np.array([r[1] for r in rows], float)
y = np.array([r[2] for r in rows], float)
rho, pval = spearmanr(x, y)
print(f"{'class':>4} {'pos':>5} {'AP':>7}")
for a, p, m in rows:
    print(f"{a:>4} {p:>5d} {m:>7.4f}")
print(f"\nSpearman rho={rho:.4f} p={pval:.4g}")

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
ax.scatter(x, y, s=38, c="#2b6cb0", zorder=3)
ax.set_xscale("log")
ax.set_xlabel("Positive samples in report split (log scale)")
ax.set_ylabel("Per-class AP")
ax.set_ylim(0, 1.0)
ax.set_xlim(6, 1400)
ax.grid(alpha=0.3, linewidth=0.5)

offsets = {"NOF": (-30, 6), "CMG": (8, 6), "ALV": (8, -14), "BRC": (-38, 6),
           "PLE": (8, 6), "PTX": (8, 6), "MAS": (12, -4), "INT": (8, -14),
           "PLM": (8, 6), "MGE": (-34, -14), "FB": (8, -14), "HRN": (8, 6)}
for a, p, m in rows:
    ax.annotate(a, (p, m), textcoords="offset points", xytext=offsets.get(a, (8, 6)),
                fontsize=8, color="#1a202c")

hl = {a: (p, m) for a, p, m in rows if a in ("PLE", "BRC")}
for a, (p, m) in hl.items():
    ax.scatter([p], [m], s=110, facecolors="none", edgecolors="#c53030",
               linewidths=1.6, zorder=4)
ple, brc = hl["PLE"], hl["BRC"]
dap = ple[1] - brc[1]
ax.annotate("", xy=ple, xytext=brc,
            arrowprops=dict(arrowstyle="<->", color="#c53030", lw=1.2,
                            connectionstyle="arc3,rad=0.25"))
ax.annotate(f"pos\u2248{ple[0]} vs {brc[0]}, \u0394AP\u2248{dap:.2f}",
            xy=((ple[0] * brc[0]) ** 0.5, (ple[1] + brc[1]) / 2 + 0.055),
            fontsize=7.5, color="#c53030", ha="center")

ax.text(0.97, 0.05, f"Spearman \u03c1 = {rho:.3f}\np = {pval:.3g}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#718096", lw=0.8))

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(BASE, f"fig1.{ext}"), dpi=300, bbox_inches="tight")
print("saved:", BASE)
