#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =========================================================
# 0) Paths (relative; repo-friendly) + output
# =========================================================
ROOT = Path(__file__).resolve().parents[1]

FILE_SIGNED = (
    ROOT
    / "outputs" / "SHAP_INTERACTION" / "IP"
    / "shap_signed_sum_low.csv"
)
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "fig_7_A_left.pdf"
OUT_SVG = OUT_DIR / "fig_7_A_left.svg"

# =========================================================
# 1) Style: default font + all black (no proprietary font)
# =========================================================
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.usetex": False,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",

    # Force all-black to avoid grey
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# seaborn base style (white background)
sns.set_style("white")

# =========================================================
# 2) Short names & order
# =========================================================
feature_name_map = {
    "indicator_sex_female":                "FP",
    "indicator_Ethnic_nonwhite":           "NWP",
    "indicator_education_noqualification": "NQP",
    "indicator_age_elder":                 "EP",
    "indicator_age_children":              "CP",
    "indicator_NS-SeC_unemployed":         "UEP",
    "indicator_NS-SeC_students":           "SP",
}
short_order = ["FP", "NWP", "NQP", "EP", "CP", "UEP", "SP"]

# =========================================================
# 3) Read matrix + rename + enforce order
# =========================================================
df_total = pd.read_csv(FILE_SIGNED, index_col=0)

# Your CSV may already use long names; robust rename both axes
df_total.rename(index=feature_name_map, columns=feature_name_map, inplace=True)

missing = [k for k in short_order if (k not in df_total.index or k not in df_total.columns)]
if missing:
    raise KeyError(
        f"Missing keys after renaming: {missing}. "
        "Please check the CSV row/column names and feature_name_map."
    )

df_total = df_total.loc[short_order, short_order].astype(float)

# Mask upper triangle + diagonal
mask = np.triu(np.ones_like(df_total, dtype=bool), k=0)
np.fill_diagonal(df_total.values, np.nan)

# =========================================================
# 4) Custom colormap (green-white-purple), centered at 0
# =========================================================
custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_shap_cmap",
    [(0.0, "#7cc0a2"), (0.5, "#FFFFFF"), (1.0, "#9294bb")],
    N=256
)

# =========================================================
# 5) Plot (journal size)
# =========================================================
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(8 * cm, 8 * cm))

hm = sns.heatmap(
    df_total,
    annot=True,
    fmt=".2f",
    cmap=custom_cmap,
    center=0,
    square=True,
    linewidths=0,
    mask=mask,
    cbar_kws={
        "label": "Total SHAP Interaction Value",
        "shrink": 0.85,
        "aspect": 40,          # smaller -> thicker; larger -> thinner
        "orientation": "horizontal",
        "pad": 0.10            # smaller -> closer to main plot
    },
    ax=ax
)

# =========================================================
# 6) Force all text black; keep default font
# =========================================================
# Cell annotations
for t in ax.texts:
    t.set_color("black")

# Axis tick labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", color="black")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="black")

# Title (remove if not needed)
ax.set_title("Total SHAP Interaction Matrix", fontsize=8, color="black", pad=4)

# Axes spines
for s in ax.spines.values():
    s.set_linewidth(0.5)
    s.set_edgecolor("black")

# Colorbar styling
cbar = ax.collections[0].colorbar
cbar.ax.xaxis.label.set_color("black")
for lbl in cbar.ax.get_xticklabels():
    lbl.set_color("black")
cbar.ax.tick_params(width=0.5, length=2, colors="black")

for s in cbar.ax.spines.values():
    s.set_linewidth(0.5)
    s.set_edgecolor("black")

# =========================================================
# 7) Optional diagonal dashed guide line (visual only)
# =========================================================
n = len(short_order)
for i in range(n):
    ax.plot([i, i + 1], [i, i + 1], color="black", lw=0.7, linestyle="--", zorder=10)

plt.tight_layout()

# =========================================================
# 8) Save: PDF + SVG
# =========================================================
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0)
fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0)

plt.show()
print("✅ Saved:", OUT_PDF)
print("✅ Saved:", OUT_SVG)
