#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Requirements (same as just now):
# - NO proprietary font (use default font)
# - Force all-black text/axes (avoid seaborn grey)
# - Repo-friendly paths (relative)
# - Export PDF + SVG

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================================================
# 0) Paths (relative; repo-friendly)
# =========================================================
ROOT = Path(__file__).resolve().parents[1]

FILE_INTER_RAW = (
    ROOT / "outputs" / "SHAP_INTERACTION" / "IA"
    / "shap_inter_raw_low.npy"
)
FILE_MASK = (
    ROOT / "outputs" / "SHAP_INTERACTION" / "IA"
    / "interaction"
    / "shap_mask_low.npy"
)  # optional (not used below)
FILE_SIGNED = (
    ROOT / "outputs" / "SHAP_INTERACTION" / "IA" / "interaction"
    / "shap_signed_sum_low.csv"
)

OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "fig_7_B_right.pdf"
OUT_SVG = OUT_DIR / "fig_7_B_right.svg"

# =========================================================
# 1) Style: default font + all black
# =========================================================
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.usetex": False,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",

    # Force all-black
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

# seaborn: keep it minimal; don't use grey themes
sns.set_style("white")

# =========================================================
# 2) Short-name mapping
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

# =========================================================
# 3) Load data + decide feature order
# =========================================================
interaction_values = np.load(FILE_INTER_RAW)  # (n_samples, n_feat, n_feat)

df_signed = pd.read_csv(FILE_SIGNED, index_col=0)
cols_long = list(df_signed.columns)
short_names = [feature_name_map.get(c, c) for c in cols_long]

n_feat = interaction_values.shape[1]
if len(short_names) != n_feat:
    cols_long = list(feature_name_map.keys())[:n_feat]
    short_names = [feature_name_map.get(c, c) for c in cols_long]

# =========================================================
# 4) Build long table (upper triangle i<j)
# =========================================================
records = []
for i in range(n_feat):
    for j in range(i + 1, n_feat):
        pair = f"{short_names[i]} & {short_names[j]}"
        vals = interaction_values[:, i, j]
        records.extend([(pair, float(v)) for v in vals])

df_box = pd.DataFrame(records, columns=["Pair", "SHAP Interaction Value"])

# Optional: keep pairs in a stable, meaningful order (rather than alpha)
pair_order = [f"{short_names[i]} & {short_names[j]}" for i in range(n_feat) for j in range(i + 1, n_feat)]
df_box["Pair"] = pd.Categorical(df_box["Pair"], categories=pair_order, ordered=True)

# =========================================================
# 5) Plot (journal size)
# =========================================================
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(10 * cm, 8 * cm))

sns.boxplot(
    x="Pair",
    y="SHAP Interaction Value",
    data=df_box,
    order=pair_order,
    showfliers=False,
    ax=ax,
    boxprops=dict(facecolor="none", edgecolor="black", linewidth=0.6),
    whiskerprops=dict(color="black", linewidth=0.6),
    medianprops=dict(color="black", linewidth=0.8),
    capprops=dict(color="black", linewidth=0.6),
)

# y=0 reference line (keep your original red dashed)
ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0, zorder=0)

# Titles/labels (default font)
ax.set_title("Distribution of SHAP Interactions Across Variable Pairs", fontsize=8, color="black", pad=4)
ax.set_xlabel("")
ax.set_ylabel("SHAP Interaction Value", fontsize=8, color="black")

# Tick labels: rotate x labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", color="black")
ax.set_yticklabels(ax.get_yticklabels(), color="black")

ax.tick_params(axis="both", which="major", width=0.5, length=2, colors="black")

# Spines: only left/bottom
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)
ax.spines["left"].set_edgecolor("black")
ax.spines["bottom"].set_edgecolor("black")

plt.tight_layout()

# =========================================================
# 6) Save: PDF + SVG
# =========================================================
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0)
fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0)

plt.close(fig)
print("✅ Saved:", OUT_PDF)
print("✅ Saved:", OUT_SVG)
