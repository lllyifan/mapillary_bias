#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import PathCollection

# =====================================================
# Paths (relative to repository root)
# =====================================================
ROOT = Path(__file__).resolve().parents[1]

SHAP_PATH = ROOT / "outputs" / "SHAP" / "IP"/ "shap_values_low.npy"
X_PATH = ROOT / "outputs" / "SHAP" / "IP" / "shap_X_val.csv"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "fig_5_A_left.pdf"
OUT_SVG = OUT_DIR / "fig_5_A_left.svg"

# =====================================================
# Matplotlib style (default font)
# =====================================================
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
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
        "legend.fontsize": 8,
        "figure.titlesize": 8,
    }
)

# =====================================================
# Feature name mapping
# =====================================================
feature_map = {
    "indicator_sex_female": "FP",
    "indicator_Ethnic_nonwhite": "NWP",
    "indicator_age_elder": "EP",
    "indicator_age_children": "CP",
    "indicator_education_noqualification": "NQP",
    "indicator_NS-SeC_unemployed": "UEP",
    "indicator_NS-SeC_students": "SP",
}

# =====================================================
# Load SHAP values and validation features
# =====================================================
shap_vals = np.load(SHAP_PATH)
X_raw = pd.read_csv(X_PATH)

X = X_raw[list(feature_map.keys())].rename(columns=feature_map)

# =====================================================
# Sort features by mean |SHAP|
# =====================================================
mean_abs = np.abs(shap_vals).mean(axis=0)
df_imp = (
    pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)

sorted_feats = df_imp["feature"].tolist()
order_idx = [X.columns.get_loc(f) for f in sorted_feats]
shap_vals_sorted = shap_vals[:, order_idx]
X_sorted = X[sorted_feats]

# =====================================================
# Custom colormap for feature values
# =====================================================
custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_shap_cmap",
    [(0.0, "#7cc0a2"), (0.5, "#FFFFFF"), (1.0, "#9294bb")],
    N=256,
)

vmin = float(np.nanmin(X_sorted.values))
vmax = float(np.nanmax(X_sorted.values))
norm = plt.Normalize(vmin, vmax)

# =====================================================
# SHAP beeswarm
# =====================================================
expl = shap.Explanation(values=shap_vals_sorted, data=X_sorted.values, feature_names=sorted_feats)
shap.plots.beeswarm(expl, color=custom_cmap, show=False, plot_size=None)

fig = plt.gcf()
fig.set_size_inches(10 / 2.54, 6 / 2.54)
main_ax = fig.axes[0]

# Style the x=0 reference line
for line in main_ax.lines:
    xdata = line.get_xdata()
    if np.allclose(xdata, [0, 0]):
        line.set_color("black")
        line.set_linewidth(1.0)
        line.set_linestyle("--")

# Make points crisp/small
for coll in main_ax.collections:
    if isinstance(coll, PathCollection):
        n = coll.get_offsets().shape[0]
        coll.set_sizes(np.full(n, 5.0))
        coll.set_linewidths(0.0)
        coll.set_rasterized(False)

# Remove SHAP-added extra axes (keep main)
extra_axes = [ax for ax in fig.axes if ax is not main_ax]
for ax_extra in extra_axes:
    fig.delaxes(ax_extra)

divider = make_axes_locatable(main_ax)

# =====================================================
# Colorbar (top)
# =====================================================
CBAR_THICKNESS = "4.5%"
CBAR_PAD = 0.16

cax = divider.append_axes("top", size=CBAR_THICKNESS, pad=CBAR_PAD)
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])

cb = plt.colorbar(sm, cax=cax, orientation="horizontal", ticks=[vmin, vmax])
cb.ax.set_xticklabels(["Low", "High"], fontsize=8)
cb.set_label("")

cax.text(
    0.5,
    -0.65,
    "Feature value",
    transform=cax.transAxes,
    ha="center",
    va="top",
    fontsize=8,
    color="black",
)

cb.ax.tick_params(width=0.5, length=2, colors="black")
for spine in cax.spines.values():
    spine.set_linewidth(0.5)
    spine.set_edgecolor("black")

# Main axis styling
for spine in main_ax.spines.values():
    spine.set_linewidth(0.5)
    spine.set_edgecolor("black")
main_ax.set_xlabel("SHAP value", fontsize=8, labelpad=6, color="black")
main_ax.tick_params(axis="both", which="major", width=0.5, length=2, labelsize=8, colors="black")

# =====================================================
# Right-side bar chart: mean |SHAP|
# =====================================================
ax_bar = divider.append_axes("right", size="26%", pad=0.30)
y_pos = np.arange(len(sorted_feats))

bars = ax_bar.barh(
    y=y_pos,
    width=df_imp["mean_abs_shap"].values,
    color="#BDBDBD",
    edgecolor="#4D4D4D",
    height=0.60,
    linewidth=0.5,
)
ax_bar.invert_yaxis()

xpad = float(df_imp["mean_abs_shap"].max()) * 0.02 if len(df_imp) else 0.0
for bar, val in zip(bars, df_imp["mean_abs_shap"].values):
    ax_bar.text(
        bar.get_width() + xpad,
        bar.get_y() + bar.get_height() / 2,
        f"{float(val):.3f}",
        va="center",
        ha="left",
        fontsize=8,
        color="black",
    )

ax_bar.set_yticks([])
ax_bar.set_xlabel("Mean |SHAP|", fontsize=8, labelpad=6, color="black")
ax_bar.tick_params(axis="x", which="major", width=0.5, length=2, labelsize=8, colors="black")

ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.spines["left"].set_linewidth(0.5)
ax_bar.spines["bottom"].set_linewidth(0.5)
ax_bar.spines["left"].set_edgecolor("black")
ax_bar.spines["bottom"].set_edgecolor("black")

# =====================================================
# Force all text/spines to black (safety)
# =====================================================
def force_all_text_black(fig_):
    for ax_ in fig_.axes:
        ax_.title.set_color("black")
        ax_.xaxis.label.set_color("black")
        ax_.yaxis.label.set_color("black")
        for t in ax_.get_xticklabels() + ax_.get_yticklabels():
            t.set_color("black")
        ax_.tick_params(axis="both", colors="black")
        for sp in ax_.spines.values():
            sp.set_edgecolor("black")
        for txt in ax_.texts:
            txt.set_color("black")


force_all_text_black(fig)

# =====================================================
# Save
# =====================================================
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0, transparent=False)
fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0, transparent=False)

plt.close(fig)
print("Saved:", OUT_PDF)
print("Saved:", OUT_SVG)
