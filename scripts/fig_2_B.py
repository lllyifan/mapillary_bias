#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from libpysal.weights import Queen
from esda import Moran, Moran_Rate

# Paths
ROOT = Path(__file__).resolve().parents[1]

SHP_PATH = ROOT / "data" / "LSOA_joined_remodeling_with_centroid.shp"
OUT_DIR = ROOT / "outputs" / "figures" 
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "fig_2_B.pdf"
OUT_SVG = OUT_DIR / "fig_2_B.svg"

# Column names
COL_NUM = "NUM"
COL_POP = "populati_1"
COL_AREA = "area"

# Style
plt.rcParams.update(
    {
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
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)
sns.set_style("white")

cm = 1 / 2.54
figsize = (16 * cm, 6 * cm)

# Load data
gdf = gpd.read_file(SHP_PATH).copy()
gdf = gdf.dropna(subset=[COL_NUM, COL_POP, COL_AREA]).copy()
gdf = gdf[(gdf[COL_POP] > 0) & (gdf[COL_AREA] > 0)].copy()

# Spatial weights
w = Queen.from_dataframe(gdf, use_index=False)
w.transform = "r"

def get_z_wz(values: np.ndarray, w):
    values = np.asarray(values, dtype=float)
    mu = np.nanmean(values)
    sd = np.nanstd(values, ddof=1)
    if (not np.isfinite(sd)) or sd == 0:
        z = np.zeros_like(values, dtype=float)
    else:
        z = (values - mu) / sd
    wz = w.sparse.dot(z)
    return z, wz

# Statistics
y_num = gdf[COL_NUM].astype(float).to_numpy()
m_num = Moran(y_num, w, permutations=999)

events = y_num
expo_pop = gdf[COL_POP].astype(float).to_numpy()
mr_ip = Moran_Rate(events, expo_pop, w, permutations=999)
rate_ip = events / expo_pop

expo_area = gdf[COL_AREA].astype(float).to_numpy()
mr_ia = Moran_Rate(events, expo_area, w, permutations=999)
rate_ia = events / expo_area

panels = [
    {
        "values_for_scatter": y_num,
        "title": f"Moran's I — IC\nI={m_num.I:.4f}, p={m_num.p_sim:.4f}",
    },
    {
        "values_for_scatter": rate_ip,
        "title": f"Moran's I — IP\nI={mr_ip.I:.4f}, p={mr_ip.p_sim:.4f}",
    },
    {
        "values_for_scatter": rate_ia,
        "title": f"Moran's I — IA\nI={mr_ia.I:.4f}, p={mr_ia.p_sim:.4f}",
    },
]

# Plot
fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

for ax, panel in zip(axes, panels):
    vals = np.asarray(panel["values_for_scatter"], dtype=float)
    z, wz = get_z_wz(vals, w)
    df_plot = pd.DataFrame({"z": z, "wz": wz})

    sns.regplot(
        x="z",
        y="wz",
        data=df_plot,
        ax=ax,
        scatter_kws={"s": 10, "alpha": 0.6, "edgecolor": "none"},
        line_kws={"linewidth": 1.2},
        ci=95,
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_title(panel["title"], fontsize=8, color="black")
    ax.set_xlabel("Standardised Value", fontsize=8, color="black")
    ax.set_ylabel("Spatial Lag", fontsize=8, color="black")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.tick_params(width=0.5, length=2, direction="in", colors="black")

    for spine_name, spine in ax.spines.items():
        if spine_name in ["left", "bottom"]:
            spine.set_linewidth(0.5)
            spine.set_edgecolor("black")

# Save
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0)
fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0)
plt.close(fig)

print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_SVG}")
