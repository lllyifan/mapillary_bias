#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Requirements (same as just now):
# - NO proprietary font (use default font)
# - Force all-black text/axes (avoid grey)
# - Repo-friendly paths (relative)
# - Export PDF + SVG
# - Keep your original colours + TwoSlopeNorm centered at 0
# - Keep per-panel symmetric colour limits (±max|slope|)
# - Keep north arrow + scalebar styling

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.colors import LinearSegmentedColormap

# =========================================================
# 0) Paths & parameters (relative)
# =========================================================
ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_PATH = (
    ROOT / "data"
    / "Regions_December_2021_EN_BFC_2022_-2718764585271217414"
    / "regions_wales_england.geojson"
)

DATA_BASE = ROOT / "outputs" / "ICE" / "IP"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

label = "low"  # or "high"

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

    # force all-black
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",

    # sizes
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# =========================================================
# 2) Colormap (keep your original palette)
# =========================================================
custom_colors = ["#7CC0A2", "#DAEBD0", "white", "#D5EDF9", "#9294BB"]
cmap_cont = LinearSegmentedColormap.from_list("custom_sym", custom_colors, N=256)

# =========================================================
# 3) Features & labels
# =========================================================
feature_cols = [
    "indicator_sex_female",
    "indicator_Ethnic_nonwhite",
    "indicator_education_noqualification",
    "indicator_age_elder",
    "indicator_age_children",
    "indicator_NS-SeC_unemployed",
    "indicator_NS-SeC_students",
]
rename_dict = {
    "indicator_sex_female": "FP",
    "indicator_Ethnic_nonwhite": "NWP",
    "indicator_education_noqualification": "NQP",
    "indicator_age_elder": "EP",
    "indicator_age_children": "CP",
    "indicator_NS-SeC_unemployed": "UEP",
    "indicator_NS-SeC_students": "SP",
}

# =========================================================
# 4) Basemap
# =========================================================
region = gpd.read_file(BOUNDARY_PATH)
if region.crs is None:
    region = region.set_crs(epsg=27700)

# =========================================================
# 5) North arrow + scalebar
# =========================================================
def draw_north_arrow_and_scalebar(ax):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # North arrow (cross)
    cx = x1 - (x1 - x0) * 0.05
    cy = y1 - (y1 - y0) * 0.05
    ax.plot([cx, cx], [cy - 10000, cy + 10000], color="black", lw=1.0, zorder=10)
    ax.plot([cx - 10000, cx + 10000], [cy, cy], color="black", lw=1.0, zorder=10)
    ax.text(cx, cy + 15000, "N", ha="center", va="bottom", fontsize=8, color="black", zorder=11)

    # Scalebar (100 km) — keep your original numbers
    scale_start = x1 - 120000
    scale_end = x1 - 20000
    scale_y = y0 + 2000
    ax.plot([scale_start, scale_end], [scale_y, scale_y], color="black", lw=0.5, zorder=10)
    ax.plot([scale_start, scale_start], [scale_y - 100, scale_y + 10000], color="black", lw=0.5, zorder=10)
    ax.plot([scale_end, scale_end], [scale_y - 100, scale_y + 10000], color="black", lw=0.5, zorder=10)
    ax.text(
        (scale_start + scale_end) / 2,
        scale_y + 10000,
        "100 km",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        zorder=11,
    )

# =========================================================
# 6) Layout
# =========================================================
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16 / 2.54, 14 / 2.54))
axes = axes.flatten()
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# =========================================================
# 7) Main loop
# =========================================================
for i, feature in enumerate(feature_cols):
    ax = axes[i]

    p = DATA_BASE / label / feature / f"slopes_{feature}_{label}.csv"
    short = rename_dict.get(feature, feature)

    if not p.exists():
        ax.axis("off")
        ax.set_title(f"{short} (missing)", fontsize=8, color="black")
        continue

    df = pd.read_csv(p)

    # Points: WGS84 -> region CRS
    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["LONG"], df["LAT"]),
        crs="EPSG:4326",
    ).to_crs(region.crs)

    # Symmetric per-panel colour range around 0
    m = float(np.nanmax(np.abs(gdf_pts["slope"].to_numpy(dtype=float))))
    if (not np.isfinite(m)) or (m == 0):
        vmin, vmax = -1e-9, 1e-9
    else:
        vmin, vmax = -m, m
    norm = mpl_colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # Plot points + boundary
    gdf_pts.plot(
        ax=ax,
        column="slope",
        cmap=cmap_cont,
        norm=norm,
        markersize=1.2,
        alpha=0.95,
        legend=False,
        zorder=2,
    )
    region.boundary.plot(ax=ax, linewidth=0.15, edgecolor="black", zorder=6)

    ax.set_title(short, fontsize=8, color="black")
    ax.axis("off")

    # North arrow + scalebar
    draw_north_arrow_and_scalebar(ax)

    # Horizontal colourbar inset
    cax = ax.inset_axes([0.18, -0.11, 0.64, 0.035], transform=ax.transAxes)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_cont)
    sm.set_array([])

    ticks = np.linspace(vmin, vmax, 5)
    if 0.0 not in np.round(ticks, 12):
        ticks = np.sort(np.unique(np.append(ticks, 0.0)))

    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks, format="%.2f")
    cbar.ax.tick_params(labelsize=7, width=0.5, length=2, colors="black")
    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(7)
        t.set_rotation(45)
        t.set_ha("right")
        t.set_color("black")

    # colourbar frame
    for spine in cax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("black")

# Remove unused last axis (2×4 but only 7 panels)
for j in range(len(feature_cols), len(axes)):
    fig.delaxes(axes[j])

# =========================================================
# 8) Save
# =========================================================
out_pdf = OUT_DIR / f"fig_8_IP_ICE_spatial_slopes_{label}.pdf"
out_svg = OUT_DIR / f"fig_8_IP_ICE_spatial_slopes_{label}.svg"

fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0)
fig.savefig(out_svg, bbox_inches="tight", pad_inches=0)

plt.close(fig)
print("✅ Saved:", out_pdf)
print("✅ Saved:", out_svg)
