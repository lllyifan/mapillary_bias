#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point
from shapely import affinity
import numpy as np

# =====================================================
# Paths (relative to repository root)
# =====================================================
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
OUTDIR = ROOT / "outputs" / "figures" 
OUTDIR.mkdir(parents=True, exist_ok=True)

SHP_PATH = DATA_DIR / "LSOA_joined_remodeling_noShapeArea.shp"
COL_TXT_PATH = DATA_DIR / "satscan_input_IP" / "IP.col.txt"
BOUNDARY_PATH = DATA_DIR / "Regions_December_2021_EN_BFC_2022_-2718764585271217414" / "regions_wales_england.geojson"

OUT_PDF = OUTDIR / "fig_3_A_left.pdf"
OUT_SVG = OUTDIR / "fig_3_A_left.svg"
OUT_TIF = OUTDIR / "fig_3_A_left.tiff"

# =====================================================
# Settings
# =====================================================
FIGSIZE = (10 / 2.54, 12 / 2.54)

LSOA_CODE_COL = "LSOA21CD"
FOCUS_IDS = [9, 33, 32, 20]
INSET_PAD_FRAC = 0.10

BOUNDARY_COLOR = "#4D4D4D"
COL_HIGH = "#9294BB"
COL_LOW = "#7CC0A2"

FS_MIN_MAIN, FS_MAX_MAIN = 5.0, 8.0
INSET_LABEL_FONTSIZE = 5

INSET_SKIP_IF_TOO_CLOSE = False
INSET_MIN_DIST_M = 6000

LABEL_MIN_DIST_M = 0

# =====================================================
# Read data
# =====================================================
gdf = gpd.read_file(SHP_PATH)
gdf_region = gpd.read_file(BOUNDARY_PATH)

if LSOA_CODE_COL not in gdf.columns:
    raise KeyError(f"Column '{LSOA_CODE_COL}' not found in {SHP_PATH.name}")

# =====================================================
# Parse SaTScan clusters (.col.txt)
# =====================================================
with open(COL_TXT_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

cluster_data = []
for line in lines:
    parts = re.split(r"\s{2,}", line.strip())
    if len(parts) < 15:
        continue
    cluster_data.append(
        {
            "ClusterID": int(parts[0]),
            "X": float(parts[2]),
            "Y": float(parts[3]),
            "Radius": float(parts[4]),
            "RelativeRisk": float(parts[14]),
        }
    )

df_cluster = pd.DataFrame(cluster_data)
if df_cluster.empty:
    raise ValueError("Parsed cluster table is empty; check IA.col.txt and parsing rules.")

gdf_cent = gpd.GeoDataFrame(
    df_cluster.copy(),
    geometry=[Point(xy) for xy in zip(df_cluster["X"], df_cluster["Y"])],
    crs=gdf.crs,
)

# =====================================================
# Radius=0 handling: attach nearest LSOA and equivalent-area radius
# =====================================================
if gdf.crs != gdf_cent.crs:
    gdf = gdf.to_crs(gdf_cent.crs)

g0 = gdf_cent[gdf_cent["Radius"] <= 0].copy()

if not g0.empty:
    gdf_lsoa = gdf[[LSOA_CODE_COL, "geometry"]].copy()
    gdf_lsoa["area_m2"] = gdf_lsoa.geometry.area

    g0_hit = gpd.sjoin_nearest(
        g0,
        gdf_lsoa[[LSOA_CODE_COL, "area_m2", "geometry"]],
        how="left",
        distance_col="dist_m",
    )
    g0_hit["equiv_radius_m"] = np.sqrt(g0_hit["area_m2"].values / np.pi)

    df_cluster = df_cluster.merge(
        g0_hit[["ClusterID", LSOA_CODE_COL, "area_m2", "equiv_radius_m", "dist_m"]],
        on="ClusterID",
        how="left",
    )
else:
    df_cluster[LSOA_CODE_COL] = np.nan
    df_cluster["area_m2"] = np.nan
    df_cluster["equiv_radius_m"] = np.nan
    df_cluster["dist_m"] = np.nan

df_cluster["Radius_vis"] = df_cluster["Radius"].astype(float).copy()
mask0 = df_cluster["Radius_vis"] <= 0
df_cluster.loc[mask0, "Radius_vis"] = df_cluster.loc[mask0, "equiv_radius_m"]
df_cluster["Radius_vis"] = df_cluster["Radius_vis"].fillna(2000.0)

# =====================================================
# Build circle polygons using Radius_vis
# =====================================================
def create_circle(x: float, y: float, radius: float, resolution: int = 100):
    return affinity.scale(Point(x, y).buffer(1, resolution=resolution), radius, radius)

df_cluster["geometry"] = df_cluster.apply(
    lambda r: create_circle(float(r["X"]), float(r["Y"]), float(r["Radius_vis"])),
    axis=1,
)
gdf_cluster = gpd.GeoDataFrame(df_cluster, geometry="geometry", crs=gdf.crs)

# =====================================================
# Map circle size to label font size (quantile + log)
# =====================================================
def fontsize_from_radius(radius_array, fs_min, fs_max, qlo=0.05, qhi=0.95):
    r = np.asarray(radius_array, dtype=float)
    if r.size == 0:
        return np.array([], dtype=float)
    lo = np.quantile(r, qlo)
    hi = np.quantile(r, qhi)
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        return np.full_like(r, (fs_min + fs_max) / 2.0, dtype=float)
    r_clip = np.clip(r, lo, hi)
    x = np.log1p(r_clip)
    x0, x1 = np.log1p(lo), np.log1p(hi)
    fs = fs_min + (x - x0) * (fs_max - fs_min) / (x1 - x0)
    return np.clip(fs, fs_min, fs_max)

df_cluster["fs_main"] = fontsize_from_radius(df_cluster["Radius_vis"].values, FS_MIN_MAIN, FS_MAX_MAIN)

# =====================================================
# Main map
# =====================================================
fig = plt.figure(figsize=FIGSIZE)
main_ax = fig.add_axes([0.02, 0.02, 0.95, 0.95])
main_ax.set_facecolor("#f5f5f5")
fig.patch.set_facecolor("#ffffff")

gdf.plot(ax=main_ax, facecolor="#e2e2e2", edgecolor="white", linewidth=0.1)
gdf_region.boundary.plot(ax=main_ax, color=BOUNDARY_COLOR, linewidth=0.25)

gdf_cluster[df_cluster["RelativeRisk"] > 1].plot(
    ax=main_ax, color=COL_HIGH, alpha=0.4, edgecolor=BOUNDARY_COLOR, linewidth=0.25, zorder=2
)
gdf_cluster[df_cluster["RelativeRisk"] <= 1].plot(
    ax=main_ax, color=COL_LOW, alpha=0.4, edgecolor=BOUNDARY_COLOR, linewidth=0.25, zorder=2
)

main_ax.set_aspect("equal")
main_ax.axis("off")

# =====================================================
# Main labels (optional distance pruning)
# =====================================================
df_lab = df_cluster[["ClusterID", "X", "Y", "Radius_vis", "fs_main"]].copy()
df_lab = df_lab.sort_values(["Radius_vis", "ClusterID"], ascending=[False, True]).reset_index(drop=True)

placed_xy = []
min_dist2 = float(LABEL_MIN_DIST_M) ** 2 if (LABEL_MIN_DIST_M and LABEL_MIN_DIST_M > 0) else None

kept_main = 0
for _, r in df_lab.iterrows():
    cid = int(r["ClusterID"])
    x, y = float(r["X"]), float(r["Y"])

    if min_dist2 is not None:
        ok = True
        for (px, py) in placed_xy:
            dx = x - px
            dy = y - py
            if dx * dx + dy * dy < min_dist2:
                ok = False
                break
        if not ok:
            continue

    main_ax.text(
        x,
        y,
        str(cid),
        fontsize=float(r["fs_main"]),
        ha="center",
        va="center",
        color="black",
        zorder=10,
        clip_on=False,
    )
    placed_xy.append((x, y))
    kept_main += 1

# =====================================================
# Inset (focus bbox defined by FOCUS_IDS)
# =====================================================
g_focus = gdf_cluster[gdf_cluster["ClusterID"].isin(FOCUS_IDS)].copy()
if g_focus.empty:
    raise ValueError(f"FOCUS_IDS not found in clusters: {FOCUS_IDS}")

fxmin, fymin, fxmax, fymax = g_focus.total_bounds
fdx = fxmax - fxmin
fdy = fymax - fymin
pad_x = INSET_PAD_FRAC * fdx if fdx > 0 else 20000
pad_y = INSET_PAD_FRAC * fdy if fdy > 0 else 20000

ixmin, ixmax = fxmin - pad_x, fxmax + pad_x
iymin, iymax = fymin - pad_y, fymax + pad_y

inset_ax = fig.add_axes([0.05, 0.62, 0.32, 0.32])
inset_ax.set_facecolor("#f5f5f5")

gdf_inset = gdf.cx[ixmin:ixmax, iymin:iymax]
reg_inset = gdf_region.cx[ixmin:ixmax, iymin:iymax]
clu_inset = gdf_cluster.cx[ixmin:ixmax, iymin:iymax]

gdf_inset.plot(ax=inset_ax, facecolor="#e2e2e2", edgecolor="white", linewidth=0.1)
reg_inset.boundary.plot(ax=inset_ax, color=BOUNDARY_COLOR, linewidth=0.25)

clu_inset[clu_inset["RelativeRisk"] > 1].plot(
    ax=inset_ax, color=COL_HIGH, alpha=0.4, edgecolor=BOUNDARY_COLOR, linewidth=0.25, zorder=2
)
clu_inset[clu_inset["RelativeRisk"] <= 1].plot(
    ax=inset_ax, color=COL_LOW, alpha=0.4, edgecolor=BOUNDARY_COLOR, linewidth=0.25, zorder=2
)

inset_ax.set_xlim(ixmin, ixmax)
inset_ax.set_ylim(iymin, iymax)
inset_ax.set_aspect("equal")
inset_ax.set_xticks([])
inset_ax.set_yticks([])

clu_inset = clu_inset.sort_values(["Radius_vis", "ClusterID"], ascending=[False, True])

placed_xy_in = []
min_dist2_in = float(INSET_MIN_DIST_M) ** 2 if (INSET_SKIP_IF_TOO_CLOSE and INSET_MIN_DIST_M > 0) else None
kept_inset = 0

for _, r in clu_inset.iterrows():
    cid = int(r["ClusterID"])
    x, y = float(r["X"]), float(r["Y"])

    if min_dist2_in is not None:
        ok = True
        for (px, py) in placed_xy_in:
            dx = x - px
            dy = y - py
            if dx * dx + dy * dy < min_dist2_in:
                ok = False
                break
        if not ok:
            continue

    inset_ax.text(
        x,
        y,
        str(cid),
        fontsize=INSET_LABEL_FONTSIZE,
        ha="center",
        va="center",
        color="black",
        zorder=10,
        clip_on=True,
    )
    placed_xy_in.append((x, y))
    kept_inset += 1

# =====================================================
# Scale bar (main map)
# =====================================================
x0, x1 = main_ax.get_xlim()
y0, y1 = main_ax.get_ylim()

scale_start = x1 - 180000
scale_end = x1 - 80000
scale_y = y0 + 2000

main_ax.plot([scale_start, scale_end], [scale_y, scale_y], color="black", lw=1, zorder=20)
main_ax.plot([scale_start, scale_start], [scale_y - 100, scale_y + 10000], color="black", lw=1, zorder=20)
main_ax.plot([scale_end, scale_end], [scale_y - 100, scale_y + 10000], color="black", lw=1, zorder=20)
main_ax.text(
    (scale_start + scale_end) / 2,
    scale_y + 10000,
    "100 km",
    ha="center",
    va="bottom",
    fontsize=8,
    zorder=20,
)

# =====================================================
# Legend
# =====================================================
patch_high = mpatches.Patch(color=COL_HIGH, label="High-IA Clusters (RR > 1)", alpha=0.4)
patch_low = mpatches.Patch(color=COL_LOW, label="Low-IA Clusters (RR ≤ 1)", alpha=0.4)

main_ax.legend(
    handles=[patch_high, patch_low],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    frameon=False,
    fontsize=8,
    handlelength=1.5,
    columnspacing=1.5,
)

# =====================================================
# Save outputs
# =====================================================
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0)
fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0)
fig.savefig(OUT_TIF, dpi=600, format="tiff")

print("---- radius=0 clusters (LSOA + area + equiv radius) ----")
print(
    df_cluster[df_cluster["Radius"] <= 0][
        ["ClusterID", "RelativeRisk", LSOA_CODE_COL, "area_m2", "equiv_radius_m", "dist_m"]
    ].sort_values("ClusterID")
)
print(f"[LABELS] main_kept={kept_main}/{len(df_lab)} | inset_kept={kept_inset}/{len(clu_inset)}")

plt.show()
