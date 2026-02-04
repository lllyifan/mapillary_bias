#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import mapclassify
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


# Paths
ROOT = Path(__file__).resolve().parents[1]

SHP_PATH = ROOT / "data" / "LSOA_joined_remodeling.shp"
BOUNDARY_PATH = ROOT / "data" / "Regions_December_2021_EN_BFC_2022_-2718764585271217414" / "regions_wales_england.geojson"
CITY_PATH = ROOT / "data" / "UACC_Dec_2016_FEB_in_the_United_Kingdom_2022_4012563770868598386" / "cities9_point.shp"
CITY_NAME_FIELD = "uacc16nm"

OUTDIR = ROOT / "outputs" / "figures" 
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUTDIR / "fig_2_A.pdf"
OUT_SVG = OUTDIR / "fig_2_A.svg"

# Settings (default Matplotlib font; no external fonts)
RENAME_CITY = {
    "Bristol, City of": "Bristol",
    "City of London": "London",
}

VARIABLES = [
    ("NUMPOINTS", "IC"),
    ("rate", "IP"),
    ("IA", "IA"),
]

CUSTOM_COLORS = ["#7CC0A2", "#DAEBD0", "#D5EDF9", "#9294BB"]
CMAP = ListedColormap(CUSTOM_COLORS)

BOUNDARY_COLOR = "#7A7A7A"
LABEL_DX_BASE, LABEL_DY_BASE = 100, 100

OFFSETS = {
    "North East": (8000, -30000),
    "North West": (-10000, 8000),
    "Yorkshire and The Humber": (8000, -6000),
    "East Midlands": (6000, -4000),
    "West Midlands": (-6000, -20000),
    "East of England": (12000, 2000),
    "London": (8000, 25000),
    "South East": (25000, -35000),
    "South West": (-25000, -6000),
    "Wales": (-8000, -18000),
}

# Load data
gdf = gpd.read_file(SHP_PATH, encoding="ISO-8859-1").to_crs(epsg=27700)
gdf_boundary = gpd.read_file(BOUNDARY_PATH).to_crs(epsg=27700)
gdf_city = gpd.read_file(CITY_PATH).to_crs(epsg=27700)

labels_gdf = gdf_boundary.dissolve(by="RGN21NM", as_index=False)[["RGN21NM", "geometry"]]

gdf_city["label_name"] = gdf_city[CITY_NAME_FIELD].astype(str).replace(RENAME_CITY)
gdf_city = gdf_city.drop_duplicates(subset=["label_name"]).copy()

manual_city_offsets = {n: (0, 0) for n in gdf_city["label_name"]}
manual_city_offsets.update(
    {
        "Newcastle upon Tyne": (-40000, 8000),
        "Manchester": (-30000, 8000),
        "Liverpool": (-30000, -20000),
        "Sheffield": (8000, -6000),
        "Nottingham": (-30000, 9000),
        "Birmingham": (-30000, 8000),
        "Cardiff": (-20000, 9000),
        "Bristol": (8000, -6000),
    }
)

lon_pt = gdf_city.loc[gdf_city["label_name"] == "London"].geometry.iloc[0]
cx, cy = lon_pt.x, lon_pt.y
LONDON_BBOX = (cx - 35000, cy - 30000, cx + 35000, cy + 30000)

# Layout
fig, axes = plt.subplots(
    3,
    3,
    figsize=(16 / 2.54, 12 / 2.54),
    gridspec_kw={"height_ratios": [1.0, 0.20, 0.14]},
)
fig.subplots_adjust(
    left=0.02,
    right=0.98,
    top=0.98,
    bottom=0.04,
    wspace=0.02,
    hspace=0.00,
)

# Plot
for i, (value_col, legend_title) in enumerate(VARIABLES):
    ax, axins, axleg = axes[0, i], axes[1, i], axes[2, i]
    axleg.axis("off")
    axleg.set_xlim(0, 1)
    axleg.set_ylim(0, 1)

    scheme = mapclassify.Quantiles(gdf[value_col], k=4)
    bins = scheme.bins.tolist()

    legend_labels = []
    for j in range(4):
        lo = bins[j - 1] if j > 0 else gdf[value_col].min()
        hi = bins[j]
        legend_labels.append(f"{lo:.2f} – {hi:.2f}")

    gdf.plot(
        column=value_col,
        cmap=CMAP,
        scheme="UserDefined",
        classification_kwds={"bins": bins},
        linewidth=0,
        edgecolor="none",
        ax=ax,
    )

    gdf_boundary.boundary.plot(ax=ax, linewidth=0.2, color=BOUNDARY_COLOR)
    gdf_city.plot(ax=ax, markersize=4, color="red", zorder=3)

    for _, r in gdf_city.iterrows():
        if r["label_name"] == "London":
            continue
        x, y = r.geometry.x, r.geometry.y
        dx, dy = manual_city_offsets.get(r["label_name"], (0, 0))
        ax.text(x + LABEL_DX_BASE + dx, y + LABEL_DY_BASE + dy, r["label_name"], fontsize=4)

    for _, r in labels_gdf.iterrows():
        x, y = r.geometry.representative_point().coords[0]
        dx, dy = OFFSETS.get(r["RGN21NM"], (0, 0))
        ax.text(
            x + dx,
            y + dy,
            r["RGN21NM"],
            fontsize=5,
            ha="center",
            va="center",
            fontweight="bold",
        )

    ax.axis("off")

    xmin, ymin, xmax, ymax = LONDON_BBOX
    gdf.cx[xmin:xmax, ymin:ymax].plot(
        column=value_col,
        cmap=CMAP,
        scheme="UserDefined",
        classification_kwds={"bins": bins},
        linewidth=0,
        edgecolor="none",
        ax=axins,
    )
    gdf_boundary.cx[xmin:xmax, ymin:ymax].boundary.plot(ax=axins, lw=0.35, color=BOUNDARY_COLOR)
    gdf_city.cx[xmin:xmax, ymin:ymax].plot(ax=axins, color="red", markersize=10, zorder=3)

    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymin, ymax)
    axins.set_aspect("equal", adjustable="box")
    axins.set_xticks([])
    axins.set_yticks([])
    axins.text(
        0.02,
        0.98,
        "London",
        transform=axins.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        fontweight="bold",
    )

    xs = np.linspace(0.25, 0.85, len(CUSTOM_COLORS))
    for c, lab, x in zip(CUSTOM_COLORS, legend_labels, xs):
        axleg.add_patch(Rectangle((x, 0.55), 0.1, 0.1, transform=axleg.transAxes, color=c))
        axleg.text(
            x + 0.05,
            0.2,
            lab,
            ha="center",
            va="top",
            transform=axleg.transAxes,
            fontsize=8,
            rotation=45,
        )
    axleg.text(
        xs[0] - 0.15,
        0.55,
        legend_title,
        ha="left",
        va="center",
        transform=axleg.transAxes,
        fontsize=8,
    )

# Position adjustments
MOVE_UP_INSET = 0.090
MOVE_UP_LEG = 0.070
for i in range(3):
    for ax_, move in [(axes[1, i], MOVE_UP_INSET), (axes[2, i], MOVE_UP_LEG)]:
        p = ax_.get_position()
        ax_.set_position([p.x0, p.y0 + move, p.width, p.height])

# Save
fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_SVG, bbox_inches="tight")
plt.show()
