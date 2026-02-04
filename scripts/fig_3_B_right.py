#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# =====================================================
# Paths (relative to repository root)
# =====================================================
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
OUTDIR = ROOT / "outputs" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

SHP_PATH = DATA_DIR / "LSOA_joined_remodeling_noShapeArea.shp"
COL_TXT_PATH = DATA_DIR / "satscan_input_IA" / "IA.col.txt"
BOUNDARY_PATH = DATA_DIR / "Regions_December_2021_EN_BFC_2022_-2718764585271217414" / "regions_wales_england.geojson"

OUT_PDF = OUTDIR / "fig_3_B_right.pdf"
OUT_SVG = OUTDIR / "fig_3_B_right.svg"

# =====================================================
# Read data (kept for consistency with your workflow)
# =====================================================
_ = gpd.read_file(SHP_PATH)
_ = gpd.read_file(BOUNDARY_PATH)

# =====================================================
# Parse SaTScan .col.txt
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

# =====================================================
# Build table data (High / Low)
# =====================================================
df_cluster["ID"] = df_cluster["ClusterID"].astype(int)
df_cluster["RR"] = df_cluster["RelativeRisk"].map(lambda x: f"{x:.2f}")

table_data = df_cluster[["ID", "RR"]].copy()
table_high = table_data[df_cluster["RelativeRisk"] > 1].reset_index(drop=True)
table_low = table_data[df_cluster["RelativeRisk"] <= 1].reset_index(drop=True)

max_rows = max(len(table_high), len(table_low))

def pad_table(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    missing = n_rows - len(df)
    if missing <= 0:
        return df
    padding = pd.DataFrame([[""] * df.shape[1]] * missing, columns=df.columns)
    return pd.concat([df, padding], ignore_index=True)

table_high = pad_table(table_high, max_rows)
table_low = pad_table(table_low, max_rows)

# =====================================================
# Plot two tables side-by-side
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(6 / 2.54, 12 / 2.54))

titles = ["High-IA Clusters", "Low-IA Clusters"]
tables = [table_high, table_low]

for ax, df, title in zip(axes, tables, titles):
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.5)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0)
        if row == 0:
            cell.set_facecolor("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f0f0f0" if row % 2 == 0 else "white")

    n_rows = len(df)
    ax.plot([0, 1], [1, 1], color="black", linewidth=0.6, transform=ax.transAxes)
    ax.plot(
        [0, 1],
        [1 - 1 / (n_rows + 1), 1 - 1 / (n_rows + 1)],
        color="black",
        linewidth=0.6,
        transform=ax.transAxes,
    )
    ax.plot([0, 1], [0, 0], color="black", linewidth=0.6, transform=ax.transAxes)

    ax.set_title(title, fontsize=8, pad=8)

plt.tight_layout()

# =====================================================
# Save
# =====================================================
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0)
fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0)

plt.show()
