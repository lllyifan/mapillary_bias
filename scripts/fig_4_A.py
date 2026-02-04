#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "outputs" / "OCI" / "IP" / "OCI_results.csv"
OUTDIR = ROOT / "outputs" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUTDIR / "fig_4_A.pdf"
OUT_SVG = OUTDIR / "fig_4_A.svg"

# Read data
df = pd.read_csv(CSV_PATH)

required = {"Group", "OCI"}
missing = required - set(df.columns)
if missing:
    raise KeyError(f"Missing required columns: {sorted(missing)}")

group_to_abbr = {
    "Sex Female": "FP",
    "Ethnic Nonwhite": "NWP",
    "Education Noqualification": "NQP",
    "Age Elder": "EP",
    "Age Children": "CP",
    "Ns Sec Unemployed": "UEP",
    "Ns Sec Students": "SP",
}

df = df.dropna(subset=["OCI"]).copy()
df["OCI"] = df["OCI"].astype(float)

df["GroupShort"] = df["Group"].map(group_to_abbr).fillna(df["Group"])
df = df.sort_values("OCI", ascending=False).reset_index(drop=True)

COL_HIGH = "#9294BB"
COL_LOW = "#7CC0A2"
df["Color"] = df["OCI"].apply(lambda x: COL_HIGH if x > 1 else COL_LOW)

# Plot
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(8 * cm, 8 * cm))

bars = ax.barh(
    y=df["GroupShort"],
    width=df["OCI"],
    color=df["Color"],
    height=0.6,
    edgecolor="none",
)

for bar, oci in zip(bars, df["OCI"].values):
    ax.text(
        bar.get_width() + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{oci:.2f}",
        va="center",
        ha="left",
        fontsize=8,
        color="black",
    )

# Reference line at OCI = 1.0
ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0, zorder=0)

# X limits with padding
xmin = float(min(df["OCI"].min(), 1.0))
xmax = float(max(df["OCI"].max(), 1.0))
rng = max(xmax - xmin, 0.1)
pad = rng * 0.08
ax.set_xlim(xmin - pad, xmax + pad)

ax.set_xlabel("OCI", fontsize=8, color="black")

ax.tick_params(axis="x", width=0.5, length=2, labelsize=8, colors="black")
ax.tick_params(axis="y", width=0.5, length=2, labelsize=8, colors="black")

for spine in ax.spines.values():
    spine.set_linewidth(0.5)
    spine.set_edgecolor("black")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.invert_yaxis()

plt.tight_layout()

# Save
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0)
fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0)

plt.show()
print("Saved:", OUT_PDF)
print("Saved:", OUT_SVG)
