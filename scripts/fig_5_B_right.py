#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parents[1]

SHAP_PATH = ROOT / "outputs" / "SHAP" / "IA" / "shap_values_low.npy"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "fig_5_B_right.pdf"
OUT_SVG = OUT_DIR / "fig_5_B_right.svg"

# Matplotlib style
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
        "font.size": 7,
        "axes.titlesize": 7,
    }
)

# Feature short names & colors
feature_map = {
    "indicator_sex_female": "FP",
    "indicator_Ethnic_nonwhite": "NWP",
    "indicator_age_elder": "EP",
    "indicator_age_children": "CP",
    "indicator_education_noqualification": "NQP",
    "indicator_NS-SeC_unemployed": "UEP",
    "indicator_NS-SeC_students": "SP",
}
short_labels = list(feature_map.values())

color_list = ["#1c1658", "#157673", "#4da6a6", "#82a6cb", "#692168", "#805da4", "#d6c0e0"]

# Load data
shap_vals = np.load(SHAP_PATH)

abs_sums = np.abs(shap_vals).sum(axis=0)
sizes = abs_sums.astype(float).tolist()
total = float(np.sum(abs_sums)) if len(sizes) else 1.0

# Plot donut
fig, ax = plt.subplots(figsize=(6 / 2.54, 7 / 2.54))

wedges, _ = ax.pie(
    sizes,
    labels=None,
    colors=color_list,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=0.6),
)

centre = plt.Circle((0, 0), 0.6, color="white")
ax.add_artist(centre)

ax.set_title("Total |SHAP| Distribution", fontsize=7, pad=1, color="black")
ax.axis("equal")

# Outside labels with simple collision avoidance
R_TEXT = 1.28
R_ARROW = 1.02
MIN_DY = 0.10
X_PAD = 0.08
N_ITER = 2

items = []
for i, w in enumerate(wedges):
    ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
    x = np.cos(ang)
    y = np.sin(ang)

    side = "right" if x >= 0 else "left"
    pct = (sizes[i] / total) * 100.0
    tag = short_labels[i] if i < len(short_labels) else ""

    text = f"{tag}\n{pct:.1f}% ({sizes[i]:.2f})"

    items.append({"i": i, "ang": ang, "x": x, "y": y, "side": side, "text": text})


def repel_side(items_side):
    if not items_side:
        return

    for it in items_side:
        it["tx"] = (R_TEXT * np.cos(it["ang"])) + (X_PAD if it["side"] == "right" else -X_PAD)
        it["ty"] = (R_TEXT * np.sin(it["ang"]))

    for _ in range(N_ITER):
        items_side.sort(key=lambda d: d["ty"])
        for k in range(1, len(items_side)):
            prev = items_side[k - 1]
            curr = items_side[k]
            if curr["ty"] - prev["ty"] < MIN_DY:
                curr["ty"] = prev["ty"] + MIN_DY


left_items = [it for it in items if it["side"] == "left"]
right_items = [it for it in items if it["side"] == "right"]
repel_side(left_items)
repel_side(right_items)

for it in items:
    if it["side"] == "left":
        it2 = next(d for d in left_items if d["i"] == it["i"])
        ha = "right"
    else:
        it2 = next(d for d in right_items if d["i"] == it["i"])
        ha = "left"

    tx, ty = it2["tx"], it2["ty"]
    x0 = R_ARROW * it["x"]
    y0 = R_ARROW * it["y"]

    ax.annotate(
        it["text"],
        xy=(x0, y0),
        xytext=(tx, ty),
        ha=ha,
        va="center",
        fontsize=7,
        color="black",
        arrowprops=dict(
            arrowstyle="-",
            color="#4D4D4D",
            lw=0.6,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="angle3,angleA=0,angleB=90",
        ),
    )

ax.set_xlim(-1.55, 1.55)
ax.set_ylim(-1.45, 1.45)

# Save
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0)
fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0)

plt.show()
