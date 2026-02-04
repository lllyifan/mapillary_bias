#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import ScaledTranslation

# =====================================================
# 0) Paths (relative to repository root)
# =====================================================
ROOT = Path(__file__).resolve().parents[1]

SHAP_VALUE_PATH = ROOT / "outputs" / "SHAP" / "IP" / "shap_values_low.npy"
X_VAL_PATH      = ROOT / "outputs" / "SHAP" / "IP" / "shap_X_val.csv"
LOWESS_DIR      = ROOT / "outputs" / "SHAP" / "IP" / "lowess_with_ci"
OUTPUT_DIR      = ROOT / "outputs" / "figures"

LOWESS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Outputs
OUT_PDF = OUTPUT_DIR / "fig_6_A.pdf"
OUT_SVG = OUTPUT_DIR / "fig_6_A.svg"

# =====================================================
# 1) Matplotlib style (default font + all black)
# =====================================================
mpl.rcParams.update(
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
    }
)

# =====================================================
# 2) Feature mapping
# =====================================================
feature_name_map = {
    "indicator_sex_female": "FP",
    "indicator_Ethnic_nonwhite": "NWP",
    "indicator_education_noqualification": "NQP",
    "indicator_age_elder": "EP",
    "indicator_age_children": "CP",
    "indicator_NS-SeC_unemployed": "UEP",
    "indicator_NS-SeC_students": "SP",
}
ordered_features = list(feature_name_map.keys())

# =====================================================
# 3) Lowess fit + bootstrap CI (save intermediates)
# =====================================================
N_BOOT = 100
FRAC = 0.3
N_XI = 200
RNG = np.random.default_rng(42)

X_val = pd.read_csv(X_VAL_PATH)
shap_values_low = np.load(SHAP_VALUE_PATH)  # (n_samples, n_features)

# Ensure column order matches SHAP columns (assumes same order as in X file)
feature_cols = X_val.columns.tolist()

# Save CI per feature
for idx, feat in enumerate(feature_cols):
    x = X_val[feat].to_numpy(dtype=float)
    y = shap_values_low[:, idx].astype(float)

    # Guard: constant x
    if np.nanmin(x) == np.nanmax(x):
        xi = np.linspace(np.nanmin(x), np.nanmax(x) + 1e-9, N_XI)
    else:
        xi = np.linspace(np.nanmin(x), np.nanmax(x), N_XI)

    # Lowess on full data
    fitted = lowess(y, x, frac=FRAC, return_sorted=True)
    x_fit, y_fit = fitted[:, 0], fitted[:, 1]

    # Bootstrap CI on xi grid
    y_boot = np.zeros((N_BOOT, len(xi)), dtype=float)
    n = len(x)

    for b in range(N_BOOT):
        idxs = RNG.choice(n, size=n, replace=True)
        xb = x[idxs]
        yb = y[idxs]
        fb = lowess(yb, xb, frac=FRAC, return_sorted=True)

        # np.interp needs ascending x; lowess returns sorted by x, but may have ties
        xfb = fb[:, 0]
        yfb = fb[:, 1]
        # If all xfb equal (degenerate), fallback constant
        if np.nanmin(xfb) == np.nanmax(xfb):
            y_boot[b] = np.full_like(xi, float(np.nanmean(yfb)))
        else:
            y_boot[b] = np.interp(xi, xfb, yfb)

    lower = np.percentile(y_boot, 2.5, axis=0)
    upper = np.percentile(y_boot, 97.5, axis=0)

    np.savez(
        LOWESS_DIR / f"lowess_{feat}_ci.npz",
        xi=xi,
        x_fit=x_fit,
        y_fit=y_fit,
        lower=lower,
        upper=upper,
    )

# =====================================================
# 4) Helper: gradient background
# =====================================================
purp_rgb = (0x92 / 255, 0x94 / 255, 0xBB / 255)
green_rgb = (0x7C / 255, 0xC0 / 255, 0xA2 / 255)

purp_cmap = LinearSegmentedColormap.from_list("purp_grad", [(*purp_rgb, 0.0), (*purp_rgb, 0.5)])
green_cmap = LinearSegmentedColormap.from_list("green_grad", [(*green_rgb, 0.0), (*green_rgb, 0.5)])


def draw_gradient(ax, x_start, x_end, cmap):
    y0, y1 = ax.get_ylim()
    if not (np.isfinite(x_start) and np.isfinite(x_end)) or (x_end <= x_start):
        return
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(
        grad,
        extent=(x_start, x_end, y0, y1),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        zorder=0,
    )


# =====================================================
# 5) Helper: place top labels with offset + rotation (avoid overlap)
# =====================================================
def place_top_labels_avoid_overlap(
    ax,
    xs,
    fmt="{:.2f}",
    y_axes=1.10,
    step_pts=3,
    max_k=10,
    fontsize=8,
    color="black",
    zorder=5,
    rotation=45,
):
    """
    Place top labels at y=y_axes in axis coordinates.
    If overlaps, shift left/right in points until no overlap.
    Rotation is applied (default 45 degrees).
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    base_tr = ax.get_xaxis_transform()

    def offset_candidates():
        yield 0
        for k in range(1, max_k + 1):
            yield +k * step_pts
            yield -k * step_pts

    placed_bboxes = []
    texts = []

    xs_sorted = np.array(xs, dtype=float)
    xs_sorted = xs_sorted[np.isfinite(xs_sorted)]
    xs_sorted.sort()

    for x0 in xs_sorted:
        label = fmt.format(float(x0))
        chosen_text = None
        chosen_bbox = None

        for dx_pts in offset_candidates():
            dx_in = dx_pts / 72.0
            offset_tr = base_tr + ScaledTranslation(dx_in, 0, fig.dpi_scale_trans)

            t = ax.text(
                float(x0),
                y_axes,
                label,
                transform=offset_tr,
                ha="center",
                va="bottom",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=fontsize,
                color=color,
                clip_on=False,
                zorder=zorder,
            )

            fig.canvas.draw()
            bb = t.get_window_extent(renderer=renderer)

            if not any(bb.overlaps(prev) for prev in placed_bboxes):
                chosen_text = t
                chosen_bbox = bb
                break

            t.remove()

        if chosen_text is None:
            # last resort: far offset
            dx_pts = max_k * step_pts
            dx_in = dx_pts / 72.0
            offset_tr = base_tr + ScaledTranslation(dx_in, 0, fig.dpi_scale_trans)
            chosen_text = ax.text(
                float(x0),
                y_axes,
                label,
                transform=offset_tr,
                ha="center",
                va="bottom",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=fontsize,
                color=color,
                clip_on=False,
                zorder=zorder,
            )
            fig.canvas.draw()
            chosen_bbox = chosen_text.get_window_extent(renderer=renderer)

        texts.append(chosen_text)
        placed_bboxes.append(chosen_bbox)

    return texts


# =====================================================
# 6) Plot (2x4; last panel legend)
# =====================================================
cm = 1 / 2.54
fig, axes = plt.subplots(
    2,
    4,
    figsize=(16 * cm, 8 * cm),  # slightly taller to avoid rotated text clipping
    sharey=False,
)

axes_flat = axes.flatten()
rng = np.random.default_rng(42)

for idx, feature in enumerate(ordered_features):
    ax = axes_flat[idx]

    # data
    x_all = X_val[feature].to_numpy(dtype=float)
    y_all = shap_values_low[:, feature_cols.index(feature)].astype(float)

    # load lowess+ci
    data = np.load(LOWESS_DIR / f"lowess_{feature}_ci.npz")
    xi = data["xi"]
    lower = data["lower"]
    upper = data["upper"]
    y_line = (lower + upper) / 2.0

    # tipping points: sign changes of y_line
    sign = np.sign(y_line)
    zero_idx = np.where(np.diff(sign) != 0)[0]
    tipping_xs = (xi[zero_idx] + xi[zero_idx + 1]) / 2.0

    # y-limits
    local_min = float(np.nanmin([np.nanmin(y_all), np.nanmin(lower)]))
    local_max = float(np.nanmax([np.nanmax(y_all), np.nanmax(upper)]))
    pad = (local_max - local_min) * 0.05 if local_max > local_min else 0.1
    y0, y1 = local_min - pad, local_max + pad
    ax.set_ylim(y0, y1)

    # gradient background alternating by segments
    segments = np.concatenate(([float(np.nanmin(xi))], tipping_xs, [float(np.nanmax(xi))]))
    segments = np.sort(np.unique(segments))
    for j in range(len(segments) - 1):
        cmap = purp_cmap if (j % 2 == 0) else green_cmap
        draw_gradient(ax, segments[j], segments[j + 1], cmap)

    # baseline
    ax.axhline(0, color="#4D4D4D", linestyle="--", linewidth=0.8)

    # scatter (subsample)
    n_samp = min(1000, len(x_all))
    sel = rng.choice(len(x_all), size=n_samp, replace=False)
    ax.scatter(
        x_all[sel],
        y_all[sel],
        facecolor="white",
        edgecolor="black",
        s=3,
        linewidth=0.2,
        zorder=2,
    )

    # lowess line
    ax.plot(xi, y_line, color="red", linewidth=1.0, zorder=3)

    # tipping vertical lines
    for tx in np.array(tipping_xs, dtype=float):
        if np.isfinite(tx):
            ax.axvline(float(tx), color="black", linestyle="--", linewidth=0.8, zorder=4)

    # top labels (rotated + offset)
    place_top_labels_avoid_overlap(
        ax,
        tipping_xs,
        fmt="{:.2f}",
        y_axes=1.10,
        step_pts=3,
        max_k=10,
        fontsize=8,
        color="black",
        zorder=5,
        rotation=45,
    )

    # labels
    short = feature_name_map[feature]
    ax.set_xlabel(short, fontsize=8)
    ax.set_ylabel("SHAP", fontsize=8)

    # x ticks: min/mid/max rounded to 0.05
    xmin = float(np.nanmin(xi))
    xmax = float(np.nanmax(xi))
    xmid = (xmin + xmax) / 2.0
    xt_raw = [xmin, xmid, xmax]
    xt = [round(v / 0.05) * 0.05 for v in xt_raw]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:.2f}" for v in xt])
    ax.tick_params(axis="x", width=0.5, length=2, colors="black")

    # y ticks: min/mid/max
    ymid = (y0 + y1) / 2.0
    yt = [y0, ymid, y1]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:.2f}" for v in yt])
    ax.tick_params(axis="y", width=0.5, length=2, colors="black")

    # spines
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color("black")

# Legend panel (last)
legend_ax = axes_flat[-1]
legend_ax.axis("off")
legend_elements = [
    Line2D([0], [0], marker="o", color="w", label="Samples",
           markerfacecolor="white", markeredgecolor="black", markersize=5),
    Line2D([0], [0], color="red", lw=1.0, label="LOWESS"),
    Line2D([0], [0], linestyle="--", color="black", lw=0.8, label="Tipping"),
]
legend_ax.legend(handles=legend_elements, loc="center", frameon=False, fontsize=8)

plt.tight_layout()

# Give a bit more padding to avoid rotated text being clipped
fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.1)
fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

print("[DONE] Saved:", OUT_PDF)
print("[DONE] Saved:", OUT_SVG)
