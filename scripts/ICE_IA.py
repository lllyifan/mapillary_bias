#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parents[1]

#inputs live under data
DATA_DIR = ROOT / "data" / "satscan_input_IA"

#RF artifacts live under out
RF_OUT_DIR = ROOT / "outputs" / "RF" / "IA"
MODEL_PATH = RF_OUT_DIR / "rf_model_ClusterType.joblib"
VAL_SET_PATH = RF_OUT_DIR / "val_set_ClusterType.csv"

#ICE artifacts live under out
ICE_OUT_DIR = ROOT / "outputs" / "ICE" / "IA"
OUTPUT_BASE = ICE_OUT_DIR / "ICE_manual_output"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Settings
FEATURE_COLS = [
    "indicator_sex_female",
    "indicator_Ethnic_nonwhite",
    "indicator_education_noqualification",
    "indicator_age_elder",
    "indicator_age_children",
    "indicator_NS-SeC_unemployed",
    "indicator_NS-SeC_students",
]
COORD_COLS = ["LAT", "LONG"]

GRID_RESOLUTION = 20
MAX_ICE_CURVES_TO_PLOT = 100

# Helpers
def manual_ice(estimator, X: pd.DataFrame, feature: str, target_class: int, grid_resolution: int = 20):
    min_val, max_val = X[feature].min(), X[feature].max()
    grid_values = np.linspace(min_val, max_val, grid_resolution)

    n_samples = X.shape[0]
    ice_curves = np.zeros((n_samples, grid_resolution), dtype=float)

    for j, val in enumerate(grid_values):
        X_tmp = X.copy()
        X_tmp[feature] = val
        ice_curves[:, j] = estimator.predict_proba(X_tmp)[:, target_class]

    return grid_values, ice_curves

def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

def save_npy(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# Load model and validation data
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing RF model: {MODEL_PATH}")
if not VAL_SET_PATH.exists():
    raise FileNotFoundError(f"Missing validation set: {VAL_SET_PATH}")

rf_model = joblib.load(MODEL_PATH)
df_val = pd.read_csv(VAL_SET_PATH)

need_cols = set(FEATURE_COLS + COORD_COLS)
missing = sorted(list(need_cols - set(df_val.columns)))
if missing:
    raise KeyError(f"Missing required columns in val_set: {missing}")

X_val = df_val[FEATURE_COLS].copy()
coords = df_val[COORD_COLS].copy()

print(f"[READ] model: {MODEL_PATH}")
print(f"[READ] val_set: {VAL_SET_PATH}")
print(f"[OUT]  ice_dir: {OUTPUT_BASE}")

# Main routine
def process_manual_ice(target_class: int, label: str):
    out_dir = OUTPUT_BASE / label
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "IP",
        "label": label,
        "target_class": int(target_class),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "val_set_path": str(VAL_SET_PATH.relative_to(ROOT)),
        "grid_resolution": int(GRID_RESOLUTION),
        "features": {},
    }

    all_slopes_long = []

    for feature in FEATURE_COLS:
        print(f"Computing ICE: label={label} feature={feature}")

        grid, ice = manual_ice(
            rf_model,
            X_val,
            feature,
            target_class,
            grid_resolution=GRID_RESOLUTION,
        )

        fbase = f"{feature}_{label}"
        fdir = out_dir / feature
        fdir.mkdir(parents=True, exist_ok=True)

        path_grid_csv = fdir / f"grid_{fbase}.csv"
        path_ice_npy = fdir / f"ice_{fbase}.npy"
        path_ice_summary_csv = fdir / f"ice_summary_{fbase}.csv"
        path_slopes_csv = fdir / f"slopes_{fbase}.csv"
        path_slopes_npy = fdir / f"slopes_{fbase}.npy"
        path_fig_ice_png = fdir / f"ICE_{fbase}.png"
        path_fig_spatial_png = fdir / f"SpatialICE_{fbase}.png"

        save_csv(pd.DataFrame({"grid": grid}), path_grid_csv)
        save_npy(ice, path_ice_npy)

        avg_curve = ice.mean(axis=0)
        std_curve = ice.std(axis=0)
        df_summary = pd.DataFrame({"grid": grid, "mean_prob": avg_curve, "std_prob": std_curve})
        save_csv(df_summary, path_ice_summary_csv)

        plt.figure(figsize=(8, 5))
        for i in range(min(MAX_ICE_CURVES_TO_PLOT, ice.shape[0])):
            plt.plot(grid, ice[i], alpha=0.3)
        plt.plot(grid, avg_curve, linewidth=2, label="Average effect")
        plt.title(f"ICE curve ({label}) — {feature}")
        plt.xlabel(feature)
        plt.ylabel(f"P(class={target_class})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_fig_ice_png, dpi=300)
        plt.close()

        denom = (grid[-1] - grid[0]) if (grid[-1] - grid[0]) != 0 else np.nan
        slopes = (ice[:, -1] - ice[:, 0]) / denom

        df_plot = coords.copy()
        df_plot["slope"] = slopes
        save_csv(df_plot, path_slopes_csv)
        save_npy(slopes, path_slopes_npy)

        tmp_long = df_plot.copy()
        tmp_long["feature"] = feature
        all_slopes_long.append(tmp_long[["LAT", "LONG", "feature", "slope"]])

        max_abs = np.nanmax(np.abs(df_plot["slope"].to_numpy()))
        vmin, vmax = (-max_abs, max_abs) if np.isfinite(max_abs) and max_abs > 0 else (-1e-9, 1e-9)

        plt.figure(figsize=(7, 6))
        sc = plt.scatter(
            df_plot["LONG"],
            df_plot["LAT"],
            c=df_plot["slope"],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            s=25,
        )
        cbar = plt.colorbar(sc, pad=0.01)
        cbar.set_label(f"ΔP/Δ{feature} (class={target_class})", rotation=270, labelpad=15)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Spatial ICE gradient — {feature} → class={target_class}")
        plt.tight_layout()
        plt.savefig(path_fig_spatial_png, dpi=300)
        plt.close()

        manifest["features"][feature] = {
            "grid_csv": str(path_grid_csv.relative_to(ROOT)),
            "ice_npy": str(path_ice_npy.relative_to(ROOT)),
            "ice_summary_csv": str(path_ice_summary_csv.relative_to(ROOT)),
            "slopes_csv": str(path_slopes_csv.relative_to(ROOT)),
            "slopes_npy": str(path_slopes_npy.relative_to(ROOT)),
            "fig_ice_png": str(path_fig_ice_png.relative_to(ROOT)),
            "fig_spatial_png": str(path_fig_spatial_png.relative_to(ROOT)),
            "stats": {
                "grid_min": float(np.min(grid)),
                "grid_max": float(np.max(grid)),
                "avg_curve_min": float(np.min(avg_curve)),
                "avg_curve_max": float(np.max(avg_curve)),
                "slope_mean": float(np.nanmean(slopes)),
                "slope_std": float(np.nanstd(slopes)),
                "slope_min": float(np.nanmin(slopes)),
                "slope_max": float(np.nanmax(slopes)),
            },
        }

    df_all_slopes = pd.concat(all_slopes_long, ignore_index=True)
    path_all_slopes_csv = out_dir / f"all_slopes_{label}.csv"
    save_csv(df_all_slopes, path_all_slopes_csv)

    path_manifest_json = out_dir / f"manifest_{label}.json"
    save_json(manifest, path_manifest_json)

    print(f"Summary saved:\n- {path_all_slopes_csv}\n- {path_manifest_json}")

# Run (keep your original hard-coded class indices)
process_manual_ice(0, "high")
process_manual_ice(1, "low")

print("All ICE outputs generated.")
