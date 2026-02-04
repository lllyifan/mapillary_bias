#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "outputs" / "RF" / "IA"
MODEL_PATH = DATA_DIR / "rf_model_ClusterType.joblib"
VAL_CSV = DATA_DIR / "val_set_ClusterType.csv"

OUTPUT_DIR = ROOT / "outputs" / "SHAP_INTERACTION" / "IA" 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTPUT_DIR / "shap_interaction_signed_sum_heatmap_low.png"
RAW_NPY = OUTPUT_DIR / "shap_inter_raw_low.npy"
SIGNED_CSV = OUTPUT_DIR / "shap_signed_sum_low.csv"
MASK_NPY = OUTPUT_DIR / "shap_mask_low.npy"
ALL_JOBLIB = OUTPUT_DIR / "shap_inter_all.joblib"

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

IDX_HIGH = 0
IDX_LOW = 1

# Load model and validation data
clf = joblib.load(MODEL_PATH)
df_val = pd.read_csv(VAL_CSV)
X_val = df_val[FEATURE_COLS]

# Compute SHAP interaction values
explainer = shap.TreeExplainer(clf)
shap_inter_all = explainer.shap_interaction_values(X_val)

if isinstance(shap_inter_all, list):
    raw = np.asarray(shap_inter_all[IDX_LOW])
else:
    arr = np.asarray(shap_inter_all)
    if arr.ndim == 4 and arr.shape[-1] == 2:
        raw = arr[..., IDX_LOW]
    elif arr.ndim == 4 and arr.shape[0] == 2 and arr.shape[1] == X_val.shape[0]:
        raw = arr[IDX_LOW, ...]
    else:
        raise ValueError(f"Unexpected shap_interaction_values shape: {arr.shape}")

print(f"X_val.shape = {X_val.shape}")
print(f"raw interaction shape = {raw.shape} (expect n_samples x n_feat x n_feat)")
print(f"Interpreting class: low (index={IDX_LOW})")

# Signed interaction aggregation
signed_sum = raw.sum(axis=0)

# Mask upper triangle and diagonal
n = signed_sum.shape[0]
mask = np.triu(np.ones((n, n), dtype=bool), k=0)

# Plot heatmap
plt.figure(figsize=(8, 6))

diag_mask = np.eye(n, dtype=bool)
offdiag_vals = signed_sum[~diag_mask]
max_abs = np.max(np.abs(offdiag_vals)) if offdiag_vals.size else 1e-9

m = np.ma.array(signed_sum, mask=mask)
im = plt.imshow(
    m,
    aspect="equal",
    cmap="bwr",
    vmin=-max_abs,
    vmax=+max_abs,
)
plt.colorbar(
    im,
    label="Sum of SHAP interaction (signed)\n(excluding self-interaction)",
)

for i in range(n):
    for j in range(n):
        if not mask[i, j]:
            val = signed_sum[i, j]
            txt_color = "white" if abs(val) > max_abs * 0.6 else "black"
            plt.text(
                j,
                i,
                f"{val:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=txt_color,
            )

plt.xticks(np.arange(n), FEATURE_COLS, rotation=45, ha="right")
plt.yticks(np.arange(n), FEATURE_COLS)
plt.title(
    "Lower-triangle signed SHAP interaction for LOW (class index 1)\n"
    "(positive = red, negative = blue)"
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()

print(f"Low-class heatmap saved to: {OUT_PNG}")

# Save outputs
np.save(RAW_NPY, raw)
pd.DataFrame(
    signed_sum,
    index=FEATURE_COLS,
    columns=FEATURE_COLS,
).to_csv(SIGNED_CSV, float_format="%.6f")
np.save(MASK_NPY, mask)
joblib.dump(shap_inter_all, ALL_JOBLIB)

print(f"Raw interaction saved to: {RAW_NPY}")
print(f"Signed sum saved to: {SIGNED_CSV}")
print(f"Mask saved to: {MASK_NPY}")
print(f"Full interaction object saved to: {ALL_JOBLIB}")
