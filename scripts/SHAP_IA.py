#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "outputs" / "RF" / "IA" / "rf_model_ClusterType.joblib"
VALSET_PATH = ROOT / "outputs" / "RF" / "IA" / "predictions_val_ClusterType.csv"
OUTPUT_DIR = ROOT / "outputs" / "SHAP" / "IA"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHAP_SUMMARY_PNG = OUTPUT_DIR / "shap_summary_bar_low.png"
SHAP_VALUES_NPY = OUTPUT_DIR / "shap_values_low.npy"
SHAP_XVAL_CSV = OUTPUT_DIR / "shap_X_val.csv"

# Settings
IDX_HIGH = 0
IDX_LOW = 1

FEATURE_COLS = [
    "indicator_sex_female",
    "indicator_Ethnic_nonwhite",
    "indicator_education_noqualification",
    "indicator_age_elder",
    "indicator_age_children",
    "indicator_NS-SeC_unemployed",
    "indicator_NS-SeC_students",
]

# Load model and validation features
print("Loading model and validation data...")
rf_model = joblib.load(MODEL_PATH)
X_val = pd.read_csv(VALSET_PATH)[FEATURE_COLS]

# Compute SHAP values
print("Computing SHAP values...")
explainer = shap.TreeExplainer(rf_model)
sv = explainer.shap_values(X_val)

print(f"sv.shape = {sv.shape} | X_val.shape = {X_val.shape}")
print("Explaining: P(low) (class index = 1)")

shap_values_low = sv[:, :, IDX_LOW]
print(f"shap_values_low.shape = {shap_values_low.shape}")

# SHAP summary plot 
print("Rendering SHAP summary plot...")
shap.summary_plot(shap_values_low, X_val, plot_type="bar", show=False)
plt.title("SHAP summary (P(low))", fontsize=11)
plt.tight_layout()
plt.savefig(SHAP_SUMMARY_PNG, dpi=300)
plt.close()

# Save outputs
print("Saving SHAP outputs...")
np.save(SHAP_VALUES_NPY, shap_values_low)
X_val.to_csv(SHAP_XVAL_CSV, index=False)

print(f"Done. Outputs saved to: {OUTPUT_DIR}")
