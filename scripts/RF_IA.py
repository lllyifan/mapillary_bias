#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Paths
ROOT = Path(__file__).resolve().parents[1]


DATA_DIR = ROOT / "data" / "satscan_input_IA"
INPUT_CSV = DATA_DIR / "matched_lsoas_from_clusters_IA.csv"


OUT_DIR = ROOT / "outputs" / "RF" / "IA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "rf_model_ClusterType.joblib"
PRED_PATH  = OUT_DIR / "predictions_val_ClusterType.csv"
TRAIN_PATH = OUT_DIR / "train_set_ClusterType.csv"
VAL_PATH   = OUT_DIR / "val_set_ClusterType.csv"


CLASSES_PATH = OUT_DIR / "label_encoder_classes.txt"
METRICS_PATH = OUT_DIR / "metrics.txt"

# Load data
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print(f"[READ] {INPUT_CSV} | rows={len(df):,}")

# Feature selection
FEATURE_COLS = [
    "indicator_sex_female",
    "indicator_Ethnic_nonwhite",
    "indicator_education_noqualification",
    "indicator_age_elder",
    "indicator_age_children",
    "indicator_NS-SeC_unemployed",
    "indicator_NS-SeC_students",
]
COORD_COLS = ["LAT", "LONG"]  # keep in output tables if you want, but not used in training

# Basic checks
need_cols = set(FEATURE_COLS + COORD_COLS + ["ClusterType"])
missing = sorted(list(need_cols - set(df.columns)))
if missing:
    raise KeyError(f"Missing required columns: {missing}")

X = df[FEATURE_COLS + COORD_COLS].copy()
y_raw = df["ClusterType"].astype(str).copy()

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_.tolist()
print(f"[LABELS] {classes}")

# Save classes
CLASSES_PATH.write_text("\n".join(classes), encoding="utf-8")

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
print(f"[SPLIT] train={len(X_train):,} | val={len(X_val):,}")

# Model training
clf = RandomForestClassifier(
    n_estimators=600,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="log2",
    bootstrap=True,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train[FEATURE_COLS], y_train)
print("[FIT] done")

# Save model
joblib.dump(clf, MODEL_PATH)
print(f"[WRITE] model -> {MODEL_PATH}")

# Prediction and evaluation
y_pred = clf.predict(X_val[FEATURE_COLS])

# AUC
metrics_lines = []
acc = accuracy_score(y_val, y_pred)
metrics_lines.append(f"Accuracy: {acc:.6f}")

auc = None
try:
    proba = clf.predict_proba(X_val[FEATURE_COLS])
    if proba.shape[1] == 2:
        y_proba = proba[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        metrics_lines.append(f"AUC: {auc:.6f}")
    else:

        auc = roc_auc_score(y_val, proba, multi_class="ovr")
        metrics_lines.append(f"AUC(ovr): {auc:.6f}")
except Exception as e:
    metrics_lines.append(f"AUC: NA ({e})")

report = classification_report(
    y_val,
    y_pred,
    target_names=classes,
)
metrics_lines.append("\nClassification report:\n" + report)

print("\n".join(metrics_lines))
METRICS_PATH.write_text("\n".join(metrics_lines), encoding="utf-8")

# Save validation predictions
val_results = X_val.copy()
val_results["True"] = le.inverse_transform(y_val)
val_results["Predicted"] = le.inverse_transform(y_pred)


try:
    proba = clf.predict_proba(X_val[FEATURE_COLS])
    if proba.shape[1] == 2 and "high" in classes:
        high_idx = classes.index("high")
        val_results["Prob_high"] = proba[:, high_idx]
except Exception:
    pass

val_results.to_csv(PRED_PATH, index=False)
print(f"[WRITE] val predictions -> {PRED_PATH}")

# Save train / validation sets
X_train.to_csv(TRAIN_PATH, index=False)
X_val.to_csv(VAL_PATH, index=False)
print(f"[WRITE] train -> {TRAIN_PATH}")
print(f"[WRITE] val   -> {VAL_PATH}")

print(f"[DONE] outputs in: {OUT_DIR}")
