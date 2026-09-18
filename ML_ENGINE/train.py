"""
================================================================================
  CODE CORTEX 3.0  -  STEP 6 : MODEL TRAINING
  Member 1 | ML / Research
================================================================================

  !!! USER RUNS THIS MANUALLY after confirming split + preprocessing !!!

  Trains three models and evaluates on the VALIDATION set:
    1. Logistic Regression   (linear baseline, interpretable)
    2. Random Forest         (ensemble, handles outliers and non-linearity)
    3. XGBoost               (gradient boosting, best-in-class for tabular)

  Evaluation metrics (val set):
    Accuracy  |  Precision  |  Recall  |  F1  |  ROC-AUC
    FPR (False Positive Rate)  |  Inference time per sample

  Model selection:  best ROC-AUC on val set.

Run:
    python ml/train.py

Reads:
    ml/artifacts/X_train.csv, X_val.csv
    ml/artifacts/y_train.csv, y_val.csv
    ml/artifacts/feature_columns.json

Outputs:
    ml/artifacts/best_model.pkl        <- winning model (pickled)
    ml/artifacts/all_models.pkl        <- dict of all three fitted models
    ml/artifacts/val_metrics.json      <- numeric results for downstream use
    ml/reports/training_report.txt
    ml/plots/05_roc_curves.png
    ml/plots/06_confusion_matrices.png
================================================================================
"""

import io
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc as sklearn_auc,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# -- Paths --------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR        = os.path.join(BASE_DIR, "ml")
ARTIFACTS_DIR = os.path.join(ML_DIR, "artifacts")
REPORTS_DIR   = os.path.join(ML_DIR, "reports")
PLOTS_DIR     = os.path.join(ML_DIR, "plots")

for d in [ARTIFACTS_DIR, REPORTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

REPORT_FILE = os.path.join(REPORTS_DIR, "training_report.txt")

# -- Logger -------------------------------------------------------------------
SEP  = "-" * 72
SEP2 = "=" * 72
report_lines: list = []

def log(msg: str = "", indent: int = 0) -> None:
    line = ("  " * indent) + str(msg)
    print(line)
    report_lines.append(line)

def section(title: str) -> None:
    log(); log(SEP); log(f"  {title}"); log(SEP)

# -- Plot style ---------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "text.color":       "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "grid.color":       "#333",
    "grid.alpha":       0.4,
})
MODEL_COLORS = {
    "LogisticRegression": "#5B9BD5",
    "RandomForest":       "#E09B5C",
    "XGBoost":            "#7ED06E",
}

# =============================================================================
log(SEP2)
log("  CODE CORTEX 3.0  -  STEP 6 : MODEL TRAINING")
log(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(SEP2)

# -- Load split data ----------------------------------------------------------
section("LOAD  -  Reading artifacts")

def load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(ARTIFACTS_DIR, f"{name}.csv")
    if not os.path.exists(path):
        log(f"[ERROR] Missing artifact: {path}")
        sys.exit(1)
    return pd.read_csv(path)

X_train = load_csv("X_train")
X_val   = load_csv("X_val")
y_train = load_csv("y_train")["legitimate"]
y_val   = load_csv("y_val")["legitimate"]

with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json")) as f:
    FEATURE_COLS = json.load(f)

# Ensure column order matches the locked feature list
X_train = X_train[FEATURE_COLS]
X_val   = X_val[FEATURE_COLS]

log(f"X_train : {X_train.shape[0]:,} x {X_train.shape[1]}", 1)
log(f"X_val   : {X_val.shape[0]:,} x {X_val.shape[1]}", 1)
log(f"y_train : {int((y_train==0).sum()):,} malware  |  "
    f"{int((y_train==1).sum()):,} legit", 1)
log(f"y_val   : {int((y_val==0).sum()):,} malware  |  "
    f"{int((y_val==1).sum()):,} legit", 1)

# -- Model definitions --------------------------------------------------------
section("MODEL DEFINITIONS")

MODELS = {
    "LogisticRegression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

if XGBOOST_AVAILABLE:
    # Compute scale_pos_weight for XGBoost (ratio malware:legit)
    n_neg  = int((y_train == 0).sum())
    n_pos  = int((y_train == 1).sum())
    spw    = n_neg / max(n_pos, 1)
    MODELS["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    log("XGBoost imported successfully.", 1)
else:
    log("[WARN] xgboost not installed. Only LR + RF will be trained.", 1)
    log("       Install with:  pip install xgboost", 2)

for name, model in MODELS.items():
    log(f"  {name}:", 1)
    log(f"    {model}", 2)

# -- Train + Evaluate ---------------------------------------------------------
section("TRAINING  &  VALIDATION EVALUATION")

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    """Return a dict of all evaluation metrics."""
    cm    = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / max(fp + tn, 1)
    fnr_val = fn / max(fn + tp, 1)
    return {
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":         float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":    float(roc_auc_score(y_true, y_prob)),
        "fpr":        float(fpr_val),
        "fnr":        float(fnr_val),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

results        = {}
trained_models = {}
roc_data       = {}

for model_name, model in MODELS.items():
    log(f"\n  [{model_name}]", 1)

    # --- Train ---
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    log(f"    Train time      : {train_time:.2f}s", 2)

    # --- Inference time (val set, averaged per sample) ---
    t1 = time.time()
    y_val_prob = model.predict_proba(X_val)[:, 1]
    infer_time = (time.time() - t1) / len(X_val) * 1000  # ms per sample
    y_val_pred = (y_val_prob >= 0.5).astype(int)

    # --- Metrics ---
    m = compute_metrics(y_val.values, y_val_pred, y_val_prob)
    m["train_time_s"]      = round(train_time, 3)
    m["infer_ms_per_sample"] = round(infer_time, 4)
    results[model_name]    = m
    trained_models[model_name] = model

    # --- ROC data for plotting ---
    fpr_arr, tpr_arr, _ = roc_curve(y_val, y_val_prob)
    roc_data[model_name] = {"fpr": fpr_arr.tolist(), "tpr": tpr_arr.tolist(),
                             "auc": m["roc_auc"]}

    log(f"    Accuracy        : {m['accuracy']*100:.3f}%", 2)
    log(f"    Precision       : {m['precision']*100:.3f}%", 2)
    log(f"    Recall          : {m['recall']*100:.3f}%", 2)
    log(f"    F1              : {m['f1']*100:.3f}%", 2)
    log(f"    ROC-AUC         : {m['roc_auc']:.6f}", 2)
    log(f"    FPR             : {m['fpr']*100:.3f}%"
        f"  (false alarms out of all legit files)", 2)
    log(f"    FNR             : {m['fnr']*100:.3f}%"
        f"  (missed malware out of all malware)", 2)
    log(f"    Confusion Matrix: TP={m['TP']:,} TN={m['TN']:,}"
        f" FP={m['FP']:,} FN={m['FN']:,}", 2)
    log(f"    Infer time      : {m['infer_ms_per_sample']:.4f} ms/sample", 2)

# -- Comparison Table ---------------------------------------------------------
section("COMPARISON TABLE  (Validation Set)")

log(f"  {'Metric':<22}", 1, )
header  = f"  {'Metric':<22}"
divider = "  " + "-" * (22 + 16 * len(results))
for name in results:
    header += f" {name:>16}"
log(header, 0)
log(divider, 0)

for metric in ["accuracy", "precision", "recall", "f1", "roc_auc",
               "fpr", "fnr", "infer_ms_per_sample"]:
    row = f"  {metric:<22}"
    for name in results:
        val = results[name][metric]
        if metric in ["accuracy", "precision", "recall", "f1", "fpr", "fnr"]:
            row += f" {val*100:>15.3f}%"
        elif metric == "roc_auc":
            row += f" {val:>16.6f}"
        else:
            row += f" {val:>15.4f}ms"
    log(row, 0)

# -- Model Selection ----------------------------------------------------------
section("MODEL SELECTION  (based on highest ROC-AUC on val set)")

best_model_name = max(results, key=lambda n: results[n]["roc_auc"])
best_metrics    = results[best_model_name]

log(f"  Selected model : {best_model_name}", 1)
log(f"  ROC-AUC        : {best_metrics['roc_auc']:.6f}", 1)
log(f"  F1             : {best_metrics['f1']:.6f}", 1)
log(f"  FPR            : {best_metrics['fpr']*100:.3f}%", 1)
log(f"  FNR            : {best_metrics['fnr']*100:.3f}%", 1)
log("", 1)
log("  Note: Final evaluation on the HELD-OUT TEST SET is done separately", 1)
log("  in Step 8, after hyperparameter tuning (Step 7).", 1)
log("  Do NOT peek at test set metrics until Step 8.", 1)

# Save best model
best_path = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
with open(best_path, "wb") as f:
    pickle.dump(trained_models[best_model_name], f)
log(f"\n  [SAVED] best_model.pkl  ({best_model_name})  -> {best_path}", 1)

# Save all models
all_path = os.path.join(ARTIFACTS_DIR, "all_models.pkl")
with open(all_path, "wb") as f:
    pickle.dump(trained_models, f)
log(f"  [SAVED] all_models.pkl  -> {all_path}", 1)

# Save metrics JSON
metrics_path = os.path.join(ARTIFACTS_DIR, "val_metrics.json")
metrics_out = {
    "best_model": best_model_name,
    "models": results,
}
with open(metrics_path, "w") as f:
    json.dump(metrics_out, f, indent=2)
log(f"  [SAVED] val_metrics.json  -> {metrics_path}", 1)

# -- Plots --------------------------------------------------------------------
section("PLOTS  -  ROC Curves + Confusion Matrices")

# Plot 05 : ROC curves
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("#1a1a2e")
ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5, label="Random classifier")
for name, rd in roc_data.items():
    color = MODEL_COLORS.get(name, "#aaa")
    lw    = 2.5 if name == best_model_name else 1.5
    ls    = "-" if name == best_model_name else "--"
    label = f"{name}  (AUC={rd['auc']:.4f})"
    if name == best_model_name:
        label += "  [BEST]"
    ax.plot(rd["fpr"], rd["tpr"], color=color, lw=lw, ls=ls, label=label)

ax.set_xlabel("False Positive Rate", fontsize=12, color="#e0e0e0")
ax.set_ylabel("True Positive Rate",  fontsize=12, color="#e0e0e0")
ax.set_title("ROC Curves - Validation Set", fontsize=14,
             fontweight="bold", color="#e0e0e0")
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
_p = os.path.join(PLOTS_DIR, "05_roc_curves.png")
plt.savefig(_p, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {_p}", 1)

# Plot 06 : Confusion matrices
n_models = len(trained_models)
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
fig.patch.set_facecolor("#1a1a2e")
if n_models == 1:
    axes = [axes]

for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred_v = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred_v)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
        linewidths=0.5,
        linecolor="#1a1a2e",
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title(f"{name}", fontsize=11, fontweight="bold", color="#e0e0e0")
    ax.set_xlabel("Predicted Label", color="#e0e0e0")
    ax.set_ylabel("True Label",      color="#e0e0e0")
    ax.set_xticklabels(["Malware (0)", "Legit (1)"], color="#ccc")
    ax.set_yticklabels(["Malware (0)", "Legit (1)"], color="#ccc", rotation=0)

plt.suptitle("Confusion Matrices - Validation Set",
             fontsize=13, fontweight="bold", color="#e0e0e0")
plt.tight_layout()
_p = os.path.join(PLOTS_DIR, "06_confusion_matrices.png")
plt.savefig(_p, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {_p}", 1)

# -- Final Banner -------------------------------------------------------------
log()
log(SEP2)
log("  STEP 6 - MODEL TRAINING COMPLETE")
log(f"  Best model : {best_model_name}  (ROC-AUC = {best_metrics['roc_auc']:.6f})")
log("  Next steps :")
log("    Step 7 : Hyperparameter tuning on train+val only")
log("    Step 8 : Final evaluation on test set (first time test is touched)")
log("    Step 9 : Probability calibration + threshold selection")
log("    Step 10: SHAP explainability")
log(f"  Finished : {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(SEP2)

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n[REPORT] Written to: {REPORT_FILE}")
