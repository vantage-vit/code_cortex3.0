"""
================================================================================
  CODE CORTEX 3.0  -  STEP 7 : HYPERPARAMETER TUNING + FLAGSHIP MODEL COMPARISON
  Member 1 | ML / Research
================================================================================

  Objectives
  ----------
  1. XGBoost Grid Search
     - Primary objective : minimise FNR  (missed malware)
     - Hard constraint   : FPR < 1.5%   (false alarms must stay low)
     - Grid              : max_depth x min_child_weight x subsample x
                           scale_pos_weight  (user-specified)
     - All combinations evaluated on VALIDATION SET only (test set SEALED).

  2. LightGBM  (new flagship)
     - Custom hard grid: num_leaves, min_child_samples, learning_rate,
       subsample, colsample_bytree, min_split_gain, reg_alpha, reg_lambda
     - Same FNR-primary, FPR<1.5% selection criterion.

  3. CatBoost  (new flagship)
     - Custom hard grid: depth, l2_leaf_reg, bagging_temperature,
       border_count, learning_rate, random_strength
     - Same selection criterion.

  4. Final Comparison Table  (all 3 tuned models + original 3 from Step 6)
     - Select champion model with lowest FNR subject to FPR < 1.5%.

  CRITICAL RULES
  --------------
  - Only X_train / y_train and X_val / y_val are loaded.
  - X_test / y_test are NEVER imported or referenced in this file.
  - All fitting happens on X_train only; evaluation on X_val only.

Run:
    python ml/tune.py

Reads:
    ml/artifacts/X_train.csv, y_train.csv
    ml/artifacts/X_val.csv,   y_val.csv
    ml/artifacts/feature_columns.json
    ml/artifacts/val_metrics.json       <- Step 6 baseline results

Outputs (ml/artifacts/):
    xgb_grid_results.csv
    lgbm_grid_results.csv
    cb_grid_results.csv
    tuned_xgb.pkl
    tuned_lgbm.pkl
    tuned_catboost.pkl
    tuned_best_model.pkl                <- overall champion
    tuned_best_params.json
    tuned_val_metrics.json
    ml/reports/tuning_report.txt
    ml/plots/07_tuned_roc_curves.png
    ml/plots/08_tuned_confusion_matrices.png
    ml/plots/09_xgb_grid_heatmap.png
    ml/plots/10_fnr_fpr_scatter.png
================================================================================
"""

import io
import itertools
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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

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

REPORT_FILE = os.path.join(REPORTS_DIR, "tuning_report.txt")

FPR_LIMIT = 0.015   # hard constraint: FPR must stay below 1.5%

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
    "XGBoost_tuned":      "#7ED06E",
    "LightGBM_tuned":     "#E8A838",
    "CatBoost_tuned":     "#D65780",
    "LogisticRegression": "#5B9BD5",
    "RandomForest":       "#E09B5C",
    "XGBoost":            "#91D4A8",
}

# =============================================================================
log(SEP2)
log("  CODE CORTEX 3.0  -  STEP 7 : HYPERPARAMETER TUNING")
log(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(SEP2)

# =============================================================================
# LOAD DATA
# =============================================================================
section("LOAD  -  Reading train/val artifacts  (test set NEVER touched)")

def _load(name: str) -> pd.DataFrame:
    p = os.path.join(ARTIFACTS_DIR, f"{name}.csv")
    if not os.path.exists(p):
        log(f"[ERROR] Missing: {p}")
        sys.exit(1)
    return pd.read_csv(p)

X_train = _load("X_train")
X_val   = _load("X_val")
y_train = _load("y_train")["legitimate"]
y_val   = _load("y_val")["legitimate"]

with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json")) as fh:
    FEATURE_COLS = json.load(fh)

X_train = X_train[FEATURE_COLS]
X_val   = X_val[FEATURE_COLS]

log(f"X_train : {X_train.shape[0]:,} rows  x  {X_train.shape[1]} features", 1)
log(f"X_val   : {X_val.shape[0]:,} rows  x  {X_val.shape[1]} features", 1)
log(f"y_train : {int((y_train==0).sum()):,} malware  |  {int((y_train==1).sum()):,} legit", 1)
log(f"y_val   : {int((y_val==0).sum()):,} malware  |  {int((y_val==1).sum()):,} legit", 1)
log(f"FPR constraint : < {FPR_LIMIT*100:.1f}%   (primary objective = min FNR)", 1)

# Load Step 6 baseline results for final comparison table
_base_path = os.path.join(ARTIFACTS_DIR, "val_metrics.json")
baseline_results = {}
if os.path.exists(_base_path):
    with open(_base_path) as fh:
        _b = json.load(fh)
    baseline_results = _b.get("models", {})
    log(f"Loaded Step 6 baselines: {list(baseline_results.keys())}", 1)

# -- Helper: compute all metrics ----------------------------------------------
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
        "fpr":       float(fp / max(fp + tn, 1)),
        "fnr":       float(fn / max(fn + tp, 1)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

def eval_model(model, threshold: float = 0.5) -> dict:
    t0 = time.time()
    y_prob = model.predict_proba(X_val)[:, 1]
    infer_ms = (time.time() - t0) / len(X_val) * 1000
    y_pred = (y_prob >= threshold).astype(int)
    m = compute_metrics(y_val.values, y_pred, y_prob)
    m["infer_ms_per_sample"] = round(infer_ms, 4)
    return m


# =============================================================================
# SECTION A -- XGBoost Grid Search
# =============================================================================
section("XGBoost Grid Search  (user-specified parameter space)")

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:
    log("[ERROR] xgboost not installed.  pip install xgboost", 1)
    XGB_OK = False

XGB_GRID = {
    "max_depth":        [4, 5, 6],
    "min_child_weight": [1, 3, 5],
    "subsample":        [0.7, 0.8, 0.9],
    "scale_pos_weight": [1.5, 2.0, 2.5, 3.0],
}
# Fixed params across all XGB runs (harder than Step 6)
XGB_FIXED = dict(
    n_estimators      = 500,        # more trees than Step 6 (300)
    learning_rate     = 0.05,       # slower LR -> more precise convergence
    colsample_bytree  = 0.75,
    colsample_bylevel = 0.75,       # per-level column subsampling (harder)
    reg_alpha         = 0.1,        # L1 regularisation
    reg_lambda        = 1.5,        # L2 regularisation
    eval_metric       = "logloss",
    use_label_encoder = False,
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = 0,
)

xgb_records   = []
xgb_combos    = list(itertools.product(
    XGB_GRID["max_depth"],
    XGB_GRID["min_child_weight"],
    XGB_GRID["subsample"],
    XGB_GRID["scale_pos_weight"],
))

log(f"Total XGBoost combinations : {len(xgb_combos)}", 1)
log(f"Fixed: n_estimators={XGB_FIXED['n_estimators']}, lr={XGB_FIXED['learning_rate']}, "
    f"reg_alpha={XGB_FIXED['reg_alpha']}, reg_lambda={XGB_FIXED['reg_lambda']}", 1)
log("", 0)

best_xgb_model  = None
best_xgb_params = None
best_xgb_fnr    = 1.0
best_xgb_m      = None

xgb_start = time.time()

if XGB_OK:
    for i, (md, mcw, ss, spw) in enumerate(xgb_combos, 1):
        params = dict(max_depth=md, min_child_weight=mcw,
                      subsample=ss, scale_pos_weight=spw)
        clf = XGBClassifier(**XGB_FIXED, **params)
        clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False)
        m = eval_model(clf)

        row = {**params, **m}
        xgb_records.append(row)

        feasible = "OK" if m["fpr"] < FPR_LIMIT else "--"
        log(
            f"  [{i:03d}/{len(xgb_combos)}]  "
            f"depth={md} mcw={mcw} ss={ss:.1f} spw={spw:.1f}  |  "
            f"FNR={m['fnr']*100:.3f}%  FPR={m['fpr']*100:.3f}%  "
            f"AUC={m['roc_auc']:.5f}  [{feasible}]"
        )

        if m["fpr"] < FPR_LIMIT and m["fnr"] < best_xgb_fnr:
            best_xgb_fnr    = m["fnr"]
            best_xgb_model  = clf
            best_xgb_params = params
            best_xgb_m      = m

xgb_elapsed = time.time() - xgb_start
log(f"\n  XGBoost grid search finished in {xgb_elapsed:.1f}s", 1)

# Save grid results
xgb_df = pd.DataFrame(xgb_records)
xgb_csv = os.path.join(ARTIFACTS_DIR, "xgb_grid_results.csv")
xgb_df.to_csv(xgb_csv, index=False)
log(f"  [SAVED] xgb_grid_results.csv  ({len(xgb_records)} rows)", 1)

if best_xgb_model is not None:
    log("\n  -- XGBoost Best Config ------------------------------------------", 1)
    log(f"  Params   : {best_xgb_params}", 1)
    log(f"  FNR      : {best_xgb_m['fnr']*100:.3f}%", 1)
    log(f"  FPR      : {best_xgb_m['fpr']*100:.3f}%", 1)
    log(f"  ROC-AUC  : {best_xgb_m['roc_auc']:.6f}", 1)
    log(f"  F1       : {best_xgb_m['f1']*100:.3f}%", 1)
    log(f"  CM       : TP={best_xgb_m['TP']} TN={best_xgb_m['TN']} "
        f"FP={best_xgb_m['FP']} FN={best_xgb_m['FN']}", 1)

    xgb_pkl = os.path.join(ARTIFACTS_DIR, "tuned_xgb.pkl")
    with open(xgb_pkl, "wb") as fh:
        pickle.dump(best_xgb_model, fh)
    log(f"  [SAVED] tuned_xgb.pkl", 1)
else:
    log("  [WARN] No XGBoost config satisfied FPR < 1.5%. Keeping best FNR overall.", 1)

# -- XGBoost grid heatmap (min FNR per depth x min_child_weight) --------------
if xgb_records:
    _df = pd.DataFrame(xgb_records)
    _pivot = _df.groupby(["max_depth", "min_child_weight"])["fnr"].min().unstack()
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#1a1a2e")
    sns.heatmap(
        _pivot * 100,
        annot=True, fmt=".2f", cmap="YlOrRd_r",
        ax=ax, cbar_kws={"label": "Min FNR (%) across ss x spw"},
        linewidths=0.5, linecolor="#1a1a2e",
    )
    ax.set_title("XGBoost Grid Search -- Min FNR by depth x min_child_weight",
                 fontsize=11, fontweight="bold", color="#e0e0e0")
    ax.set_xlabel("min_child_weight", color="#e0e0e0")
    ax.set_ylabel("max_depth",        color="#e0e0e0")
    plt.tight_layout()
    _pp = os.path.join(PLOTS_DIR, "09_xgb_grid_heatmap.png")
    plt.savefig(_pp, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    log(f"  [PLOT] -> {_pp}", 1)


# =============================================================================
# SECTION B -- LightGBM  (new flagship)
# =============================================================================
section("LightGBM Flagship  (hard parameter grid)")

try:
    import lightgbm as lgb
    LGBM_OK = True
    log(f"LightGBM {lgb.__version__} available.", 1)
except ImportError:
    log("[ERROR] lightgbm not installed.  pip install lightgbm", 1)
    LGBM_OK = False

LGBM_GRID = {
    "num_leaves":        [63, 127, 255],
    "min_child_samples": [20, 50, 100],
    "learning_rate":     [0.02, 0.05],
    "subsample":         [0.7, 0.85],
    "scale_pos_weight":  [1.5, 2.0, 3.0],
}
LGBM_FIXED = dict(
    n_estimators     = 700,         # large tree budget
    max_depth        = -1,          # unlimited depth, controlled by num_leaves
    colsample_bytree = 0.75,
    min_split_gain   = 0.1,         # gain threshold for splitting (harder)
    reg_alpha        = 0.2,         # L1
    reg_lambda       = 2.0,         # L2
    n_jobs           = -1,
    random_state     = 42,
    verbosity        = -1,
)

lgbm_records  = []
lgbm_combos   = list(itertools.product(
    LGBM_GRID["num_leaves"],
    LGBM_GRID["min_child_samples"],
    LGBM_GRID["learning_rate"],
    LGBM_GRID["subsample"],
    LGBM_GRID["scale_pos_weight"],
))

log(f"Total LightGBM combinations : {len(lgbm_combos)}", 1)
log(f"Fixed: n_estimators={LGBM_FIXED['n_estimators']}, reg_alpha={LGBM_FIXED['reg_alpha']}, "
    f"reg_lambda={LGBM_FIXED['reg_lambda']}, min_split_gain={LGBM_FIXED['min_split_gain']}", 1)
log("", 0)

best_lgbm_model  = None
best_lgbm_params = None
best_lgbm_fnr    = 1.0
best_lgbm_m      = None

lgbm_start = time.time()

if LGBM_OK:
    for i, (nl, mcs, lr, ss, spw) in enumerate(lgbm_combos, 1):
        params = dict(num_leaves=nl, min_child_samples=mcs,
                      learning_rate=lr, subsample=ss, scale_pos_weight=spw)
        clf = lgb.LGBMClassifier(**LGBM_FIXED, **params)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        m = eval_model(clf)

        row = {**params, **m}
        lgbm_records.append(row)

        feasible = "OK" if m["fpr"] < FPR_LIMIT else "--"
        log(
            f"  [{i:03d}/{len(lgbm_combos)}]  "
            f"nl={nl} mcs={mcs} lr={lr:.2f} ss={ss:.2f} spw={spw:.1f}  |  "
            f"FNR={m['fnr']*100:.3f}%  FPR={m['fpr']*100:.3f}%  "
            f"AUC={m['roc_auc']:.5f}  [{feasible}]"
        )

        if m["fpr"] < FPR_LIMIT and m["fnr"] < best_lgbm_fnr:
            best_lgbm_fnr    = m["fnr"]
            best_lgbm_model  = clf
            best_lgbm_params = params
            best_lgbm_m      = m

lgbm_elapsed = time.time() - lgbm_start
log(f"\n  LightGBM grid search finished in {lgbm_elapsed:.1f}s", 1)

lgbm_csv = os.path.join(ARTIFACTS_DIR, "lgbm_grid_results.csv")
if lgbm_records:
    pd.DataFrame(lgbm_records).to_csv(lgbm_csv, index=False)
    log(f"  [SAVED] lgbm_grid_results.csv  ({len(lgbm_records)} rows)", 1)

if best_lgbm_model is not None:
    log("\n  -- LightGBM Best Config -----------------------------------------", 1)
    log(f"  Params   : {best_lgbm_params}", 1)
    log(f"  FNR      : {best_lgbm_m['fnr']*100:.3f}%", 1)
    log(f"  FPR      : {best_lgbm_m['fpr']*100:.3f}%", 1)
    log(f"  ROC-AUC  : {best_lgbm_m['roc_auc']:.6f}", 1)
    log(f"  F1       : {best_lgbm_m['f1']*100:.3f}%", 1)
    log(f"  CM       : TP={best_lgbm_m['TP']} TN={best_lgbm_m['TN']} "
        f"FP={best_lgbm_m['FP']} FN={best_lgbm_m['FN']}", 1)

    lgbm_pkl = os.path.join(ARTIFACTS_DIR, "tuned_lgbm.pkl")
    with open(lgbm_pkl, "wb") as fh:
        pickle.dump(best_lgbm_model, fh)
    log(f"  [SAVED] tuned_lgbm.pkl", 1)


# =============================================================================
# SECTION C -- CatBoost  (new flagship)
# =============================================================================
section("CatBoost Flagship  (hard parameter grid)")

try:
    from catboost import CatBoostClassifier
    CB_OK = True
    import catboost
    log(f"CatBoost {catboost.__version__} available.", 1)
except ImportError:
    log("[ERROR] catboost not installed.  pip install catboost", 1)
    CB_OK = False

CB_GRID = {
    "depth":               [6, 8, 10],
    "l2_leaf_reg":         [3, 5, 7],
    "bagging_temperature": [0.5, 1.0, 2.0],
    "learning_rate":       [0.02, 0.05],
}
CB_FIXED = dict(
    iterations         = 700,
    border_count       = 254,       # max histogram resolution (harder, more accurate)
    random_strength    = 1.5,       # regularisation via randomness in splitting
    od_type            = "Iter",    # overfitting detector
    od_wait            = 50,        # stop if no improvement after 50 rounds
    auto_class_weights = "Balanced",
    random_seed        = 42,
    thread_count       = -1,
    verbose            = False,
    allow_writing_files = False,
)

cb_records  = []
cb_combos   = list(itertools.product(
    CB_GRID["depth"],
    CB_GRID["l2_leaf_reg"],
    CB_GRID["bagging_temperature"],
    CB_GRID["learning_rate"],
))

log(f"Total CatBoost combinations : {len(cb_combos)}", 1)
log(f"Fixed: iterations={CB_FIXED['iterations']}, border_count={CB_FIXED['border_count']}, "
    f"random_strength={CB_FIXED['random_strength']}, od_wait={CB_FIXED['od_wait']}", 1)
log("", 0)

best_cb_model  = None
best_cb_params = None
best_cb_fnr    = 1.0
best_cb_m      = None

cb_start = time.time()

if CB_OK:
    for i, (depth, l2, bt, lr) in enumerate(cb_combos, 1):
        params = dict(depth=depth, l2_leaf_reg=l2,
                      bagging_temperature=bt, learning_rate=lr)
        clf = CatBoostClassifier(**CB_FIXED, **params)
        clf.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False,
        )
        m = eval_model(clf)

        row = {**params, **m}
        cb_records.append(row)

        feasible = "OK" if m["fpr"] < FPR_LIMIT else "--"
        log(
            f"  [{i:03d}/{len(cb_combos)}]  "
            f"depth={depth} l2={l2} bt={bt:.1f} lr={lr:.2f}  |  "
            f"FNR={m['fnr']*100:.3f}%  FPR={m['fpr']*100:.3f}%  "
            f"AUC={m['roc_auc']:.5f}  [{feasible}]"
        )

        if m["fpr"] < FPR_LIMIT and m["fnr"] < best_cb_fnr:
            best_cb_fnr    = m["fnr"]
            best_cb_model  = clf
            best_cb_params = params
            best_cb_m      = m

cb_elapsed = time.time() - cb_start
log(f"\n  CatBoost grid search finished in {cb_elapsed:.1f}s", 1)

cb_csv = os.path.join(ARTIFACTS_DIR, "cb_grid_results.csv")
if cb_records:
    pd.DataFrame(cb_records).to_csv(cb_csv, index=False)
    log(f"  [SAVED] cb_grid_results.csv  ({len(cb_records)} rows)", 1)

if best_cb_model is not None:
    log("\n  -- CatBoost Best Config -----------------------------------------", 1)
    log(f"  Params   : {best_cb_params}", 1)
    log(f"  FNR      : {best_cb_m['fnr']*100:.3f}%", 1)
    log(f"  FPR      : {best_cb_m['fpr']*100:.3f}%", 1)
    log(f"  ROC-AUC  : {best_cb_m['roc_auc']:.6f}", 1)
    log(f"  F1       : {best_cb_m['f1']*100:.3f}%", 1)
    log(f"  CM       : TP={best_cb_m['TP']} TN={best_cb_m['TN']} "
        f"FP={best_cb_m['FP']} FN={best_cb_m['FN']}", 1)

    cb_pkl = os.path.join(ARTIFACTS_DIR, "tuned_catboost.pkl")
    with open(cb_pkl, "wb") as fh:
        pickle.dump(best_cb_model, fh)
    log(f"  [SAVED] tuned_catboost.pkl", 1)


# =============================================================================
# SECTION D -- FINAL COMPARISON TABLE
# =============================================================================
section("FINAL COMPARISON TABLE  (Step 6 baselines  +  Step 7 tuned models)")

tuned_models  = {}
tuned_results = {}
roc_data      = {}

# Load Step 6 trained models to re-evaluate (for consistency)
_all_models_pkl = os.path.join(ARTIFACTS_DIR, "all_models.pkl")
step6_models = {}
if os.path.exists(_all_models_pkl):
    with open(_all_models_pkl, "rb") as fh:
        step6_models = pickle.load(fh)

# Evaluate Step 6 baselines on val set
for name, mdl in step6_models.items():
    m_tmp = eval_model(mdl)
    tuned_results[name] = m_tmp
    roc_y = mdl.predict_proba(X_val)[:, 1]
    fp_arr, tp_arr, _ = roc_curve(y_val, roc_y)
    roc_data[name] = {"fpr": fp_arr.tolist(), "tpr": tp_arr.tolist(),
                      "auc": m_tmp["roc_auc"]}

# Add tuned models
tuned_new = {}
if best_xgb_model  is not None: tuned_new["XGBoost_tuned"]  = (best_xgb_model,  best_xgb_m)
if best_lgbm_model is not None: tuned_new["LightGBM_tuned"] = (best_lgbm_model, best_lgbm_m)
if best_cb_model   is not None: tuned_new["CatBoost_tuned"] = (best_cb_model,   best_cb_m)

for name, (mdl, m_tmp) in tuned_new.items():
    tuned_results[name] = m_tmp
    tuned_models[name]  = mdl
    roc_y = mdl.predict_proba(X_val)[:, 1]
    fp_arr, tp_arr, _ = roc_curve(y_val, roc_y)
    roc_data[name] = {"fpr": fp_arr.tolist(), "tpr": tp_arr.tolist(),
                      "auc": m_tmp["roc_auc"]}

# Print comparison table
all_names = list(tuned_results.keys())
col_w = 25
header  = f"  {'Metric':<{col_w}}"
divider = "  " + "-" * (col_w + 17 * len(all_names))
for nm in all_names:
    header += f" {nm:>16}"

log(header, 0)
log(divider, 0)

METRICS_PRINT = ["accuracy", "precision", "recall", "f1",
                 "roc_auc", "fpr", "fnr", "infer_ms_per_sample"]

for metric in METRICS_PRINT:
    row = f"  {metric:<{col_w}}"
    for nm in all_names:
        val = tuned_results[nm].get(metric, float("nan"))
        if metric in ("accuracy", "precision", "recall", "f1", "fpr", "fnr"):
            row += f" {val*100:>15.3f}%"
        elif metric == "roc_auc":
            row += f" {val:>16.6f}"
        else:
            row += f" {val:>15.4f}ms"
    log(row, 0)

log(divider, 0)

# FPR constraint annotation
log("", 0)
log(f"  FPR constraint : < {FPR_LIMIT*100:.1f}%   (OK = feasible,  -- = violates)", 1)
constraint_row = f"  {'FPR feasible':<{col_w}}"
for nm in all_names:
    feasible = "OK" if tuned_results[nm]["fpr"] < FPR_LIMIT else "--"
    constraint_row += f" {feasible:>16}"
log(constraint_row, 0)


# =============================================================================
# SECTION E -- CHAMPION SELECTION
# =============================================================================
section("CHAMPION SELECTION  (lowest FNR with FPR < 1.5%)")

feasible_names = [nm for nm in all_names
                  if tuned_results[nm]["fpr"] < FPR_LIMIT]

if feasible_names:
    champion_name = min(feasible_names,
                        key=lambda nm: tuned_results[nm]["fnr"])
else:
    log("  [WARN] No model satisfies FPR < 1.5%. Selecting by lowest FPR.", 1)
    champion_name = min(all_names,
                        key=lambda nm: tuned_results[nm]["fpr"])

champ_m = tuned_results[champion_name]
log(f"  CHAMPION  :  {champion_name}", 1)
log(f"  FNR       :  {champ_m['fnr']*100:.3f}%  "
    f"(missed malware out of all malware)", 1)
log(f"  FPR       :  {champ_m['fpr']*100:.3f}%  "
    f"(false alarms out of all legit)", 1)
log(f"  ROC-AUC   :  {champ_m['roc_auc']:.6f}", 1)
log(f"  F1        :  {champ_m['f1']*100:.3f}%", 1)
log(f"  Precision :  {champ_m['precision']*100:.3f}%", 1)
log(f"  Recall    :  {champ_m['recall']*100:.3f}%", 1)
log(f"  CM        :  TP={champ_m['TP']} TN={champ_m['TN']} "
    f"FP={champ_m['FP']} FN={champ_m['FN']}", 1)
log(f"  Infer     :  {champ_m['infer_ms_per_sample']:.4f} ms/sample", 1)

# Save champion model object
if champion_name in tuned_models:
    champion_model = tuned_models[champion_name]
elif champion_name in step6_models:
    champion_model = step6_models[champion_name]
else:
    champion_model = None

if champion_model is not None:
    best_pkl = os.path.join(ARTIFACTS_DIR, "tuned_best_model.pkl")
    with open(best_pkl, "wb") as fh:
        pickle.dump(champion_model, fh)
    log(f"\n  [SAVED] tuned_best_model.pkl  ({champion_name})  ->  {best_pkl}", 1)

# Save best params for downstream use
best_param_record = {
    "champion":       champion_name,
    "params":         (best_xgb_params  if "XGBoost"   in champion_name
                       else best_lgbm_params if "LightGBM" in champion_name
                       else best_cb_params   if "CatBoost" in champion_name
                       else "see original train.py"),
    "val_metrics":    champ_m,
    "fpr_constraint": FPR_LIMIT,
    "objective":      "min_fnr",
    "note":           ("tuned_best_model.pkl is the model going forward. "
                       "Test set evaluation in Step 8."),
}
bp_json = os.path.join(ARTIFACTS_DIR, "tuned_best_params.json")
with open(bp_json, "w") as fh:
    json.dump(best_param_record, fh, indent=2)
log(f"  [SAVED] tuned_best_params.json", 1)

# Save full tuned metrics
tv_json = os.path.join(ARTIFACTS_DIR, "tuned_val_metrics.json")
with open(tv_json, "w") as fh:
    json.dump({"champion": champion_name, "models": tuned_results}, fh, indent=2)
log(f"  [SAVED] tuned_val_metrics.json", 1)


# =============================================================================
# SECTION F -- PLOTS
# =============================================================================
section("PLOTS")

# -- Plot 07 : Tuned ROC curves -----------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#1a1a2e")
ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random")

for nm, rd in roc_data.items():
    color = MODEL_COLORS.get(nm, "#aaa")
    is_champ = (nm == champion_name)
    lw = 2.8 if is_champ else 1.4
    ls = "-"  if is_champ else "--"
    label = f"{nm}  (AUC={rd['auc']:.5f})"
    if is_champ:
        label += "  [CHAMPION]"
    ax.plot(rd["fpr"], rd["tpr"], color=color, lw=lw, ls=ls, label=label)

ax.axvline(x=FPR_LIMIT, color="#FF6B6B", linestyle=":", lw=1.5,
           alpha=0.7, label=f"FPR limit ({FPR_LIMIT*100:.0f}%)")
ax.set_xlabel("False Positive Rate", fontsize=12, color="#e0e0e0")
ax.set_ylabel("True Positive Rate",  fontsize=12, color="#e0e0e0")
ax.set_title("ROC Curves -- Step 7 Final Comparison (Validation Set)",
             fontsize=13, fontweight="bold", color="#e0e0e0")
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
_p7 = os.path.join(PLOTS_DIR, "07_tuned_roc_curves.png")
plt.savefig(_p7, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {_p7}", 1)

# -- Plot 08 : Tuned confusion matrices ---------------------------------------
all_model_objs = {**step6_models, **tuned_models}
n_models = len(tuned_results)
n_cols   = min(n_models, 3)
n_rows   = (n_models + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(7 * n_cols, 5.5 * n_rows))
fig.patch.set_facecolor("#1a1a2e")
axes_flat = np.array(axes).flatten() if n_models > 1 else [axes]

for idx, nm in enumerate(tuned_results.keys()):
    ax = axes_flat[idx]
    mdl_obj = all_model_objs.get(nm)
    if mdl_obj is None:
        ax.set_visible(False)
        continue
    y_prob_tmp = mdl_obj.predict_proba(X_val)[:, 1]
    y_pred_tmp = (y_prob_tmp >= 0.5).astype(int)
    cm_tmp = confusion_matrix(y_val, y_pred_tmp)
    sns.heatmap(
        cm_tmp,
        annot=True, fmt="d", cmap="Blues",
        ax=ax, cbar=False,
        linewidths=0.5, linecolor="#1a1a2e",
        annot_kws={"size": 12, "weight": "bold"},
    )
    title_str = nm + ("  [CHAMPION]" if nm == champion_name else "")
    ax.set_title(title_str, fontsize=10, fontweight="bold", color="#e0e0e0")
    ax.set_xlabel("Predicted",  color="#e0e0e0")
    ax.set_ylabel("True Label", color="#e0e0e0")
    ax.set_xticklabels(["Malware (0)", "Legit (1)"], color="#ccc")
    ax.set_yticklabels(["Malware (0)", "Legit (1)"], color="#ccc", rotation=0)

for j in range(len(tuned_results), len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle("Confusion Matrices -- Step 7 Final Comparison (Validation Set)",
             fontsize=13, fontweight="bold", color="#e0e0e0", y=1.01)
plt.tight_layout()
_p8 = os.path.join(PLOTS_DIR, "08_tuned_confusion_matrices.png")
plt.savefig(_p8, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {_p8}", 1)

# -- Plot 10 : FNR vs FPR scatter (all grid combos) ---------------------------
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("#1a1a2e")

if xgb_records:
    _df = pd.DataFrame(xgb_records)
    ok = _df["fpr"] < FPR_LIMIT
    if (~ok).any():
        ax.scatter(_df.loc[~ok, "fpr"]*100, _df.loc[~ok, "fnr"]*100,
                   c="#555", s=20, alpha=0.4, label="XGBoost (infeasible)")
    if ok.any():
        ax.scatter(_df.loc[ok, "fpr"]*100, _df.loc[ok, "fnr"]*100,
                   c="#7ED06E", s=30, alpha=0.8, label="XGBoost (feasible)")

if lgbm_records:
    _df2 = pd.DataFrame(lgbm_records)
    ok2 = _df2["fpr"] < FPR_LIMIT
    if ok2.any():
        ax.scatter(_df2.loc[ok2, "fpr"]*100, _df2.loc[ok2, "fnr"]*100,
                   c="#E8A838", s=30, alpha=0.8, label="LightGBM (feasible)")

if cb_records:
    _df3 = pd.DataFrame(cb_records)
    ok3 = _df3["fpr"] < FPR_LIMIT
    if ok3.any():
        ax.scatter(_df3.loc[ok3, "fpr"]*100, _df3.loc[ok3, "fnr"]*100,
                   c="#D65780", s=30, alpha=0.8, label="CatBoost (feasible)")

ax.scatter([champ_m["fpr"]*100], [champ_m["fnr"]*100],
           c="#FFD700", s=220, zorder=5, marker="*",
           label=f"Champion ({champion_name})")
ax.axvline(x=FPR_LIMIT*100, color="#FF6B6B", lw=1.5, ls=":",
           alpha=0.7, label="FPR limit (1.5%)")
ax.set_xlabel("FPR (%)", fontsize=12, color="#e0e0e0")
ax.set_ylabel("FNR (%)", fontsize=12, color="#e0e0e0")
ax.set_title("FNR vs FPR -- All Grid Combinations (Validation Set)",
             fontsize=12, fontweight="bold", color="#e0e0e0")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
_p10 = os.path.join(PLOTS_DIR, "10_fnr_fpr_scatter.png")
plt.savefig(_p10, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {_p10}", 1)


# =============================================================================
# FINAL BANNER
# =============================================================================
log()
log(SEP2)
log("  STEP 7 - HYPERPARAMETER TUNING + FLAGSHIP COMPARISON COMPLETE")
log(f"  Champion model : {champion_name}")
log(f"  FNR            : {champ_m['fnr']*100:.3f}%  "
    f"(missed malware, primary objective)")
log(f"  FPR            : {champ_m['fpr']*100:.3f}%  "
    f"(constraint < {FPR_LIMIT*100:.1f}%  "
    f"{'PASSED' if champ_m['fpr'] < FPR_LIMIT else 'FAILED'})")
log(f"  ROC-AUC        : {champ_m['roc_auc']:.6f}")
log("")
log("  Grid artifacts :")
log("    xgb_grid_results.csv  |  lgbm_grid_results.csv  |  cb_grid_results.csv")
log("  Model artifacts:")
log("    tuned_xgb.pkl  |  tuned_lgbm.pkl  |  tuned_catboost.pkl")
log(f"    tuned_best_model.pkl  ({champion_name})")
log("  Plots:")
log("    07_tuned_roc_curves.png  |  08_tuned_confusion_matrices.png")
log("    09_xgb_grid_heatmap.png  |  10_fnr_fpr_scatter.png")
log("")
log("  NEXT STEPS:")
log("    Step 8  : Evaluate champion on HELD-OUT TEST SET (first time test touched)")
log("    Step 9  : Probability calibration + optimal threshold selection")
log("    Step 10 : SHAP explainability")
log(f"  Finished : {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(SEP2)

# Write report
with open(REPORT_FILE, "w", encoding="utf-8") as fh:
    fh.write("\n".join(report_lines))

print(f"\n[REPORT] Written to: {REPORT_FILE}")
