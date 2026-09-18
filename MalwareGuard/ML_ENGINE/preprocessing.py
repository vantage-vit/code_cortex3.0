"""
================================================================================
  CODE CORTEX 3.0  -  STEPS 3 / 4 / 5 : SPLIT -> PREPROCESS -> FEATURE ENG.
  Member 1 | ML / Research
================================================================================

Flow (strictly in this order):
  Step 3 : Stratified 70 / 15 / 15 train / val / test split
           Split FIRST, before any statistics are computed on features.

  Step 4 : StandardScaler fit on X_train ONLY.
           Transform X_val and X_test with the same fitted scaler.
           (prevents data leakage from val/test into the scaler)

  Step 5 : Feature Engineering  (all decisions based on X_train only)
           a) Near-constant filter  - drop features where >95% of X_train
                                      values are a single value.
           b) High-correlation drop - compute Pearson r on X_train_scaled;
                                      for each pair with |r| > 0.95 keep the
                                      feature with higher variance and drop
                                      the other. Same drops applied to val/test.
           c) Derived features      - domain-specific PE-header ratios created
                                      BEFORE scaling (same formula, no fitting).
           d) Final feature list locked -> feature_columns.json

Run:
    python ml/preprocessing.py

Reads:
    data/processed_malware.csv
    data/labels.csv

Outputs  (ml/artifacts/):
    X_train.csv  X_val.csv  X_test.csv
    y_train.csv  y_val.csv  y_test.csv
    split_indices.json        <- row positions in original dataset
    preprocessor.pkl          <- fitted StandardScaler
    feature_columns.json      <- final locked feature list
    dropped_features.json     <- what was dropped and why
    reports/preprocessing_report.txt
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# -- Paths --------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
ML_DIR        = os.path.join(BASE_DIR, "ml")
ARTIFACTS_DIR = os.path.join(ML_DIR, "artifacts")
REPORTS_DIR   = os.path.join(ML_DIR, "reports")

for d in [ARTIFACTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

REPORT_FILE = os.path.join(REPORTS_DIR, "preprocessing_report.txt")

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

# =============================================================================
log(SEP2)
log("  CODE CORTEX 3.0  -  STEPS 3-4-5 : SPLIT / PREPROCESS / FEATURE ENG.")
log(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(SEP2)

# -- Load clean data ----------------------------------------------------------
section("LOAD  -  Reading processed_malware.csv + labels.csv")

X_raw = pd.read_csv(os.path.join(DATA_DIR, "processed_malware.csv"))
y_raw = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))["legitimate"]

assert len(X_raw) == len(y_raw), "Row mismatch!"
log(f"Loaded : {X_raw.shape[0]:,} rows x {X_raw.shape[1]} features", 1)
log(f"Labels : {len(y_raw):,}  (0=malware {int((y_raw==0).sum()):,} "
    f"| 1=legit {int((y_raw==1).sum()):,})", 1)

# =============================================================================
# STEP 5a - DERIVED FEATURES (created on raw values, before any split/scale)
# Applied identically to all splits -> no fitting needed.
# =============================================================================
section("STEP 5a  -  DERIVED FEATURE CREATION  (pre-split, no fitting)")

X = X_raw.copy()
eps = 1e-9   # small constant to avoid division by zero

# entropy_range : how wide the entropy spread across sections
X["entropy_range"] = X["SectionsMaxEntropy"] - X["SectionsMinEntropy"]

# import_density : average imports per section (import-heavy = suspicious)
X["import_density"] = X["ImportsNb"] / (X["SectionsNb"] + eps)

# resource_entropy_spread : entropy variation in resource section
X["resource_entropy_spread"] = (X["ResourcesMaxEntropy"]
                                 - X["ResourcesMinEntropy"])

# code_to_image_ratio : fraction of PE image that is actual code
X["code_to_image_ratio"] = X["SizeOfCode"] / (X["SizeOfImage"] + eps)

# section_fill_ratio : how densely packed the sections are
X["section_fill_ratio"] = (X["SectionsMeanRawsize"]
                            / (X["SectionMaxRawsize"] + eps))

derived_features = [
    "entropy_range",
    "import_density",
    "resource_entropy_spread",
    "code_to_image_ratio",
    "section_fill_ratio",
]

log(f"Derived features added : {len(derived_features)}", 1)
for df_name in derived_features:
    log(f"  + {df_name}", 2)
log(f"Total features now     : {X.shape[1]}", 1)

# =============================================================================
# STEP 3 - STRATIFIED TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# Split HERE before any fitting. Indices saved for identifier join later.
# =============================================================================
section("STEP 3  -  STRATIFIED SPLIT  70 / 15 / 15")

RANDOM_STATE = 42

# First cut: 70% train | 30% temp
X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
    X, y_raw, X.index.tolist(),
    test_size=0.30,
    stratify=y_raw,
    random_state=RANDOM_STATE,
)

# Second cut: 15% val | 15% test  (50/50 of the 30% temp)
X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
    X_temp, y_temp, idx_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE,
)

# Reset indices (all three splits are now 0-indexed internally)
for df in [X_train, X_val, X_test]:
    df.reset_index(drop=True, inplace=True)
y_train = y_train.reset_index(drop=True)
y_val   = y_val.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

log(f"  Total  : {len(X):,} samples", 1)
log(f"  Train  : {len(X_train):,}  ({len(X_train)/len(X)*100:.1f}%)"
    f"  |  Malware {int((y_train==0).sum()):,}  Legit {int((y_train==1).sum()):,}", 1)
log(f"  Val    : {len(X_val):,}  ({len(X_val)/len(X)*100:.1f}%)"
    f"  |  Malware {int((y_val==0).sum()):,}  Legit {int((y_val==1).sum()):,}", 1)
log(f"  Test   : {len(X_test):,}  ({len(X_test)/len(X)*100:.1f}%)"
    f"  |  Malware {int((y_test==0).sum()):,}  Legit {int((y_test==1).sum()):,}", 1)

# Verify class proportions are maintained
for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    mal_pct = (y_split == 0).mean() * 100
    leg_pct = (y_split == 1).mean() * 100
    log(f"  {split_name} class %  ->  Malware {mal_pct:.1f}%  |  Legit {leg_pct:.1f}%", 2)

# Save original row indices for identifier alignment
split_indices = {
    "train": [int(i) for i in idx_train],
    "val":   [int(i) for i in idx_val],
    "test":  [int(i) for i in idx_test],
}
with open(os.path.join(ARTIFACTS_DIR, "split_indices.json"), "w") as f:
    json.dump(split_indices, f)
log("  [SAVED] split_indices.json -> ml/artifacts/", 1)

# =============================================================================
# STEP 5b - NEAR-CONSTANT FEATURE REMOVAL  (decided on X_train only)
# Drop features where >95% of training samples have the SAME single value.
# =============================================================================
section("STEP 5b  -  NEAR-CONSTANT FEATURE REMOVAL  (threshold: 95%)")

NEAR_CONST_THRESHOLD = 0.95
near_const_drops = []

for col in X_train.columns:
    top_freq = X_train[col].value_counts(normalize=True).iloc[0]
    if top_freq >= NEAR_CONST_THRESHOLD:
        near_const_drops.append(col)
        log(f"  DROP '{col}'  -> top value covers {top_freq*100:.1f}% of training set", 1)

if not near_const_drops:
    log("  No near-constant features found (all features vary sufficiently).", 1)

X_train.drop(columns=near_const_drops, inplace=True)
X_val.drop(columns=near_const_drops,   inplace=True)
X_test.drop(columns=near_const_drops,  inplace=True)
log(f"  Features remaining : {X_train.shape[1]}", 1)

# =============================================================================
# STEP 4 - STANDARD SCALING  (fit on X_train ONLY)
# =============================================================================
section("STEP 4  -  STANDARD SCALER  (fit on train, transform val+test)")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
)

log(f"  Scaler fit on X_train  ({X_train_scaled.shape[0]:,} rows)", 1)
log(f"  Transformed X_val      ({X_val_scaled.shape[0]:,} rows)", 1)
log(f"  Transformed X_test     ({X_test_scaled.shape[0]:,} rows)", 1)
log("  All features now have mean~0, std~1  (train-derived statistics).", 1)

# Save scaler
scaler_path = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
log(f"  [SAVED] preprocessor.pkl -> {scaler_path}", 1)

# =============================================================================
# STEP 5c - HIGH-CORRELATION DROP  (computed on X_train_scaled)
# For each pair |r| > 0.95, drop the feature with LOWER variance on train.
# =============================================================================
section("STEP 5c  -  HIGH-CORRELATION FEATURE DROP  (threshold |r| > 0.95)")

CORR_THRESHOLD = 0.95
corr_matrix    = X_train_scaled.corr(method="pearson").abs()

# Upper triangle mask
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
)

corr_drop_set   = set()
corr_drop_pairs = []

cols_list = X_train_scaled.columns.tolist()
train_var = X_train_scaled.var()

for col in upper.columns:
    if col in corr_drop_set:
        continue
    high_corr = upper[col][upper[col] > CORR_THRESHOLD].index.tolist()
    for partner in high_corr:
        if partner in corr_drop_set:
            continue
        # Keep the one with higher variance; drop the other
        if train_var[col] >= train_var[partner]:
            to_drop = partner
        else:
            to_drop = col
        corr_drop_set.add(to_drop)
        corr_drop_pairs.append({
            "kept": col if to_drop == partner else partner,
            "dropped": to_drop,
            "r": float(corr_matrix.loc[col, partner]),
        })

if corr_drop_pairs:
    log(f"  Pairs above threshold ({len(corr_drop_pairs)}):", 1)
    log(f"  {'Kept':<35} {'Dropped':<35} {'r':>8}", 1)
    log("  " + "-" * 82, 1)
    for rec in corr_drop_pairs:
        log(f"  {rec['kept']:<35} {rec['dropped']:<35} {rec['r']:>8.4f}", 1)
else:
    log(f"  No feature pairs exceed |r| > {CORR_THRESHOLD} threshold.", 1)

corr_drops = list(corr_drop_set)
X_train_scaled.drop(columns=corr_drops, inplace=True)
X_val_scaled.drop(columns=corr_drops,   inplace=True)
X_test_scaled.drop(columns=corr_drops,  inplace=True)

log(f"\n  Features removed : {len(corr_drops)}", 1)
if corr_drops:
    log(f"  {corr_drops}", 2)
log(f"  Features remaining : {X_train_scaled.shape[1]}", 1)

# =============================================================================
# STEP 5d - LOCK FINAL FEATURE LIST
# =============================================================================
section("STEP 5d  -  LOCK FINAL FEATURE LIST")

FINAL_FEATURES = X_train_scaled.columns.tolist()

log(f"  Final feature count : {len(FINAL_FEATURES)}", 1)
log(f"  Original (post-clean): {X_raw.shape[1]}", 1)
log(f"  + Derived features   : {len(derived_features)}", 1)
log(f"  - Near-constant      : {len(near_const_drops)}", 1)
log(f"  - High-corr drops    : {len(corr_drops)}", 1)
log(f"  = Final              : {len(FINAL_FEATURES)}", 1)
log("", 1)
log("  Final feature list:", 1)
for i, f in enumerate(FINAL_FEATURES, 1):
    log(f"  {i:>3}. {f}", 2)

# Save feature list
feat_path = os.path.join(ARTIFACTS_DIR, "feature_columns.json")
with open(feat_path, "w") as f:
    json.dump(FINAL_FEATURES, f, indent=2)
log(f"\n  [SAVED] feature_columns.json -> {feat_path}", 1)

# Save dropped features manifest
drops_manifest = {
    "near_constant_drops": near_const_drops,
    "high_corr_drops": corr_drops,
    "high_corr_pairs": corr_drop_pairs,
    "derived_features_added": derived_features,
}
drops_path = os.path.join(ARTIFACTS_DIR, "dropped_features.json")
with open(drops_path, "w") as f:
    json.dump(drops_manifest, f, indent=2)
log(f"  [SAVED] dropped_features.json -> {drops_path}", 1)

# =============================================================================
# SAVE SPLIT DATA
# =============================================================================
section("SAVE  -  Writing train / val / test CSVs")

splits = {
    "X_train": X_train_scaled,
    "X_val":   X_val_scaled,
    "X_test":  X_test_scaled,
    "y_train": y_train.to_frame("legitimate"),
    "y_val":   y_val.to_frame("legitimate"),
    "y_test":  y_test.to_frame("legitimate"),
}

for name, df_out in splits.items():
    path = os.path.join(ARTIFACTS_DIR, f"{name}.csv")
    df_out.to_csv(path, index=False)
    log(f"  {name}.csv   -> {df_out.shape[0]:,} rows x {df_out.shape[1]} cols", 1)

# =============================================================================
# FINAL SUMMARY BANNER
# =============================================================================
log()
log(SEP2)
log("  STEPS 3-4-5  COMPLETE  -  FINAL SUMMARY")
log(SEP2)
log(f"  Rows (total)       : {len(X):,}", 1)
log(f"  Train              : {len(X_train_scaled):,}  (70%)", 1)
log(f"  Val                : {len(X_val_scaled):,}  (15%)", 1)
log(f"  Test               : {len(X_test_scaled):,}  (15%)", 1)
log(f"  Original features  : {X_raw.shape[1]}", 1)
log(f"  + Derived features : {len(derived_features)}", 1)
log(f"  - Near-constant    : {len(near_const_drops)}", 1)
log(f"  - High-corr drops  : {len(corr_drops)}", 1)
log(f"  = FINAL features   : {len(FINAL_FEATURES)}", 1)
log("  Artifacts saved:", 1)
for name in ["X_train.csv", "X_val.csv", "X_test.csv",
             "y_train.csv", "y_val.csv", "y_test.csv",
             "preprocessor.pkl", "feature_columns.json",
             "dropped_features.json", "split_indices.json"]:
    log(f"    ml/artifacts/{name}", 2)
log(f"  Finished : {time.strftime('%Y-%m-%d %H:%M:%S')}", 1)
log(SEP2)

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n[REPORT] Written to: {REPORT_FILE}")
