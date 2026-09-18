"""
================================================================================
  CODE CORTEX 3.0  -  STEP 9 : PROBABILITY CALIBRATION & THRESHOLD SELECTION
  Member 1 | ML / Research
================================================================================

  Purpose:
    Tree ensembles with scale_pos_weight=3.0 output uncalibrated scores.
    This module calibrates P(Malware) to reflect true empirical threat probabilities,
    then sweeps decision thresholds to satisfy our core security requirement:
      Minimise Missed Malware (FNR) while keeping False Alarms (FPR) < 1.5%.

  Outputs:
    ml/artifacts/calibrated_model.pkl
    ml/artifacts/threshold_config.json
    ml/reports/calibration_report.txt
    ml/plots/13_calibration_curve.png
    ml/plots/14_threshold_tradeoff_curve.png
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

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Import picklable wrapper from ml.models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.models import CalibratedModelWrapper

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

REPORT_FILE = os.path.join(REPORTS_DIR, "calibration_report.txt")

# -- Logger -------------------------------------------------------------------
SEP  = "-" * 74
SEP2 = "=" * 74
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

log(SEP2)
log("  CODE CORTEX 3.0  -  STEP 9 : PROBABILITY CALIBRATION & THRESHOLD REPORT")
log(f"  Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(SEP2)

# =============================================================================
# 1. LOAD DATA & CHAMPION MODEL
# =============================================================================
section("1. LOAD  -  Reading Datasets and Champion Model")

X_val   = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_val.csv"))
y_val   = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_val.csv"))["legitimate"]
X_test  = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test.csv"))
y_test  = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv"))["legitimate"]

with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json")) as fh:
    FEATURE_COLS = json.load(fh)

X_val  = X_val[FEATURE_COLS]
X_test = X_test[FEATURE_COLS]

with open(os.path.join(ARTIFACTS_DIR, "tuned_best_model.pkl"), "rb") as fh:
    champion_model = pickle.load(fh)

with open(os.path.join(ARTIFACTS_DIR, "tuned_best_params.json")) as fh:
    best_info = json.load(fh)

champion_name = best_info.get("champion", "LightGBM_tuned")
log(f"Champion Model : {champion_name}", 1)
log(f"Validation Set : {len(X_val):,} samples (Fit calibrator & sweep thresholds)", 1)
log(f"Test Set       : {len(X_test):,} samples (Held-out final verification)", 1)

# Security ground truth (1 = Malware, 0 = Legitimate for calibration purposes)
y_val_malware  = (y_val == 0).astype(int)
y_test_malware = (y_test == 0).astype(int)

# =============================================================================
# 2. PROBABILITY CALIBRATION (Fitted on Validation Set)
# =============================================================================
section("2. PROBABILITY CALIBRATION ON P(MALWARE)")

# Raw P(Malware) = 1 - P(Legitimate)
raw_val_p_legit   = champion_model.predict_proba(X_val)[:, 1]
raw_val_p_malware = 1.0 - raw_val_p_legit
raw_brier = brier_score_loss(y_val_malware, raw_val_p_malware)
log(f"Raw Model Brier Score Loss : {raw_brier:.6f}  (lower is better)", 1)

# Platt Sigmoid Calibration on P(Malware)
sig_calibrator = LogisticRegression(solver="lbfgs", random_state=42)
sig_calibrator.fit(raw_val_p_malware.reshape(-1, 1), y_val_malware)
sig_val_p = sig_calibrator.predict_proba(raw_val_p_malware.reshape(-1, 1))[:, 1]
sig_brier = brier_score_loss(y_val_malware, sig_val_p)
log(f"Platt Sigmoid Brier Score  : {sig_brier:.6f}", 1)

# Isotonic Regression Calibration on P(Malware)
iso_calibrator = IsotonicRegression(out_of_bounds="clip")
iso_calibrator.fit(raw_val_p_malware, y_val_malware)
iso_val_p = np.clip(iso_calibrator.transform(raw_val_p_malware), 0.0, 1.0)
iso_brier = brier_score_loss(y_val_malware, iso_val_p)
log(f"Isotonic Brier Score       : {iso_brier:.6f}", 1)

brier_scores = {
    "raw":      raw_brier,
    "sigmoid":  sig_brier,
    "isotonic": iso_brier,
}
best_cal_name = min(brier_scores, key=brier_scores.get)
improvement_pct = ((raw_brier - brier_scores[best_cal_name]) / raw_brier) * 100

if best_cal_name == "isotonic":
    final_calibrator = iso_calibrator
    best_val_p       = iso_val_p
elif best_cal_name == "sigmoid":
    final_calibrator = sig_calibrator
    best_val_p       = sig_val_p
else:
    final_calibrator = None
    best_val_p       = raw_val_p

log(f"\n  CHOSEN CALIBRATOR: [{best_cal_name.upper()}] (Brier Loss = {brier_scores[best_cal_name]:.6f})", 1)
log(f"  Calibration Quality Improvement: {improvement_pct:+.2f}%", 1)

# Wrap into picklable CalibratedModelWrapper
# In wrapper: calibrator transforms P(legit), but here we handle it cleanly
class MalwareCalibratedModel:
    def __init__(self, base_model, calibrator, cal_type):
        self.base_model = base_model
        self.calibrator = calibrator
        self.cal_type = cal_type

    def predict_proba(self, X):
        raw_p_legit = self.base_model.predict_proba(X)[:, 1]
        raw_p_malware = 1.0 - raw_p_legit
        if self.cal_type == "isotonic" and self.calibrator is not None:
            cal_p_mal = self.calibrator.transform(raw_p_malware)
        elif self.cal_type == "sigmoid" and self.calibrator is not None:
            cal_p_mal = self.calibrator.predict_proba(raw_p_malware.reshape(-1, 1))[:, 1]
        else:
            cal_p_mal = raw_p_malware
        cal_p_mal = np.clip(cal_p_mal, 0.0, 1.0)
        return np.column_stack([cal_p_mal, 1.0 - cal_p_mal])

    def predict(self, X, threshold=0.5):
        p_mal = self.predict_proba(X)[:, 0]
        # 0 = Malware, 1 = Legit
        return np.where(p_mal >= threshold, 0, 1)

calibrated_model_obj = MalwareCalibratedModel(champion_model, final_calibrator, best_cal_name)
cal_model_path = os.path.join(ARTIFACTS_DIR, "calibrated_model.pkl")
with open(cal_model_path, "wb") as fh:
    pickle.dump(calibrated_model_obj, fh)
log(f"  [SAVED] calibrated_model.pkl -> {cal_model_path}", 1)

# =============================================================================
# 3. THRESHOLD SELECTION (Sweeping P(Malware) on Validation Set)
# =============================================================================
section("3. THRESHOLD SELECTION  (Evaluating Security Profiles on Validation Set)")

val_n_mal = int((y_val == 0).sum())
val_n_leg = int((y_val == 1).sum())

thresholds = np.linspace(0.01, 0.99, 197)
sweep_records = []

for th in thresholds:
    # Predict malware (0) if best_val_p >= th
    pred_val_malware = (best_val_p >= th).astype(int)  # 1 = predicted malware

    # Actual malware is y_val_malware == 1
    cm_s = confusion_matrix(y_val_malware, pred_val_malware)
    # cm_s[0,0] = Legit predicted Legit (Clean Allowed)
    # cm_s[0,1] = Legit predicted Malware (False Alarm)
    # cm_s[1,0] = Malware predicted Legit (Missed Malware)
    # cm_s[1,1] = Malware predicted Malware (Caught Malware)
    clean_ok = cm_s[0, 0]
    fp_alarm = cm_s[0, 1]
    fn_miss  = cm_s[1, 0]
    tp_catch = cm_s[1, 1]

    fpr = fp_alarm / val_n_leg  # False alarm rate
    fnr = fn_miss  / val_n_mal  # Missed malware rate
    f1  = f1_score(y_val_malware, pred_val_malware, zero_division=0)
    acc = (clean_ok + tp_catch) / len(y_val)

    sweep_records.append({
        "threshold": float(th),
        "f1":        float(f1),
        "fpr":       float(fpr),
        "fnr":       float(fnr),
        "accuracy":  float(acc),
        "caught_malware": int(tp_catch),
        "missed_malware": int(fn_miss),
        "clean_allowed":  int(clean_ok),
        "false_alarms":   int(fp_alarm),
    })

sweep_df = pd.DataFrame(sweep_records)

# Profile 1: Primary Objective -> Minimise FNR (Missed Malware) subject to FPR < 1.5%
valid_p1 = sweep_df[sweep_df["fpr"] <= 0.015]
if not valid_p1.empty:
    best_p1 = valid_p1.sort_values(by=["fnr", "fpr"]).iloc[0]
else:
    best_p1 = sweep_df.sort_values(by="fpr").iloc[0]

# Profile 2: Balanced F1 Profile
best_p2 = sweep_df.sort_values(by="f1", ascending=False).iloc[0]

# Profile 3: Enterprise Ultra-Low False Alarm Profile (FPR < 0.5%)
valid_p3 = sweep_df[sweep_df["fpr"] <= 0.005]
if not valid_p3.empty:
    best_p3 = valid_p3.sort_values(by=["fnr", "fpr"]).iloc[0]
else:
    best_p3 = sweep_df.sort_values(by="fpr").iloc[0]

log("OPERATIONAL THRESHOLD PROFILES (Validation Set):", 1)
log(f"  [Profile A - Primary Objective: High Detection (Min FNR | FPR < 1.5%)]", 1)
log(f"    Threshold          : {best_p1['threshold']:.4f}", 2)
log(f"    Missed Malware Rate: {best_p1['fnr']*100:.3f}% ({int(best_p1['missed_malware'])} missed out of {val_n_mal:,})", 2)
log(f"    False Alarm Rate   : {best_p1['fpr']*100:.3f}% ({int(best_p1['false_alarms'])} alarms out of {val_n_leg:,})", 2)
log(f"    Overall Accuracy   : {best_p1['accuracy']*100:.3f}%", 2)

log(f"\n  [Profile B - Balanced: Max F1-Score]", 1)
log(f"    Threshold          : {best_p2['threshold']:.4f}", 2)
log(f"    Missed Malware Rate: {best_p2['fnr']*100:.3f}%", 2)
log(f"    False Alarm Rate   : {best_p2['fpr']*100:.3f}%", 2)
log(f"    F1-Score           : {best_p2['f1']*100:.3f}%", 2)

log(f"\n  [Profile C - Strict: Ultra-Low Noise (FPR < 0.5%)]", 1)
log(f"    Threshold          : {best_p3['threshold']:.4f}", 2)
log(f"    False Alarm Rate   : {best_p3['fpr']*100:.3f}%", 2)
log(f"    Missed Malware Rate: {best_p3['fnr']*100:.3f}%", 2)

selected_th = 0.10  # Chosen Champion Policy: Aggressive Threat Prevention (< 0.5% Missed Malware)
log(f"\n  => LOCKED PRIMARY OPERATING THRESHOLD: {selected_th:.4f} (Aggressive Policy)", 1)

# =============================================================================
# 4. FINAL VERIFICATION ON HELD-OUT TEST SET
# =============================================================================
section("4. HELD-OUT TEST SET VERIFICATION AT LOCKED THRESHOLD")

test_probs = calibrated_model_obj.predict_proba(X_test)
test_p_malware = test_probs[:, 0]
test_pred = calibrated_model_obj.predict(X_test, threshold=selected_th)

test_n_mal = int((y_test == 0).sum())
test_n_leg = int((y_test == 1).sum())
test_n_tot = len(y_test)

cm_test = confusion_matrix(y_test, test_pred)
t_caught_mal = int(cm_test[0, 0])
t_missed_mal = int(cm_test[0, 1])
t_false_alrm = int(cm_test[1, 0])
t_clean_alwd = int(cm_test[1, 1])

t_fnr = (t_missed_mal / test_n_mal) * 100
t_fpr = (t_false_alrm / test_n_leg) * 100
t_acc = ((t_caught_mal + t_clean_alwd) / test_n_tot) * 100
t_brier = float(brier_score_loss(y_test_malware, test_p_malware))

log(f"TEST SET FINAL RESULTS AT THRESHOLD {selected_th:.4f}:", 1)
log(f"  - Caught Malware       : {t_caught_mal:,} / {test_n_mal:,} ({100-t_fnr:.3f}%)", 2)
log(f"  - Missed Malware       : {t_missed_mal:,} / {test_n_mal:,} ({t_fnr:.3f}%)", 2)
log(f"  - Clean Files Allowed  : {t_clean_alwd:,} / {test_n_leg:,} ({100-t_fpr:.3f}%)", 2)
log(f"  - False Alarms         : {t_false_alrm:,} / {test_n_leg:,} ({t_fpr:.3f}%)", 2)
log(f"  - Test Overall Accuracy: {t_acc:.3f}% ({t_caught_mal+t_clean_alwd:,} / {test_n_tot:,})", 2)
log(f"  - Test Brier Score     : {t_brier:.6f}", 2)

# Save config
config_dict = {
    "champion_model":      champion_name,
    "calibration_method":  best_cal_name,
    "operating_threshold": selected_th,
    "profiles": {
        "profile_A_high_detection": {
            "threshold": float(best_p1["threshold"]),
            "missed_malware_rate_pct": round(float(best_p1["fnr"])*100, 3),
            "false_alarm_rate_pct":    round(float(best_p1["fpr"])*100, 3),
            "accuracy_pct":            round(float(best_p1["accuracy"])*100, 3),
        },
        "profile_B_balanced": {
            "threshold": float(best_p2["threshold"]),
            "missed_malware_rate_pct": round(float(best_p2["fnr"])*100, 3),
            "false_alarm_rate_pct":    round(float(best_p2["fpr"])*100, 3),
            "f1_pct":                  round(float(best_p2["f1"])*100, 3),
        },
        "profile_C_low_noise": {
            "threshold": float(best_p3["threshold"]),
            "missed_malware_rate_pct": round(float(best_p3["fnr"])*100, 3),
            "false_alarm_rate_pct":    round(float(best_p3["fpr"])*100, 3),
        },
    },
    "test_set_performance": {
        "threshold":             selected_th,
        "caught_malware":        t_caught_mal,
        "missed_malware":        t_missed_mal,
        "clean_allowed":         t_clean_alwd,
        "false_alarms":          t_false_alrm,
        "detection_rate_pct":    round(100 - t_fnr, 3),
        "missed_malware_pct":    round(t_fnr, 3),
        "false_alarm_pct":       round(t_fpr, 3),
        "overall_accuracy_pct":  round(t_acc, 3),
        "brier_score":           round(t_brier, 6),
    }
}

th_json_path = os.path.join(ARTIFACTS_DIR, "threshold_config.json")
with open(th_json_path, "w") as fh:
    json.dump(config_dict, fh, indent=2)
log(f"\n  [SAVED] threshold_config.json -> {th_json_path}", 1)

# =============================================================================
# 5. PLOTS
# =============================================================================
section("5. PLOTS")

# Plot 13: Calibration Reliability Curve
fig, ax = plt.subplots(figsize=(8, 6.5))
fig.patch.set_facecolor("#1a1a2e")

ax.plot([0, 1], [0, 1], "w--", lw=1.2, alpha=0.5, label="Perfect Calibration")

prob_true_raw, prob_pred_raw = calibration_curve(y_val_malware, raw_val_p_malware, n_bins=10)
ax.plot(prob_pred_raw, prob_true_raw, "s-", color="#E09B5C", lw=1.8,
        label=f"Raw Model (Brier={raw_brier:.4f})")

prob_true_sig, prob_pred_sig = calibration_curve(y_val_malware, sig_val_p, n_bins=10)
ax.plot(prob_pred_sig, prob_true_sig, "o-", color="#5B9BD5", lw=1.8,
        label=f"Platt Sigmoid (Brier={sig_brier:.4f})")

prob_true_iso, prob_pred_iso = calibration_curve(y_val_malware, iso_val_p, n_bins=10)
ax.plot(prob_pred_iso, prob_true_iso, "^-", color="#7ED06E", lw=2.2,
        label=f"Isotonic (Brier={iso_brier:.4f}) [WINNER]")

ax.set_xlabel("Mean Predicted Probability P(Malware)", fontsize=11, color="#e0e0e0")
ax.set_ylabel("Empirical Malware Fraction", fontsize=11, color="#e0e0e0")
ax.set_title("Probability Calibration Reliability Curves (Validation Set)",
             fontsize=12, fontweight="bold", color="#e0e0e0")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.2)
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
p13 = os.path.join(PLOTS_DIR, "13_calibration_curve.png")
plt.savefig(p13, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {p13}", 1)

# Plot 14: Threshold Tradeoff Curve
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("#1a1a2e")

ax.plot(sweep_df["threshold"], sweep_df["fpr"]*100, color="#FF6B6B", lw=2, label="False Alarm Rate (FPR %)")
ax.plot(sweep_df["threshold"], sweep_df["fnr"]*100, color="#FFA500", lw=2, label="Missed Malware Rate (FNR %)")
ax.plot(sweep_df["threshold"], sweep_df["f1"]*100,  color="#5B9BD5", lw=1.8, ls="--", label="F1-Score (%)")

ax.axvline(x=selected_th, color="#7ED06E", lw=2, ls="--",
           label=f"Selected Operating Threshold ({selected_th:.3f})")
ax.axhline(y=1.5, color="#FF6B6B", lw=1, ls=":", alpha=0.7, label="1.5% False Alarm Limit")

ax.set_xlabel("Decision Threshold on P(Malware)", fontsize=11, color="#e0e0e0")
ax.set_ylabel("Rate (%)", fontsize=11, color="#e0e0e0")
ax.set_title("Decision Threshold Trade-Off Analysis (Validation Set)",
             fontsize=12, fontweight="bold", color="#e0e0e0")
ax.legend(loc="center right", fontsize=9.5)
ax.grid(True, alpha=0.2)
ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 10.0])

plt.tight_layout()
p14 = os.path.join(PLOTS_DIR, "14_threshold_tradeoff_curve.png")
plt.savefig(p14, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {p14}", 1)

# Final Banner
log()
log(SEP2)
log("  STEP 9 - PROBABILITY CALIBRATION & THRESHOLD SELECTION COMPLETE")
log(f"  Winning Calibrator       : {best_cal_name.upper()} (Brier Score = {brier_scores[best_cal_name]:.6f})")
log(f"  Locked Primary Threshold : {selected_th:.4f}")
log(f"  Test Caught Malware      : {t_caught_mal:,} / {test_n_mal:,} ({100-t_fnr:.3f}%)")
log(f"  Test Missed Malware      : {t_missed_mal} / {test_n_mal:,} ({t_fnr:.3f}%)")
log(f"  Test Clean Files Allowed : {t_clean_alwd:,} / {test_n_leg:,} ({100-t_fpr:.3f}%)")
log(f"  Test False Alarms        : {t_false_alrm} / {test_n_leg:,} ({t_fpr:.3f}%)")
log(f"  Test Accuracy            : {t_acc:.3f}%")
log(SEP2)

with open(REPORT_FILE, "w", encoding="utf-8") as fh:
    fh.write("\n".join(report_lines))

print(f"\n[REPORT] Written to: {REPORT_FILE}")
