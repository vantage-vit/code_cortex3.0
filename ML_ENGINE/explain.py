"""
================================================================================
  CODE CORTEX 3.0  -  STEP 10 : SHAP EXPLAINABILITY
  Member 1 | ML / Research
================================================================================

  Purpose:
    Provide transparent, defensible Explainable AI (XAI) for cybersecurity analysts.
    Uses SHAP (SHapley Additive exPlanations) via TreeExplainer to compute exact
    feature attributions for PE binary malware detections.

  Features:
    1. Global Explainability:
       - Computes mean |SHAP| across background validation samples.
       - Identifies and locks top 5 global malware drivers into top5_features.json.
       - Plots SHAP summary bar chart and beeswarm distribution plot.

    2. Local Sample Explainability:
       - Generates local feature contributions in the exact required JSON schema:
         [{"feature": "<FeatureName>", "impact": <float>}, ...]
       - Formats top 5 risk factors driving each prediction.

    3. Fast Production Serialisation:
       - Saves fitted TreeExplainer and background reference to
         ml/artifacts/shap_explainer.pkl for zero-latency inference in predict.py.

  Outputs:
    ml/artifacts/top5_features.json
    ml/artifacts/shap_explainer.pkl
    ml/reports/explainability_report.txt
    ml/plots/15_shap_summary_bar.png
    ml/plots/16_shap_beeswarm.png
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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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

REPORT_FILE = os.path.join(REPORTS_DIR, "explainability_report.txt")

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

log(SEP2)
log("  CODE CORTEX 3.0  -  STEP 10 : SHAP EXPLAINABILITY")
log(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(SEP2)

if not SHAP_AVAILABLE:
    log("[ERROR] The 'shap' library is not installed. Run: pip install shap", 1)
    sys.exit(1)

# =============================================================================
# 1. LOAD MODEL AND DATA
# =============================================================================
section("1. LOAD  -  Reading Model and Feature Sets")

with open(os.path.join(ARTIFACTS_DIR, "tuned_best_model.pkl"), "rb") as fh:
    champion_model = pickle.load(fh)

with open(os.path.join(ARTIFACTS_DIR, "tuned_best_params.json")) as fh:
    best_info = json.load(fh)

with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json")) as fh:
    FEATURE_COLS = json.load(fh)

X_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train.csv"))[FEATURE_COLS]
X_val   = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_val.csv"))[FEATURE_COLS]
y_val   = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_val.csv"))["legitimate"]

champion_name = best_info.get("champion", "Tuned Model")
log(f"Champion Model : {champion_name}", 1)
log(f"Features       : {len(FEATURE_COLS)} locked attributes", 1)

# =============================================================================
# 2. BUILD TREE EXPLAINER & COMPUTE SHAP VALUES
# =============================================================================
section("2. SHAP TREE EXPLAINER INITIALISATION & COMPUTATION")

t0 = time.time()
explainer = shap.TreeExplainer(champion_model)
log(f"TreeExplainer created in {time.time()-t0:.2f}s", 1)

# Representative sample of validation set (2,000 stratified samples for fast, robust global estimation)
sample_size = min(2000, len(X_val))
val_sample = X_val.sample(n=sample_size, random_state=42)
y_sample   = y_val.loc[val_sample.index]

log(f"Computing SHAP values for {sample_size:,} background validation samples...", 1)
t1 = time.time()
shap_values = explainer.shap_values(val_sample)
compute_time = time.time() - t1
log(f"SHAP computation completed in {compute_time:.2f}s  ({(compute_time/sample_size)*1000:.2f} ms/sample)", 1)

# LightGBM binary classifier returns either a list of 2 arrays [neg, pos] or 1 array (log-odds for pos class)
# Standardize to class 0 (Malware) or class 1 (Legit)
# In our project: class 0 = Malware, class 1 = Legit
if isinstance(shap_values, list):
    # Take class 0 (impact towards Malware) or class 1
    # For malware detection impact, positive impact indicates malware risk:
    shap_matrix = -shap_values[1]  # positive = pushes toward Malware (0)
else:
    shap_matrix = -shap_values     # positive = pushes toward Malware (0)

# =============================================================================
# 3. GLOBAL FEATURE IMPORTANCE & TOP 5 SELECTION
# =============================================================================
section("3. GLOBAL FEATURE IMPORTANCE (Mean |SHAP| Ranking)")

mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
importance_df = pd.DataFrame({
    "feature":        FEATURE_COLS,
    "mean_abs_shap":  mean_abs_shap,
}).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

top5 = importance_df.head(5).to_dict(orient="records")
top5_features_list = [f["feature"] for f in top5]

log("Top 10 Global Features by Mean |SHAP Impact|:", 1)
for rank, row in importance_df.head(10).iterrows():
    log(f"  #{rank+1:02d}  {row['feature']:<30}  Mean |SHAP| = {row['mean_abs_shap']:.4f}", 1)

log("\nLOCKED TOP 5 GLOBAL MALWARE DRIVERS:", 1)
for rank, f_dict in enumerate(top5, 1):
    log(f"  ★ Rank {rank}: {f_dict['feature']}  (Mean Impact: {f_dict['mean_abs_shap']:.4f})", 1)

# Save top 5 JSON
top5_json_path = os.path.join(ARTIFACTS_DIR, "top5_features.json")
top5_payload = {
    "champion_model": champion_name,
    "top5_features":  top5,
    "feature_rankings": importance_df.to_dict(orient="records"),
}
with open(top5_json_path, "w") as fh:
    json.dump(top5_payload, fh, indent=2)
log(f"\n  [SAVED] top5_features.json -> {top5_json_path}", 1)

# Save explainer package for predict.py
explainer_bundle = {
    "explainer":       explainer,
    "feature_columns": FEATURE_COLS,
    "model_name":      champion_name,
    "top5_global":     top5_features_list,
}
explainer_pkl_path = os.path.join(ARTIFACTS_DIR, "shap_explainer.pkl")
with open(explainer_pkl_path, "wb") as fh:
    pickle.dump(explainer_bundle, fh)
log(f"  [SAVED] shap_explainer.pkl -> {explainer_pkl_path}", 1)

# =============================================================================
# 4. LOCAL EXPLANATION VERIFICATION (Exact JSON Schema)
# =============================================================================
section("4. LOCAL SAMPLE EXPLANATIONS (Demonstrating Target JSON Contract)")

def explain_sample(sample_row: pd.Series, top_k: int = 5) -> list:
    """Return top_k local SHAP impact factors in exact schema:
       [{"feature": "<name>", "impact": <float>}, ...]
    """
    sample_df = pd.DataFrame([sample_row])[FEATURE_COLS]
    s_values = explainer.shap_values(sample_df)
    if isinstance(s_values, list):
        s_contrib = -s_values[1][0]
    else:
        s_contrib = -s_values[0]

    factors = []
    for feat_name, impact in zip(FEATURE_COLS, s_contrib):
        factors.append({
            "feature": feat_name,
            "impact":  round(float(impact), 4),
        })
    # Sort by absolute impact descending
    factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return factors[:top_k]

# Test on 2 sample files from validation set: 1 malware sample and 1 legit sample
malware_idx = y_val[y_val == 0].index[0]
legit_idx   = y_val[y_val == 1].index[0]

sample_mal_exp   = explain_sample(X_val.loc[malware_idx])
sample_legit_exp = explain_sample(X_val.loc[legit_idx])

log("Sample 1: True Malware Sample Explanation:", 1)
log(json.dumps(sample_mal_exp, indent=2), 2)

log("\nSample 2: True Legitimate Sample Explanation:", 1)
log(json.dumps(sample_legit_exp, indent=2), 2)

# =============================================================================
# 5. VISUALISATIONS
# =============================================================================
section("5. VISUALISATIONS  -  Generating SHAP Summary Plots")

# Plot 15: Global Feature Importance Bar Chart
plt.figure(figsize=(10, 7))
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#1a1a2e")

top15_df = importance_df.head(15).iloc[::-1]
bars = ax.barh(top15_df["feature"], top15_df["mean_abs_shap"], color="#7ED06E", height=0.65)
ax.set_xlabel("Mean |SHAP Value| (Average Impact Magnitude)", fontsize=11, color="#e0e0e0")
ax.set_title("Global Feature Importance — Top 15 Malware Indicators", fontsize=13, fontweight="bold", color="#e0e0e0")
ax.grid(True, axis="x", alpha=0.2)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.3f}",
            va="center", ha="left", color="#ccc", fontsize=9)

plt.tight_layout()
p15 = os.path.join(PLOTS_DIR, "15_shap_summary_bar.png")
plt.savefig(p15, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {p15}", 1)

# Plot 16: SHAP Beeswarm Distribution Plot
plt.figure(figsize=(11, 7.5))
fig = plt.gcf()
fig.patch.set_facecolor("#1a1a2e")

shap.summary_plot(
    shap_matrix,
    val_sample,
    max_display=15,
    show=False,
    color_bar=True,
    plot_type="dot",
)
plt.title("SHAP Feature Attribution Distribution (Top 15)", fontsize=12, fontweight="bold", color="#e0e0e0")
plt.tight_layout()
p16 = os.path.join(PLOTS_DIR, "16_shap_beeswarm.png")
plt.savefig(p16, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {p16}", 1)

# Final Banner
log()
log(SEP2)
log("  STEP 10 - SHAP EXPLAINABILITY COMPLETE")
log(f"  Top 5 Locked Drivers: {', '.join(top5_features_list)}")
log(f"  Explainer Serialised: shap_explainer.pkl (Ready for Step 14 predict.py)")
log(f"  Reports & Artifacts : explainability_report.txt  |  top5_features.json")
log(f"  Plots Generated     : 15_shap_summary_bar.png  |  16_shap_beeswarm.png")
log(SEP2)

with open(REPORT_FILE, "w", encoding="utf-8") as fh:
    fh.write("\n".join(report_lines))

print(f"\n[REPORT] Written to: {REPORT_FILE}")
