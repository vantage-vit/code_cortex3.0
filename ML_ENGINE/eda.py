"""
================================================================================
  CODE CORTEX 3.0  -  STEP 2 : EXPLORATORY DATA ANALYSIS
  Member 1 | ML / Research
================================================================================
Run:
    python ml/eda.py

Reads:
    data/processed_malware.csv   (104,589 x 54 numeric features)
    data/labels.csv              (104,589 x 1 target: 0=malware 1=legit)

Outputs:
    ml/reports/eda_report.txt
    ml/plots/01_class_distribution.png
    ml/plots/02_feature_boxplots.png
    ml/plots/03_correlation_heatmap.png
    ml/plots/04_outlier_summary.png
================================================================================
"""

import io
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive / headless
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# -- Paths --------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
ML_DIR      = os.path.join(BASE_DIR, "ml")
PLOTS_DIR   = os.path.join(ML_DIR, "plots")
REPORTS_DIR = os.path.join(ML_DIR, "reports")
ARTIFACTS_DIR = os.path.join(ML_DIR, "artifacts")

for d in [PLOTS_DIR, REPORTS_DIR, ARTIFACTS_DIR]:
    os.makedirs(d, exist_ok=True)

REPORT_FILE = os.path.join(REPORTS_DIR, "eda_report.txt")

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

# -- Global plot style --------------------------------------------------------
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
CLR = {0: "#E05C5C", 1: "#5B9BD5"}   # red=malware, blue=legitimate

# =============================================================================
log(SEP2)
log("  CODE CORTEX 3.0  -  STEP 2 : EXPLORATORY DATA ANALYSIS")
log(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(SEP2)

# -- 1. Load ------------------------------------------------------------------
section("1. LOAD DATA")

X_path = os.path.join(DATA_DIR, "processed_malware.csv")
y_path = os.path.join(DATA_DIR, "labels.csv")

if not os.path.exists(X_path):
    log(f"[ERROR] Missing: {X_path}"); sys.exit(1)
if not os.path.exists(y_path):
    log(f"[ERROR] Missing: {y_path}"); sys.exit(1)

X = pd.read_csv(X_path)
y = pd.read_csv(y_path)["legitimate"]

assert len(X) == len(y), f"Row mismatch: X={len(X)}, y={len(y)}"

log(f"Feature matrix : {X.shape[0]:,} rows x {X.shape[1]} columns", 1)
log(f"Label series   : {len(y):,} entries  (0 = malware, 1 = legitimate)", 1)

# -- 2. Class Distribution ----------------------------------------------------
section("2. CLASS DISTRIBUTION")

total = len(y)
n_mal = int((y == 0).sum())
n_leg = int((y == 1).sum())
ratio = max(n_mal, n_leg) / max(min(n_mal, n_leg), 1)

log(f"  {'Class':<20} {'Count':>10}  {'Percent':>9}", 1)
log("  " + "-" * 44, 1)
log(f"  Malware    (0)     {n_mal:>10,}  {n_mal/total*100:>8.2f}%", 1)
log(f"  Legitimate (1)     {n_leg:>10,}  {n_leg/total*100:>8.2f}%", 1)
log("  " + "-" * 44, 1)
log(f"  Total              {total:>10,}   100.00%", 1)
log(f"  Imbalance ratio  : {ratio:.4f} : 1", 1)

if ratio > 1.5:
    log(f"[WARN] Imbalanced dataset (ratio {ratio:.2f} > threshold 1.5).", 1)
    log("  -> Use class_weight='balanced' in sklearn models.", 2)
    log("  -> Prefer ROC-AUC / PR-AUC / F1 over raw accuracy.", 2)
    log("  -> SMOTE only inside CV fold if used (avoid leakage).", 2)
else:
    log("  Classes are acceptably balanced - no resampling required.", 1)

# Plot 01 ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor("#1a1a2e")

ax = axes[0]
bars = ax.bar(["Malware (0)", "Legitimate (1)"], [n_mal, n_leg],
              color=[CLR[0], CLR[1]], width=0.5,
              edgecolor="#1a1a2e", linewidth=1.5)
for bar, cnt in zip(bars, [n_mal, n_leg]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f"{cnt:,}\n({cnt/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=11,
            fontweight="bold", color="#e0e0e0")
ax.set_title("Class Counts", fontsize=13, fontweight="bold", color="#e0e0e0")
ax.set_ylabel("Sample Count", color="#e0e0e0")
ax.set_ylim(0, max(n_mal, n_leg) * 1.20)
ax.spines[["top", "right"]].set_visible(False)

ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(
    [n_mal, n_leg],
    labels=["Malware (0)", "Legitimate (1)"],
    colors=[CLR[0], CLR[1]],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2.5},
    textprops={"color": "#e0e0e0", "fontsize": 11},
)
for at in autotexts:
    at.set_fontweight("bold")
ax2.set_title("Class Split", fontsize=13, fontweight="bold", color="#e0e0e0")

plt.suptitle("MalwareGuard - Class Distribution", fontsize=15,
             fontweight="bold", color="#e0e0e0", y=1.02)
plt.tight_layout()
_p = os.path.join(PLOTS_DIR, "01_class_distribution.png")
plt.savefig(_p, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {_p}", 1)

# -- 3. Per-class Feature Statistics ------------------------------------------
section("3. PER-CLASS FEATURE STATISTICS (mean comparison, top-15 discriminating)")

df_all = X.copy()
df_all["_label"] = y.values

mal_mean  = df_all[df_all["_label"] == 0][X.columns].mean()
leg_mean  = df_all[df_all["_label"] == 1][X.columns].mean()
mean_diff = (mal_mean - leg_mean).abs()
top15_feats = mean_diff.nlargest(15).index.tolist()

log(f"  {'Feature':<35} {'Mal Mean':>16} {'Leg Mean':>16} {'|delta|':>14}", 1)
log("  " + "-" * 85, 1)
for feat in top15_feats:
    log(f"  {feat:<35} {mal_mean[feat]:>16.4f} {leg_mean[feat]:>16.4f}"
        f" {mean_diff[feat]:>14.4f}", 1)

# Plot 02 - box plots per class for top 12 ---
n_box = min(12, len(top15_feats))
ncols = 4
nrows = -(-n_box // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 4.5))
fig.patch.set_facecolor("#1a1a2e")
axes_flat = axes.flatten()

p01 = X[top15_feats].quantile(0.01)
p99 = X[top15_feats].quantile(0.99)

for i, feat in enumerate(top15_feats[:n_box]):
    ax = axes_flat[i]
    d0 = df_all[df_all["_label"] == 0][feat].clip(p01[feat], p99[feat])
    d1 = df_all[df_all["_label"] == 1][feat].clip(p01[feat], p99[feat])

    bp = ax.boxplot(
        [d0, d1],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2.5},
        whiskerprops={"color": "#aaa", "linewidth": 1.2},
        capprops={"color": "#aaa", "linewidth": 1.2},
        flierprops={"marker": ".", "markersize": 1.5, "alpha": 0.3,
                    "markerfacecolor": "#888"},
    )
    for patch, color in zip(bp["boxes"], [CLR[0], CLR[1]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#1a1a2e")

    ax.set_title(feat, fontsize=8.5, fontweight="bold", color="#e0e0e0", pad=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Malware", "Legit"], fontsize=8, color="#ccc")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#ccc")

for j in range(n_box, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle(
    "Top-12 Discriminating Features - Distribution by Class\n"
    "(Values clipped to 1st-99th percentile for visibility)",
    fontsize=13, fontweight="bold", color="#e0e0e0",
)
plt.tight_layout()
_p = os.path.join(PLOTS_DIR, "02_feature_boxplots.png")
plt.savefig(_p, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {_p}", 1)

# -- 4. Outlier Inspection -----------------------------------------------------
section("4. OUTLIER INSPECTION  (IQR x 1.5 mild  |  IQR x 3.0 extreme)")

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
out_recs = []

for col in num_cols:
    q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        continue
    n_mild    = int(((X[col] < q1 - 1.5 * iqr) | (X[col] > q3 + 1.5 * iqr)).sum())
    n_extreme = int(((X[col] < q1 - 3.0 * iqr) | (X[col] > q3 + 3.0 * iqr)).sum())
    out_recs.append({
        "Feature":  col,
        "Mild":     n_mild,
        "Mild%":    n_mild / len(X) * 100,
        "Extreme":  n_extreme,
        "Extreme%": n_extreme / len(X) * 100,
    })

out_df = pd.DataFrame(out_recs).sort_values("Extreme", ascending=False).reset_index(drop=True)

log(f"  {'Feature':<38} {'Mild(1.5x)':>11} {'Mild%':>7}"
    f" {'Extreme(3x)':>12} {'Ext%':>6}", 1)
log("  " + "-" * 80, 1)
for _, row in out_df.head(20).iterrows():
    flag = "  <- HIGH" if row["Extreme%"] > 10 else ""
    log(f"  {row['Feature']:<38} {int(row['Mild']):>11,} {row['Mild%']:>6.1f}%"
        f" {int(row['Extreme']):>12,} {row['Extreme%']:>5.1f}%{flag}", 1)

log("", 1)
log("  Decision : KEEP all outliers - do not remove.", 1)
log("  Rationale: PE-header values in packed/obfuscated malware are", 1)
log("             intentionally extreme and ARE the signal.", 1)
log("             RF + XGBoost are split-based -> outlier-robust.", 1)
log("             Logistic Regression will be z-score scaled (Step 4),", 1)
log("             further mitigating outlier impact on LR.", 1)

# Plot 04 ---
top15_out = out_df.head(15).copy()
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("#1a1a2e")
colors_bar = [
    "#E05C5C" if v > 10 else "#E09B5C" if v > 5 else "#5B9BD5"
    for v in top15_out["Extreme%"][::-1]
]
ax.barh(top15_out["Feature"][::-1], top15_out["Extreme%"][::-1],
        color=colors_bar, edgecolor="#1a1a2e", height=0.65)
ax.axvline(10, color="orange", ls="--", lw=1.8,
           label="10% threshold (HIGH)", alpha=0.9)
ax.axvline(5, color="yellow", ls=":", lw=1.2,
           label="5% threshold (MED)", alpha=0.7)
ax.set_xlabel("% Samples Flagged as Extreme Outliers (3 x IQR)",
              fontsize=11, color="#e0e0e0")
ax.set_title("Top-15 Features - Extreme Outlier Rate",
             fontsize=13, fontweight="bold", color="#e0e0e0")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
for bar, val in zip(ax.patches, top15_out["Extreme%"][::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=8.5, color="#e0e0e0")
plt.tight_layout()
_p = os.path.join(PLOTS_DIR, "04_outlier_summary.png")
plt.savefig(_p, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {_p}", 1)

# -- 5. Correlation Analysis --------------------------------------------------
section("5. CORRELATION ANALYSIS  (Pearson)")

corr = X[num_cols].corr(method="pearson")

pairs = []
cols_arr = list(corr.columns)
for i in range(len(cols_arr)):
    for j in range(i + 1, len(cols_arr)):
        r = corr.iloc[i, j]
        if abs(r) > 0.70:
            pairs.append({"Feature A": cols_arr[i],
                          "Feature B": cols_arr[j],
                          "r": r})

pairs_df = (pd.DataFrame(pairs)
              .sort_values("r", key=lambda s: s.abs(), ascending=False)
              .reset_index(drop=True))

log(f"  Pairs |r| > 0.70 : {len(pairs_df)}", 1)
log(f"  Pairs |r| > 0.90 : {int((pairs_df['r'].abs() > 0.90).sum())}", 1)
log(f"  Pairs |r| > 0.95 : {int((pairs_df['r'].abs() > 0.95).sum())}"
    f"  <- will be dropped in Step 5 (Feature Engineering)", 1)
log("", 1)
log(f"  {'Feature A':<35} {'Feature B':<35} {'r':>8}", 1)
log("  " + "-" * 82, 1)
for _, row in pairs_df.head(30).iterrows():
    flag = "  <- DROP (Step 5)" if abs(row["r"]) > 0.95 else ""
    log(f"  {row['Feature A']:<35} {row['Feature B']:<35}"
        f" {row['r']:>8.4f}{flag}", 1)

# Save pairs for Step 5
pairs_df.to_csv(os.path.join(ARTIFACTS_DIR, "correlation_pairs.csv"), index=False)
log(f"\n  [SAVED] correlation_pairs.csv -> ml/artifacts/", 1)

# Plot 03 - correlation heatmap ---
fig, ax = plt.subplots(figsize=(22, 19))
fig.patch.set_facecolor("#1a1a2e")
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.25,
    linecolor="#1a1a2e",
    annot=False,
    cbar_kws={"shrink": 0.65, "label": "Pearson r"},
    ax=ax,
)
ax.set_title("Pearson Correlation Matrix - All 54 Features",
             fontsize=15, fontweight="bold", color="#e0e0e0", pad=18)
ax.tick_params(labelsize=8, colors="#ccc")
plt.tight_layout()
_p = os.path.join(PLOTS_DIR, "03_correlation_heatmap.png")
plt.savefig(_p, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
log(f"  [PLOT] -> {_p}", 1)

# -- 6. EDA Summary for hand-off ----------------------------------------------
section("6. EDA FINDINGS SUMMARY  (hand-off notes for Steps 3-5)")

log("  CLASS BALANCE", 1)
log(f"    Malware : {n_mal:,}  ({n_mal/total*100:.1f}%)  |  "
    f"Legitimate : {n_leg:,}  ({n_leg/total*100:.1f}%)", 2)
log(f"    Ratio {ratio:.2f}:1  -> use class_weight='balanced'", 2)

log("  TOP DISCRIMINATING FEATURES  (largest mean gap):", 1)
for i, f in enumerate(top15_feats[:5], 1):
    log(f"    {i}. {f}  (|delta mean| = {mean_diff[f]:.2f})", 2)

log("  OUTLIERS", 1)
high_out = out_df[out_df["Extreme%"] > 10]["Feature"].tolist()
log(f"    {len(high_out)} features with >10% extreme outliers - KEEPING all.", 2)
if high_out:
    log(f"    ({', '.join(high_out[:5])}{'...' if len(high_out) > 5 else ''})", 2)

log("  HIGH CORRELATION", 1)
n_drop_corr = int((pairs_df["r"].abs() > 0.95).sum())
log(f"    {n_drop_corr} pairs with |r| > 0.95 -> one from each pair dropped in Step 5.", 2)

log("  SCALING", 1)
log("    All 54 features are numeric -> StandardScaler applied in Step 4.", 2)

log()
log(SEP2)
log("  STEP 2 - EDA COMPLETE")
log(f"  Finished : {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Report   : {REPORT_FILE}")
log(f"  Plots    : {PLOTS_DIR}")
log(SEP2)

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n[REPORT] Written to: {REPORT_FILE}")
