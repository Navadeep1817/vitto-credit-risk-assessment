"""
Credit Default Risk Analysis — Vitto DS Intern Assessment
UCI Credit Card Default Dataset (Taiwan, 2005)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sqlite3, json

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

# ── Palette ──────────────────────────────────────────────────────────────────
C_DEFAULT  = "#E63946"
C_NO_DEF   = "#457B9D"
C_ACCENT   = "#2D6A4F"
C_BG       = "#F8F9FA"
PLOT_DIR   = "plots"

plt.rcParams.update({
    "figure.facecolor": C_BG,
    "axes.facecolor":   C_BG,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD & BASIC QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("="*60)
print("SECTION 0 — LOAD & QUALITY REPORT")
print("="*60)

df = pd.read_csv("UCI_Credit_Card.csv")
df.rename(columns={"default.payment.next.month": "DEFAULT"}, inplace=True)

print(f"Shape          : {df.shape}")
print(f"Dtypes         :\n{df.dtypes.value_counts()}")
print(f"Null counts    :\n{df.isnull().sum().sum()} total nulls")
print(f"Class balance  :\n{df['DEFAULT'].value_counts()}")
print(f"Default rate   : {df['DEFAULT'].mean()*100:.1f}%")

# Anomaly flags
neg_bill = (df[["BILL_AMT1","BILL_AMT2","BILL_AMT3",
                 "BILL_AMT4","BILL_AMT5","BILL_AMT6"]] < 0).sum()
print("\nNegative BILL_AMT (overpayments):")
print(neg_bill[neg_bill > 0])

print("\nEDUCATION value counts (raw):")
print(df["EDUCATION"].value_counts().sort_index())
print("\nMARRIAGE value counts (raw):")
print(df["MARRIAGE"].value_counts().sort_index())

# ─────────────────────────────────────────────────────────────────────────────
# 1. EDA PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 1 — EDA")
print("="*60)

# ── Plot 1: Distributions of LIMIT_BAL, AGE, PAY_0 ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Figure 1 — Key Variable Distributions", fontsize=14, fontweight="bold", y=1.02)

for ax, col, color, label in zip(
        axes,
        ["LIMIT_BAL", "AGE", "PAY_0"],
        [C_NO_DEF, C_ACCENT, C_DEFAULT],
        ["Credit Limit (NT$)", "Client Age (years)", "Sep 2005 Repayment Status"]):
    sns.histplot(df[col], ax=ax, color=color, bins=40, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(label);  ax.set_ylabel("Count")
    ax.set_title(col)

axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,_: f"{x/1e3:.0f}k"))
axes[2].axvline(0, color="gray", ls="--", lw=1, label="Paid duly (0)")
axes[2].legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot 01")

# ── Plot 2: Default rate by demographic groups ────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
fig.suptitle("Figure 2 — Default Rate by Demographic Group", fontsize=14, fontweight="bold", y=1.02)

SEX_MAP  = {1:"Male", 2:"Female"}
EDU_MAP  = {1:"Grad", 2:"Univ", 3:"High Sch", 4:"Other"}
MAR_MAP  = {1:"Married", 2:"Single", 3:"Other"}

df["AGE_BAND"] = pd.cut(df["AGE"], bins=[20,30,40,50,60,80],
                         labels=["21-30","31-40","41-50","51-60","61+"])
df["SEX_LBL"]  = df["SEX"].map(SEX_MAP)

for ax, col, lbl, cmap in zip(
        axes,
        ["SEX_LBL",  "EDUCATION", "MARRIAGE", "AGE_BAND"],
        ["Sex",      "Education", "Marital Status", "Age Band"],
        [C_NO_DEF,   C_ACCENT,    C_DEFAULT,  "#6D6875"]):
    rates = df.groupby(col)["DEFAULT"].mean() * 100
    bars = ax.bar(rates.index.astype(str), rates.values, color=cmap, edgecolor="white")
    ax.set_title(lbl);  ax.set_ylabel("Default Rate (%)")
    ax.set_ylim(0, 35)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4,
                f"{b.get_height():.1f}%", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_default_by_demographics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot 02")

# ── Plot 3: Repayment delay patterns PAY_0–PAY_6 ─────────────────────────────
pay_cols = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
months   = ["Sep","Aug","Jul","Jun","May","Apr"]

fig, ax = plt.subplots(figsize=(11, 5))
for outcome, color, label in [(0, C_NO_DEF, "No Default"), (1, C_DEFAULT, "Default")]:
    sub = df[df["DEFAULT"] == outcome][pay_cols].mean()
    ax.plot(months, sub.values, marker="o", linewidth=2.5,
            color=color, label=label, markersize=7)
ax.set_title("Figure 3 — Average Repayment Delay Score by Month & Default Outcome")
ax.set_xlabel("Statement Month (2005)")
ax.set_ylabel("Mean PAY Score\n(higher = more delayed)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_repayment_delay.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot 03")

# ── Plot 4: Correlation heatmap ───────────────────────────────────────────────
num_cols = ["LIMIT_BAL","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
            "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
            "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","DEFAULT"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, annot=False, linewidths=0.3, linecolor="white",
            cbar_kws={"shrink":0.7})
ax.set_title("Figure 4 — Feature Correlation Matrix", pad=12)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

top5_corr = corr["DEFAULT"].drop("DEFAULT").abs().sort_values(ascending=False).head(5)
print("Top 5 features correlated with DEFAULT:")
print(top5_corr)
print("Saved plot 04")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 2 — FEATURE ENGINEERING")
print("="*60)

bill_cols   = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
pay_amt_cols= ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]

# AVG_UTIL_RATE — capped at 1 to handle overpayments (negative bills → util < 0 nonsensical)
util = df[bill_cols].clip(lower=0).div(df["LIMIT_BAL"], axis=0)
df["AVG_UTIL_RATE"] = util.mean(axis=1)

# AVG_PAY_RATIO — average (payment / bill) where bill > 0
ratios = []
for b, p in zip(bill_cols, pay_amt_cols):
    mask = df[b] > 0
    r = np.where(mask, df[p] / df[b], np.nan)
    ratios.append(r)
df["AVG_PAY_RATIO"] = np.nanmean(ratios, axis=0)
df["AVG_PAY_RATIO"] = df["AVG_PAY_RATIO"].fillna(0).clip(upper=5)  # cap extreme ratios

# TOTAL_DELAY_MONTHS — count of PAY_x > 0
df["TOTAL_DELAY_MONTHS"] = (df[pay_cols] > 0).sum(axis=1)

# EDUCATION encoding — merge 0,5,6 → 4 (other)
df["EDUCATION"] = df["EDUCATION"].replace({0:4, 5:4, 6:4})

# MARRIAGE encoding — merge 0 → 3 (other)
df["MARRIAGE"] = df["MARRIAGE"].replace({0:3})

print("Feature engineering complete.")
print(f"  AVG_UTIL_RATE  — mean: {df['AVG_UTIL_RATE'].mean():.3f}, max: {df['AVG_UTIL_RATE'].max():.2f}")
print(f"  AVG_PAY_RATIO  — mean: {df['AVG_PAY_RATIO'].mean():.3f}")
print(f"  TOTAL_DELAY_MONTHS — value counts:\n{df['TOTAL_DELAY_MONTHS'].value_counts().sort_index()}")
print(f"  EDUCATION (recoded): {df['EDUCATION'].value_counts().sort_index().to_dict()}")
print(f"  MARRIAGE  (recoded): {df['MARRIAGE'].value_counts().sort_index().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL DEVELOPMENT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 3 — MODEL DEVELOPMENT")
print("="*60)

feature_cols = (pay_cols +
                bill_cols + pay_amt_cols +
                ["LIMIT_BAL","AGE","EDUCATION","MARRIAGE","SEX",
                 "AVG_UTIL_RATE","AVG_PAY_RATIO","TOTAL_DELAY_MONTHS"])

X = df[feature_cols].copy()
y = df["DEFAULT"].copy()

# SMOTE on training folds (applied inside CV for correctness)
# We do a single train/test split for final evaluation, SMOTE on train only
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_tr_res, y_tr_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — train shape: {X_tr_res.shape}, class balance: {pd.Series(y_tr_res).value_counts().to_dict()}")

# ── Logistic Regression baseline ──────────────────────────────────────────────
lr = Pipeline([("scaler", StandardScaler()),
               ("clf", LogisticRegression(max_iter=1000, random_state=42,
                                          class_weight="balanced"))])
lr.fit(X_tr_res, y_tr_res)
y_pred_lr  = lr.predict(X_test)
y_prob_lr  = lr.predict_proba(X_test)[:,1]

print("\n--- Logistic Regression ---")
print(classification_report(y_test, y_pred_lr))
auc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"AUC-ROC: {auc_lr:.4f}")

# ── XGBoost ────────────────────────────────────────────────────────────────────
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb_clf = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
xgb_clf.fit(X_tr_res, y_tr_res)
y_pred_xgb = xgb_clf.predict(X_test)
y_prob_xgb = xgb_clf.predict_proba(X_test)[:,1]

print("\n--- XGBoost ---")
print(classification_report(y_test, y_pred_xgb))
auc_xgb = roc_auc_score(y_test, y_prob_xgb)
print(f"AUC-ROC: {auc_xgb:.4f}")

# ── Stratified 5-fold CV ───────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_lr  = cross_val_score(lr,     X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
cv_scores_xgb = cross_val_score(xgb_clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

print(f"\nCV AUC — LR : {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")
print(f"CV AUC — XGB: {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")

# ── Feature importance (XGBoost) ───────────────────────────────────────────────
feat_imp = pd.Series(xgb_clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 XGBoost features:")
print(feat_imp.head(10))

# ── Plot 5: ROC Curves ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ROC
fpr_lr,  tpr_lr,  _ = roc_curve(y_test, y_prob_lr)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
ax = axes[0]
ax.plot(fpr_lr,  tpr_lr,  color=C_NO_DEF, lw=2.5, label=f"Logistic Reg (AUC={auc_lr:.3f})")
ax.plot(fpr_xgb, tpr_xgb, color=C_DEFAULT, lw=2.5, label=f"XGBoost     (AUC={auc_xgb:.3f})")
ax.plot([0,1],[0,1],"--", color="gray", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Figure 5a — ROC Curves")
ax.legend(loc="lower right")

# Confusion matrix (XGBoost)
cm = confusion_matrix(y_test, y_pred_xgb)
ax = axes[1]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Predicted No Default","Predicted Default"],
            yticklabels=["Actual No Default","Actual Default"],
            linewidths=0.5, linecolor="white")
ax.set_title("Figure 5b — XGBoost Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_roc_confusion.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot 05")

# ── Plot 6: Feature Importance ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
top15 = feat_imp.head(15)
colors = [C_DEFAULT if i < 5 else C_NO_DEF for i in range(len(top15))]
bars = ax.barh(top15.index[::-1], top15.values[::-1], color=colors[::-1], edgecolor="white")
ax.set_title("Figure 6 — XGBoost Top 15 Feature Importances")
ax.set_xlabel("Importance Score")
red_patch  = mpatches.Patch(color=C_DEFAULT, label="Top 5 features")
blue_patch = mpatches.Patch(color=C_NO_DEF,  label="Other features")
ax.legend(handles=[red_patch, blue_patch])
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot 06")

# ── Plot 7: SHAP Beeswarm ─────────────────────────────────────────────────────
print("\nComputing SHAP values (sample of 500)...")
explainer = shap.TreeExplainer(xgb_clf)
X_sample  = X_test.sample(500, random_state=42)
shap_vals  = explainer.shap_values(X_sample)

fig = plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_sample, show=False, plot_size=None, color_bar=True)
plt.title("Figure 7 — SHAP Beeswarm Plot (500 test samples)", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot 07")

# ── Plot 8: Fairness — FPR by SEX and EDUCATION ───────────────────────────────
print("\nFairness analysis...")

def fpr_for_group(mask):
    cm_g = confusion_matrix(y_test[mask], y_pred_xgb[mask])
    if cm_g[0].sum() == 0: return np.nan
    return cm_g[0,1] / cm_g[0].sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# By SEX
sex_test = df.loc[X_test.index, "SEX"].map({1:"Male",2:"Female"})
fpr_sex  = {g: fpr_for_group(sex_test == g) for g in ["Male","Female"]}
axes[0].bar(fpr_sex.keys(), [v*100 for v in fpr_sex.values()], color=[C_NO_DEF, C_DEFAULT], edgecolor="white")
axes[0].set_title("Figure 8a — False Positive Rate by Sex")
axes[0].set_ylabel("FPR (%)"); axes[0].set_ylim(0, 35)
for i,(k,v) in enumerate(fpr_sex.items()):
    axes[0].text(i, v*100+0.5, f"{v*100:.1f}%", ha="center", fontsize=10)

# By EDUCATION
edu_labels = {1:"Grad",2:"Univ",3:"High Sch",4:"Other"}
edu_test = df.loc[X_test.index, "EDUCATION"].map(edu_labels)
fpr_edu  = {g: fpr_for_group(edu_test == g) for g in ["Grad","Univ","High Sch","Other"]}
axes[1].bar(fpr_edu.keys(), [v*100 for v in fpr_edu.values()], color=C_ACCENT, edgecolor="white")
axes[1].set_title("Figure 8b — False Positive Rate by Education")
axes[1].set_ylabel("FPR (%)"); axes[1].set_ylim(0, 35)
for i,(k,v) in enumerate(fpr_edu.items()):
    axes[1].text(i, v*100+0.5, f"{v*100:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/08_fairness.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot 08")
print("FPR by SEX:", {k: f"{v*100:.1f}%" for k,v in fpr_sex.items()})
print("FPR by EDU:", {k: f"{v*100:.1f}%" for k,v in fpr_edu.items()})

# ── SQL bonus ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 4 — SQL QUERIES")
print("="*60)

conn = sqlite3.connect(":memory:")
df.to_sql("credit", conn, index=False)

q1 = pd.read_sql("""
  SELECT EDUCATION,
         COUNT(*)                               AS clients,
         ROUND(AVG("DEFAULT")*100,2)            AS default_pct
  FROM credit
  GROUP BY EDUCATION ORDER BY default_pct DESC
""", conn)
print("\nQ1 — Default rate by education:\n", q1.to_string(index=False))

q2 = pd.read_sql("""
  SELECT CASE WHEN AGE < 30 THEN '< 30'
              WHEN AGE < 40 THEN '30-39'
              WHEN AGE < 50 THEN '40-49'
              ELSE '50+' END AS age_band,
         ROUND(AVG(LIMIT_BAL),0)        AS avg_credit_limit,
         ROUND(AVG("DEFAULT")*100,2)    AS default_pct
  FROM credit
  GROUP BY age_band ORDER BY age_band
""", conn)
print("\nQ2 — Credit limit & default by age band:\n", q2.to_string(index=False))

q3 = pd.read_sql("""
  SELECT MARRIAGE, SEX,
         COUNT(*)                        AS clients,
         ROUND(AVG("DEFAULT")*100,2)     AS default_pct
  FROM credit
  GROUP BY MARRIAGE, SEX ORDER BY default_pct DESC LIMIT 10
""", conn)
print("\nQ3 — Default rate by marital status × sex:\n", q3.to_string(index=False))
conn.close()

# ── Collect metrics for write-up ──────────────────────────────────────────────
metrics = {
    "lr_precision":  precision_score(y_test, y_pred_lr),
    "lr_recall":     recall_score(y_test, y_pred_lr),
    "lr_f1":         f1_score(y_test, y_pred_lr),
    "lr_auc":        auc_lr,
    "xgb_precision": precision_score(y_test, y_pred_xgb),
    "xgb_recall":    recall_score(y_test, y_pred_xgb),
    "xgb_f1":        f1_score(y_test, y_pred_xgb),
    "xgb_auc":       auc_xgb,
    "cv_lr_mean":    cv_scores_lr.mean(),
    "cv_lr_std":     cv_scores_lr.std(),
    "cv_xgb_mean":   cv_scores_xgb.mean(),
    "cv_xgb_std":    cv_scores_xgb.std(),
    "top5_features": feat_imp.head(5).to_dict(),
    "fpr_sex":       {k: round(v*100,1) for k,v in fpr_sex.items()},
    "fpr_edu":       {k: round(v*100,1) for k,v in fpr_edu.items()},
}

with open("outputs/metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

print("\nAll analysis complete. Metrics saved to outputs/metrics.json")
print("Plots saved to plots/")
