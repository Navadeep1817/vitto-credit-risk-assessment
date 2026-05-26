# Credit Default Risk Analysis
### Vitto DS Intern Assessment · UCI Credit Card Default Dataset

---

## Overview

End-to-end credit default prediction pipeline built on 30,000 Taiwanese credit card clients (2005). Covers data quality auditing, exploratory analysis, feature engineering, binary classification modelling, fairness evaluation, SHAP interpretability, and SQL business queries.

---

## Setup & Reproduction

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap reportlab nbformat
```

Python 3.9+ recommended.

### Run

```bash
# 1. Place UCI_Credit_Card.csv in this directory
# 2. Run the full analysis (generates plots/ and outputs/metrics.json)
python3 analysis.py

# 3. Open the Jupyter notebook
jupyter notebook outputs/credit_risk_analysis.ipynb

# 4. Rebuild the write-up PDF (optional)
python3 build_pdf.py
```

---

## Dataset

**Source:** UCI Machine Learning Repository — [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)  
**License:** CC BY 4.0  
**Rows:** 30,000 | **Columns:** 25 | **Period:** April–September 2005 | **Region:** Taiwan

**Target variable:** `default.payment.next.month` (1 = default, 0 = no default) — 22.1% positive rate.

---

## Methodology Summary

### Data Quality
| Issue | Handling |
|-------|----------|
| EDUCATION values 0, 5, 6 | Merged into category 4 (Other) — undocumented, 1.6% of data |
| MARRIAGE value 0 | Merged into 3 (Other) — 54 rows, 0.18% of data |
| Negative BILL_AMT (overpayments) | Clipped to 0 for utilisation calc; raw values retained |
| PAY_x = -2 (no consumption) | Treated as non-delayed |
| Class imbalance (22.1% default) | SMOTE on training split only (no leakage) |

### Feature Engineering
| Feature | Description |
|---------|-------------|
| `AVG_UTIL_RATE` | Mean credit utilisation across 6 months (bills clipped to 0 for negatives) |
| `AVG_PAY_RATIO` | Mean payment-to-bill ratio where bill > 0; measures repayment consistency |
| `TOTAL_DELAY_MONTHS` | Count of PAY_x > 0 across 6 months; cumulative delinquency signal |

### Models
- **Logistic Regression** (baseline) with StandardScaler + `class_weight='balanced'`
- **XGBoost** with `scale_pos_weight` and SMOTE-augmented training data
- **Evaluation:** Stratified 5-fold cross-validation, AUC-ROC as primary metric (appropriate for imbalanced data)

---

## Key Findings

| Metric | Logistic Regression | XGBoost |
|--------|--------------------:|--------:|
| Test AUC-ROC | 0.713 | **0.748** |
| CV AUC (5-fold) | 0.755 ± 0.004 | **0.780 ± 0.005** |
| Recall (default class) | 0.57 | **0.77** |
| Precision (default class) | 0.40 | 0.33 |

**XGBoost is the best model.** Higher AUC and dramatically better recall — critical for a credit risk application where missing a true defaulter is more costly than a false alarm.

### Top 5 Predictive Features (XGBoost)
1. `TOTAL_DELAY_MONTHS` — history of delayed payments is the strongest signal
2. `SEX`
3. `MARRIAGE`
4. `PAY_2` — August 2005 repayment status
5. `PAY_0` — September 2005 repayment status

### Business Actions
1. **Flag customers with ≥3 months of delayed payment** for early intervention (payment plans, counselling).
2. **Pause automatic credit limit increases** for customers consistently using >80% of their limit.

### Fairness Notes
- Males show a False Positive Rate of ~57% vs ~35% for females — the model incorrectly flags creditworthy males at a higher rate. Should be monitored in production.
- High school educated borrowers have a lower FPR (~34%) than university graduates (~49%).

---

## Assumptions
- PAY_x = -2 is treated as equivalent to -1 (no delinquency) for TOTAL_DELAY_MONTHS.
- Negative BILL_AMT values are legitimate overpayments, not data errors.
- EDUCATION and MARRIAGE undocumented values are treated as "Other" (not dropped).
- SMOTE is applied only inside the training fold to prevent leakage into the test set.
- Default threshold of 0.5 used for confusion matrix; in production this should be tuned against a business cost matrix.

---

## Deliverables

| File | Description |
|------|-------------|
| `outputs/credit_risk_analysis.ipynb` | Full annotated Jupyter notebook |
| `outputs/writeup.pdf` | 2-page write-up PDF |
| `README.md` | This file |
| `analysis.py` | Standalone Python script (runs end-to-end) |
| `plots/*.png` | All 8 publication-quality plots |
| `outputs/metrics.json` | Model metrics in JSON |

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn · SHAP · Seaborn · Matplotlib · SQLite · ReportLab
