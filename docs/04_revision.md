# 04_revision.md — ML Concepts & Modeling Reference

---

## 1. Lending Club: How It Worked

Lending Club connected borrowers directly with investors, cutting out traditional banks.

**Loan lifecycle:**
1. Borrower applies — provides income, FICO, employment details
2. LC assigns a risk grade (A = safest, G = riskiest) and sets an interest rate
3. Investors fund the loan in small increments
4. Borrower makes monthly payments, distributed to investors
5. If 120+ days late → loan is "Charged Off" → investors lose their share

**Our goal:** Predict charge-off before the loan is approved, using only application-time data.

---

## 2. Why Accuracy Fails on Imbalanced Data

With 80% non-default and 20% default, a model that predicts "no default" for everyone achieves
80% accuracy while catching zero actual defaults. Accuracy is therefore useless for credit risk.

**Use instead:** ROC-AUC, Precision-Recall AUC, F1-Score.

---

## 3. Evaluation Metrics

### Confusion Matrix Terms

| Term | Meaning |
|------|---------|
| True Positive (TP) | Predicted default, actually defaulted |
| True Negative (TN) | Predicted paid, actually paid |
| False Positive (FP) | Predicted default, actually paid (false alarm) |
| False Negative (FN) | Predicted paid, actually defaulted (missed default) |

### Key Metrics

**Recall** — what fraction of actual defaults did we catch?

    Recall = TP / (TP + FN)

High recall = catch most defaulters, but may reject some good borrowers too.

**Precision** — when we predict default, how often are we right?

    Precision = TP / (TP + FP)

High precision = rejections are well-justified, but we may miss some defaults.

**F1-Score** — harmonic mean of precision and recall; useful when both matter.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

**ROC-AUC** — probability that the model ranks a random defaulter above a random non-defaulter.
Threshold-independent and unaffected by class imbalance.

| AUC | Interpretation |
|-----|----------------|
| 0.50 | Random (useless) |
| 0.70 | Minimum acceptable for credit models |
| 0.80 | Strong |
| 0.90+ | Exceptional (rare in real-world credit data) |

**Metric to use:** ROC-AUC as primary. Precision-Recall AUC as secondary (better for
highly imbalanced data). Never use accuracy.

---

## 4. Statistical Testing for Categorical Features

### Chi-Square Test
Tests whether two categorical variables are independent.
- Returns a p-value: probability of observing the pattern if no relationship exists
- p < 0.05 → relationship is statistically significant
- With 1M+ rows, almost everything will be significant — use effect size to filter

### Cramer's V (Effect Size)
Measures the strength of a categorical association on a 0–1 scale.

    V = sqrt(chi2 / (n * min(rows-1, cols-1)))

| Cramer's V | Interpretation |
|------------|----------------|
| 0.00–0.05 | Negligible |
| 0.05–0.10 | Weak |
| 0.10–0.20 | Moderate |
| 0.20–0.40 | Strong |
| 0.40+ | Very strong |

**Rule:** Use chi-square to confirm a relationship exists, use Cramer's V to decide whether it
is strong enough to be useful as a feature.

### Large Sample Caution
With 1M+ records, a difference of 0.2% between groups can yield p < 0.001. Always check
Cramer's V — statistical significance does not imply predictive usefulness.

---

## 5. Feature Selection Approach

**Step 1 — Domain filter:** Drop columns not available at origination, identifiers, and
LC's own risk outputs (grade, int_rate, etc.)

**Step 2 — Univariate analysis:** Plot default rate by bin/category. Keep features with
a clear trend. Drop flat or noisy features.

**Step 3 — Model-based selection:** Train XGBoost on all candidates. Drop features in the
bottom 10% by gain importance. Use L1 regularization for linear models.

**Do not rely solely on chi-square p-values** — a feature can be statistically significant
but practically useless, or vice versa.

---

## 6. Modeling Strategy

### Recommended Models

| Model | Best For | Notes |
|-------|----------|-------|
| XGBoost / LightGBM | Primary model | Handles mixed types, finds non-linear patterns automatically |
| Logistic Regression | Interpretability baseline | Requires WOE-transformed features |
| Random Forest | Feature importance validation | Usually outperformed by boosting |

### Feature Engineering by Model Type

**Tree models (XGBoost, LightGBM):**
- Raw numeric values work fine — trees find thresholds internally
- Dates: convert to numeric (e.g., `months_since_earliest_cr_line`)
- Categoricals: one-hot encode or use native categorical support

**Linear models (Logistic Regression):**
- Apply WOE (Weight of Evidence) encoding to numeric and categorical features
- WOE = ln(% non-defaults in bin / % defaults in bin)
- Fit WOE on training fold only, then apply to validation/test

### Practical Workflow
1. Create one master dataset with all engineered features
2. Train XGBoost first — fastest path to a strong baseline
3. Validate feature importance from XGBoost against domain expectations
4. Train Logistic Regression with WOE as interpretability baseline
5. Compare ROC-AUC and Precision-Recall AUC across models
