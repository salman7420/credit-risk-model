# Project Charter: Credit Risk Modeling

**Owner**: Salman Rasheed
**Dataset**: Lending Club Accepted Loans (2007–2018)

---

## Business Problem

Lending Club issued 2.26 million personal loans between 2007 and 2018. Approximately 15–20% of
these loans were charged off, resulting in significant financial losses for investors. This project
builds a Probability of Default (PD) model to predict whether a borrower will default before a
loan is approved.

---

## Scope

**In scope**
- Binary classification: Charged Off (default) vs. Fully Paid (non-default)
- Features available at loan origination only (no post-disbursement data)
- Streamlit application for interactive loan assessment

**Out of scope**
- Loss Given Default (LGD) and Exposure at Default (EAD) modeling
- Real-time streaming predictions
- A/B testing framework

---

## Success Criteria

| Metric | Target |
|--------|--------|
| ROC-AUC | > 0.70 |
| Precision at 80% recall | > 0.25 |
| Interpretability | Required (SHAP-based explanations) |

---

## Methodology

| Phase | Deliverable |
|-------|-------------|
| Data engineering | Cleaned Parquet, filtered to closed loans |
| EDA | Insights report, univariate and bivariate analysis |
| Feature engineering | Numerical and categorical transformer pipelines |
| Modeling | XGBoost with Optuna tuning, tracked in MLflow |
| Threshold optimization | F1-maximizing threshold on validation set |
| Deployment | Streamlit app with per-borrower SHAP explanations |

---

## Risks

| Risk | Mitigation |
|------|------------|
| Data leakage | Strict exclusion of post-origination columns |
| Class imbalance | scale_pos_weight, AUC-PR as primary metric |
| Overfitting | Cross-validation, early stopping |
