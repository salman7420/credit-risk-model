# Business context, objectives

# Project Charter: Credit Risk Modeling

**Document Owner**: Salman Rasheed
**Date**: December 18, 2025  
**Status**: Data documentation (pre-coding)

---

## 1. Business Problem

### 1.1 Context
Lending Club, a peer-to-peer lending platform, operated from 2007-2018 and issued 2.26 million personal loans. Approximately 15-20% of these loans defaulted (were not fully paid back), resulting in significant financial losses for individual investors who funded the loans.

### 1.2 Problem Statement:
We need to predict whether a loan applicant will default before approving the loan in order to make better lending decisions and reduce investor losses. This allows:
- Better approval/rejection decisions
- Risk-based pricing (higher interest rates for riskier borrowers)
- Portfolio risk management

### 1.3 Success Criteria
| Metric | Target | Rationale |
|--------|--------|-----------|
| ROC-AUC | > 0.70 | Industry benchmark for credit models |
| Precision (at 80% recall) | > 0.25 | Catch 80% of defaults while minimizing false alarms |
| Model Interpretability | Required | Regulatory compliance (Fair Lending Act) |

---

## 2. Scope

### 2.1 In Scope
- Predict binary outcome: Default (Charged Off) vs. Paid (Fully Paid)
- Use data available **at loan origination** only
- Build 3-4 candidate models
- Deploy a Streamlit demo app

### 2.2 Out of Scope
- Predicting exact loss amount (Loss Given Default - LGD)
- Real-time streaming predictions
- A/B testing framework

---

## 3. Data

### 3.1 Source
- **Dataset**: Lending Club Accepted Loans (2007-2018)
- **Size**: 2.26M rows, 153 columns, ~1.6GB

### 3.2 Target Variable
- `loan_status` → Binary classification
  - **Class 0 (Negative)**: "Fully Paid"
  - **Class 1 (Positive)**: "Charged Off"
  - **Excluded**: "Current", "Late", "In Grace Period" (incomplete loans)

### 3.3 Class Imbalance
- Expected: 80-85% Fully Paid, 15-20% Charged Off
- **Mitigation**: Use `class_weight='balanced'`, SMOTE, or adjust decision threshold

---

## 4. Methodology

### 4.1 Project Phases
| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 0. Planning & Research | 2-3 days | This document + domain knowledge |
| 1. Data Engineering | 2 days | Clean Parquet file |
| 2. EDA | 3-4 days | Insights report + visualizations |
| 3. Feature Engineering | 3-4 days | Feature set v1, v2, v3 |
| 4. Feature Selection | 2 days | Selected features (40-60 columns) |
| 5. Modeling | 5-7 days | Trained models + MLflow runs |
| 6. Evaluation | 2-3 days | Performance report + SHAP plots |
| 7. Deployment | 2-3 days | Streamlit app + Docker container |

**Total Estimated Time**: 3-4 weeks (part-time)

### 4.2 Algorithms to Test
1. **Logistic Regression** (Baseline, interpretable)
2. **Random Forest** (Non-linear, robust)
3. **XGBoost** (Industry standard)
4. **LightGBM** (Faster alternative)

---

## 5. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data leakage | High accuracy but useless model | Strict column filtering, temporal validation |
| Class imbalance | Model predicts "Paid" for everything | Use ROC-AUC, not accuracy; balance classes |
| Overfitting | Works on train, fails on test | Cross-validation, regularization |

---

## 6. Stakeholders (In Real Bank Context)

| Role | Responsibility | Our Project |
|------|----------------|-------------|
| Risk Manager | Approves model for production use | You (learning) |
| Data Scientist | Builds and validates model | You (coding) |
| Compliance Officer | Ensures fair lending compliance | N/A (educational) |
| IT/DevOps | Deploys model to production | You (Streamlit) |

---
