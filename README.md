# Credit Risk Modeling: Lending Club Loan Default Prediction

## Overview

A production-grade Probability of Default (PD) model built on historical Lending Club loan data
(2007–2018). The system predicts the likelihood of borrower default using a fully automated
machine learning pipeline — from raw data ingestion through feature engineering, model training,
threshold optimization, and live inference via a deployed web application.

The project is structured to reflect industry standards in financial ML: modular `src/` code,
experiment tracking with MLflow, reproducible training splits, and a Streamlit app for
interactive loan assessment.

Live Demo: https://credit-risk-model-salman.streamlit.app

---

## Business Context

Credit default prediction is a core function in consumer lending. This model supports
underwriting decisions by surfacing a calibrated default probability for each loan application,
alongside the top risk drivers specific to that borrower. The three-zone decision framework
(Approve / Manual Review / Reject) mirrors real-world credit policy design.

---

## Tech Stack

| Layer               | Tools                                              |
|---------------------|----------------------------------------------------|
| Data processing     | pandas 2.1.4, numpy 1.26.2, pyarrow 14.0.1        |
| Feature engineering | scikit-learn 1.3.2 (custom transformers)           |
| Modeling            | XGBoost 2.0.3                                      |
| Hyperparameter tuning | Optuna 3.5.0                                     |
| Experiment tracking | MLflow 2.11.0                                      |
| Explainability      | SHAP 0.44.0                                        |
| Web application     | Streamlit 1.41.0                                   |
| Model persistence   | joblib 1.3.2, cloudpickle                          |
| Visualization       | Matplotlib 3.8.2, Seaborn 0.13.0, Plotly 5.18.0   |

Python version: 3.11.x

---

## Project Structure

lending-club-credit-risk/
├── data/
│ ├── raw/ # Original and preprocessed loan CSV/Parquet files
│ ├── processed/ # Feature-selected dataset
│ └── splits/ # Train / validation / test splits
├── docs/ # Project documentation (see below)
├── models/ # Saved pipeline, threshold, and evaluation artifacts
├── notebooks/ # Exploratory analysis
├── src/
│ ├── feature_engineering/ # Custom sklearn transformers (numerical + categorical)
│ ├── pipeline/ # Master pipeline assembly
│ ├── preprocessing/ # Raw data filtering and column selection
│ └── training/ # Model training, threshold tuning, evaluation
├── streamlit_app/
│ ├── pages/ # Loan assessment page
│ └── utils/ # Predictor logic, config
├── .streamlit/ # Streamlit configuration
├── requirements.txt
└── README.md


---

## Machine Learning Pipeline

**1. Preprocessing**
Raw loan data is filtered to closed loans (fully paid or charged off) with a target variable
`default` (1 = charged off, 0 = fully paid). Post-filter dataset: ~1.1M loans.

**2. Feature Engineering**
Numerical pipeline: imputation, outlier capping (p99.5), binary flags, ratio features,
product interactions, log-stress features, and binning — producing 51 features from 35 inputs.

Categorical pipeline: rule-based cleaning (employment stability bucketing, regional grouping,
credit maturity binning, purpose risk bucketing), followed by OHE and ordinal encoding —
producing 11 features from 7 inputs.

Total features passed to model: 62.

**3. Model**
XGBoost gradient boosting classifier, tuned with Optuna (Bayesian optimization, 50 trials).
Class imbalance handled via `scale_pos_weight`. Evaluation metric: AUC-PR (precision-recall),
appropriate for the imbalanced default classification task.

**4. Threshold Optimization**
A custom threshold sweep selects the operating point that maximizes F1 on the validation set,
stored in `models/best_threshold.json`.

**5. Inference**
At prediction time, the full sklearn pipeline applies all transforms on raw input before
passing to XGBoost. SHAP TreeExplainer provides per-borrower feature attributions.
Decision zones: Approve (< 0.35), Manual Review (0.35–0.55), Reject (> 0.55).

---

## Documentation

Full documentation is available in the `docs/` folder:

- [Project Charter](docs/01_project_charter.md) — Business context and objectives
- [Domain Knowledge](docs/02_domain_knowledge.md) — Credit risk fundamentals
- [Data Documentation](docs/03_data_documentation.md) — Dataset overview and feature definitions
- [EDA Insights](docs/04_eda_insights.md) — Numerical and categorical analysis findings

---

## Local Setup

```bash
git clone https://github.com/salman7420/credit-risk-model.git
cd credit-risk-model
python -m venv env && source env/bin/activate
pip install -r requirements.txt

# Train model (requires data in data/splits/)
PYTHONPATH=src python -m training.train

# Launch app
streamlit run streamlit_app/app.py

Contact
LinkedIn: https://www.linkedin.com/in/salman-rasheed-ai/
Email: salmandatascience25@gmail.com