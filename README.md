# Credit Risk Modeling: Lending Club Loan Default Prediction

## Overview

A production-grade Probability of Default (PD) model built on historical Lending Club loan data
(2007–2018). The system predicts the likelihood of borrower default using a fully automated
machine learning pipeline — from raw data ingestion through feature engineering, model training,
threshold optimization, and live inference via a deployed web application.

The project is structured to reflect industry standards in financial ML: modular `src/` code,
experiment tracking with MLflow, reproducible training splits, a Streamlit app for interactive
loan assessment, and an LLM-powered narrative layer that explains each decision in plain English.

Live Demo: https://credit-risk-model-salman.streamlit.app

***

## Business Context

Credit default prediction is a core function in consumer lending. This model supports
underwriting decisions by surfacing a calibrated default probability for each loan application,
alongside the top risk drivers specific to that borrower. The three-zone decision framework
(Approve / Manual Review / Reject) mirrors real-world credit policy design.

Each assessment is augmented by an AI-generated risk narrative — a structured plain-English
explanation of the model's decision, written in the voice of a senior credit analyst. This
bridges the gap between raw model output and actionable underwriter insight.

***

## Tech Stack

| Layer                  | Tools                                              |
|------------------------|----------------------------------------------------|
| Data processing        | pandas 2.1.4, numpy 1.26.2      |
| Feature engineering    | scikit-learn 1.3.2 (custom transformers)           |
| Modeling               | XGBoost 2.0.3                                      |
| Hyperparameter tuning  | Optuna 3.5.0                                       |
| Experiment tracking    | MLflow 2.11.0                                      |
| Explainability         | SHAP 0.44.0                                        |
| Web application        | Streamlit 1.41.0                                   |
| LLM narrative layer    | Groq API (LLaMA 3.3 70B)                           |
| Model persistence      | joblib 1.3.2, cloudpickle                          |
| Visualization          | Matplotlib 3.8.2, Seaborn 0.13.0, Plotly 5.18.0   |

Python version: 3.11.x

***

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

***

## LLM Narrative Layer

After each prediction, an AI-generated risk narrative is produced via the Groq API
(LLaMA 3.3 70B). The narrative is structured as a senior credit analyst's assessment and
contains three sections:

- **Risk Assessment Summary** — one-paragraph verdict with the key reason for the decision
- **Top Risk Drivers** — plain-English explanation of the SHAP factors driving the score
- **Recommendation** — actionable guidance for the underwriter

The LLM pipeline is fully modular and lives in `llm/`:

```
llm/
├── data/               # ApplicantFeatures + PredictionResult dataclasses
├── prompts/            # system_prompt.md, feature_glossary.md, user_prompt_builder.py
├── client/             # llm_client.py — Groq API wrapper
├── narrative/          # narrative.py — response parser
└── orchestrator/       # llm_orchestrator.py — single entry point
```

The system prompt and feature glossary are loaded once per session (cached via `lru_cache`).
The user prompt is dynamically built per borrower from their transformed features and SHAP
drivers. Narrative generation gracefully degrades — if the API call fails, the prediction
result is still shown and the error is surfaced as a non-blocking warning.

***

## Project Structure

```
lending-club-credit-risk/
├── app.py                          # Streamlit entry point
├── .env                            # GROQ_API_KEY (not committed)
├── requirements.txt
│
├── streamlit_app/
│   └── pages/
│       ├── 1_loan_assessment.py    # Borrower input form + prediction + LLM narrative
│       └── 2_model_insights.py     # Model performance + SHAP analysis
│
├── llm/                            # LLM narrative pipeline
│   ├── data/
│   ├── prompts/
│   ├── client/
│   ├── narrative/
│   └── orchestrator/
│
├── src/
│   ├── config/                     # master_config, numerical_config, categorical_config
│   ├── pipeline/                   # numerical, categorical, master pipelines
│   ├── transformers/               # custom sklearn transformers
│   └── training/                   # train.py, evaluate.py, threshold_tuner.py
│
├── models/
│   ├── xgboost_model.pkl
│   ├── master_pipeline.pkl
│   └── best_threshold.json
│
└── docs/
    ├── 01_project_charter.md
    ├── 02_domain_knowledge.md
    ├── 03_data_documentation.md
    └── 04_eda_insights.md
```

***

## Documentation

Full documentation is available in the `docs/` folder:

- [Project Charter](docs/01_project_charter.md) — Business context and objectives
- [Domain Knowledge](docs/02_domain_knowledge.md) — Credit risk fundamentals
- [Data Documentation](docs/03_data_documentation.md) — Dataset overview and feature definitions
- [EDA Insights](docs/04_eda_insights.md) — Numerical and categorical analysis findings

***

## Local Setup

```bash
git clone https://github.com/salman7420/credit-risk-model.git
cd credit-risk-model
python -m venv env && source env/bin/activate
pip install -r requirements.txt
```

**Set up your Groq API key** (required for LLM narratives):

```bash
# Create a .env file in the project root
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

Get a free key at [console.groq.com](https://console.groq.com) — no credit card required.

```bash
# Train model (requires data in data/splits/)
PYTHONPATH=src python -m training.train

# Launch app
streamlit run app.py
```

> **Note:** The LLM narrative layer requires a valid `GROQ_API_KEY` in `.env`. If the key is
> missing, the app still runs and shows predictions — only the narrative section is skipped.

***

## Contact

LinkedIn: https://www.linkedin.com/in/salman-rasheed-ai/  
Email: salmandatascience25@gmail.com
