# Project Execution Plan: Credit Risk Modeling
## End-to-End Coding Roadmap with MLOps Best Practices

---

## 1. Technology Stack & Dependencies

### 1.1 Python Version
**Selected**: Python 3.11.x (latest stable)

**Rationale**:
- XGBoost 3.x requires Python ≥3.10 [web:9]
- Python 3.11 is mature and stable (3.12 is cutting edge but some ML libraries still catching up)
- Full compatibility with all major ML/data libraries
- Better performance than 3.10 (10-60% speedup in many operations)

### 1.2 Core Dependencies (requirements.txt)

```txt
# ============================================================================
# DATA PROCESSING & LOADING
# ============================================================================
polars==0.20.3              # Fast dataframe library for large files
pyarrow==14.0.1             # Parquet file support
pandas==2.1.4               # For compatibility with some ML libraries
numpy==1.26.2               # Numerical operations

# ============================================================================
# EXPLORATORY DATA ANALYSIS & VISUALIZATION
# ============================================================================
matplotlib==3.8.2           # Plotting
seaborn==0.13.0             # Statistical visualizations
plotly==5.18.0              # Interactive plots
sweetviz==2.3.1             # Automated EDA reports
ydata-profiling==4.6.0      # Comprehensive data profiling (formerly pandas-profiling)

# ============================================================================
# MACHINE LEARNING
# ============================================================================
scikit-learn==1.3.2         # Core ML algorithms (LogReg, RandomForest)
xgboost==2.0.3              # Gradient boosting
lightgbm==4.1.0             # Alternative gradient boosting
catboost==1.2.2             # Categorical feature handling
imbalanced-learn==0.11.0    # SMOTE and class imbalance handling

# ============================================================================
# MODEL INTERPRETABILITY
# ============================================================================
shap==0.44.0                # SHAP values for model interpretation
eli5==0.13.0                # Model interpretation
lime==0.2.0.1               # Local interpretable model explanations

# ============================================================================
# MLOPS & EXPERIMENT TRACKING
# ============================================================================
mlflow==2.9.2               # Experiment tracking, model registry, versioning
dvc==3.37.0                 # Data version control
dagshub==0.3.14             # Git-based MLflow hosting (optional, free tier)

# ============================================================================
# DEPLOYMENT & API
# ============================================================================
streamlit==1.29.0           # Interactive web app
fastapi==0.108.0            # REST API (optional, for production-grade serving)
uvicorn==0.25.0             # ASGI server for FastAPI
pydantic==2.5.3             # Data validation

# ============================================================================
# UTILITIES & ENVIRONMENT
# ============================================================================
python-dotenv==1.0.0        # Environment variable management
pyyaml==6.0.1               # Config file parsing
loguru==0.7.2               # Better logging
tqdm==4.66.1                # Progress bars
jupyter==1.0.0              # Notebooks for exploration
ipykernel==6.28.0           # Jupyter kernel
black==23.12.1              # Code formatting
ruff==0.1.9                 # Fast linter
pre-commit==3.6.0           # Git hooks for code quality
```

### 1.3 Installation Command

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

---

## 2. Project Directory Structure

```
lending-club-fraud-detection/
│
├── README.md                          # Project overview
├── requirements.txt                   # Dependencies
├── .env                               # Environment variables (API keys, paths)
├── .gitignore                         # Git ignore file
├── pyproject.toml                     # Project config (black, ruff settings)
│
├── docs/                              # Documentation
│   ├── 01_project_charter.md         # [DONE] Business objectives
│   ├── 02_domain_knowledge.md        # [DONE] Credit risk concepts
│   ├── 03_data_documentation.md      # [DONE] Feature inventory
│   ├── 04_eda_insights.md            # [TODO] EDA findings
│   ├── 05_feature_engineering.md     # [TODO] Feature engineering decisions
│   ├── 06_modeling_approach.md       # [TODO] Model selection & tuning
│   ├── 07_model_performance.md       # [TODO] Evaluation results
│   ├── 08_model_interpretation.md    # [TODO] SHAP analysis
│   ├── 09_deployment_guide.md        # [TODO] Deployment instructions
│   └── 10_lessons_learned.md         # [TODO] Retrospective
│
├── data/                              # Data storage (gitignored except README)
│   ├── raw/                           # Raw CSV from Lending Club (1.6GB)
│   │   └── accepted_loans.csv
│   ├── processed/                     # Cleaned parquet files
│   │   ├── base_dataset.parquet       # Filtered, 95 features + target
│   │   ├── train.parquet              # Training set (with feature engineering)
│   │   ├── test.parquet               # Test set
│   │   └── validation.parquet         # Holdout validation set (optional)
│   └── engineered/                    # Feature store
│       ├── features_v1.parquet        # First feature set
│       ├── features_v2.parquet        # Second iteration
│       └── features_v3.parquet        # Final feature set
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_data_loading.ipynb          # Initial data load & profiling
│   ├── 02_eda_univariate.ipynb        # Univariate analysis
│   ├── 03_eda_bivariate.ipynb         # Bivariate analysis (vs target)
│   ├── 04_feature_engineering.ipynb   # Feature creation experiments
│   ├── 05_baseline_modeling.ipynb     # Logistic Regression baseline
│   ├── 06_tree_models.ipynb           # RF, XGBoost, LightGBM, CatBoost
│   └── 07_model_interpretation.ipynb  # SHAP, feature importance
│
├── src/                               # Source code (production-ready modules)
│   ├── __init__.py
│   ├── config.py                      # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_raw.py                # Load raw CSV -> base parquet
│   │   ├── preprocess.py              # Data cleaning & filtering
│   │   └── feature_engineering.py     # Feature creation pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py                # Logistic Regression
│   │   ├── tree_models.py             # RF, XGBoost, LightGBM, CatBoost
│   │   ├── train.py                   # Training orchestration
│   │   └── evaluate.py                # Evaluation metrics
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── eda_plots.py               # EDA visualization functions
│   │   └── model_plots.py             # ROC curves, confusion matrix, etc.
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                  # Logging setup
│       └── helpers.py                 # Utility functions
│
├── mlflow/                            # MLflow artifacts (gitignored)
│   ├── mlruns/                        # Experiment runs
│   └── models/                        # Registered models
│
├── streamlit_app/                     # Streamlit deployment
│   ├── app.py                         # Main Streamlit app
│   ├── pages/
│   │   ├── 1_Data_Explorer.py         # Data exploration page
│   │   ├── 2_Model_Performance.py     # Model metrics & visualizations
│   │   └── 3_Predict.py               # Live prediction interface
│   └── utils/
│       └── load_model.py              # Model loading utilities
│
├── tests/                             # Unit tests (pytest)
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   └── test_models.py
│
├── scripts/                           # Executable scripts
│   ├── 01_convert_to_parquet.py       # CSV -> Parquet conversion
│   ├── 02_run_eda.py                  # Generate EDA report
│   ├── 03_train_models.py             # Train all models with MLflow
│   ├── 04_evaluate_models.py          # Compare model performance
│   └── 05_deploy_streamlit.py         # Launch Streamlit app
│
└── .dvc/                              # DVC configuration (data versioning)
```

---

## 3. Data Strategy: Versions & Layering

### 3.1 Data Versioning Approach

We'll create **multiple data versions** as we progress through the pipeline:

| Version | File | Purpose | Size Estimate |
|---------|------|---------|---------------|
| **Raw** | `data/raw/accepted_loans.csv` | Original Lending Club data | 1.6 GB |
| **Base** | `data/processed/base_dataset.parquet` | Filtered rows (Fully Paid/Charged Off), 95 features + target | ~800 MB |
| **Features V1** | `data/engineered/features_v1.parquet` | Base + simple features (FICO mid, DTI bins, utilization flags) | ~900 MB |
| **Features V2** | `data/engineered/features_v2.parquet` | V1 + interaction features, aggregates | ~1 GB |
| **Features V3** | `data/engineered/features_v3.parquet` | V2 + domain-specific ratios, temporal features | ~1.1 GB |
| **Train/Test** | `data/processed/train.parquet`, `test.parquet` | Final split for modeling | ~800 MB + 200 MB |

### 3.2 Why Multiple Feature Versions?

1. **Baseline Model** (LogReg): Use Features V1 (simple, interpretable)
2. **Tree Models** (RF, XGBoost): Use Features V2/V3 (can handle complexity)
3. **Ablation Studies**: Compare V1 vs V2 vs V3 to measure feature engineering impact
4. **Model Comparison**: Ensure fair comparison by using same feature set within each experiment group

### 3.3 Data Version Control with DVC

We'll use **DVC (Data Version Control)** to track data versions without storing large files in Git:

```bash
# Initialize DVC
dvc init

# Track raw data
dvc add data/raw/accepted_loans.csv
git add data/raw/accepted_loans.csv.dvc .gitignore

# Track processed data
dvc add data/processed/base_dataset.parquet
git add data/processed/base_dataset.parquet.dvc

# Push to remote storage (e.g., AWS S3, Google Drive, local NAS)
dvc remote add -d myremote s3://my-bucket/lending-club-data
dvc push
```

**Benefits**:
- Git tracks metadata (.dvc files), not large data files
- Team members can `dvc pull` to get exact data versions
- Reproducibility: every Git commit links to specific data version

---

## 4. Model Strategy: What Models, When, and Why

### 4.1 Model Selection Plan

We'll train **4 models** in sequence, from simplest to most complex:

| # | Model | Library | When to Use | Feature Set | Priority |
|---|-------|---------|-------------|-------------|----------|
| 1 | **Logistic Regression** | scikit-learn | Baseline, interpretable, fast | Features V1 (simple) | ✅ Phase 1 |
| 2 | **Random Forest** | scikit-learn | Non-linear, robust, feature importance | Features V2 | ✅ Phase 1 |
| 3 | **XGBoost** | xgboost | Industry standard, handles missing data, regularization | Features V3 | ✅ Phase 1 |
| 4 | **LightGBM** | lightgbm | Faster than XGBoost, similar performance | Features V3 | ⚠️ Optional (Phase 2) |

### 4.2 Why XGBoost as Primary Focus?

- **Industry Standard**: Most widely used in credit risk modeling
- **Handles Missing Data**: Native support for missing values (no imputation needed)
- **Regularization**: L1/L2 penalties prevent overfitting
- **Feature Importance**: Built-in SHAP value integration
- **Class Imbalance**: `scale_pos_weight` parameter for imbalanced classes
- **Production-Ready**: Stable, well-documented, widely deployed

### 4.3 Model Training Strategy

**Phase 1: Quick Prototyping (Week 1-2)**
- Train 3 models (LogReg, RF, XGBoost) with **default hyperparameters**
- Use **5-fold cross-validation** on training set
- Evaluate on held-out test set
- Document baseline performance in `07_model_performance.md`

**Phase 2: Hyperparameter Tuning (Week 2-3)**
- Focus on **best-performing model** from Phase 1 (likely XGBoost)
- Use **Optuna** or **scikit-learn GridSearchCV** for hyperparameter tuning
- Track experiments with **MLflow**
- Goal: +2-5% improvement in ROC-AUC

**Phase 3: Ensemble (Optional, Week 3-4)**
- If time permits, create **voting ensemble** or **stacking** with top 2-3 models
- Typically yields +1-3% improvement

---

## 5. MLOps & Best Practices

### 5.1 Experiment Tracking with MLflow

**Why MLflow?**
- **Experiment Tracking**: Log metrics, parameters, artifacts
- **Model Registry**: Version and stage models (Staging → Production)
- **Reproducibility**: Link models to exact code/data/config
- **UI Dashboard**: Compare runs visually
- **Industry Standard**: Used by 1000s of companies

**MLflow Workflow**:

```python
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Start experiment
mlflow.set_experiment("lending_club_default_prediction")

with mlflow.start_run(run_name="xgboost_v1"):
    # Log parameters
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("feature_set", "features_v3")

    # Train model
    model = xgb.XGBClassifier(...)
    model.fit(X_train, y_train)

    # Log metrics
    mlflow.log_metric("roc_auc_train", 0.75)
    mlflow.log_metric("roc_auc_test", 0.72)
    mlflow.log_metric("precision_at_80_recall", 0.28)

    # Log model
    mlflow.xgboost.log_model(model, "model")

    # Log artifacts (plots, feature importance)
    mlflow.log_artifact("plots/roc_curve.png")
    mlflow.log_artifact("plots/feature_importance.png")
```

### 5.2 Model Versioning Strategy

**Model Registry Flow**:
1. **Development**: Train in notebook/script, log to MLflow
2. **Staging**: Promote best model to "Staging" in Model Registry
3. **Validation**: Test on held-out validation set (or via A/B test in production)
4. **Production**: Promote to "Production" stage
5. **Archiving**: Older models marked as "Archived"

**Naming Convention**:
```
Model Name: lending_club_xgboost
Version: 1, 2, 3, ...
Stage: None → Staging → Production → Archived
Tags: {"feature_set": "v3", "data_version": "2024-12-26"}
```

### 5.3 Code Quality & Reproducibility

**Pre-commit Hooks** (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
```

**Config Management** (`src/config.py`):
```python
from pathlib import Path
from pydantic import BaseModel

class Config(BaseModel):
    # Paths
    DATA_RAW: Path = Path("data/raw")
    DATA_PROCESSED: Path = Path("data/processed")
    DATA_ENGINEERED: Path = Path("data/engineered")
    MLFLOW_URI: str = "file:./mlflow"

    # Model hyperparameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5

    # XGBoost defaults
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_LEARNING_RATE: float = 0.1
    XGBOOST_N_ESTIMATORS: int = 100

config = Config()
```

### 5.4 Logging with Loguru

```python
from loguru import logger

# Configure logger
logger.add("logs/training_{time}.log", rotation="500 MB")

# Usage
logger.info("Starting data loading...")
logger.warning("Found 10% missing values in 'emp_title'")
logger.error("Model training failed: {error}", error=e)
```

---

## 6. Streamlit App Strategy

### 6.1 App Structure (3 Pages)

**Page 1: Data Explorer**
- Upload new data or use sample
- Display summary statistics
- Interactive filters (FICO range, DTI, state)
- Distribution plots (FICO, DTI, loan amount)

**Page 2: Model Performance**
- Model comparison table (ROC-AUC, precision, recall)
- ROC curves (all models overlaid)
- Confusion matrices
- Feature importance (top 20 features)
- SHAP summary plot

**Page 3: Predict (Single Loan)**
- Input form: FICO, DTI, loan amount, purpose, etc.
- Real-time prediction: Probability of default
- SHAP force plot: Why did model predict this?
- Risk category: Low / Medium / High

### 6.2 Deployment Options

**Local Development**:
```bash
streamlit run streamlit_app/app.py
```

**Cloud Deployment**:
- **Streamlit Cloud** (free tier): Easy, 1-click GitHub integration
- **Heroku** (free tier deprecated, use paid): Requires Procfile
- **AWS EC2**: Full control, requires setup
- **Render** (free tier): Simple Docker deployment

---

## 7. Step-by-Step Execution Flow

### **PHASE 0: Setup (Day 0) ✅ DONE**
- [x] Create project charter
- [x] Document domain knowledge
- [x] Create data documentation (feature inventory)

---

### **PHASE 1: Data Preparation (Days 1-2)**

#### Step 1.1: Convert CSV to Parquet ⏳ NEXT
**File**: `scripts/01_convert_to_parquet.py`

**Tasks**:
1. Load raw CSV with Polars (lazy loading for memory efficiency)
2. Filter rows: keep only `loan_status` in ['Fully Paid', 'Charged Off']
3. Select 95 features + target
4. Inspect data types, missing values, row counts
5. Save as `data/processed/base_dataset.parquet`
6. **Document**: Update `03_data_documentation.md` Section 5 with actual stats

**Deliverable**: 
- `data/processed/base_dataset.parquet` (~800 MB)
- Updated `03_data_documentation.md` with data quality metrics

**Code Skeleton**:
```python
import polars as pl

# Load raw CSV (lazy)
df = pl.scan_csv("data/raw/accepted_loans.csv")

# Filter rows
df_filtered = df.filter(
    pl.col("loan_status").is_in(["Fully Paid", "Charged Off"])
)

# Select columns (95 features + target)
features_to_use = [...]  # From data documentation
df_final = df_filtered.select(features_to_use + ["loan_status"])

# Collect and save
df_final.collect().write_parquet("data/processed/base_dataset.parquet")
```

---

#### Step 1.2: Data Profiling ⏳
**File**: `notebooks/01_data_loading.ipynb` OR `scripts/02_run_eda.py`

**Tasks**:
1. Load `base_dataset.parquet`
2. Generate automated EDA report with `ydata-profiling`
3. Inspect:
   - Missing value % per column
   - Data types
   - Target distribution
   - Outliers in FICO, DTI, income
4. **Document**: Create initial draft of `04_eda_insights.md`

**Deliverable**:
- HTML report: `reports/data_profile.html`
- Updated `03_data_documentation.md` Section 5

---

### **PHASE 2: Exploratory Data Analysis (Days 3-5)**

#### Step 2.1: Univariate Analysis
**File**: `notebooks/02_eda_univariate.ipynb`

**Tasks**:
1. Distribution plots for numeric features (histograms, box plots)
2. Frequency tables for categorical features (purpose, state, home ownership)
3. Identify:
   - Skewed distributions (income, loan amount)
   - High-cardinality categoricals (state, emp_title)
4. **Document**: Add to `04_eda_insights.md`

**Deliverable**: 
- Plots saved to `reports/figures/univariate/`
- Section in `04_eda_insights.md`

---

#### Step 2.2: Bivariate Analysis (vs Target)
**File**: `notebooks/03_eda_bivariate.ipynb`

**Tasks**:
1. Default rate by:
   - FICO bins (< 650, 650-699, 700-749, 750+)
   - DTI bins (< 20%, 20-30%, 30-40%, 40-50%, > 50%)
   - Revolving utilization bins
   - Loan purpose
   - Home ownership
   - Term (36 vs 60 months)
2. Statistical tests (chi-square for categorical, t-test for numeric)
3. Correlation heatmap (numeric features vs target)
4. **Document**: Add to `04_eda_insights.md`

**Deliverable**:
- Plots saved to `reports/figures/bivariate/`
- Key insights in `04_eda_insights.md`

**Expected Insights** (to validate domain knowledge):
- FICO < 650 → 25-30% default rate
- FICO 750+ → 5-10% default rate
- DTI > 40% → elevated default
- 60-month term → higher default than 36-month
- Small business, credit card → higher default

---

#### Step 2.3: Multivariate Analysis
**File**: `notebooks/03_eda_bivariate.ipynb` (continued)

**Tasks**:
1. Interaction effects: FICO × DTI, FICO × revolving utilization
2. Segmentation: High-risk profiles (low FICO + high DTI + high utilization)
3. **Document**: Add to `04_eda_insights.md`

**Deliverable**:
- Advanced plots (3D scatter, heatmaps)
- Section in `04_eda_insights.md`

---

### **PHASE 3: Feature Engineering (Days 6-8)**

#### Step 3.1: Create Features V1 (Simple)
**File**: `notebooks/04_feature_engineering.ipynb` OR `src/data/feature_engineering.py`

**Tasks**:
1. **Numeric transformations**:
   - `fico_mid = (ficoRangeLow + ficoRangeHigh) / 2`
   - `credit_history_years = (current_date - earliestCrLine).years`
   - `loan_to_income = loanAmnt / annualInc`
   - `installment_to_income = installment / (annualInc / 12)` (recompute installment)
2. **Categorical encoding**:
   - One-hot encode: `purpose`, `homeOwnership`, `term`
   - Ordinal encode: Home ownership (RENT < OTHER < MORTGAGE < OWN)
   - Group rare categories in `addrState` (< 1% → "OTHER")
3. **Flags**:
   - `high_dti = (dti > 43)`
   - `high_utilization = (revolUtil > 70)`
   - `has_delinq = (delinq2Yrs > 0)`
   - `has_public_record = (pubRec > 0)`
4. **Binning**:
   - FICO bins: [< 650, 650-699, 700-749, 750+]
   - DTI bins: [< 20, 20-30, 30-40, 40-50, > 50]
5. Save as `features_v1.parquet`
6. **Document**: Create `05_feature_engineering.md`

**Deliverable**:
- `data/engineered/features_v1.parquet`
- `05_feature_engineering.md` with feature definitions

---

#### Step 3.2: Create Features V2 (Interaction Terms)
**File**: Same as 3.1

**Tasks**:
1. **Interactions**:
   - `fico_dti_interaction = fico_mid * dti`
   - `fico_utilization_interaction = fico_mid * revolUtil`
2. **Aggregates**:
   - `total_derogatory_marks = pubRec + tax_liens + chargeoff_within_12_mths`
   - `total_inquiries = inqLast6Mths + inq_last_12m`
3. Save as `features_v2.parquet`
4. **Document**: Update `05_feature_engineering.md`

**Deliverable**:
- `data/engineered/features_v2.parquet`

---

#### Step 3.3: Create Features V3 (Advanced)
**File**: Same as 3.1

**Tasks**:
1. **Domain-specific ratios**:
   - `new_accounts_ratio = accOpenPast24Mths / totalAcc`
   - `active_to_total_ratio = num_actv_rev_tl / num_rev_accts`
2. **Temporal features** (if `issue_d` available):
   - Loan issue year, month, quarter
   - Loan age (months since origination)
3. **Text features** (if time permits):
   - Extract keywords from `emp_title` (engineer, manager, driver, etc.)
4. Save as `features_v3.parquet`
5. **Document**: Finalize `05_feature_engineering.md`

**Deliverable**:
- `data/engineered/features_v3.parquet`
- Complete `05_feature_engineering.md`

---

### **PHASE 4: Baseline Modeling (Days 9-10)**

#### Step 4.1: Train/Test Split
**File**: `src/data/preprocess.py`

**Tasks**:
1. Load `features_v1.parquet`
2. Create binary target: `y = (loan_status == 'Charged Off').astype(int)`
3. Split: 80% train, 20% test (stratified by target)
4. Save splits: `train.parquet`, `test.parquet`

**Deliverable**:
- `data/processed/train.parquet`
- `data/processed/test.parquet`

---

#### Step 4.2: Logistic Regression Baseline
**File**: `notebooks/05_baseline_modeling.ipynb` OR `src/models/baseline.py`

**Tasks**:
1. Train Logistic Regression with `class_weight='balanced'`
2. 5-fold cross-validation
3. Evaluate:
   - ROC-AUC (train, test)
   - Precision at 80% recall
   - Confusion matrix
4. Log to MLflow
5. **Document**: Create `06_modeling_approach.md`

**Deliverable**:
- Trained model logged in MLflow
- Initial `06_modeling_approach.md`

---

### **PHASE 5: Tree-Based Models (Days 11-14)**

#### Step 5.1: Random Forest
**File**: `notebooks/06_tree_models.ipynb` OR `src/models/tree_models.py`

**Tasks**:
1. Train Random Forest (features_v2)
2. Hyperparameters: `n_estimators=100, max_depth=10, min_samples_split=100`
3. Evaluate and log to MLflow
4. Feature importance plot
5. **Document**: Update `06_modeling_approach.md`

---

#### Step 5.2: XGBoost (Primary Focus)
**File**: Same as 5.1

**Tasks**:
1. Train XGBoost (features_v3)
2. Hyperparameters:
   - `max_depth=6`
   - `learning_rate=0.1`
   - `n_estimators=100`
   - `scale_pos_weight = (# Fully Paid) / (# Charged Off)`
   - `eval_metric='auc'`
3. 5-fold cross-validation
4. Evaluate and log to MLflow
5. SHAP value computation
6. **Document**: Update `06_modeling_approach.md`

**Deliverable**:
- Best XGBoost model in MLflow
- SHAP plots

---

#### Step 5.3: LightGBM (Optional)
**File**: Same as 5.1

**Tasks**:
1. Train LightGBM (features_v3)
2. Compare speed and performance to XGBoost
3. Log to MLflow
4. **Document**: Update `06_modeling_approach.md`

---

### **PHASE 6: Model Evaluation & Interpretation (Days 15-16)**

#### Step 6.1: Model Comparison
**File**: `scripts/04_evaluate_models.py` OR `notebooks/07_model_interpretation.ipynb`

**Tasks**:
1. Load all models from MLflow
2. Create comparison table:
   | Model | ROC-AUC (Train) | ROC-AUC (Test) | Precision @ 80% Recall | Training Time |
3. Plot overlaid ROC curves
4. **Document**: Create `07_model_performance.md`

**Deliverable**:
- Comparison plots
- `07_model_performance.md`

---

#### Step 6.2: Model Interpretation (SHAP)
**File**: Same as 6.1

**Tasks**:
1. For **best model** (likely XGBoost):
   - SHAP summary plot (top 20 features)
   - SHAP waterfall plot (individual predictions)
   - SHAP dependence plots (FICO, DTI, revolUtil)
2. Feature importance: Compare SHAP vs model's native importance
3. **Document**: Create `08_model_interpretation.md`

**Deliverable**:
- SHAP plots saved to `reports/figures/interpretation/`
- `08_model_interpretation.md`

---

### **PHASE 7: Deployment (Days 17-19)**

#### Step 7.1: Build Streamlit App
**File**: `streamlit_app/app.py`

**Tasks**:
1. Create 3 pages (Data Explorer, Model Performance, Predict)
2. Load best model from MLflow Model Registry
3. Implement prediction logic with SHAP force plot
4. Test locally
5. **Document**: Create `09_deployment_guide.md`

**Deliverable**:
- Working Streamlit app
- `09_deployment_guide.md`

---

#### Step 7.2: Deploy to Cloud
**File**: `streamlit_app/` (with deployment config)

**Tasks**:
1. Deploy to Streamlit Cloud (easiest) or Render
2. Set up environment variables (MLflow URI, model version)
3. Test public URL
4. **Document**: Update `09_deployment_guide.md` with deployment steps

**Deliverable**:
- Live app URL
- Complete `09_deployment_guide.md`

---

### **PHASE 8: Documentation & Wrap-Up (Day 20)**

#### Step 8.1: Lessons Learned
**File**: `docs/10_lessons_learned.md`

**Tasks**:
1. Reflect on what worked well
2. What would you do differently?
3. Challenges faced
4. Future improvements

**Deliverable**:
- Complete `10_lessons_learned.md`

---

#### Step 8.2: Update README
**File**: `README.md`

**Tasks**:
1. Project overview
2. Installation instructions
3. How to run scripts
4. How to access Streamlit app
5. Links to documentation

**Deliverable**:
- Polished `README.md`

---

## 8. Success Metrics (Checkpoints)

| Checkpoint | Metric | Target | Status |
|------------|--------|--------|--------|
| **Phase 1** | Data loaded to Parquet | < 1 minute load time | ⏳ |
| **Phase 2** | EDA insights documented | ≥ 10 key findings | ⏳ |
| **Phase 3** | Features engineered | 3 versions created | ⏳ |
| **Phase 4** | Baseline model trained | ROC-AUC > 0.65 | ⏳ |
| **Phase 5** | XGBoost trained | ROC-AUC > 0.70 | ⏳ |
| **Phase 6** | Model interpreted | SHAP plots generated | ⏳ |
| **Phase 7** | Streamlit deployed | Public URL live | ⏳ |
| **Phase 8** | Documentation complete | All 10 docs done | ⏳ |

---

## 9. Timeline (Part-Time, 3-4 Weeks)

| Week | Days | Phase | Deliverables |
|------|------|-------|--------------|
| **Week 1** | 1-2 | Data Preparation | Parquet conversion, data profiling |
| | 3-5 | EDA | Univariate, bivariate, multivariate analysis |
| | 6-7 | Feature Engineering (start) | Features V1 created |
| **Week 2** | 8 | Feature Engineering (finish) | Features V2, V3 created |
| | 9-10 | Baseline Modeling | Logistic Regression trained |
| | 11-14 | Tree Models | RF, XGBoost, LightGBM trained |
| **Week 3** | 15-16 | Evaluation & Interpretation | Model comparison, SHAP analysis |
| | 17-19 | Deployment | Streamlit app built & deployed |
| **Week 4** | 20 | Wrap-Up | Documentation finalized |

---

## 10. Key Reminders for Success

### ✅ Do's
- **Commit frequently** to Git (feature branches → main)
- **Log every experiment** to MLflow (even failed runs)
- **Document as you go** (don't wait until the end)
- **Version data** with DVC (easy rollbacks)
- **Write modular code** in `src/` (not just notebooks)
- **Use config files** (`config.py`, YAML) instead of hardcoding
- **Test on test set only once** (avoid overfitting to test)

### ❌ Don'ts
- **Don't skip data profiling** (causes downstream issues)
- **Don't ignore class imbalance** (use `class_weight`, `scale_pos_weight`)
- **Don't use test set for feature engineering** (leakage!)
- **Don't hardcode file paths** (use Path, config)
- **Don't commit large files** to Git (use .gitignore, DVC)
- **Don't skip model interpretation** (SHAP is required for trust)

---

## 11. Next Immediate Action

**START HERE**: 
```bash
cd lending-club-fraud-detection
python scripts/01_convert_to_parquet.py
```

This will:
1. Load the raw 1.6GB CSV with Polars (fast!)
2. Filter to resolved loans (Fully Paid / Charged Off)
3. Select 95 features + target
4. Save as Parquet (~800 MB)
5. Print data quality stats

**Expected runtime**: 2-5 minutes (Polars is fast!)

---

## 12. Current Work Steps here:
Complete Analysis Workflow Per Feature
Step-by-Step Process:

For each categorical feature (e.g., home_ownership):

✅ Frequency Distribution → Identify rare categories
✅ Default Rate by Category → Calculate risk levels
✅ Chi-Square Test → Check statistical significance
✅ Cramér's V → Measure effect size
✅ Data Quality Check → Standardize text, check for placeholders
✅ Grouping Decision → Apply business logic if needed
✅ Final Decision: Keep / Group / Drop / Transform

---


