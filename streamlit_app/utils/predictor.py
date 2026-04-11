"""
predictor.py
------------
Central bridge between the Streamlit UI and the trained ML pipeline.

Changes vs original:
  - predict() now also returns `transformed_features` — a dict of
    {feature_name: value} for all 62 pipeline output features.
    Used by llm/applicant_data/ to build the full applicant snapshot.
"""

import json
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import cloudpickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH  = ROOT / "models" / "pipeline.pkl"
THRESHOLD_PATH = ROOT / "models" / "best_threshold.json"

APPROVE_BELOW = 0.35
REJECT_ABOVE  = 0.55

# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading model pipeline...")
def _load_pipeline():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"pipeline.pkl not found at {PIPELINE_PATH}\n"
            f"Run: PYTHONPATH=src python -m training.train"
        )
    with open(PIPELINE_PATH, "rb") as f:
        return cloudpickle.load(f)

@st.cache_resource(show_spinner="⏳ Loading decision threshold...")
def _load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"best_threshold.json not found at {THRESHOLD_PATH}\n"
            f"Run: PYTHONPATH=src python -m training.threshold_tuner"
        )
    with open(THRESHOLD_PATH) as f:
        return float(json.load(f)["best_threshold"])

@st.cache_resource(show_spinner="⏳ Building SHAP explainer...")
def _load_explainer(_model):
    return shap.TreeExplainer(_model)

# ─────────────────────────────────────────────────────────────────────────────
# DECISION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def _get_decision(probability: float) -> dict:
    if probability < APPROVE_BELOW:
        return {
            "label":       "APPROVE",
            "emoji":       "✅",
            "color":       "green",
            "description": "Low default risk. Borrower meets lending criteria.",
        }
    elif probability < REJECT_ABOVE:
        return {
            "label":       "MANUAL REVIEW",
            "emoji":       "🟡",
            "color":       "orange",
            "description": "Borderline risk. Recommend human review before decision.",
        }
    else:
        return {
            "label":       "REJECT",
            "emoji":       "🔴",
            "color":       "red",
            "description": "High default risk. Does not meet lending criteria.",
        }

# ─────────────────────────────────────────────────────────────────────────────
# SHAP — TOP FACTORS
# ─────────────────────────────────────────────────────────────────────────────
def _get_top_shap_factors(
    shap_values: np.ndarray,
    feature_names: list,
    top_n: int = 5,
) -> pd.DataFrame:
    shap_series = pd.Series(shap_values, index=feature_names)
    top_idx = shap_series.abs().nlargest(top_n).index
    result = pd.DataFrame({
        "feature":  top_idx,
        "shap_val": shap_series[top_idx].values,
    })
    result["direction"] = result["shap_val"].apply(
        lambda v: "⬆️ increases risk" if v > 0 else "⬇️ decreases risk"
    )
    result["shap_abs"] = result["shap_val"].abs()
    return result.sort_values("shap_abs", ascending=False).drop(columns="shap_abs").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NAME RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
def _get_feature_names(preprocessor, n_features: int) -> list:
    """
    Hardcoded feature names matching the exact pipeline output order.
    Numerical pipeline outputs first (51), then categorical (11) = 62 total.
    """
    # ── Numerical features (51) — in order from numerical_pipeline output
    numerical_names = [
        # Raw numericals (scaled)
        "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
        "open_acc", "revol_util", "tot_cur_bal", "open_acc_6m", "open_act_il",
        "open_il_12m", "open_il_24m", "il_util", "open_rv_12m", "open_rv_24m",
        "all_util", "total_rev_hi_lim", "inq_fi", "inq_last_12m", "bc_util",
        "percent_bc_gt_75", "acc_open_past_24mths", "pub_rec_bankruptcies",
        "mort_acc", "mo_sin_old_rev_tl_op", "num_rev_accts", "tot_hi_cred_lim",
        "total_bc_limit", "avg_cur_bal", "num_op_rev_tl", "num_actv_rev_tl",
        "num_rev_tl_bal_gt_0",
        # Binary flags (from imputer step 1)
        "has_delinq_2yrs", "has_pub_rec",
        # Engineered ratio features
        "rev_accts_to_age", "indebt_rev_ratio", "new_account_share",
        "bc_limit_util", "revol_bal_to_income", "il_recent_share",
        # Engineered product features
        "actv_rev_util", "bc_util_stress", "revol_stress_score", "il_recent_intensive",
        # Binary flag features
        "high_revol_util", "recent_intensive",
        # Binning features (ordinal encoded)
        "revol_util_tier",
        # Log-stress features
        "stress_util_income", "util_total_rev_stress_log",
        # PTI feature
        "pti",
        # Installment (derived from loan_amnt + term)
        "installment",
    ]

    # ── Categorical features (11) — in order from categorical_pipeline output
    categorical_names = [
        "emp_length_stability",   # stable / transitional / unstable
        "home_ownership",         # RENT / OWN / MORTGAGE (label encoded)
        "purpose_bucket",         # low_risk / medium_risk / high_risk
        "addr_region",            # northeast / southeast / midwest / west
        "credit_maturity",        # new / moderate / established / veteran
        "term",                   # 36 / 60 (encoded)
        "verification_status",    # Not Verified / Source Verified / Verified
        # One-hot or ordinal encoded categoricals (remaining slots)
        "home_ownership_enc",
        "purpose_bucket_enc",
        "addr_region_enc",
        "credit_maturity_enc",
    ]

    combined = numerical_names + categorical_names

    if len(combined) == n_features:
        return combined

    print(f"[predictor] WARNING: hardcoded names={len(combined)} != n_features={n_features}")
    print(f"[predictor] Numerical: {len(numerical_names)}, Categorical: {len(categorical_names)}")
    return [f"feature_{i}" for i in range(n_features)]

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict(input_df: pd.DataFrame) -> dict:
    """
    Full prediction pipeline for a single borrower row.

    Args:
        input_df : pd.DataFrame — 1 row of raw feature values from the form

    Returns dict with keys:
        probability         float        — default probability 0.0 – 1.0
        probability_pct     float        — probability as 0–100 percentage
        threshold           float        — best_threshold from training
        decision            dict         — label / emoji / color / description
        shap_factors        pd.DataFrame — top 5 SHAP drivers for this borrower
        feature_names       list[str]    — all 62 feature names post-transform
        transformed_features dict        — {feature_name: value} for all 62 features
                                           (NEW — used by llm/applicant_data/)
    """
    pipeline  = _load_pipeline()
    threshold = _load_threshold()

    preprocessor = pipeline.named_steps["preprocessor"]
    model        = pipeline.named_steps["model"]

    # ── Transform input (uses stored train statistics — no refit)
    X_transformed = preprocessor.transform(input_df)
    n_features    = X_transformed.shape[1]

    # ── Resolve feature names
    feature_names = _get_feature_names(preprocessor, n_features)

    print(f"[predictor] X_transformed : {X_transformed.shape}")
    print(f"[predictor] feature_names : {len(feature_names)} → {feature_names[:5]}...")

    # ── Predict default probability
    probability = float(model.predict_proba(X_transformed)[0, 1])
    decision    = _get_decision(probability)

    # ── SHAP values
    explainer   = _load_explainer(model)
    shap_output = explainer.shap_values(X_transformed)

    if isinstance(shap_output, list):
        shap_row = shap_output[1][0]      # older SHAP: [neg_class, pos_class]
    elif shap_output.ndim == 3:
        shap_row = shap_output[0, :, 1]  # newer SHAP: (samples, features, classes)
    else:
        shap_row = shap_output[0]         # standard 2D: (samples, features)

    if len(shap_row) != len(feature_names):
        print(f"[predictor] WARNING: shap_row={len(shap_row)} != feature_names={len(feature_names)}")
        feature_names = [f"feature_{i}" for i in range(len(shap_row))]

    top_factors = _get_top_shap_factors(shap_row, feature_names, top_n=5)

    # ── NEW: Build named dict of all transformed feature values
    # X_transformed is a numpy array — convert to flat dict using feature_names
    transformed_features: dict = dict(zip(feature_names, X_transformed[0].tolist()))

    return {
        "probability":          probability,
        "probability_pct":      round(probability * 100, 1),
        "threshold":            threshold,
        "decision":             decision,
        "shap_factors":         top_factors,
        "feature_names":        feature_names,
        "transformed_features": transformed_features,   # ← NEW
    }