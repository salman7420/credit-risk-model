"""
snapshot_builder.py
-------------------
The single entry point for building the full applicant snapshot.

Call build_snapshot() after predictor.predict() returns — it takes
the raw input DataFrame and the predictor result dict and produces
both ApplicantFeatures and PredictionResult objects, ready to be
passed to prompt_builder.py.

Usage:
    from llm.applicant_data.snapshot_builder import build_snapshot

    result   = predict(input_df)
    features, prediction = build_snapshot(input_df, result)
"""

import pandas as pd
from llm.data.applicant_data import ApplicantFeatures
from llm.data.prediction_data  import PredictionResult, ShapDriver


# Verdict → recommendation mapping
_RECOMMENDATION_MAP = {
    "APPROVE":       "Approve",
    "MANUAL REVIEW": "Manual Review",
    "REJECT":        "Decline",
}


def build_snapshot(
    input_df: pd.DataFrame,
    predictor_result: dict,
) -> tuple[ApplicantFeatures, PredictionResult]:
    """
    Build ApplicantFeatures + PredictionResult from predictor output.

    Args:
        input_df          : pd.DataFrame — 1-row raw form input
        predictor_result  : dict from predictor.predict() — must contain:
                              probability, probability_pct, threshold,
                              decision, shap_factors, feature_names,
                              transformed_features   ← (new field in predictor)

    Returns:
        (ApplicantFeatures, PredictionResult) — both ready for prompt_builder
    """
    decision             = predictor_result["decision"]
    transformed_features = predictor_result["transformed_features"]
    feature_names        = predictor_result["feature_names"]
    probability          = predictor_result["probability"]
    threshold            = predictor_result["threshold"]

    # ── Build ApplicantFeatures ───────────────────────────────────────────────
    features = ApplicantFeatures(
        all_features  = transformed_features,    # all 62 pipeline output features
        feature_names = feature_names,
    )

    # ── Build SHAP drivers list ───────────────────────────────────────────────
    shap_drivers = []
    for _, row in predictor_result["shap_factors"].iterrows():
        feature_val = transformed_features.get(row["feature"], None)
        direction   = "increases risk" if row["shap_val"] > 0 else "decreases risk"

        shap_drivers.append(ShapDriver(
            feature       = row["feature"],
            shap_value    = round(float(row["shap_val"]), 4),
            direction     = direction,
            feature_value = round(float(feature_val), 4) if feature_val is not None else None,
        ))

    # ── Build PredictionResult ────────────────────────────────────────────────
    prediction = PredictionResult(
        probability         = round(probability, 4),
        probability_pct     = round(probability * 100, 1),
        threshold           = round(threshold, 4),
        threshold_pct       = round(threshold * 100, 1),
        verdict             = decision["label"],
        recommendation      = _RECOMMENDATION_MAP.get(decision["label"], "Manual Review"),
        verdict_description = decision["description"],
        shap_drivers        = shap_drivers,
    )

    return features, prediction