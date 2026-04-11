"""
user_prompt_builder.py
----------------------
Builds the dynamic user prompt for the LLM narrative engine.

Takes ApplicantFeatures + PredictionResult and produces a
structured text prompt tailored to the verdict zone:

    APPROVE       → summary mode       (strengths-focused)
    REJECT        → risk report mode   (red flags, specific)
    MANUAL REVIEW → analyst mode       (balanced, both sides)

Usage:
    from llm.prompts.user_prompt_builder import build_user_prompt

    prompt = build_user_prompt(features, prediction)
    # → pass to llm_client.py as the user message
"""

from llm.data.applicant_data import ApplicantFeatures
from llm.data.prediction_data import PredictionResult


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NAME → PLAIN ENGLISH MAP
# Mirrors the translation table in system_prompt.md
# Used to make SHAP driver names human-readable in the prompt
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_LABEL_MAP = {
    "stress_util_income":        "Income-Adjusted Credit Stress",
    "revol_util":                "Revolving Credit Utilization",
    "bc_util_stress":            "Bankcard Stress Index",
    "pti":                       "Payment-to-Income Ratio",
    "new_account_share":         "Recent Account Growth Rate",
    "dti":                       "Debt-to-Income Ratio",
    "il_recent_intensive":       "Recent Installment Loan Intensity",
    "revol_stress_score":        "Revolving Credit Stress Score",
    "bc_util":                   "Bankcard Utilization",
    "percent_bc_gt_75":          "Bankcards Over 75% Utilized",
    "indebt_rev_ratio":          "Revolving Indebtedness Ratio",
    "revol_bal_to_income":       "Revolving Balance to Income",
    "high_revol_util":           "High Revolving Utilization Flag",
    "annual_inc":                "Annual Income",
    "loan_amnt":                 "Loan Amount",
    "installment":               "Monthly Installment",
    "open_acc":                  "Open Credit Accounts",
    "inq_last_6mths":            "Recent Credit Inquiries (6 months)",
    "acc_open_past_24mths":      "New Accounts (Last 24 Months)",
    "delinq_2yrs":               "Delinquencies (Last 2 Years)",
    "pub_rec_bankruptcies":      "Bankruptcies on Record",
    "mort_acc":                  "Mortgage Accounts",
    "mo_sin_old_rev_tl_op":      "Age of Oldest Revolving Account",
    "tot_cur_bal":               "Total Current Balance",
    "tot_hi_cred_lim":           "Total High Credit Limit",
    "all_util":                  "Overall Credit Utilization",
    "util_total_rev_stress_log": "Total Revolving Stress (Log)",
    "actv_rev_util":             "Active Revolving Utilization",
    "num_actv_rev_tl":           "Active Revolving Accounts",
    "il_recent_share":           "Recent Installment Share",
    "has_delinq_2yrs":           "Has Recent Delinquency",
    "has_pub_rec":               "Has Public Record",
    "recent_intensive":          "Recent Credit Intensity Flag",
    "emp_length_stability":      "Employment Stability",
    "purpose_bucket":            "Loan Purpose Risk Tier",
    "credit_maturity":           "Credit History Maturity",
    "addr_region":               "Geographic Region",
    "home_ownership":            "Housing Status",
    "verification_status":       "Income Verification Status",
}


def _label(feature_name: str) -> str:
    """Translate raw feature name to plain English label."""
    return FEATURE_LABEL_MAP.get(feature_name, feature_name.replace("_", " ").title())


def _format_value(value: float | None) -> str:
    """Format a feature value for display."""
    if value is None:
        return "N/A"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) < 0.01:
        return f"{value:.4f}"
    return f"{value:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# SHARED SECTIONS
# These blocks appear in all three prompt templates
# ─────────────────────────────────────────────────────────────────────────────

def _build_verdict_block(prediction: PredictionResult) -> str:
    return f"""
=== VERDICT ===
Decision:           {prediction.verdict}
Recommendation:     {prediction.recommendation}
Default Likelihood: {prediction.probability_pct}%
Decision Threshold: {prediction.threshold_pct}%
""".strip()


def _build_borrower_block(features: ApplicantFeatures) -> str:
    f = features.all_features
    return f"""
=== BORROWER PROFILE ===
Loan Amount:         ${_format_value(f.get('loan_amnt'))}
Loan Term:           {"60 months" if f.get('term', 0) > 1 else "36 months"}
Annual Income:       ${_format_value(f.get('annual_inc'))}
Monthly Installment: ${_format_value(f.get('installment'))}
Debt-to-Income:      {_format_value(f.get('dti'))}%
Revolving Util:      {_format_value(f.get('revol_util'))}%
Employment Stability:{f.get('emp_length_stability', 'N/A')}
Home Ownership:      {f.get('home_ownership', 'N/A')}
Loan Purpose Tier:   {f.get('purpose_bucket', 'N/A')}
Credit Maturity:     {f.get('credit_maturity', 'N/A')}
""".strip()


def _build_shap_block(prediction: PredictionResult) -> str:
    lines = ["=== KEY RISK DRIVERS (ranked by impact) ==="]
    for i, driver in enumerate(prediction.shap_drivers, start=1):
        arrow = "↑" if driver.direction == "increases risk" else "↓"
        label = _label(driver.feature)
        val   = _format_value(driver.feature_value)
        lines.append(
            f"{i}. {label:<40} {arrow} {driver.direction:<20} "
            f"| value: {val:<10} | impact: {abs(driver.shap_value):.4f}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ZONE TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

def _build_approve_prompt(
    features: ApplicantFeatures,
    prediction: PredictionResult,
) -> str:
    return f"""
{_build_verdict_block(prediction)}

{_build_borrower_block(features)}

{_build_shap_block(prediction)}

=== YOUR TASK — APPROVE: SUMMARY MODE ===
This borrower has been APPROVED by the risk assessment.
Write a concise approval summary following the APPROVE format
defined in your instructions.

Focus on:
- The 2–3 strongest signals that supported approval
- Any minor flags worth monitoring (if present in the drivers above)
- Keep it brief — the loan officer needs a quick, confident confirmation

Do NOT restate the raw numbers verbatim — translate them into
meaningful sentences a loan officer can act on.
""".strip()


def _build_reject_prompt(
    features: ApplicantFeatures,
    prediction: PredictionResult,
) -> str:
    return f"""
{_build_verdict_block(prediction)}

{_build_borrower_block(features)}

{_build_shap_block(prediction)}

=== YOUR TASK — REJECT: RISK REPORT MODE ===
This borrower has been REJECTED by the risk assessment.
Write a detailed risk report following the REJECT format
defined in your instructions.

Focus on:
- The top risk drivers listed above — explain each one clearly
  in plain business language, why it matters, and what it signals
- The combined picture — how do these factors together create
  an elevated default risk profile?
- Be direct and specific — do not use vague or softened language

The loan officer needs to understand exactly why this borrower
does not meet lending criteria.
""".strip()


def _build_manual_prompt(
    features: ApplicantFeatures,
    prediction: PredictionResult,
) -> str:
    return f"""
{_build_verdict_block(prediction)}

{_build_borrower_block(features)}

{_build_shap_block(prediction)}

=== YOUR TASK — MANUAL REVIEW: ANALYST MODE ===
This borrower is in the BORDERLINE zone and requires human review.
Write a balanced analyst assessment following the MANUAL REVIEW
format defined in your instructions.

Focus on:
- Risk signals: which drivers are elevated and why they matter
- Mitigating factors: which drivers are working in the borrower's favor
- Verification steps: what specific information should the reviewer
  request or check before making a final decision?

Be balanced — present both sides honestly so the loan officer
can make an informed manual decision.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(
    features: ApplicantFeatures,
    prediction: PredictionResult,
) -> str:
    """
    Build the complete user prompt string for the LLM.

    Routes to the correct zone template based on the model verdict:
        APPROVE       → _build_approve_prompt()
        REJECT        → _build_reject_prompt()
        MANUAL REVIEW → _build_manual_prompt()

    Args:
        features   : ApplicantFeatures from snapshot_builder
        prediction : PredictionResult  from snapshot_builder

    Returns:
        str — complete user prompt ready to send to the LLM API
    """
    verdict = prediction.verdict

    if verdict == "APPROVE":
        return _build_approve_prompt(features, prediction)
    elif verdict == "REJECT":
        return _build_reject_prompt(features, prediction)
    else:
        return _build_manual_prompt(features, prediction)