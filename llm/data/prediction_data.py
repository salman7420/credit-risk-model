"""
prediction_result.py
--------------------
Holds the model's output for a single borrower — the default probability,
the threshold-based verdict, the recommendation, and the top SHAP drivers
that explain WHY the model made this decision.

This is the object the LLM uses to write the risk narrative.
The verdict and recommendation come FROM THE MODEL — the LLM never
decides these. It only explains them.

Data source:
  - probability, threshold  → predictor result
  - verdict, recommendation → predictor result["decision"]
  - shap_factors            → predictor result["shap_factors"] (DataFrame)
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ShapDriver:
    """One SHAP feature driver — what moved the needle and by how much."""
    feature:        str           # e.g. "stress_util_income"
    shap_value:     float         # raw SHAP value — positive = increases default risk
    direction:      str           # "increases risk" | "decreases risk"
    feature_value:  float | None  # the actual transformed value for this applicant


@dataclass
class PredictionResult:
    """
    Complete model output for one borrower assessment.

    Attributes:
        probability         float  — raw default probability (0.0 – 1.0)
        probability_pct     float  — percentage form (e.g. 71.0)
        threshold           float  — best_threshold from training
        threshold_pct       float  — percentage form (e.g. 46.6)

        verdict             str    — model + rule decision:
                                     "APPROVE" | "MANUAL REVIEW" | "REJECT"
        recommendation      str    — human-readable: "Approve" | "Manual Review" | "Decline"
        verdict_description str    — short explanation of the verdict

        shap_drivers        list   — top 5 ShapDriver objects, ranked by
                                     absolute SHAP magnitude (most impactful first)
    """

    # ── Numeric output
    probability:         float = 0.0
    probability_pct:     float = 0.0
    threshold:           float = 0.0
    threshold_pct:       float = 0.0

    # ── Decision (owned by model + threshold rule — NOT the LLM)
    verdict: Literal[
        "APPROVE", "MANUAL REVIEW", "REJECT"
    ] = "MANUAL REVIEW"

    recommendation: Literal[
        "Approve", "Manual Review", "Decline"
    ] = "Manual Review"

    verdict_description: str = ""

    # ── SHAP explanation
    shap_drivers: list[ShapDriver] = field(default_factory=list)