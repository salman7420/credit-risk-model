"""
applicant_features.py
---------------------
Holds ALL features that went into the model for a single borrower — both
the raw inputs from the form AND the engineered/transformed features the
pipeline created.

Why both?
  - Raw inputs are human-readable (loan_amnt=12000, purpose="debt_consolidation")
  - Engineered features are what the model actually used (stress_util_income,
    revol_util_tier, pti, installment, etc.)
  - The LLM prompt needs both: raw for context, engineered for accuracy.

Data source:
  - raw_inputs       → input_df.iloc[0].to_dict()  (from Streamlit form)
  - all_features     → predictor result["transformed_features"]  (all 62)
  - feature_names    → predictor result["feature_names"]
"""

from dataclasses import dataclass, field


@dataclass
class ApplicantFeatures:
    """
    All 62 pipeline output features for one borrower.
    These are the exact values the model received — scaled numericals,
    engineered ratios, binary flags, and encoded categoricals.
    """
    all_features:  dict       = field(default_factory=dict)
    feature_names: list[str]  = field(default_factory=list)