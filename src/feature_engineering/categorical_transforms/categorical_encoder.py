"""
encode_categorical.py
---------------------
Builds the ColumnTransformer responsible for encoding all categorical
columns after clean_categorical.py has run.

Two encoding strategies:
  OrdinalEncoder → columns with meaningful order (stability, risk, maturity)
  OneHotEncoder  → columns with no meaningful order (ownership, region, term, status)

Usage:
    from encode_categorical import build_categorical_encoder
    encoder = build_categorical_encoder()
    # encoder is an unfitted ColumnTransformer — pipeline calls fit/transform
"""

import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Import all ordering configs from categorical_config.py
from feature_engineering.utils.configs.categorical_config import (
    EMP_LENGTH_STABILITY_ORDER,
    PURPOSE_BUCKET_ORDER,
    CREDIT_MATURITY_ORDER,
    OHE_COLS,
    ORDINAL_COLS,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)


def build_categorical_encoder() -> ColumnTransformer:
    """
    Builds and returns an unfitted ColumnTransformer for categorical encoding.

    Transformers
    ------------
    1. ordinal_encoder : OrdinalEncoder
       Columns  : emp_length_stability, purpose_bucket, credit_maturity
       Why      : These have meaningful order — model should know
                  "stable" > "transitional" > "unstable"
       Handles unknown at inference → encodes as NaN (safe, won't crash)

    2. ohe_encoder : OneHotEncoder
       Columns  : home_ownership, addr_region, term, verification_status
       Why      : No meaningful order — each category is independent
       drop="first" → removes one category per column to avoid
                      dummy variable trap (multicollinearity)
       handle_unknown="ignore" → unseen categories at inference → all zeros

    remainder="passthrough"
       Any columns not listed above pass through untouched.
       Numerical columns will not reach this transformer (handled separately)
       but passthrough ensures nothing is accidentally dropped.

    Returns
    -------
    ColumnTransformer (unfitted) — call pipeline.fit() to fit this
    """
    logger.info("Building categorical ColumnTransformer")

    # ------------------------------------------------------------------
    # OrdinalEncoder
    # Order lists must match column order in ORDINAL_COLS exactly
    # ------------------------------------------------------------------
    ordinal_encoder = OrdinalEncoder(
        categories=[
            EMP_LENGTH_STABILITY_ORDER,   # emp_length_stability
            PURPOSE_BUCKET_ORDER,          # purpose_bucket
            CREDIT_MATURITY_ORDER,         # credit_maturity
        ],
        handle_unknown="use_encoded_value",
        unknown_value=-1,   # unseen category at inference → -1 (identifiable)
        dtype=float,
    )
    logger.debug(f"  OrdinalEncoder configured for: {ORDINAL_COLS}")

    # ------------------------------------------------------------------
    # OneHotEncoder
    # drop="first" removes reference category per column
    # sparse_output=False → returns dense array (easier to work with)
    # ------------------------------------------------------------------
    ohe_encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",   # unseen category → all zeros (won't crash)
        sparse_output=False,
        dtype=float,
    )
    logger.debug(f"  OneHotEncoder configured for: {OHE_COLS}")

    # ------------------------------------------------------------------
    # ColumnTransformer — combines both encoders
    # ------------------------------------------------------------------
    categorical_encoder = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_encoder, ORDINAL_COLS),
            ("ohe",     ohe_encoder,     OHE_COLS),
        ],
        remainder="passthrough",  # non-listed cols pass through untouched
        verbose_feature_names_out=False,  # cleaner output col names
    )

    logger.info("Categorical ColumnTransformer built successfully")
    logger.info(f"  Ordinal cols : {ORDINAL_COLS}")
    logger.info(f"  OHE cols     : {OHE_COLS}")

    return categorical_encoder
