"""
impute_and_cap.py
-----------------
Custom sklearn transformer for numerical feature imputation and outlier capping.
Runs as STEP 1 in numerical_pipeline before feature engineering and scaling.

Responsibilities:
    1. Fill missing values — zero-fill or median-fill depending on column semantics
    2. Cap outliers      — upper cap at p99.5 learned from training data only
    3. Create binary flags — has_delinq_2yrs, has_pub_rec (before any dropping)

What this file does NOT do:
    - Feature engineering  → feature_creators.py
    - Scaling              → numerical_pipeline.py (RobustScaler)
    - Drop engineered-redundant cols → feature_creators.create_all_features()

Data Leakage Contract:
    fit(X_train)   → learns self.medians_ and self.caps_ from training data ONLY
    transform(X)   → applies stored values — never recomputes from input data
    Test data never influences any fill value or cap threshold.

Usage:
    from impute_and_cap import NumericalImputer

    imputer = NumericalImputer()
    imputer.fit(X_train_num)
    X_train_clean = imputer.transform(X_train_num)
    X_test_clean  = imputer.transform(X_test_num)    # uses train medians + caps
"""

import logging
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engineering.utils.configs.numerical_config import (
    COLS_ZERO_FILL,
    COLS_MEDIAN_FILL,
    COLS_CAP_ONLY,
    COLS_ZERO_FILL_AND_CAP,
    COLS_MEDIAN_FILL_AND_CAP,
    CAP_QUANTILE,            # 0.995
    DTI_LOWER_CLIP,          # 0.0
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)


class NumericalImputer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer: imputes missing values and caps outliers
    for all numerical features.

    Why a custom class and not SimpleImputer + ColumnTransformer:
        Cap thresholds (p99.5) must be LEARNED from training data at fit()
        and stored — applying training percentiles to test data prevents
        data leakage. SimpleImputer cannot learn and store percentile caps.

    Attributes learned at fit()
    ---------------------------
    self.medians_ : dict  {col_name: median_value}
        Computed from X_train only. Used in transform() for median-fill cols.

    self.caps_ : dict  {col_name: upper_cap_value}
        p99.5 computed from X_train only. Used in transform() for all cap cols.

    fit() columns
    -------------
    Learns medians for : COLS_MEDIAN_FILL + COLS_MEDIAN_FILL_AND_CAP
    Learns caps for    : COLS_CAP_ONLY + COLS_ZERO_FILL_AND_CAP + COLS_MEDIAN_FILL_AND_CAP

    transform() operations (in order)
    ----------------------------------
    1. Binary flags   → has_delinq_2yrs, has_pub_rec   (BEFORE any dropping)
    2. Zero-fill      → COLS_ZERO_FILL + COLS_ZERO_FILL_AND_CAP
    3. Median-fill    → COLS_MEDIAN_FILL + COLS_MEDIAN_FILL_AND_CAP (stored medians)
    4. Upper cap      → all cap groups  (stored p99.5 thresholds)
    5. dti lower clip → clip(lower=0) so negative dti values don't survive
    """

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learns medians and p99.5 caps from training data only.
        Stores results in self.medians_ and self.caps_.
        Never called on test data.
        """
        logger.info("NumericalImputer.fit() — learning medians and caps from training data")
        X = X.copy()

        # ── Learn medians ─────────────────────────────────────────────────────
        median_cols = COLS_MEDIAN_FILL + COLS_MEDIAN_FILL_AND_CAP
        self.medians_ = {}
        for col in median_cols:
            if col in X.columns:
                self.medians_[col] = X[col].median()
                logger.debug(f"  median learned — '{col}': {self.medians_[col]:.4f}")
            else:
                logger.warning(f"  median col not found in data: '{col}' — skipping")

        # ── Learn p99.5 caps ──────────────────────────────────────────────────
        cap_cols = COLS_CAP_ONLY + COLS_ZERO_FILL_AND_CAP + COLS_MEDIAN_FILL_AND_CAP
        self.caps_ = {}
        for col in cap_cols:
            if col in X.columns:
                self.caps_[col] = X[col].quantile(CAP_QUANTILE)
                logger.debug(f"  cap learned — '{col}' p{CAP_QUANTILE*100}: {self.caps_[col]:.4f}")
            else:
                logger.warning(f"  cap col not found in data: '{col}' — skipping")

        logger.info(f"  fit() complete — {len(self.medians_)} medians, {len(self.caps_)} caps stored")
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Applies imputation and capping using ONLY stored fit() values.
        Never recomputes medians or caps from input X.

        Operations in strict order:
            1. Binary flags    (must be BEFORE dropping anything)
            2. Zero-fill
            3. Median-fill     (uses self.medians_)
            4. Upper cap       (uses self.caps_)
            5. dti lower clip
        """
        logger.info("NumericalImputer.transform() — applying imputation and capping")
        logger.info(f"  Input shape: {X.shape}")
        X = X.copy()

        # ── STEP 1: Binary flags ──────────────────────────────────────────────
        # Must happen BEFORE any dropping — raw source columns still present
        logger.info("-- STEP 1: Binary flags --")

        if "delinq_2yrs" in X.columns:
            X["has_delinq_2yrs"] = (X["delinq_2yrs"] > 0).astype(int)
            logger.debug(f"  'has_delinq_2yrs' — {X['has_delinq_2yrs'].sum()} positives")

        if "pub_rec" in X.columns:
            X["has_pub_rec"] = (X["pub_rec"] > 0).astype(int)
            logger.debug(f"  'has_pub_rec' — {X['has_pub_rec'].sum()} positives")

        # ── STEP 2: Zero-fill ─────────────────────────────────────────────────
        logger.info("-- STEP 2: Zero-fill --")
        zero_fill_all = COLS_ZERO_FILL + COLS_ZERO_FILL_AND_CAP
        for col in zero_fill_all:
            if col in X.columns:
                missing = X[col].isna().sum()
                if missing > 0:
                    X[col] = X[col].fillna(0)
                    logger.debug(f"  zero-filled '{col}': {missing} rows")

        # ── STEP 3: Median-fill ───────────────────────────────────────────────
        logger.info("-- STEP 3: Median-fill (stored train medians) --")
        for col, median_val in self.medians_.items():
            if col in X.columns:
                missing = X[col].isna().sum()
                if missing > 0:
                    X[col] = X[col].fillna(median_val)
                    logger.debug(f"  median-filled '{col}': {missing} rows → {median_val:.4f}")

        # ── STEP 4: Upper cap ─────────────────────────────────────────────────
        logger.info("-- STEP 4: Upper cap (stored train p99.5) --")
        for col, cap_val in self.caps_.items():
            if col in X.columns:
                over_cap = (X[col] > cap_val).sum()
                X[col] = X[col].clip(upper=cap_val)
                if over_cap > 0:
                    logger.debug(f"  capped '{col}' at {cap_val:.4f} — {over_cap} rows clipped")

        # ── STEP 5: dti lower clip ────────────────────────────────────────────
        # Negative dti values are data errors — clip to 0 floor
        logger.info("-- STEP 5: dti lower clip → 0 --")
        if "dti" in X.columns:
            neg_dti = (X["dti"] < DTI_LOWER_CLIP).sum()
            X["dti"] = X["dti"].clip(lower=DTI_LOWER_CLIP)
            if neg_dti > 0:
                logger.debug(f"  clipped {neg_dti} negative dti values to 0")

        logger.info(f"  Output shape: {X.shape}")
        logger.info("NumericalImputer.transform() complete")
        return X
