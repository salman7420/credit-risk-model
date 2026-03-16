"""
numerical_pipeline.py
---------------------
Builds the complete end-to-end sklearn Pipeline for numerical features.
"""

import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engineering.numerical_transforms.impute_and_cap import NumericalImputer
from feature_engineering.numerical_transforms.numerical_feature_creators import create_all_features
from feature_engineering.utils.configs.numerical_config import COLS_REVOL_UTIL_TIER

from sklearn.preprocessing import OrdinalEncoder

REVOL_UTIL_TIER_ORDER = [["low", "moderate", "high", "critical"]]  # risk order: 0→3


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )                               # ← Fix 1: missing closing ) here
    logger.addHandler(handler)


def _build_scaler(numeric_cols: list) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that applies RobustScaler to numeric columns
    and passes string columns (revol_util_tier) through untouched.
    """
    return ColumnTransformer(
        transformers=[
            ("scaler", RobustScaler(), numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )                               # ← Fix 2: missing closing ) here


class _ScalerWrapper(BaseEstimator, TransformerMixin):
    """
    Dynamically identifies numeric vs string columns at fit() time,
    scales numeric cols with RobustScaler, passes string cols through.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_cols_ = [
            col for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col])
        ]
        self.string_cols_ = [
            col for col in X.columns
            if not pd.api.types.is_numeric_dtype(X[col])
        ]
        logger.info(f"_ScalerWrapper.fit() — {len(self.numeric_cols_)} numeric cols to scale")
        logger.info(f"  Passthrough string cols: {self.string_cols_}")

        self.ct_ = _build_scaler(self.numeric_cols_)
        self.ct_.fit(X)

        # ✅ Fit OrdinalEncoder on revol_util_tier (and any other string cols)
        if self.string_cols_:
            self.ord_enc_ = OrdinalEncoder(
                categories=REVOL_UTIL_TIER_ORDER,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=float,
            )
            self.ord_enc_.fit(X[self.string_cols_])

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        logger.info("_ScalerWrapper.transform() — applying RobustScaler + OrdinalEncoder")

        scaled = self.ct_.transform(X)
        scaled_df = pd.DataFrame(scaled, columns=self.numeric_cols_, index=X.index)

        # ✅ Encode revol_util_tier → float (low=0, moderate=1, high=2, critical=3)
        if self.string_cols_:
            encoded = self.ord_enc_.transform(X[self.string_cols_])
            encoded_df = pd.DataFrame(encoded, columns=self.string_cols_, index=X.index)
            return pd.concat([scaled_df, encoded_df], axis=1)
        
        return scaled_df


def build_numerical_pipeline() -> Pipeline:
    """
    Builds and returns the complete unfitted numerical Pipeline.

    Steps:
        1. impute_cap  → NumericalImputer()
        2. engineer    → FunctionTransformer(create_all_features)
        3. scale       → _ScalerWrapper() (RobustScaler + string passthrough)
    """
    logger.info("Building numerical pipeline → impute_cap + engineer + scale")

    pipeline = Pipeline(steps=[
        ("impute_cap", NumericalImputer()),
        ("engineer",   FunctionTransformer(create_all_features)),
        ("scale",      _ScalerWrapper()),
    ])

    logger.info("Numerical pipeline built successfully")
    return pipeline
