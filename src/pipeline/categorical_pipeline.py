"""
categorical_pipeline.py
-----------------------
Builds the complete end-to-end sklearn Pipeline for categorical features.

This file combines:
    1. build_categorical_cleaner()  → sklearn Pipeline     (from clean_categorical.py)
    2. build_categorical_encoder()  → ColumnTransformer    (from encode_categorical.py)

Into one single Pipeline object that can be fit on train and transform on both
train and test without any data leakage.

Pipeline Flow:
    Raw categorical DataFrame
        ↓ Step 1 — clean   (Pipeline of 5 cleaning steps)
        ↓ Step 2 — encode  (ColumnTransformer → OrdinalEncoder + OHE)
    Encoded numpy array (ready for model)

Usage:
    from categorical_pipeline import build_categorical_pipeline

    cat_pipeline = build_categorical_pipeline()

    X_train_encoded = cat_pipeline.fit_transform(X_train_cat)
    X_test_encoded  = cat_pipeline.transform(X_test_cat)
"""

import logging
from sklearn.pipeline import Pipeline

from feature_engineering.categorical_transforms.clean_categorical import build_categorical_cleaner
from feature_engineering.categorical_transforms.categorical_encoder import build_categorical_encoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)


def build_categorical_pipeline() -> Pipeline:
    """
    Builds and returns the complete unfitted categorical Pipeline.

    Steps:
        1. clean  → build_categorical_cleaner()   Pipeline with 5 sub-steps:
                        - EmpLengthCleaner()                (stateful — learns median)
                        - FunctionTransformer(home_ownership) (stateless)
                        - FunctionTransformer(purpose)        (stateless)
                        - FunctionTransformer(addr_state)     (stateless)
                        - FunctionTransformer(earliest_cr_line)(stateless)

        2. encode → build_categorical_encoder()   ColumnTransformer with 2 sub-steps:
                        - OrdinalEncoder  on [emp_length_stability, credit_maturity]
                        - OneHotEncoder   on [home_ownership, purpose_bucket, addr_region]

    Returns:
        sklearn.pipeline.Pipeline (unfitted)
    """
    logger.info("Building categorical pipeline → cleaner + encoder")

    pipeline = Pipeline(steps=[
        ("clean",  build_categorical_cleaner()),
        ("encode", build_categorical_encoder()),
    ])

    logger.info("Categorical pipeline built successfully")
    return pipeline
