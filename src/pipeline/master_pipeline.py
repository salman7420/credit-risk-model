"""
master_pipeline.py
------------------
Assembles the complete end-to-end preprocessing + model pipeline
for the credit risk ML project.

Architecture:
    Raw DataFrame (numerical + categorical columns mixed)
        ↓
    ColumnTransformer  (runs numerical and categorical in PARALLEL)
        ├── numerical_pipeline   → NumericalImputer + FeatureEngineering + RobustScaler
        └── categorical_pipeline → CategoricalCleaner + CategoricalEncoder
        ↓
    Fully encoded + scaled numpy array
        ↓
    Model  (any sklearn-compatible estimator passed in)

⚠️  revol_util_tier handling:
    numerical_pipeline produces 'revol_util_tier' as a string column.
    It is defined in NUMERICAL_COLS list with its source columns so
    ColumnTransformer passes it to categorical_pipeline automatically —
    master_pipeline routes it by including it in CAT_COLS.

Usage:
    from pipeline.master_pipeline import build_master_pipeline
    from xgboost import XGBClassifier

    pipeline = build_master_pipeline(model=XGBClassifier())
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
"""

import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from pipeline.categorical_pipeline import build_categorical_pipeline
from pipeline.numerical_pipeline import build_numerical_pipeline
from feature_engineering.utils.configs.master_config import NUM_COLS, CAT_COLS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)


def build_master_pipeline(model) -> Pipeline:
    """
    Builds and returns the complete unfitted end-to-end Pipeline.

    Parameters
    ----------
    model : any sklearn-compatible estimator
            e.g. XGBClassifier(), LogisticRegression(), RandomForestClassifier()

    Returns
    -------
    sklearn.pipeline.Pipeline (unfitted)

    Steps
    -----
    1. preprocessor  → ColumnTransformer
                        ├── numerical   → build_numerical_pipeline()  on NUM_COLS
                        └── categorical → build_categorical_pipeline() on CAT_COLS
    2. model         → passed-in estimator
    """
    logger.info("Building master pipeline")
    logger.info(f"  Numerical cols  : {len(NUM_COLS)}")
    logger.info(f"  Categorical cols: {len(CAT_COLS)}")
    logger.info(f"  Model           : {type(model).__name__}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical",   build_numerical_pipeline(),   NUM_COLS),
            ("categorical", build_categorical_pipeline(), CAT_COLS),
        ],
        remainder="drop",              # drop anything not in NUM_COLS or CAT_COLS
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        model),
    ])

    logger.info("Master pipeline built successfully")
    return pipeline
