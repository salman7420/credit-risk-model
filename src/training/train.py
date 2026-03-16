"""
train.py
--------
End-to-end training script for the credit risk XGBoost model.

Flow:
    1. Load X_train, X_val, X_test, y_train, y_val, y_test
    2. Start MLflow experiment
    3. Optuna hyperparameter search (cross-val on X_train only)
    4. Retrain best model on full X_train
    5. Evaluate on X_val  → confirm no overfitting
    6. Evaluate on X_test → final honest performance (ONCE only)
    7. Log everything to MLflow + save pipeline.pkl

Usage:
    python -m src.training.train
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from optuna.integration import OptunaSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report
)
from dotenv import load_dotenv

from pipeline.master_pipeline import build_master_pipeline
from training.evaluate import evaluate_split, compute_ks
from training.plots.plot_roc import plot_roc_curve
from training.plots.plot_confusion import plot_confusion_matrix
from training.plots.plot_feature_importance import plot_feature_importance

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR   = "data/splits" 
MODELS_DIR      = "models"
EXPERIMENT_NAME = "credit_risk_xgboost"
N_TRIALS        = 50
CV_FOLDS        = 5
RANDOM_STATE    = 42
os.makedirs(MODELS_DIR, exist_ok=True)
TARGET_COL = "target"  


# ── Step 1: Load Data ─────────────────────────────────────────────────────────
def load_data():
    logger.info("Loading train / val / test splits...")

    train = pd.read_parquet(f"{DATA_DIR}/train.parquet")
    val   = pd.read_parquet(f"{DATA_DIR}/val.parquet")
    test  = pd.read_parquet(f"{DATA_DIR}/test.parquet")

    # Separate X and y from each combined file
    X_train, y_train = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
    X_val,   y_val   = val.drop(  columns=[TARGET_COL]), val[TARGET_COL]
    X_test,  y_test  = test.drop( columns=[TARGET_COL]), test[TARGET_COL]

    logger.info(f"  X_train: {X_train.shape}  |  default rate: {y_train.mean():.3f}")
    logger.info(f"  X_val:   {X_val.shape}    |  default rate: {y_val.mean():.3f}")
    logger.info(f"  X_test:  {X_test.shape}   |  default rate: {y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Step 2: Optuna Objective ──────────────────────────────────────────────────
def build_objective(X_train, y_train):
    """
    Returns the Optuna objective function.

    Each trial:
        1. Optuna suggests hyperparameters
        2. Builds master_pipeline with those params
        3. 5-fold stratified cross-val on X_train → mean AUC
        4. Returns mean AUC (Optuna maximizes this)

    Stratified CV ensures each fold has same class distribution
    as full training set — important for imbalanced credit data.
    """
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 3.0),
            # handles class imbalance — ratio of negative to positive class
            "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            "random_state":      RANDOM_STATE,
            "eval_metric":       "auc",
            #"use_label_encoder": False,
        }

        pipeline = build_master_pipeline(XGBClassifier(**params))

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1       # use all CPU cores
        )

        mean_auc = scores.mean()
        logger.info(f"  Trial {trial.number:>3} | AUC={mean_auc:.4f} ± {scores.std():.4f} "
                    f"| params={trial.params}")
        return mean_auc

    return objective


# ── Step 3: Run Optuna Study ──────────────────────────────────────────────────
def run_optuna(X_train, y_train):
    logger.info(f"Starting Optuna study — {N_TRIALS} trials")

    # suppress optuna's own logs — we handle logging ourselves
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",       # maximize AUC
        study_name=EXPERIMENT_NAME,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),  # Bayesian search
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10) # prune bad trials early
    )

    study.optimize(
        build_objective(X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    logger.info(f"Optuna complete — Best AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params, study.best_value


# ── Main Training Function ────────────────────────────────────────────────────
def train():
    # ── Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # ── Start MLflow run
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_optuna") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # ── STEP 3: Hyperparameter search
        best_params, best_cv_auc = run_optuna(X_train, y_train)

        # ── STEP 4: Retrain on full X_train with best params
        logger.info("Retraining best model on full X_train...")
        best_pipeline = build_master_pipeline(
            XGBClassifier(**best_params)
        )
        best_pipeline.fit(X_train, y_train)
        logger.info("Retraining complete ✅")

        # ── STEP 5: Evaluate on X_val (overfitting check)
        logger.info("Evaluating on validation set...")
        val_metrics = evaluate_split(best_pipeline, X_val, y_val, split_name="val")

        # ── STEP 6: Evaluate on X_test (final honest performance)
        logger.info("Evaluating on test set (FINAL — looked at once)...")
        test_metrics = evaluate_split(best_pipeline, X_test, y_test, split_name="test")

        # ── Overfitting check log
        auc_drop = val_metrics["auc"] - test_metrics["auc"]
        cv_to_val_drop = best_cv_auc - val_metrics["auc"]
        logger.info(f"CV AUC       : {best_cv_auc:.4f}")
        logger.info(f"Val AUC      : {val_metrics['auc']:.4f}  (drop from CV: {cv_to_val_drop:.4f})")
        logger.info(f"Test AUC     : {test_metrics['auc']:.4f}  (drop from val: {auc_drop:.4f})")

        if cv_to_val_drop > 0.03:
            logger.warning("⚠️  CV → Val AUC drop > 0.03 — possible overfitting, consider retuning")

        # ── STEP 7: Log to MLflow ─────────────────────────────────────────────

        # Log best hyperparams
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("cv_folds", CV_FOLDS)

        # Log all metrics
        mlflow.log_metric("cv_auc",          best_cv_auc)
        mlflow.log_metric("val_auc",         val_metrics["auc"])
        mlflow.log_metric("val_f1",          val_metrics["f1"])
        mlflow.log_metric("val_precision",   val_metrics["precision"])
        mlflow.log_metric("val_recall",      val_metrics["recall"])
        mlflow.log_metric("val_ks",          val_metrics["ks"])
        mlflow.log_metric("test_auc",        test_metrics["auc"])
        mlflow.log_metric("test_f1",         test_metrics["f1"])
        mlflow.log_metric("test_precision",  test_metrics["precision"])
        mlflow.log_metric("test_recall",     test_metrics["recall"])
        mlflow.log_metric("test_ks",         test_metrics["ks"])

        # Generate + log plots as artifacts
        y_val_proba  = best_pipeline.predict_proba(X_val)[:, 1]
        y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

        plot_roc_curve(y_val, y_val_proba, y_test, y_test_proba,
                       save_path="models/roc_curve.png")
        plot_confusion_matrix(y_test, best_pipeline.predict(X_test),
                              save_path="models/confusion_matrix.png")
        plot_feature_importance(best_pipeline,
                                save_path="models/feature_importance.png")

        mlflow.log_artifact("models/roc_curve.png")
        mlflow.log_artifact("models/confusion_matrix.png")
        mlflow.log_artifact("models/feature_importance.png")

        # Save full pipeline
        pipeline_path = f"{MODELS_DIR}/pipeline.pkl"
        joblib.dump(best_pipeline, pipeline_path)
        logger.info(f"Pipeline saved → {pipeline_path} ✅")

        # Log sklearn model to MLflow model registry
        mlflow.sklearn.log_model(best_pipeline, artifact_path="pipeline")
        logger.info(f"MLflow run complete → {run.info.run_id} ✅")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
