"""
threshold_tuner.py
------------------
Finds the optimal decision threshold for the trained pipeline.

Why this exists:
Default threshold of 0.5 is almost never optimal for imbalanced
credit data. This script sweeps thresholds on X_val and finds the
one that maximizes F1. Applies best threshold to X_test for final
honest evaluation.

Usage:
    PYTHONPATH=src python -m training.threshold_tuner

Output:
    models/best_threshold.json       ← threshold value to use at inference
    models/threshold_sweep.png       ← precision/recall/f1 vs threshold plot
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, precision_recall_curve,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
# Suppress noisy pipeline logs — we only want tuner output
logging.getLogger("feature_engineering").setLevel(logging.WARNING)
logging.getLogger("pipeline").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

DATA_DIR   = "data/splits"
MODELS_DIR = "models"
TARGET_COL = "target"


def load_data():
    logger.info("Loading val and test splits...")
    val  = pd.read_parquet(f"{DATA_DIR}/val.parquet")
    test = pd.read_parquet(f"{DATA_DIR}/test.parquet")
    X_val,  y_val  = val.drop(columns=[TARGET_COL]),  val[TARGET_COL]
    X_test, y_test = test.drop(columns=[TARGET_COL]), test[TARGET_COL]
    logger.info(f"  X_val:  {X_val.shape} | X_test: {X_test.shape}")
    return X_val, y_val, X_test, y_test


def sweep_thresholds_fast(y_true, y_proba):
    """
    Vectorized threshold sweep using sklearn's precision_recall_curve.
    Returns arrays of precision, recall, f1, and thresholds.
    Much faster than looping — computes all thresholds in one call.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns N+1 points — trim last point
    precisions = precisions[:-1]
    recalls    = recalls[:-1]
    f1s        = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    return precisions, recalls, f1s, thresholds


def plot_threshold_sweep(thresholds, precisions, recalls, f1s, best_thresh, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, label="Precision", color="blue",  linewidth=1.5)
    ax.plot(thresholds, recalls,    label="Recall",    color="red",   linewidth=1.5)
    ax.plot(thresholds, f1s,        label="F1",        color="green", linewidth=2.5)
    ax.axvline(
        best_thresh, color="black", linestyle="--", linewidth=1.5,
        label=f"Best threshold = {best_thresh:.2f}",
    )
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score",     fontsize=12)
    ax.set_title("Precision / Recall / F1 vs Decision Threshold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Threshold sweep plot saved → {save_path}")


def tune():
    # ── Load saved pipeline
    pipeline_path = f"{MODELS_DIR}/pipeline.pkl"
    logger.info(f"Loading pipeline from {pipeline_path}...")
    pipeline = joblib.load(pipeline_path)

    # ── Split pipeline into preprocessor + model
    # pipeline[:-1] = ColumnTransformer (all preprocessing steps)
    # pipeline[-1]  = XGBClassifier (model only)
    preprocessor = pipeline[:-1]
    model        = pipeline[-1]
    logger.info(f"  Preprocessor steps : {[s[0] for s in pipeline.steps[:-1]]}")
    logger.info(f"  Model              : {type(model).__name__}")

    # ── Load raw data
    X_val, y_val, X_test, y_test = load_data()

    # ── Transform ONCE — no re-fitting, uses stored train statistics
    logger.info("Transforming val set...")
    X_val_t  = preprocessor.transform(X_val)
    logger.info("Transforming test set...")
    X_test_t = preprocessor.transform(X_test)
    logger.info(f"  Transformed shapes — val: {X_val_t.shape} | test: {X_test_t.shape}")

    # ── Get predicted probabilities from model directly
    logger.info("Computing predicted probabilities...")
    y_val_proba  = model.predict_proba(X_val_t)[:, 1]
    y_test_proba = model.predict_proba(X_test_t)[:, 1]

    # ── Vectorized threshold sweep on VAL set only (no test leakage)
    logger.info("Sweeping thresholds on val set...")
    precisions, recalls, f1s, thresholds = sweep_thresholds_fast(y_val, y_val_proba)

    best_idx    = np.argmax(f1s)
    best_thresh = float(thresholds[best_idx])

    logger.info(f"Best threshold found : {best_thresh:.4f}")
    logger.info(f"  Val Precision      : {precisions[best_idx]:.4f}")
    logger.info(f"  Val Recall         : {recalls[best_idx]:.4f}")
    logger.info(f"  Val F1             : {f1s[best_idx]:.4f}")

    # ── Apply best threshold to TEST set (honest final evaluation)
    logger.info("── TEST Metrics at Best Threshold ──────────────────")
    y_test_pred = (y_test_proba >= best_thresh).astype(int)

    test_f1        = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall    = recall_score(y_test, y_test_pred)

    logger.info(f"  Threshold  : {best_thresh:.4f}")
    logger.info(f"  F1         : {test_f1:.4f}   (was 0.4423 at threshold 0.5)")
    logger.info(f"  Precision  : {test_precision:.4f}   (was 0.3580)")
    logger.info(f"  Recall     : {test_recall:.4f}   (was 0.5784)")
    logger.info("\n" + classification_report(
        y_test, y_test_pred,
        target_names=["no_default", "default"],
    ))

    # ── Save best threshold to JSON
    threshold_path = f"{MODELS_DIR}/best_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({
            "best_threshold": round(best_thresh, 4),
            "val_f1":         round(float(f1s[best_idx]), 4),
            "val_precision":  round(float(precisions[best_idx]), 4),
            "val_recall":     round(float(recalls[best_idx]), 4),
            "test_f1":        round(test_f1, 4),
            "test_precision": round(test_precision, 4),
            "test_recall":    round(test_recall, 4),
        }, indent=2)
    logger.info(f"Best threshold saved → {threshold_path}")

    # ── Save threshold sweep plot
    plot_threshold_sweep(
        thresholds, precisions, recalls, f1s,
        best_thresh=best_thresh,
        save_path=f"{MODELS_DIR}/threshold_sweep.png",
    )


if __name__ == "__main__":
    tune()
