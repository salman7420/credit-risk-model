"""
evaluate.py
-----------
Computes all evaluation metrics for a given split.
Called from train.py for both val and test sets.
"""

import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_score, recall_score
)

logger = logging.getLogger(__name__)


def compute_ks(y_true, y_proba) -> float:
    """
    Computes KS (Kolmogorov-Smirnov) statistic.
    Industry-standard credit risk metric.
    Measures max separation between default and non-default score distributions.
    Higher KS = better model separation.
    """
    df = sorted(zip(y_proba, np.array(y_true)), reverse=True)
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos

    tp, fp = 0, 0
    max_ks = 0.0
    for _, label in df:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        max_ks = max(max_ks, abs(tpr - fpr))

    return max_ks


def evaluate_split(pipeline, X, y, split_name: str) -> dict:
    """
    Evaluates pipeline on a given split and returns metrics dict.

    Parameters
    ----------
    pipeline   : fitted sklearn pipeline
    X          : feature DataFrame
    y          : true labels
    split_name : "val" or "test" — used for logging only

    Returns
    -------
    dict with keys: auc, f1, precision, recall, ks
    """
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred  = pipeline.predict(X)

    metrics = {
        "auc":       roc_auc_score(y, y_proba),
        "f1":        f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall":    recall_score(y, y_pred),
        "ks":        compute_ks(y, y_proba),
    }

    logger.info(f"── {split_name.upper()} Metrics ──────────────────")
    logger.info(f"  AUC-ROC   : {metrics['auc']:.4f}")
    logger.info(f"  F1        : {metrics['f1']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  KS        : {metrics['ks']:.4f}")

    return metrics
