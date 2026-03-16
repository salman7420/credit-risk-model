"""
plot_roc.py
-----------
Plots ROC curves for validation and test splits on the same chart.
Saved as PNG and logged to MLflow as artifact.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

logger = logging.getLogger(__name__)


def plot_roc_curve(y_val, y_val_proba,
                   y_test, y_test_proba,
                   save_path: str = "models/roc_curve.png") -> None:
    """
    Plots ROC curves for val and test splits on one chart.

    Having both on the same chart lets you instantly see:
        - Are val and test curves close?  → model generalizes well ✅
        - Big gap between them?           → overfitting ❌

    Parameters
    ----------
    y_val        : true labels for validation set
    y_val_proba  : predicted probabilities for validation set
    y_test       : true labels for test set
    y_test_proba : predicted probabilities for test set
    save_path    : file path to save PNG
    """
    val_auc  = roc_auc_score(y_val,  y_val_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    val_fpr,  val_tpr,  _ = roc_curve(y_val,  y_val_proba)
    test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(val_fpr,  val_tpr,
            color="steelblue", lw=2,
            label=f"Validation  (AUC = {val_auc:.4f})")

    ax.plot(test_fpr, test_tpr,
            color="darkorange", lw=2,
            label=f"Test        (AUC = {test_auc:.4f})")

    # Random baseline
    ax.plot([0, 1], [0, 1],
            color="grey", lw=1, linestyle="--",
            label="Random baseline (AUC = 0.50)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Validation vs Test", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"ROC curve saved → {save_path}  "
                f"(val AUC={val_auc:.4f}, test AUC={test_auc:.4f})")
