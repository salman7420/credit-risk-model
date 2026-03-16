"""
plot_confusion.py
-----------------
Plots a labelled confusion matrix heatmap for the test set.
Saved as PNG and logged to MLflow as artifact.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_true, y_pred,
                          save_path: str = "models/confusion_matrix.png") -> None:
    """
    Plots a colour-coded confusion matrix heatmap with counts + percentages.

    Each cell shows:
        - Raw count        (top)
        - % of total rows  (bottom, in brackets)

    Labels:
        0 → "Fully Paid"   (non-default)
        1 → "Charged Off"  (default)

    Parameters
    ----------
    y_true    : true labels (test set)
    y_pred    : predicted labels (threshold = 0.5)
    save_path : file path to save PNG
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_pct  = cm / cm.sum() * 100   # percentage of total

    # Build annotation strings: "1234\n(12.3%)"
    annot = np.array([
        [f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"
         for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])

    labels = ["Fully Paid\n(0)", "Charged Off\n(1)"]

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 13}
    )

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")

    # Highlight the costly cell — False Negatives (bottom-left)
    ax.add_patch(plt.Rectangle((0, 1), 1, 1,
                                fill=False, edgecolor="red",
                                lw=2.5, label="False Negatives (costly)"))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    tn, fp, fn, tp = cm.ravel()
    logger.info(f"Confusion matrix saved → {save_path}")
    logger.info(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    logger.info(f"  Costly False Negatives: {fn:,} ({fn/len(y_true)*100:.1f}% of total)")
