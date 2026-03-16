"""
plot_feature_importance.py
--------------------------
Plots top 20 XGBoost feature importances (by gain).
Saved as PNG and logged to MLflow as artifact.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_feature_importance(pipeline,
                            top_n: int = 20,
                            save_path: str = "models/feature_importance.png") -> None:
    """
    Extracts feature importances from the XGBoost model inside the
    master pipeline and plots a horizontal bar chart of top N features.

    Uses 'gain' importance — measures how much each feature improves
    the model when used in a split. More meaningful than 'weight'
    (which just counts how often a feature is used).

    Parameters
    ----------
    pipeline  : fitted master_pipeline (sklearn Pipeline object)
    top_n     : number of top features to show (default 20)
    save_path : file path to save PNG
    """
    # ── Extract model from pipeline ───────────────────────────────────────────
    model = pipeline.named_steps["model"]

    # ── Get feature importances (gain) ────────────────────────────────────────
    importance_dict = model.get_booster().get_score(importance_type="gain")

    if not importance_dict:
        logger.warning("No feature importances found — skipping plot")
        return

    # ── Build sorted DataFrame ────────────────────────────────────────────────
    importance_df = (
        pd.DataFrame.from_dict(importance_dict, orient="index", columns=["importance"])
        .sort_values("importance", ascending=True)   # ascending for horizontal bar
        .tail(top_n)
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.barh(
        importance_df.index,
        importance_df["importance"],
        color="steelblue",
        edgecolor="white",
        height=0.7
    )

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width * 1.01, bar.get_y() + bar.get_height() / 2,
            f"{width:,.0f}",
            va="center", ha="left", fontsize=9, color="dimgray"
        )

    ax.set_xlabel("Importance (Gain)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost — Gain)",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Feature importance plot saved → {save_path}  (top {top_n} features)")
    logger.info(f"  Top 3: {list(importance_df.index[-3:])[::-1]}")
