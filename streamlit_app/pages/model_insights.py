"""
2_model_insights.py
-------------------
Model performance dashboard.
Displays AUC, confusion matrix, feature importance,
and threshold sweep — all loaded from pre-saved artifacts in models/.
No re-computation needed — everything was saved during training.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Model Insights",
    page_icon="📊",
    layout="wide",
)

ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
THRESHOLD_PATH = MODELS_DIR / "best_threshold.json"

st.title("📊 Model Insights")
st.markdown("Performance metrics and diagnostics for the trained XGBoost credit risk model.")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — KEY METRICS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🎯 Test Set Performance")

if THRESHOLD_PATH.exists():
    with open(THRESHOLD_PATH) as f:
        thresh_data = json.load(f)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AUC-ROC",          "0.7242",                    help="Area under ROC curve — model's ranking ability")
    col2.metric("F1 Score",         f'{thresh_data.get("test_f1", "N/A")}',        help="Balance of precision and recall at best threshold")
    col3.metric("Precision",        f'{thresh_data.get("test_precision", "N/A")}', help="Of all predicted defaults, how many were correct")
    col4.metric("Recall",           f'{thresh_data.get("test_recall", "N/A")}',    help="Of all actual defaults, how many were caught")
    col5.metric("Best Threshold",   f'{thresh_data.get("best_threshold", "N/A")}', help="Optimal decision cutoff tuned on validation set")
else:
    st.warning("⚠️ best_threshold.json not found. Run threshold_tuner.py first.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PLOTS (saved during training)
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 ROC Curve",
    "🗂️ Confusion Matrix",
    "🏆 Feature Importance",
    "⚖️ Threshold Sweep",
])

def _show_image(tab, path: Path, caption: str):
    with tab:
        if path.exists():
            img = Image.open(path)
            st.image(img, caption=caption, use_container_width=True)
        else:
            st.warning(f"⚠️ Plot not found at `{path.name}`. Run training first.")

_show_image(tab1, MODELS_DIR / "roc_curve.png",          "ROC Curve — Val vs Test AUC")
_show_image(tab2, MODELS_DIR / "confusion_matrix.png",   "Confusion Matrix — Test Set")
_show_image(tab3, MODELS_DIR / "feature_importance.png", "Top 20 Feature Importances (XGBoost gain)")
_show_image(tab4, MODELS_DIR / "threshold_sweep.png",    "Precision / Recall / F1 vs Decision Threshold")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DECISION ZONE EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🚦 Risk Decision Zones")
st.markdown("""
The model outputs a **default probability (0–100%)** for each borrower.
This probability is mapped to a decision using the following zones:
""")

zone_df = pd.DataFrame({
    "Zone":        ["🟢 APPROVE",       "🟡 MANUAL REVIEW", "🔴 REJECT"],
    "Probability": ["Below 35%",        "35% – 55%",        "Above 55%"],
    "Meaning":     [
        "Low default risk — borrower meets lending criteria",
        "Borderline risk — recommend human review before decision",
        "High default risk — does not meet lending criteria",
    ],
})
st.dataframe(zone_df, use_container_width=True, hide_index=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MODEL DETAILS
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🔧 Model Details")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Algorithm:** XGBoost Classifier  
    **Hyperparameter Tuning:** Optuna (50 trials, Bayesian TPE)  
    **Cross-Validation:** 5-fold Stratified  
    **Training Data:** Lending Club loans (730K rows)  
    **Class Imbalance Handling:** `scale_pos_weight` tuned by Optuna  
    """)

with col2:
    st.markdown("""
    **Feature Engineering:**
    - 37 raw numerical features → 51 engineered features
    - Ratio, product, binary flag, log-stress, PTI features
    - Categorical features: emp_length, home_ownership, purpose, addr_state, earliest_cr_line
    - RobustScaler + OrdinalEncoder for all features
    """)
