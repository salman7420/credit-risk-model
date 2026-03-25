"""
app.py
------
Entry point for the Lending Club Credit Risk Tool.
Sets global page config and renders the landing page.
"""

import streamlit as st

st.set_page_config(
    page_title="Credit Risk Assessment Tool",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏦 Credit Risk Assessment Tool")
st.markdown("""
Welcome to the **Lending Club Credit Risk Assessment Tool**.

Use the sidebar to navigate:
- **Loan Assessment** — Enter borrower details and get an instant risk decision
- **Model Insights** — View model performance, feature importance, and threshold analysis
""")

st.info("👈 Select **Loan Assessment** from the sidebar to get started.")
