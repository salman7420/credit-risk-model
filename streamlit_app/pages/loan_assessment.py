"""
1_loan_assessment.py
--------------------
Single borrower risk assessment form.
Collects all raw features exactly as seen during training,
passes to predictor.py, displays risk score + decision.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(
    page_title="Loan Assessment",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Loan Risk Assessment")
st.markdown("Fill in the borrower details below and click **Assess Risk** to get a decision.")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAN DETAILS
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🏷️ Loan Details")
col1, col2, col3 = st.columns(3)

with col1:
    loan_amnt = st.number_input(
        "Loan Amount ($)", min_value=500, max_value=40000,
        value=10000, step=500,
        help="Total loan amount requested",
    )

with col2:
    term = st.selectbox(
        "Loan Term",
        options=[" 36 months", " 60 months"],
        help="Number of months for repayment",
    )

with col3:
    purpose = st.selectbox(
        "Loan Purpose",
        options=[
            "debt_consolidation", "credit_card", "home_improvement",
            "other", "major_purchase", "medical", "small_business",
            "car", "vacation", "moving", "house",
            "wedding", "renewable_energy", "educational",
        ],
        help="Primary reason for the loan",
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — BORROWER PROFILE
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("👤 Borrower Profile")
col1, col2, col3 = st.columns(3)

with col1:
    annual_inc = st.number_input(
        "Annual Income ($)", min_value=0, max_value=500000,
        value=60000, step=1000,
        help="Self-reported annual income",
    )
    emp_length = st.selectbox(
        "Employment Length",
        options=[
            "< 1 year", "1 year", "2 years", "3 years", "4 years",
            "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years",
        ],
        index=6,
        help="Years at current employer",
    )

with col2:
    home_ownership = st.selectbox(
        "Home Ownership",
        options=["RENT", "MORTGAGE", "OWN"],
        help="Current housing status",
    )
    verification_status = st.selectbox(
        "Income Verification Status",
        options=["Not Verified", "Verified", "Source Verified"],
        help="Whether income was verified by Lending Club",
    )

with col3:
    dti = st.number_input(
        "Debt-to-Income Ratio (DTI)",
        min_value=0.0, max_value=50.0,
        value=15.0, step=0.1,
        help="Monthly debt payments / monthly income × 100",
    )
    addr_state = st.selectbox(
        "State",
        options=[
            "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
            "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
            "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
            "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
            "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC",
        ],
        index=4,
        help="Borrower's state of residence",
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CREDIT HISTORY
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Credit History")
col1, col2, col3 = st.columns(3)

with col1:
    earliest_cr_line = st.text_input(
        "Earliest Credit Line (MMM-YYYY)",
        value="Jan-2005",
        help="Month and year of oldest credit account e.g. Jan-2005",
    )
    delinq_2yrs = st.number_input(
        "Delinquencies (Last 2 Years)",
        min_value=0, max_value=20, value=0,
        help="Number of 30+ day late payments in past 2 years",
    )
    inq_last_6mths = st.number_input(
        "Credit Inquiries (Last 6 Months)",
        min_value=0, max_value=10, value=1,
        help="Number of hard credit inquiries in last 6 months",
    )

with col2:
    open_acc = st.number_input(
        "Open Credit Accounts",
        min_value=0, max_value=50, value=10,
        help="Number of currently open credit accounts",
    )
    pub_rec = st.number_input(
        "Public Record Derogatory Marks",
        min_value=0, max_value=10, value=0,
        help="Number of derogatory public records",
    )
    pub_rec_bankruptcies = st.number_input(
        "Public Record Bankruptcies",
        min_value=0, max_value=5, value=0,
        help="Number of public record bankruptcies",
    )

with col3:
    mort_acc = st.number_input(
        "Mortgage Accounts",
        min_value=0, max_value=20, value=1,
        help="Number of mortgage accounts",
    )
    mo_sin_old_rev_tl_op = st.number_input(
        "Months Since Oldest Revolving Account",
        min_value=0, max_value=600, value=164,
        help="Months since oldest revolving account opened",
    )
    acc_open_past_24mths = st.number_input(
        "Accounts Opened (Last 24 Months)",
        min_value=0, max_value=30, value=4,
        help="Number of new accounts opened in past 24 months",
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — REVOLVING CREDIT
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("💳 Revolving Credit")
col1, col2, col3 = st.columns(3)

with col1:
    revol_bal = st.number_input(
        "Revolving Balance ($)", min_value=0, max_value=300000,
        value=15000, step=500,
        help="Total current revolving credit balance",
    )
    revol_util = st.slider(
        "Revolving Utilization (%)",
        min_value=0.0, max_value=100.0, value=45.0, step=0.1,
        help="Revolving credit used / total revolving credit limit",
    )

with col2:
    total_rev_hi_lim = st.number_input(
        "Total Revolving High Credit Limit ($)",
        min_value=0, max_value=500000, value=30000, step=1000,
        help="Total high credit/credit limit on revolving accounts",
    )
    num_rev_accts = st.number_input(
        "Number of Revolving Accounts",
        min_value=0, max_value=60, value=13,
        help="Total number of revolving accounts",
    )

with col3:
    num_actv_rev_tl = st.number_input(
        "Active Revolving Accounts",
        min_value=0, max_value=30, value=5,
        help="Number of currently active revolving accounts",
    )
    num_rev_tl_bal_gt_0 = st.number_input(
        "Revolving Accounts with Balance > 0",
        min_value=0, max_value=30, value=5,
        help="Revolving accounts currently carrying a balance",
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — INSTALLMENT & UTILIZATION
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🏦 Installment & Utilization")
col1, col2, col3 = st.columns(3)

with col1:
    tot_cur_bal = st.number_input(
        "Total Current Balance ($)",
        min_value=0, max_value=2000000, value=80000, step=1000,
        help="Total current balance of all accounts",
    )
    tot_hi_cred_lim = st.number_input(
        "Total High Credit Limit ($)",
        min_value=0, max_value=2000000, value=112000, step=1000,
        help="Total high credit/credit limit across all accounts",
    )

with col2:
    bc_util = st.number_input(
        "Bankcard Utilization (%)",
        min_value=0.0, max_value=120.0, value=55.0, step=0.1,
        help="Ratio of total current balance to high credit/limit for bankcard accounts",
    )
    total_bc_limit = st.number_input(
        "Total Bankcard Limit ($)",
        min_value=0, max_value=300000, value=15000, step=500,
        help="Total bankcard high credit/credit limit",
    )

with col3:
    percent_bc_gt_75 = st.number_input(
        "% Bankcards > 75% Utilized",
        min_value=0.0, max_value=100.0, value=40.0, step=0.1,
        help="Percentage of all bankcard accounts with utilization > 75%",
    )
    avg_cur_bal = st.number_input(
        "Average Current Balance ($)",
        min_value=0, max_value=200000, value=7000, step=500,
        help="Average current balance of all accounts",
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — INSTALLMENT LOAN ACTIVITY
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Installment Loan Activity")
col1, col2, col3 = st.columns(3)

with col1:
    open_acc_6m = st.number_input(
        "Open Accounts (Last 6 Months)",
        min_value=0, max_value=10, value=1,
        help="Number of new accounts opened in last 6 months",
    )
    open_act_il = st.number_input(
        "Active Installment Accounts",
        min_value=0, max_value=30, value=3,
        help="Number of currently active installment accounts",
    )
    open_il_12m = st.number_input(
        "Installment Accounts Opened (Last 12m)",
        min_value=0, max_value=10, value=1,
        help="Number of installment accounts opened in last 12 months",
    )
    inq_last_12m = st.number_input(
        "Credit Inquiries (Last 12 Months)",
        min_value=0, max_value=30, value=3,
        help="Number of credit inquiries in last 12 months",
    )
    num_op_rev_tl = st.number_input(
        "Open Revolving Accounts",
        min_value=0, max_value=40, value=7,
        help="Number of open revolving accounts",
    )

with col2:
    open_il_24m = st.number_input(
        "Installment Accounts Opened (Last 24m)",
        min_value=0, max_value=15, value=2,
        help="Number of installment accounts opened in last 24 months",
    )
    il_util = st.number_input(
        "Installment Loan Utilization (%)",
        min_value=0.0, max_value=150.0, value=60.0, step=0.1,
        help="Ratio of total current balance to high credit/limit on installment accounts",
    )
    open_rv_12m = st.number_input(
        "Revolving Accounts Opened (Last 12m)",
        min_value=0, max_value=15, value=2,
        help="Number of revolving accounts opened in last 12 months",
    )

with col3:
    open_rv_24m = st.number_input(
        "Revolving Accounts Opened (Last 24m)",
        min_value=0, max_value=20, value=4,
        help="Number of revolving accounts opened in last 24 months",
    )
    all_util = st.number_input(
        "Balance to Credit Limit (All Accounts %)",
        min_value=0.0, max_value=150.0, value=55.0, step=0.1,
        help="Balance to credit limit on all trades",
    )
    inq_fi = st.number_input(
        "Finance Inquiries",
        min_value=0, max_value=20, value=1,
        help="Number of personal finance inquiries",
    )

# ─────────────────────────────────────────────────────────────────────────────
# SUBMIT BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    submitted = st.button("🔍 Assess Risk", use_container_width=True, type="primary")

if submitted:
    # ── Build raw input DataFrame exactly matching training column names
    input_data = pd.DataFrame([{
        "loan_amnt":            loan_amnt,
        "term":                 term,
        "annual_inc":           annual_inc,
        "emp_length":           emp_length,
        "home_ownership":       home_ownership,
        "verification_status":  verification_status,
        "purpose":              purpose,
        "addr_state":           addr_state,
        "earliest_cr_line":     earliest_cr_line,
        "dti":                  dti,
        "delinq_2yrs":          delinq_2yrs,
        "inq_last_6mths":       inq_last_6mths,
        "open_acc":             open_acc,
        "pub_rec":              pub_rec,
        "revol_bal":            revol_bal,
        "revol_util":           revol_util,
        "tot_cur_bal":          tot_cur_bal,
        "open_acc_6m":          open_acc_6m,
        "open_act_il":          open_act_il,
        "open_il_12m":          open_il_12m,
        "open_il_24m":          open_il_24m,
        "il_util":              il_util,
        "open_rv_12m":          open_rv_12m,
        "open_rv_24m":          open_rv_24m,
        "all_util":             all_util,
        "total_rev_hi_lim":     total_rev_hi_lim,
        "inq_fi":               inq_fi,
        "inq_last_12m":         inq_last_12m,
        "bc_util":              bc_util,
        "percent_bc_gt_75":     percent_bc_gt_75,
        "acc_open_past_24mths": acc_open_past_24mths,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "mort_acc":             mort_acc,
        "mo_sin_old_rev_tl_op": mo_sin_old_rev_tl_op,
        "num_rev_accts":        num_rev_accts,
        "tot_hi_cred_lim":      tot_hi_cred_lim,
        "total_bc_limit":       total_bc_limit,
        "avg_cur_bal":          avg_cur_bal,
        "num_op_rev_tl":        num_op_rev_tl,
        "num_actv_rev_tl":      num_actv_rev_tl,
        "num_rev_tl_bal_gt_0":  num_rev_tl_bal_gt_0,
    }])

    # ── Run prediction
    with st.spinner("🔍 Analyzing borrower risk..."):
        from streamlit_app.utils.predictor import predict
        result = predict(input_data)

    st.divider()

    # ── DECISION BANNER
    decision = result["decision"]
    if decision["label"] == "APPROVE":
        st.success(f'{decision["emoji"]} **{decision["label"]}** — {decision["description"]}')
    elif decision["label"] == "MANUAL REVIEW":
        st.warning(f'{decision["emoji"]} **{decision["label"]}** — {decision["description"]}')
    else:
        st.error(f'{decision["emoji"]} **{decision["label"]}** — {decision["description"]}')

    # ── SCORE METRICS ROW
    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability",  f'{result["probability_pct"]}%')
    col2.metric("Decision Threshold",   f'{round(result["threshold"] * 100, 1)}%')
    col3.metric(
        "Risk Zone",
        decision["label"],
        delta="above threshold" if result["probability"] >= result["threshold"] else "below threshold",
        delta_color="inverse",
    )

    # ── SHAP FACTORS
    st.divider()
    st.subheader("🔍 Top Risk Drivers")
    st.caption("The 5 features with the highest impact on this borrower's risk score.")

    for _, row in result["shap_factors"].iterrows():
        icon = "🔴" if row["shap_val"] > 0 else "🟢"
        st.markdown(
            f"{icon} **{row['feature']}** — SHAP impact: "
            f"`{row['shap_val']:+.4f}` &nbsp; {row['direction']}"
        )

    # ── RAW INPUT PREVIEW
    with st.expander("📄 Raw Input Preview"):
        preview = input_data.T.rename(columns={0: "Value"})
        preview["Value"] = preview["Value"].astype(str)
        st.dataframe(preview)
