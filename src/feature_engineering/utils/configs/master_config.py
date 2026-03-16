"""
master_config.py
----------------
Defines the exact column lists fed into each sub-pipeline
inside the ColumnTransformer in master_pipeline.py.

NUM_COLS  → raw numerical columns going into numerical_pipeline
CAT_COLS  → raw categorical columns going into categorical_pipeline

Rules:
- Every column in the dataset must appear in exactly ONE list
- Columns in neither list are silently dropped (remainder="drop")
- revol_util_tier is NOT listed here — it is created inside
  numerical_pipeline and handled by _ScalerWrapper passthrough,
  then picked up by master_pipeline's categorical transformer
  since ColumnTransformer re-assembles outputs column by column
"""

# ── Numerical columns → numerical_pipeline ───────────────────────────────────
NUM_COLS = [
    "loan_amnt",
    "term",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",                # raw → has_pub_rec flag, dropped after
    "revol_bal",              # raw → ratio features, dropped after
    "revol_util",
    "tot_cur_bal",
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "il_util",
    "open_rv_12m",
    "open_rv_24m",
    "all_util",
    "total_rev_hi_lim",
    "inq_fi",
    "inq_last_12m",
    "bc_util",
    "percent_bc_gt_75",
    "acc_open_past_24mths",
    "pub_rec_bankruptcies",
    "mort_acc",
    "mo_sin_old_rev_tl_op",
    "num_rev_accts",
    "tot_hi_cred_lim",
    "total_bc_limit",
    "avg_cur_bal",
    "num_op_rev_tl",
    "num_actv_rev_tl",
    "num_rev_tl_bal_gt_0",
]

# ── Categorical columns → categorical_pipeline ────────────────────────────────
CAT_COLS = [
    "emp_length",             # → emp_length_stability (EmpLengthCleaner)
    "home_ownership",         # → cleaned in-place
    "purpose",                # → purpose_bucket
    "addr_state",             # → addr_region
    "earliest_cr_line",       # → credit_maturity
    "term",
    "verification_status"
]
