"""
numerical_config.py
--------------------
Column group definitions and constants for NumericalImputer.

All cap thresholds are learned dynamically at fit() using CAP_QUANTILE.
No hardcoded percentile values — learned from training data only.
"""

# ── Quantile used for all upper caps ─────────────────────────────────────────
CAP_QUANTILE = 0.995   # p99.5 consistently across all columns

# ── Special clip constants ────────────────────────────────────────────────────
DTI_LOWER_CLIP          = 0.0   # negative dti = data error
NEW_ACCOUNT_SHARE_UPPER = 1.0   # ratio cap applied in feature_creators
IL_RECENT_SHARE_UPPER   = 1.0   # ratio cap applied in feature_creators

# ── Fill with 0 (no cap) ──────────────────────────────────────────────────────
COLS_ZERO_FILL = [
    "pub_rec_bankruptcies",   # 0 = no bankruptcy — absence is meaningful
]

# ── Fill with median (no cap) ─────────────────────────────────────────────────
COLS_MEDIAN_FILL = [
    "percent_bc_gt_75",       # no capping needed per analysis
]

# ── Cap only — no missing values ─────────────────────────────────────────────
COLS_CAP_ONLY = [
    "loan_amnt",               # 0% missing, heavy right tail
    "annual_inc",             # 0% missing, heavy right tail
    "open_acc",               # 0% missing
    "delinq_2yrs",            # 0% missing, flagged too
    "revol_bal",              # 0% missing — kept for feature_creators, dropped there
]

# ── Fill with 0 THEN cap ─────────────────────────────────────────────────────
COLS_ZERO_FILL_AND_CAP = [
    "open_acc_6m",            # 61.68% missing — absence = no new accounts
    "open_act_il",            # 61.68% missing
    "open_il_12m",            # 61.68% missing
    "open_il_24m",            # 61.68% missing
    "il_util",                # 66.81% missing
    "open_rv_12m",            # 61.68% missing
    "open_rv_24m",            # 61.68% missing
    "all_util",               # 61.68% missing
    "inq_fi",                 # 61.68% missing
    "inq_last_12m",           # 61.68% missing
    "mort_acc",               # low missing, 0 = no mortgages
    "inq_last_6mths",         # ~0% missing, 0 = no recent inquiries
    "pub_rec",                # 0% missing — flagged as has_pub_rec, dropped in feature_creators
]

# ── Fill with median THEN cap ─────────────────────────────────────────────────
COLS_MEDIAN_FILL_AND_CAP = [
    "dti",                    # 0.02% missing, also lower-clipped to 0
    "revol_util",             # 0.06% missing
    "tot_cur_bal",            # 5.18% missing
    "bc_util",                # 4.71% missing
    "acc_open_past_24mths",   # 3.63% missing
    "mo_sin_old_rev_tl_op",   # 0% missing but heavy tail
    "num_rev_accts",          # used in rev_accts_to_age ratio
    "tot_hi_cred_lim",        # impute median, cap tail
    "total_bc_limit",         # used in bc_limit_util ratio
    "avg_cur_bal",            # impute median, cap tail
    "num_op_rev_tl",          # impute median, cap tail
    "num_actv_rev_tl",        # used in actv_rev_util product
    "num_rev_tl_bal_gt_0",    # used in indebt_ratio
    "total_rev_hi_lim",       # used in util_total_rev_stress_log
]

COLS_REVOL_UTIL_TIER = ["revol_util_tier"]