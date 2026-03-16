# categorical_config.py contains:
# --- emp_length ---
EMP_LENGTH_MAP = {
    "< 1 year":  0,
    "1 year":    1,
    "2 years":   2,
    "3 years":   3,
    "4 years":   4,
    "5 years":   5,
    "6 years":   6,
    "7 years":   7,
    "8 years":   8,
    "9 years":   9,
    "10+ years": 10,
}

EMP_LENGTH_BINS   = [0, 2, 5, 10]
EMP_LENGTH_LABELS = ["unstable", "transitional", "stable"]

# --- home_ownership ---
HOME_OWNERSHIP_MERGE = {"ANY": "OWN", "OTHER": "OWN"}
HOME_OWNERSHIP_VALID = ["RENT", "OWN", "MORTGAGE"]

# --- purpose risk buckets ---
PURPOSE_BUCKET_MAP = {
    # High risk — default rate > 22%
    "small_business":    "high_risk",
    "renewable_energy":  "high_risk",
    "moving":            "high_risk",
    # Medium risk — default rate 18–22%
    "medical":           "medium_risk",
    "house":             "medium_risk",
    "debt_consolidation":"medium_risk",
    "other":             "medium_risk",
    "vacation":          "medium_risk",
    "major_purchase":    "medium_risk",
    # Low risk — default rate < 18%
    "home_improvement":  "low_risk",
    "educational":       "low_risk",
    "credit_card":       "low_risk",
    "car":               "low_risk",
    "wedding":           "low_risk",
}



# --- addr_state → US regions ---
STATE_REGION_MAP = {
    # Northeast
    "CT": "northeast", "ME": "northeast", "MA": "northeast",
    "NH": "northeast", "RI": "northeast", "VT": "northeast",
    "NJ": "northeast", "NY": "northeast", "PA": "northeast",
    # Southeast
    "DE": "southeast", "FL": "southeast", "GA": "southeast",
    "MD": "southeast", "NC": "southeast", "SC": "southeast",
    "VA": "southeast", "WV": "southeast", "DC": "southeast",
    "AL": "southeast", "KY": "southeast", "MS": "southeast",
    "TN": "southeast", "AR": "southeast", "LA": "southeast",
    "OK": "southeast", "TX": "southeast",
    # Midwest
    "IL": "midwest", "IN": "midwest", "MI": "midwest",
    "OH": "midwest", "WI": "midwest", "IA": "midwest",
    "KS": "midwest", "MN": "midwest", "MO": "midwest",
    "NE": "midwest", "ND": "midwest", "SD": "midwest",
    # West
    "AZ": "west", "CO": "west", "ID": "west", "MT": "west",
    "NV": "west", "NM": "west", "UT": "west", "WY": "west",
    "AK": "west", "CA": "west", "HI": "west", "OR": "west",
    "WA": "west",
}

# --- earliest_cr_line → credit_maturity ---
CREDIT_HISTORY_REFERENCE_YEAR = 2015
CREDIT_MATURITY_BINS   = [0, 5, 15, 30, 65]   # 65 as hard upper after clip
CREDIT_MATURITY_LABELS = ["new", "moderate", "established", "veteran"]


# OrdinalEncoder — order lists (low → high, model learns increasing signal)
EMP_LENGTH_STABILITY_ORDER = ["unstable", "transitional", "stable"]
PURPOSE_BUCKET_ORDER       = ["low_risk", "medium_risk", "high_risk"]
CREDIT_MATURITY_ORDER      = ["new", "moderate", "established", "veteran"]

# Column name lists — must match order of categories lists above
ORDINAL_COLS = ["emp_length_stability", "purpose_bucket", "credit_maturity"]

# OHE columns — no order needed
OHE_COLS = ["home_ownership", "addr_region", "term", "verification_status"]