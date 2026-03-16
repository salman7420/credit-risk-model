"""
clean_categorical.py
----------
Categorical pre-cleaning and feature engineering functions.
Runs as STEP 1 in categorical_pipeline before any imputation or encoding.

Columns handled here:
  1. emp_length       → string → integer → clip → 3 stability bins
  2. home_ownership   → group ANY/NONE/OTHER → OWN (domain logic)
  3. purpose          → 14 values → 3 risk buckets
  4. addr_state       → 51 states → 4 US regions
  5. earliest_cr_line → date string → credit_history_years,
                        credit_history_months, credit_maturity bins


Output contract:
  - Every column this file touches is LEFT IN a state ready for
    encoding (clean string) or imputation (integer/NaN).
  - Original raw columns are dropped at end of create_all_cat_features().
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
 

from feature_engineering.utils.configs.categorical_config import (
    EMP_LENGTH_MAP, EMP_LENGTH_BINS, EMP_LENGTH_LABELS,
    HOME_OWNERSHIP_MERGE, HOME_OWNERSHIP_VALID,
    PURPOSE_BUCKET_MAP, STATE_REGION_MAP,
    CREDIT_HISTORY_REFERENCE_YEAR,
    CREDIT_MATURITY_BINS, CREDIT_MATURITY_LABELS,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)


# --- emp_length ---
class EmpLengthCleaner(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer for emp_length column.

    Converts emp_length string → ordinal stability bins in one step:
      string → integer → median impute → clip → bin → drop originals

    Why a class and not a function:
      Median is learned from training data at fit() and stored as
      self.median_ — applied to test data at transform() without
      recalculating. Prevents data leakage.

    Fit stores  : self.median_ (computed from X_train only)
    Transform   : uses self.median_ — never recomputes from input data

    Output column : emp_length_stability ("unstable"/"transitional"/"stable")
    Dropped       : emp_length (raw string)
    """

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learns median from training data only.
        Maps strings → integers first, then computes median on valid values.
        """
        emp_num = X["emp_length"].map(EMP_LENGTH_MAP)
        self.median_ = emp_num.median()
        logger.info(f"EmpLengthCleaner.fit() — stored median: {self.median_}")
        return self  # must return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Applies full emp_length cleaning using stored self.median_.
        Never recomputes median from X — uses only what fit() learned.
        """

        X = X.copy()

        # Step 1 — string → integer
        X["emp_length_num"] = X["emp_length"].map(EMP_LENGTH_MAP)
        unmapped = X["emp_length_num"].isna().sum()
        if unmapped > 0:
            logger.warning(f"  {unmapped} rows NaN → imputing with train median ({self.median_})")

        # Step 2 — impute using STORED train median (no leakage)
        X["emp_length_num"] = X["emp_length_num"].fillna(self.median_)

        # Step 3 — clip: values > 10 → 10 so they fall in "stable" bin
        X["emp_length_num"] = X["emp_length_num"].clip(upper=10)
        logger.debug(f"  After clip — min={X['emp_length_num'].min()}, "
                     f"max={X['emp_length_num'].max()}")

        # Step 4 — bin into stability categories
        X["emp_length_stability"] = pd.cut(
            X["emp_length_num"],
            bins=EMP_LENGTH_BINS,
            labels=EMP_LENGTH_LABELS,
            include_lowest=True,  # ensures 0 → "unstable"
            right=True,           # 10 included in "stable"
        )
        logger.info(f"  Distribution: {X['emp_length_stability'].value_counts().to_dict()}")

        # Step 5 — drop raw string + intermediate integer column
        X.drop(columns=["emp_length", "emp_length_num"], inplace=True)
        logger.info("  Dropped: 'emp_length', 'emp_length_num'")

        return X


# --- home_ownership ---
def clean_home_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans home_ownership to 3 valid categories in-place.

    Step 1 — Merge ANY → OWN, OTHER → OWN (similar default rates ~19%)
    Step 2 — Result: RENT / OWN / MORTGAGE only

    Modified in-place : home_ownership (same column name, clean values)
    No new columns    : encoding handled in encoder.py
    """
    logger.info("Cleaning 'home_ownership' → RENT / OWN / MORTGAGE")

    # Step 1 — merge ANY + OTHER → OWN
    df["home_ownership"] = df["home_ownership"].replace(HOME_OWNERSHIP_MERGE)
    logger.info(f"  Merged ANY → OWN, OTHER → OWN")

    # Step 2 - Verify only valid categories remain
    unexpected = set(df["home_ownership"].unique()) - set(HOME_OWNERSHIP_VALID)
    if unexpected:
        logger.warning(f"  Unexpected categories still present: {unexpected}")

    return df



# --- purpose risk buckets ---
def clean_purpose(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups 14 purpose categories into 3 risk buckets based on default rates:
      high_risk   → > 22%  : small_business, renewable_energy, moving
      medium_risk → 18-22% : medical, house, debt_consolidation, other,
                             vacation, major_purchase
      low_risk    → < 18%  : home_improvement, educational, credit_card,
                             car, wedding

    Unknown categories at inference → "medium_risk" (safe default)

    Modified in-place : purpose column replaced by purpose_bucket
    Dropped           : original purpose column
    """
    logger.info("Cleaning 'purpose' → 'purpose_bucket' (3 risk buckets)")

    df["purpose_bucket"] = df["purpose"].map(PURPOSE_BUCKET_MAP)

    # Handle unseen categories at inference time → default to medium_risk
    unseen = df["purpose_bucket"].isna().sum()
    if unseen > 0:
        logger.warning(f"  {unseen} rows had unknown purpose values → defaulting to 'medium_risk'")
        df["purpose_bucket"] = df["purpose_bucket"].fillna("medium_risk")

    logger.info(f"  Distribution: {df['purpose_bucket'].value_counts().to_dict()}")

    df.drop(columns=["purpose"], inplace=True)
    logger.info("  Dropped: 'purpose'")

    return df


# --- addr_state → US regions ---
def clean_addr_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps 51 US state codes to 4 geographic regions:
      northeast / southeast / midwest / west

    Rationale: 51 OHE columns is too high cardinality. Regional grouping
    preserves geographic economic patterns that influence default risk.

    Modified in-place : addr_state replaced by addr_region
    Dropped           : original addr_state column
    """
    logger.info("Cleaning 'addr_state' → 'addr_region' (4 US regions)")

    df["addr_region"] = df["addr_state"].map(STATE_REGION_MAP)

    unmapped = df["addr_region"].isna().sum()
    if unmapped > 0:
        logger.warning(f"  {unmapped} rows had unrecognised state codes → NaN")

    logger.info(f"  Distribution: {df['addr_region'].value_counts().to_dict()}")

    df.drop(columns=["addr_state"], inplace=True)
    logger.info("  Dropped: 'addr_state'")
    return df

    
# --- earliest_cr_line → credit_maturity ---

def clean_earliest_cr_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts earliest_cr_line date string into credit_maturity bins only.

    Rationale: credit_history_years/months skipped — highly collinear with
    mo_sin_old_rev_tl_op (already in numerical features). credit_maturity
    captures non-linear threshold effects as ordinal categories instead.

    Step 1 — Parse date string → extract year
    Step 2 — credit_history_years = REFERENCE_YEAR - extracted_year
              clip(lower=0, upper=64) handles:
                - Future dates / bad data → 0 → "new"
                - Very old accounts (1951) → 64 → "veteran"
    Step 3 — pd.cut into 4 maturity bins:
              [0–5)   → "new"
              [5–15)  → "moderate"
              [15–30) → "established"
              [30–65] → "veteran"

    New column : credit_maturity (ordinal string → OHE in encoder.py)
    Dropped    : earliest_cr_line
    """
    logger.info("Cleaning 'earliest_cr_line' → 'credit_maturity'")

    # Step 1 — parse date, extract year
    parsed = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    nat_count = parsed.isna().sum()
    if nat_count > 0:
        logger.warning(f"  {nat_count} rows could not be parsed → NaT → "
                       f"credit_history_years = NaN → pd.cut → NaN category")

    # Step 2 — compute years, clip to valid range
    credit_history_years = (CREDIT_HISTORY_REFERENCE_YEAR - parsed.dt.year)
    credit_history_years = credit_history_years.clip(lower=0, upper=64)
    logger.debug(f"  credit_history_years — min={credit_history_years.min()}, "
                 f"max={credit_history_years.max()}, "
                 f"nulls={credit_history_years.isna().sum()}")

    # Step 3 — bin into maturity categories
    df["credit_maturity"] = pd.cut(
        credit_history_years,
        bins=CREDIT_MATURITY_BINS,
        labels=CREDIT_MATURITY_LABELS,
        include_lowest=True,  # ensures 0 → "new"
        right=False,          # [left, right) — left edge included
    )

    logger.info(f"  Distribution: {df['credit_maturity'].value_counts().to_dict()}")

    df.drop(columns=["earliest_cr_line"], inplace=True)
    logger.info("  Dropped: 'earliest_cr_line'")

    return df

# ── BUILDER FUNCTION ──────────────────────────────────────────────────────────
def build_categorical_cleaner() -> Pipeline:
    """
    Builds and returns an unfitted sklearn Pipeline for categorical cleaning.

    Step order matters — each step receives the DataFrame output of the
    previous step. EmpLengthCleaner goes first because it has fit() logic
    (learns median from train). All other steps are stateless functions
    wrapped with FunctionTransformer.
    
    Steps:
        1. emp_length_cleaner  → EmpLengthCleaner()        (stateful — learns median)
        2. home_ownership      → FunctionTransformer(...)   (stateless)
        3. purpose             → FunctionTransformer(...)   (stateless)
        4. addr_state          → FunctionTransformer(...)   (stateless)
        5. earliest_cr_line    → FunctionTransformer(...)   (stateless)

    Returns:
        sklearn.pipeline.Pipeline (unfitted)
    """
    return Pipeline(steps=[
        ("emp_length_cleaner",  EmpLengthCleaner()),
        ("home_ownership",      FunctionTransformer(clean_home_ownership)),
        ("purpose",             FunctionTransformer(clean_purpose)),
        ("addr_state",          FunctionTransformer(clean_addr_state)),
        ("earliest_cr_line",    FunctionTransformer(clean_earliest_cr_line)),
    ])