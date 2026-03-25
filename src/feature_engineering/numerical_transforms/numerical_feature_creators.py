"""
numerical_feature_creators.py
------------------------------
Custom reusable feature engineering functions for the credit risk ML pipeline.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)


# ===========================================================================
# GROUP 1 — REUSABLE RATIO FUNCTION (A / B)
# ===========================================================================

def make_ratio(df: pd.DataFrame,
               numerator: str,
               denominator: str,
               new_col: str,
               offset: float = 0.0,
               clip_lower: float = None,
               clip_upper: float = None) -> pd.DataFrame:
    """
    Creates a new ratio column: new_col = numerator / (denominator + offset)

    The offset is added to the denominator ONLY during division —
    the original denominator column is NOT modified.
    """
    logger.info(f"Creating ratio feature: '{new_col}' = {numerator} / ({denominator} + {offset})")

    denom_vals = df[denominator] + offset
    zero_count = (denom_vals == 0).sum()
    if zero_count > 0:
        logger.warning(f"  '{denominator}' still has {zero_count} zeros after offset={offset}. "
                       f"Result will contain inf/nan for those rows.")

    df[new_col] = df[numerator] / denom_vals

    if clip_lower is not None or clip_upper is not None:
        df[new_col] = df[new_col].clip(lower=clip_lower, upper=clip_upper)
        logger.debug(f"  '{new_col}' clipped — lower={clip_lower}, upper={clip_upper}")

    logger.debug(f"  '{new_col}' — min={df[new_col].min():.4f}, "
                 f"max={df[new_col].max():.4f}, "
                 f"nulls={df[new_col].isna().sum()}")
    return df


# ===========================================================================
# GROUP 2 — REUSABLE PRODUCT FUNCTION (A × B)
# ===========================================================================

def make_product(df: pd.DataFrame,
                 col_a: str,
                 col_b: str,
                 new_col: str,
                 scale: float = 1.0,
                 clip_upper: float = None) -> pd.DataFrame:
    """
    Creates a new product column: new_col = col_a * col_b * scale
    """
    logger.info(f"Creating product feature: '{new_col}' = {col_a} × {col_b} × {scale}")

    df[new_col] = df[col_a] * df[col_b] * scale

    if clip_upper is not None:
        df[new_col] = df[new_col].clip(upper=clip_upper)
        logger.debug(f"  '{new_col}' clipped upper at {clip_upper}")

    logger.debug(f"  '{new_col}' — min={df[new_col].min():.4f}, "
                 f"max={df[new_col].max():.4f}, "
                 f"nulls={df[new_col].isna().sum()}")
    return df


# ===========================================================================
# GROUP 3 — BINARY FLAG FEATURES
# ===========================================================================

def make_high_revol_util(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating binary flag: 'high_revol_util' = (revol_util >= 75)")
    df["high_revol_util"] = (df["revol_util"] >= 75).astype(int)
    logger.debug(f"  'high_revol_util' — {df['high_revol_util'].sum()} positives "
                 f"({df['high_revol_util'].mean()*100:.1f}%)")
    return df


def make_recent_intensive(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating intensity feature: 'recent_intensive' = open_acc_6m²")
    df["recent_intensive"] = df["open_acc_6m"] ** 2
    logger.debug(f"  'recent_intensive' — min={df['recent_intensive'].min():.2f}, "
                 f"max={df['recent_intensive'].max():.2f}, "
                 f"nulls={df['recent_intensive'].isna().sum()}")
    return df


# ===========================================================================
# GROUP 4 — BINNING FEATURES
# ===========================================================================

def make_revol_util_tier(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating bin feature: 'revol_util_tier' from revol_util")
    df["revol_util_tier"] = pd.cut(
        df["revol_util"],
        bins=[0, 30, 60, 80, 100],
        labels=["low", "moderate", "high", "critical"],
        include_lowest=True,
        right=False,
    )
    logger.debug(f"  'revol_util_tier' distribution: "
                 f"{df['revol_util_tier'].value_counts().to_dict()}")
    return df


# ===========================================================================
# GROUP 5 — LOG-STRESS FEATURES
# ===========================================================================

def make_stress_util_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates: stress_util_income = (all_util / 100) / log1p(annual_inc)

    Fix: annual_inc clipped to minimum 1 ONLY in the denominator during
    division — the annual_inc column itself is NOT modified.
    """
    logger.info("Creating log-stress feature: 'stress_util_income'")

    # ✅ Fix: clip lower=1 only on denominator — annual_inc column unchanged
    safe_log_income = np.log1p(df["annual_inc"].clip(lower=1))

    df["stress_util_income"] = (df["all_util"] / 100) / safe_log_income

    logger.debug(f"  'stress_util_income' — min={df['stress_util_income'].min():.6f}, "
                 f"max={df['stress_util_income'].max():.6f}, "
                 f"nulls={df['stress_util_income'].isna().sum()}")
    return df


def make_util_rev_stress_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates: util_total_rev_stress_log = log1p(all_util * total_rev_hi_lim)
    No division — safe from inf, no fix needed.
    """
    logger.info("Creating log-stress feature: 'util_total_rev_stress_log'")
    raw_product = df["all_util"] * df["total_rev_hi_lim"]
    df["util_total_rev_stress_log"] = np.log1p(raw_product)
    logger.debug(f"  'util_total_rev_stress_log' — min={df['util_total_rev_stress_log'].min():.4f}, "
                 f"max={df['util_total_rev_stress_log'].max():.4f}, "
                 f"nulls={df['util_total_rev_stress_log'].isna().sum()}")
    return df


# ===========================================================================
# GROUP 6 — INSTALLMENT + PTI
# ===========================================================================

def make_installment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers installment from loan_amnt + term.
    Drops term from numerical pipeline — term stays in CAT_COLS separately.
    """
    term_months = (
        df["term"]
        .astype(str)
        .str.extract(r"(\d+)")[0]   # ← [0] instead of .squeeze()
        .astype(float)
    )
    term_months = term_months.clip(1, 360)   # ← positional (min, max), no kwargs
    term_months = term_months.replace(0, 36)

    df["installment"] = df["loan_amnt"] / term_months
    df = df.drop(columns=["term"])
    return df



def make_pti(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates: pti = (installment / (annual_inc / 12)) * 100

    Fix: annual_inc clipped to minimum 1 ONLY in the denominator during
    division — the annual_inc column itself is NOT modified.
    """
    logger.info("Creating PTI feature: 'pti' = (installment / (annual_inc / 12)) * 100")

    # ✅ Fix: clip lower=1 only on denominator — annual_inc column unchanged
    safe_monthly_income = df["annual_inc"].clip(lower=1) / 12

    df["pti"] = (df["installment"] / safe_monthly_income) * 100

    logger.debug(f"  'pti' — min={df['pti'].min():.4f}, "
                 f"max={df['pti'].max():.4f}, "
                 f"nulls={df['pti'].isna().sum()}")
    return df


# ===========================================================================
# MASTER CALLER — used as function inside sklearn FunctionTransformer
# ===========================================================================

def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering function — called by sklearn FunctionTransformer
    inside numerical_pipeline.
    """
    logger.info("=" * 60)
    logger.info("START: create_all_features()")
    logger.info(f"  Input shape: {df.shape}")

    df = df.copy()

    # STEP 1: Ratio features
    logger.info("-- STEP 1: Ratio features --")
    df = make_installment(df)
    df = make_ratio(df, "num_rev_accts", "mo_sin_old_rev_tl_op", "rev_accts_to_age",
                    offset=1.0)
    df = make_ratio(df, "num_rev_tl_bal_gt_0", "num_rev_accts", "indebt_rev_ratio",
                    offset=1.0)
    df = make_ratio(df, "acc_open_past_24mths", "open_acc", "new_account_share",
                    offset=1.0, clip_lower=0.0, clip_upper=1.0)
    df = make_ratio(df, "total_bc_limit", "revol_bal", "bc_limit_util",
                    offset=1.0)
    df = make_ratio(df, "revol_bal", "annual_inc", "revol_bal_to_income",
                    offset=1.0)                   # ✅ offset=1 prevents 0 annual_inc issue
    df = make_ratio(df, "open_il_12m", "open_act_il", "il_recent_share",
                    offset=1.0, clip_upper=1.0)

    # STEP 2: Product features
    logger.info("-- STEP 2: Product features --")
    df = make_product(df, "num_actv_rev_tl", "revol_util", "actv_rev_util")
    df = make_product(df, "bc_util", "percent_bc_gt_75", "bc_util_stress", scale=1/100)
    df = make_product(df, "revol_util", "revol_bal", "revol_stress_score")
    df = make_product(df, "open_act_il", "open_il_12m", "il_recent_intensive", clip_upper=50.0)

    # STEP 3: Binary flag features
    logger.info("-- STEP 3: Binary flag features --")
    df = make_high_revol_util(df)
    df = make_recent_intensive(df)

    # STEP 4: Binning features
    logger.info("-- STEP 4: Binning features --")
    df = make_revol_util_tier(df)

    # STEP 5: Log-stress features
    logger.info("-- STEP 5: Log-stress features --")
    df = make_stress_util_income(df)
    df = make_util_rev_stress_log(df)

    # STEP 6: PTI
    logger.info("-- STEP 6: PTI feature --")
    df = make_pti(df)

    # STEP 7: Drop raw redundant columns
    cols_to_drop = ["revol_bal", "pub_rec"]
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    logger.info(f"-- STEP 7: Dropping raw columns: {existing_drops} --")
    df.drop(columns=existing_drops, inplace=True)

    logger.info(f"  Output shape: {df.shape}")
    logger.info("END: create_all_features()")
    logger.info("=" * 60)

    return df
