
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "loan_selected.parquet")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")

# ─────────────────────────────────────────────
# SPLIT CONFIG
# ─────────────────────────────────────────────
TARGET_COL   = "target"
TEST_SIZE    = 0.15   # 15% test
VAL_SIZE     = 0.15   # 15% val  (from remaining 85%)
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────

def load_data(input_path: str = INPUT_PATH) -> pd.DataFrame:
    """Load selected parquet file."""
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    logger.success(f"Loaded. Shape: {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str   = TARGET_COL,
    test_size: float  = TEST_SIZE,
    val_size: float   = VAL_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Stratified train / val / test split.

    Returns
    -------
    X_train, X_val, X_test : feature DataFrames
    y_train, y_val, y_test : target Series
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"Full dataset  → X: {X.shape}  |  Default rate: {y.mean():.2%}")

    # ── Step 1: split off TEST first ──────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size    = test_size,       # 15% → test
        stratify     = y,               # preserve 20% default ratio
        random_state = random_state,
    )

    # ── Step 2: split remaining 85% into TRAIN + VAL ──────────────
    # val_size relative to the 85% remaining = 0.15/0.85 ≈ 0.176
    val_size_adjusted = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size    = val_size_adjusted,
        stratify     = y_temp,
        random_state = random_state,
    )

    # ── Log split summary ─────────────────────────────────────────
    logger.info("─" * 45)
    logger.info(f"Train  → X: {X_train.shape}  | Default rate: {y_train.mean():.2%}")
    logger.info(f"Val    → X: {X_val.shape}    | Default rate: {y_val.mean():.2%}")
    logger.info(f"Test   → X: {X_test.shape}   | Default rate: {y_test.mean():.2%}")
    logger.info("─" * 45)

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    splits_dir: str = SPLITS_DIR,
):
    """
    Merge X + y back, saving each split as parquet.
    """
    os.makedirs(splits_dir, exist_ok=True)

    train = X_train.copy(); train[TARGET_COL] = y_train.values
    val   = X_val.copy();   val[TARGET_COL]   = y_val.values
    test  = X_test.copy();  test[TARGET_COL]  = y_test.values

    train.to_parquet(f"{splits_dir}/train.parquet", index=False)
    val.to_parquet(  f"{splits_dir}/val.parquet",   index=False)
    test.to_parquet( f"{splits_dir}/test.parquet",  index=False)

    logger.success(f"Saved splits → {splits_dir}/")
    logger.success(f"  train.parquet  → {train.shape}")
    logger.success(f"  val.parquet    → {val.shape}")
    logger.success(f"  test.parquet   → {test.shape}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_split(
    input_path: str = INPUT_PATH,
    splits_dir: str = SPLITS_DIR,
):
    """Full split pipeline"""
    
    logger.info("=" * 55)
    logger.info("STARTING SPLIT PIPELINE")
    logger.info("=" * 55)

    df = load_data(input_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    save_splits(X_train, X_val, X_test, y_train, y_val, y_test, splits_dir)

    logger.info("=" * 55)
    logger.info("SPLIT COMPLETE — Files saved to data/splits/")
    logger.info("=" * 55)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    run_split()
