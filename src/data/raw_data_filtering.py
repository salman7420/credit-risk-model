"""
Purpose: Clean the raw data by removing unwanted columns discussed during initial planning and filtering output column to include the desired target categories only.
Steps:
1) Function to remove unwanted columns (the 25 "Use = No" columns)
2) Function to filter loan_status to only {'Fully Paid', 'Charged Off'}
3) Function to create binary target variable (0 = Fully Paid, 1 = Charged Off)

Output: Save filtered CSV to data/raw/loan_preprocessed.csv (or intermediate location)
"""

import pandas as pd
from loguru import logger


def read_raw_data():
    """Read raw data from CSV file."""
    
    FILE_PATH = 'data/raw/loan.csv'
    try:
        logger.info(f"Reading raw data from {FILE_PATH}")
        df = pd.read_csv(FILE_PATH)
        logger.success(f"Loaded {len(df)} rows and {len(df.columns)} columns from raw data.")
        return df

    except FileNotFoundError:
        logger.error(f"File not found at path {FILE_PATH}. Please check if file exist at the specified path.")
        return None
    
    except Exception as e: 
        logger.error(f"Unexpected error reading CSV: {e}")
        return None


def remove_unwanted_columns(df):
    """Remove unwanted columns from the DataFrame."""
    if df is None:
        logger.error("DataFrame is None. Cannot remove unwanted columns.")
        return None
    # Removing total 24 unwanted columns that either provide future data or are not relevant to the model training.
    COLUMNS_TO_REMOVE = [
    # Identifiers
    'id', 'member_id', 'url',
    # LC Outputs
    'grade', 'sub_grade', 'int_rate', 'installment', 'initial_list_status', 'issue_d',
    # Payment/Performance (19)
    'funded_amnt', 'funded_amnt_inv', 'out_prncp', 'out_prncp_inv',
    'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
    'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
    'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
    'pymnt_plan', 'policy_code', 'mths_since_recent_bc_dlq',
    # Hardship (15)
    'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
    'deferral_term', 'hardship_amount', 'hardship_start_date', 'hardship_end_date',
    'payment_plan_start_date', 'hardship_length', 'hardship_dpd',
    'hardship_loan_status', 'orig_projected_additional_accrued_interest',
    'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
    # Settlement (7)
    'debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_status',
    'settlement_date', 'settlement_amount', 'settlement_percentage', 'settlement_term',
    # Other
    'desc', 'title', 'zip_code', 'disbursement_method', 'num_accts_ever_120_pd'
]
                        
    try:
        logger.info("Removing unwanted columns from the DataFrame.")
        df = df.drop(columns=COLUMNS_TO_REMOVE,errors='ignore')
        logger.success("Unwanted columns removed successfully.")
        logger.success(f"DataFrame now has {len(df.columns)} columns.")
        return df

    except KeyError as e:
        logger.error(f"Error removing columns: {e}")
        return None
    


def filter_loan_status(df):
    """Filter loan_status to include only 'Fully Paid' and 'Charged Off'."""
    
    if df is None:
        logger.error("DataFrame is None. Cannot filter loan_status.")
        return None
    
    elif 'loan_status' not in df.columns:
        logger.error("loan_status column not found in DataFrame.")
        return None
    
    try:
        logger.info("Filtering loan_status to include only 'Fully Paid' and 'Charged off'")
        df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])]
        if df.empty:
            logger.warning("No rows found with valid loan_status")
            return None
        return df
    
    except KeyError as e:
        logger.error(f"Error filtering loan_status: {e}")
        return None
    


def save_raw_processed_data(df):
    """Save the processed DataFrame to a CSV file: data/raw/loan_preprocessed.csv"""
    if df is not None:
        try:
            OUTPUT_FILE_PATH = 'data/raw/loan_preprocessed.csv'
            logger.info(f"Saving processed data to {OUTPUT_FILE_PATH}")
            df.to_csv(OUTPUT_FILE_PATH, index=False)
            logger.success("Processed data saved successfully.")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    else:
        logger.error("DataFrame is None. Cannot save processed data.")

if __name__ == "__main__":  
    df = read_raw_data()
    df = remove_unwanted_columns(df)
    df = filter_loan_status(df)
    save_raw_processed_data(df)
