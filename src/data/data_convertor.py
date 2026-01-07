"""
Purpose: Convert the processed data into parquet format for efficient storage and retrieval.
1) Function to convert CSV to Parquet.

Output: Save filtered Parquet file to data/raw/loan_preprocessed.parquet. 
"""
import pandas as pd
from loguru import logger

def convert_csv_to_parquet(input_csv_path, output_parquet_path):
    logger.info(f"Converting CSV data from {input_csv_path} to Parquet format.")
    
    
    try:
        logger.info(f"Reading CSV data from {input_csv_path}.")
        df = pd.read_csv(input_csv_path)
        logger.success(f"CSV data read successfully with {len(df)} rows and {len(df.columns)} columns.")
        try: 
            logger.info(f"Converting DataFrame to Parquet and saving to {output_parquet_path}.")
            df.to_parquet(output_parquet_path, index=False)
            logger.success("Data successfully converted to Parquet format.")
        except Exception as e:
            logger.error(f"Error converting to Parquet: {e}")
            return
    except Exception as e:
        logger.error(f"Error reading CSV data: {e}")
        return
    
    

if __name__ == "__main__": 
    input_csv_path = 'data/raw/loan_preprocessed.csv'
    output_parquet_path = 'data/raw/loan_preprocessed.parquet'
    convert_csv_to_parquet(input_csv_path, output_parquet_path)