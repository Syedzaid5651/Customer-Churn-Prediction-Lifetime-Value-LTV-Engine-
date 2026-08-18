"""
Data Ingestion Script
Loads the Telco Customer Churn CSV dataset into a PostgreSQL database.

Usage:
    python src/data_ingestion.py

Prerequisites:
    1. PostgreSQL must be running
    2. Database 'churn_db' must exist
    3. .env file must have correct credentials
    4. CSV file must be in data/raw/telco_customer_churn.csv
"""

import os
import sys
import pandas as pd
from sqlalchemy import text

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_engine, setup_logger, TABLE_NAME

logger = setup_logger("data_ingestion")


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load the Telco Customer Churn CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the raw data
    """
    logger.info(f"📂 Loading CSV from: {filepath}")
    
    if not os.path.exists(filepath):
        logger.error(f"❌ File not found: {filepath}")
        logger.info("💡 Download the dataset from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial data cleaning before loading into database.
    
    Steps:
        1. Convert TotalCharges to numeric (has some blank strings)
        2. Strip whitespace from column names
        3. Log basic data quality info
    
    Args:
        df: Raw DataFrame from CSV
        
    Returns:
        Cleaned DataFrame ready for database insertion
    """
    logger.info("🧹 Performing basic data cleaning...")
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # TotalCharges has some blank strings — convert to numeric
    # Blank strings become NaN, which we'll handle later in preprocessing
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Log data quality summary
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.info("⚠️  Null values found:")
        for col, count in null_counts[null_counts > 0].items():
            logger.info(f"    {col}: {count} null values")
    else:
        logger.info("✅ No null values found")
    
    # Log churn distribution
    churn_dist = df["Churn"].value_counts()
    logger.info(f"📊 Churn Distribution:")
    logger.info(f"    No (Retained):  {churn_dist.get('No', 0)} ({churn_dist.get('No', 0)/len(df)*100:.1f}%)")
    logger.info(f"    Yes (Churned):  {churn_dist.get('Yes', 0)} ({churn_dist.get('Yes', 0)/len(df)*100:.1f}%)")
    
    return df


def load_to_postgres(df: pd.DataFrame, table_name: str = TABLE_NAME) -> None:
    """
    Load DataFrame into PostgreSQL table.
    
    Args:
        df: Cleaned DataFrame to load
        table_name: Target table name in PostgreSQL
    """
    logger.info(f"📤 Loading data into PostgreSQL table: '{table_name}'...")
    
    try:
        engine = get_engine()
        
        # Write DataFrame to PostgreSQL
        # if_exists='replace' — drops and recreates the table each time
        # This is fine for development; in production, use 'append' with upsert logic
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
            method="multi",   # Faster bulk insert
            chunksize=1000     # Insert in chunks of 1000 rows
        )
        
        # Verify the load
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.scalar()
        
        logger.info(f"✅ Successfully loaded {row_count} rows into '{table_name}'")
        
    except Exception as e:
        logger.error(f"❌ Failed to load data into PostgreSQL: {e}")
        logger.info("💡 Make sure PostgreSQL is running and .env has correct credentials")
        raise


def load_to_csv_fallback(df: pd.DataFrame) -> None:
    """
    Fallback: If PostgreSQL is not available, save cleaned data as CSV.
    This allows the project to continue with file-based storage.
    
    Args:
        df: Cleaned DataFrame
    """
    output_path = os.path.join("data", "processed", "cleaned_telco_data.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"💾 Saved cleaned data to: {output_path}")


def main():
    """Main ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("🚀 STARTING DATA INGESTION PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Load CSV
    csv_path = os.path.join("data", "raw", "telco_customer_churn.csv")
    df = load_csv(csv_path)
    
    # Step 2: Basic cleaning
    df = basic_cleaning(df)
    
    # Step 3: Try loading to PostgreSQL, fallback to CSV
    try:
        load_to_postgres(df)
    except Exception:
        logger.warning("⚠️  PostgreSQL not available. Using CSV fallback...")
        load_to_csv_fallback(df)
    
    logger.info("=" * 60)
    logger.info("✅ DATA INGESTION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
