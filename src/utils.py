"""
Utility functions for the Customer Churn & LTV Engine.
Contains database connection helpers, logging setup, and common functions.
"""

import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()

# ─── Logging Setup ───────────────────────────────────────────────────────────

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Create and configure a logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s │ %(name)-20s │ %(levelname)-8s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ─── Database Connection ─────────────────────────────────────────────────────

def get_db_url() -> str:
    """Build PostgreSQL connection URL from environment variables."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "churn_db")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "password")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def get_engine():
    """Create and return a SQLAlchemy engine for PostgreSQL."""
    url = get_db_url()
    engine = create_engine(url, echo=False)
    return engine


def test_connection() -> bool:
    """Test if PostgreSQL connection is working."""
    logger = setup_logger("db_test")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("✅ Database connection successful!")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


# ─── Common Constants ────────────────────────────────────────────────────────

# Target column name
TARGET_COL = "Churn"

# Table name in PostgreSQL
TABLE_NAME = "telco_customers"

# Categorical columns in the dataset
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# Numerical columns in the dataset
NUMERICAL_COLS = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
]

# Columns to drop before modeling
DROP_COLS = ["customerID"]
