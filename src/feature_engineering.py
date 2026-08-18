"""
Feature Engineering Module
Creates new features from existing data to improve model performance.
"""

import pandas as pd
import numpy as np
from src.utils import setup_logger

logger = setup_logger("feature_engineering")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from the raw Telco dataset.
    
    Call this BEFORE encoding/scaling — it works on the original columns.
    
    New features created:
        - avg_monthly_charge: Average charge per month of tenure
        - tenure_group: Bucketed tenure (new, short, medium, long, loyal)
        - is_long_term: Binary flag for tenure > 24 months
        - total_services: Count of services the customer uses
        - has_premium_support: Has both OnlineSecurity and TechSupport
        - charge_tenure_ratio: MonthlyCharges relative to tenure
        - is_autopay: Uses automatic payment method
        - has_streaming: Uses any streaming service
    
    Args:
        df: Raw DataFrame (before encoding)
        
    Returns:
        DataFrame with new features added
    """
    logger.info("⚙️  Creating new features...")
    df = df.copy()
    
    # ─── 1. Average Monthly Charge ────────────────────────────────────
    # TotalCharges / tenure gives average monthly charge over lifetime
    # Handle tenure=0 (new customers) to avoid division by zero
    df["avg_monthly_charge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )
    logger.info("    ✅ avg_monthly_charge")
    
    # ─── 2. Tenure Group (Buckets) ────────────────────────────────────
    # Segment customers by how long they've been with the company
    bins = [0, 6, 12, 24, 48, 72]
    labels = ["0-6m", "6-12m", "12-24m", "24-48m", "48-72m"]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, include_lowest=True
    )
    logger.info("    ✅ tenure_group")
    
    # ─── 3. Is Long-Term Customer ─────────────────────────────────────
    df["is_long_term"] = (df["tenure"] > 24).astype(int)
    logger.info("    ✅ is_long_term")
    
    # ─── 4. Total Services Count ──────────────────────────────────────
    # Count how many optional services each customer uses
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    existing_service_cols = [c for c in service_cols if c in df.columns]
    df["total_services"] = df[existing_service_cols].apply(
        lambda row: sum(1 for val in row if val in ["Yes", "Fiber optic", "DSL"]),
        axis=1
    )
    logger.info("    ✅ total_services")
    
    # ─── 5. Has Premium Support ───────────────────────────────────────
    # Customers who have both security AND tech support
    if "OnlineSecurity" in df.columns and "TechSupport" in df.columns:
        df["has_premium_support"] = (
            (df["OnlineSecurity"] == "Yes") & (df["TechSupport"] == "Yes")
        ).astype(int)
        logger.info("    ✅ has_premium_support")
    
    # ─── 6. Charge-to-Tenure Ratio ────────────────────────────────────
    # Higher ratio = paying more relative to how long they've been a customer
    df["charge_tenure_ratio"] = np.where(
        df["tenure"] > 0,
        df["MonthlyCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )
    logger.info("    ✅ charge_tenure_ratio")
    
    # ─── 7. Is Auto-Pay ──────────────────────────────────────────────
    # Customers using automatic payment are less likely to churn
    if "PaymentMethod" in df.columns:
        df["is_autopay"] = df["PaymentMethod"].apply(
            lambda x: 1 if "automatic" in str(x).lower() else 0
        )
        logger.info("    ✅ is_autopay")
    
    # ─── 8. Has Streaming ─────────────────────────────────────────────
    if "StreamingTV" in df.columns and "StreamingMovies" in df.columns:
        df["has_streaming"] = (
            (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
        ).astype(int)
        logger.info("    ✅ has_streaming")
    
    logger.info(f"📊 Total features after engineering: {len(df.columns)}")
    
    return df
