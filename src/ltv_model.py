"""
Customer Lifetime Value (LTV) Model
Predicts the expected remaining tenure and calculates LTV for each customer.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.utils import setup_logger

logger = setup_logger("ltv_model")


def calculate_simple_ltv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a simple LTV based on existing data.
    
    Formula: LTV = Monthly Charges × Expected Remaining Tenure
    
    Where Expected Remaining Tenure is estimated from historical patterns.
    
    Args:
        df: DataFrame with customer data
        
    Returns:
        DataFrame with LTV columns added
    """
    logger.info("💰 Calculating Customer Lifetime Value...")
    df = df.copy()
    
    # Simple LTV = MonthlyCharges * tenure (historical value already generated)
    df["historical_ltv"] = df["MonthlyCharges"] * df["tenure"]
    
    # Average contract lifetime by contract type (estimated from data)
    avg_lifetime = {
        "Month-to-month": 18,   # avg ~18 months
        "One year": 36,         # avg ~36 months
        "Two year": 60          # avg ~60 months
    }
    
    # Predicted remaining tenure
    if "Contract" in df.columns:
        df["expected_total_tenure"] = df["Contract"].map(avg_lifetime).fillna(24)
    else:
        df["expected_total_tenure"] = 36  # default
    
    df["remaining_tenure"] = np.maximum(
        df["expected_total_tenure"] - df["tenure"], 0
    )
    
    # Predicted LTV = Monthly Charges × Remaining Tenure
    df["predicted_ltv"] = df["MonthlyCharges"] * df["remaining_tenure"]
    
    # Total LTV = Historical + Predicted Future
    df["total_ltv"] = df["historical_ltv"] + df["predicted_ltv"]
    
    logger.info(f"    Average Historical LTV:  ${df['historical_ltv'].mean():.2f}")
    logger.info(f"    Average Predicted LTV:   ${df['predicted_ltv'].mean():.2f}")
    logger.info(f"    Average Total LTV:       ${df['total_ltv'].mean():.2f}")
    
    return df


def segment_customers(df: pd.DataFrame, ltv_col: str = "total_ltv") -> pd.DataFrame:
    """
    Segment customers into value tiers based on LTV.
    
    Segments:
        🟢 High Value   — Top 20% LTV
        🟡 Medium Value — Middle 60% LTV
        🔴 Low Value    — Bottom 20% LTV
    
    Args:
        df: DataFrame with LTV column
        ltv_col: Name of the LTV column to use
        
    Returns:
        DataFrame with 'ltv_segment' column added
    """
    logger.info("📊 Segmenting customers by LTV...")
    
    # Calculate percentile thresholds
    p20 = df[ltv_col].quantile(0.20)
    p80 = df[ltv_col].quantile(0.80)
    
    df["ltv_segment"] = pd.cut(
        df[ltv_col],
        bins=[-np.inf, p20, p80, np.inf],
        labels=["Low Value", "Medium Value", "High Value"]
    )
    
    # Log segment distribution
    segment_counts = df["ltv_segment"].value_counts()
    for segment, count in segment_counts.items():
        avg_ltv = df[df["ltv_segment"] == segment][ltv_col].mean()
        logger.info(f"    {segment}: {count} customers (avg LTV: ${avg_ltv:.2f})")
    
    return df


def train_ltv_regression(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = "total_ltv"
) -> tuple:
    """
    Train a regression model to predict LTV from customer features.
    
    Args:
        df: DataFrame with features and LTV
        feature_cols: List of feature columns to use
        target_col: LTV column to predict
        
    Returns:
        Tuple of (model, metrics_dict)
    """
    logger.info("🏋️ Training LTV regression model...")
    
    if feature_cols is None:
        feature_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    
    existing_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[existing_cols].fillna(0)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    
    logger.info(f"    MAE:  ${metrics['mae']:.2f}")
    logger.info(f"    RMSE: ${metrics['rmse']:.2f}")
    logger.info(f"    R²:   {metrics['r2']:.4f}")
    
    return model, metrics


def save_ltv_model(model, path: str = "models/ltv_model.pkl") -> None:
    """Save the LTV model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"💾 LTV model saved to: {path}")
    
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data/raw/telco_customer_churn.csv")

    # Clean TotalCharges because some values may be blank/text
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    ).fillna(0)

    # Calculate LTV
    df = calculate_simple_ltv(df)

    # Train LTV regression model
    model, metrics = train_ltv_regression(df)

    # Save model
    save_ltv_model(model)

    print("\nLTV model training completed successfully!")