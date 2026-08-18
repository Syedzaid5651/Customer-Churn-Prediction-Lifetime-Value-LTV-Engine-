"""
Data Preprocessing Module
Handles missing values, encoding, scaling, and train-test splitting.

This module is used by both notebooks (for exploration) and
the model training pipeline (for production).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

from src.utils import (
    setup_logger, CATEGORICAL_COLS, NUMERICAL_COLS, 
    DROP_COLS, TARGET_COL
)

logger = setup_logger("preprocessing")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    The Telco dataset has missing TotalCharges for customers with 0 tenure
    (new customers who haven't been billed yet). We fill these with 0.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info("🔧 Handling missing values...")
    
    # TotalCharges: fill NaN with 0 (new customers)
    missing_before = df["TotalCharges"].isnull().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    logger.info(f"    TotalCharges: filled {missing_before} missing values with 0")
    
    # Check for any other missing values
    remaining = df.isnull().sum().sum()
    if remaining > 0:
        logger.warning(f"    ⚠️ {remaining} missing values still remain")
        # Fill numeric with median, categorical with mode
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    else:
        logger.info("    ✅ All missing values handled")
    
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the target variable (Churn) to binary: Yes=1, No=0.
    
    Args:
        df: DataFrame with string target
        
    Returns:
        DataFrame with binary target
    """
    logger.info("🏷️  Encoding target variable (Churn)...")
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    logger.info(f"    Churn=1 (Yes): {df[TARGET_COL].sum()}")
    logger.info(f"    Churn=0 (No):  {(df[TARGET_COL] == 0).sum()}")
    return df


def encode_categorical(df: pd.DataFrame, method: str = "onehot") -> pd.DataFrame:
    """
    Encode categorical variables.
    
    Args:
        df: DataFrame with categorical columns
        method: 'onehot' for One-Hot Encoding, 'label' for Label Encoding
        
    Returns:
        DataFrame with encoded categorical variables
    """
    logger.info(f"🔤 Encoding categorical variables (method: {method})...")
    
    cat_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    
    if method == "onehot":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
        logger.info(f"    Created {len(df.columns)} columns after one-hot encoding")
    elif method == "label":
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        logger.info(f"    Label encoded {len(cat_cols)} columns")
    
    return df


def scale_features(df: pd.DataFrame, fit: bool = True, scaler_path: str = "models/scaler.pkl") -> pd.DataFrame:
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df: DataFrame with numerical columns
        fit: If True, fit a new scaler. If False, load existing scaler.
        scaler_path: Path to save/load the scaler
        
    Returns:
        DataFrame with scaled numerical features
    """
    logger.info("📏 Scaling numerical features...")
    
    num_cols = [col for col in NUMERICAL_COLS if col in df.columns]
    
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        
        # Save scaler for inference
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"    💾 Scaler saved to: {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        df[num_cols] = scaler.transform(df[num_cols])
        logger.info(f"    📂 Loaded scaler from: {scaler_path}")
    
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split data into train and test sets.
    
    Args:
        df: Preprocessed DataFrame
        test_size: Fraction of data for testing (default: 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"✂️  Splitting data (test_size={test_size})...")
    
    # Separate features and target
    X = df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"    Train set: {len(X_train)} samples")
    logger.info(f"    Test set:  {len(X_test)} samples")
    logger.info(f"    Train churn rate: {y_train.mean()*100:.1f}%")
    logger.info(f"    Test churn rate:  {y_test.mean()*100:.1f}%")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(
    df: pd.DataFrame,
    encode_method: str = "onehot",
    scale: bool = True,
    fit_scaler: bool = True
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Steps:
        1. Handle missing values
        2. Encode target variable
        3. Encode categorical variables
        4. Scale numerical features
    
    Args:
        df: Raw DataFrame
        encode_method: 'onehot' or 'label'
        scale: Whether to scale numerical features
        fit_scaler: Whether to fit a new scaler or load existing
        
    Returns:
        Fully preprocessed DataFrame
    """
    logger.info("=" * 50)
    logger.info("🔄 STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 50)
    
    df = handle_missing_values(df)
    df = encode_target(df)
    df = encode_categorical(df, method=encode_method)
    
    if scale:
        df = scale_features(df, fit=fit_scaler)
    
    logger.info("=" * 50)
    logger.info(f"✅ PREPROCESSING COMPLETE — Shape: {df.shape}")
    logger.info("=" * 50)
    
    return df
