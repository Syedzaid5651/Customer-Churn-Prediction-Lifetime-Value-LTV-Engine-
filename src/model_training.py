"""
Model Training Module
Trains churn prediction models, evaluates them, and saves the best one.
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

from src.utils import setup_logger

logger = setup_logger("model_training")


def get_models() -> dict:
    """
    Return a dictionary of models to train and compare.
    
    Returns:
        Dict of model_name -> model_instance
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # Handle class imbalance
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=3,  # Handle class imbalance (~73% No vs ~27% Yes)
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
    }
    return models


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> dict:
    """
    Train all models, evaluate them, and return results.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        
    Returns:
        Dictionary with model names as keys and evaluation metrics as values
    """
    models = get_models()
    results = {}
    
    logger.info("=" * 60)
    logger.info("🏋️ TRAINING MODELS")
    logger.info("=" * 60)
    
    for name, model in models.items():
        logger.info(f"\n🔄 Training: {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "model_instance": model
        }
        
        results[name] = metrics
        
        logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall:    {metrics['recall']:.4f}  ← (Most important for churn!)")
        logger.info(f"    F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"    ROC AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"    Confusion Matrix: {metrics['confusion_matrix']}")
    
    return results


def select_best_model(results: dict, metric: str = "recall") -> tuple:
    """
    Select the best model based on a given metric.
    
    For churn prediction, we prioritize RECALL because:
    - Missing a churning customer (false negative) is worse than
    - Flagging a loyal customer (false positive)
    
    Args:
        results: Dict from train_and_evaluate()
        metric: Metric to optimize ('recall', 'f1_score', 'roc_auc')
        
    Returns:
        Tuple of (best_model_name, best_model_instance, best_score)
    """
    logger.info(f"\n🏆 Selecting best model by: {metric}")
    
    best_name = max(results, key=lambda k: results[k][metric])
    best_score = results[best_name][metric]
    best_model = results[best_name]["model_instance"]
    
    logger.info(f"    Winner: {best_name} ({metric}={best_score:.4f})")
    
    return best_name, best_model, best_score


def save_model(model, model_name: str, model_path: str = "models/churn_model.pkl") -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model instance
        model_name: Name of the model (for logging)
        model_path: Path to save the model file
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"💾 Saved '{model_name}' to: {model_path}")


def save_results_report(results: dict, report_path: str = "models/training_report.json") -> None:
    """
    Save training results as a JSON report.
    
    Args:
        results: Dict from train_and_evaluate()
        report_path: Path to save the report
    """
    # Remove model instances (not JSON serializable)
    report = {}
    for name, metrics in results.items():
        report[name] = {k: v for k, v in metrics.items() if k != "model_instance"}
    
    report["timestamp"] = datetime.now().isoformat()
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📄 Training report saved to: {report_path}")


def print_comparison_table(results: dict) -> None:
    """Print a formatted comparison table of all model results."""
    logger.info("\n" + "=" * 70)
    logger.info("📊 MODEL COMPARISON TABLE")
    logger.info("=" * 70)
    logger.info(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    logger.info("-" * 70)
    
    for name, metrics in results.items():
        logger.info(
            f"{name:<25} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1_score']:>10.4f} "
            f"{metrics['roc_auc']:>10.4f}"
        )
    logger.info("=" * 70)
