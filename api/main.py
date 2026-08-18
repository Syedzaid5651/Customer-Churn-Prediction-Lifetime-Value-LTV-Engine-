"""
FastAPI Application — Customer Churn Prediction & LTV Engine API

Endpoints:
    GET  /health          → Health check
    POST /predict/churn   → Single customer churn prediction
    POST /predict/batch   → Batch churn predictions
    POST /predict/ltv     → Customer LTV prediction

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    
Docs available at:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    CustomerInput, ChurnPredictionResponse,
    LTVPredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthResponse
)
from src.feature_engineering import create_features
from src.preprocessing import encode_categorical, handle_missing_values

# ─── Global model storage ────────────────────────────────────────────────────
churn_model = None
ltv_model = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, cleanup on shutdown."""
    global churn_model, ltv_model, scaler
    
    model_dir = "models"
    
    # Load churn model
    churn_path = os.path.join(model_dir, "churn_model.pkl")
    if os.path.exists(churn_path):
        churn_model = joblib.load(churn_path)
        print(f"✅ Churn model loaded from {churn_path}")
    else:
        print(f"⚠️  Churn model not found at {churn_path}")
    
    # Load LTV model
    ltv_path = os.path.join(model_dir, "ltv_model.pkl")
    if os.path.exists(ltv_path):
        ltv_model = joblib.load(ltv_path)
        print(f"✅ LTV model loaded from {ltv_path}")
    else:
        print(f"⚠️  LTV model not found at {ltv_path}")
    
    # Load scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded from {scaler_path}")
    
    yield  # App is running
    
    # Cleanup (if needed)
    print("🔒 Shutting down API...")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn & LTV Prediction API",
    description=(
        "A production-level predictive analytics API for telecom customer "
        "churn prediction and lifetime value estimation."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helper Functions ────────────────────────────────────────────────────────

def customer_to_dataframe(customer: CustomerInput) -> pd.DataFrame:
    """Convert a CustomerInput Pydantic model to a preprocessed DataFrame."""
    data = customer.model_dump()
    df = pd.DataFrame([data])
    return df


def get_risk_level(probability: float) -> str:
    """Classify churn probability into risk levels."""
    if probability >= 0.7:
        return "🔴 High Risk"
    elif probability >= 0.4:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API and models are healthy."""
    return HealthResponse(
        status="healthy",
        model_loaded=churn_model is not None,
        version="1.0.0"
    )


@app.post("/predict/churn", response_model=ChurnPredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerInput):
    """
    Predict churn probability for a single customer.
    
    Returns the binary prediction (0/1), probability, and risk level.
    """
    if churn_model is None:
        raise HTTPException(
            status_code=503,
            detail="Churn model not loaded. Please train the model first."
        )
    
    try:
        df = customer_to_dataframe(customer)
        
        # Apply feature engineering
        df = create_features(df)
        
        # Encode and preprocess (must match training pipeline)
        df = encode_categorical(df, method="onehot")
        
        # Align columns with training data
        # (ensure same columns exist, fill missing with 0)
        model_features = churn_model.feature_names_in_ if hasattr(churn_model, 'feature_names_in_') else None
        if model_features is not None:
            for col in model_features:
                if col not in df.columns:
                    df[col] = 0
            df = df[model_features]
        
        # Predict
        prediction = int(churn_model.predict(df)[0])
        probability = float(churn_model.predict_proba(df)[0][1])
        
        return ChurnPredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=get_risk_level(probability)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers at once.
    
    Accepts a list of customers and returns predictions for all.
    """
    if churn_model is None:
        raise HTTPException(
            status_code=503,
            detail="Churn model not loaded. Please train the model first."
        )
    
    predictions = []
    for customer in request.customers:
        result = await predict_churn(customer)
        predictions.append(result)
    
    high_risk = sum(1 for p in predictions if p.churn_prediction == 1)
    avg_prob = sum(p.churn_probability for p in predictions) / len(predictions)
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_customers=len(predictions),
        high_risk_count=high_risk,
        avg_churn_probability=round(avg_prob, 4)
    )


@app.post("/predict/ltv", response_model=LTVPredictionResponse, tags=["Predictions"])
async def predict_ltv(customer: CustomerInput):
    """
    Predict the Customer Lifetime Value (LTV) for a single customer.
    """
    if ltv_model is None:
        # Fallback: calculate simple LTV without ML model
        remaining = max(36 - customer.tenure, 0)
        predicted_ltv = customer.MonthlyCharges * remaining
        
        if predicted_ltv >= 3000:
            segment = "High Value"
        elif predicted_ltv >= 1000:
            segment = "Medium Value"
        else:
            segment = "Low Value"
        
        return LTVPredictionResponse(
            predicted_ltv=round(predicted_ltv, 2),
            ltv_segment=segment
        )
    
    try:
        df = customer_to_dataframe(customer)
        features = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]]
        predicted_ltv = float(ltv_model.predict(features)[0])
        
        if predicted_ltv >= 3000:
            segment = "High Value"
        elif predicted_ltv >= 1000:
            segment = "Medium Value"
        else:
            segment = "Low Value"
        
        return LTVPredictionResponse(
            predicted_ltv=round(predicted_ltv, 2),
            ltv_segment=segment
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LTV prediction failed: {str(e)}")


# ─── Run directly ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
