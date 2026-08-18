"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class CustomerInput(BaseModel):
    """Schema for a single customer's features for prediction."""
    
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=844.20)


class ChurnPredictionResponse(BaseModel):
    """Response schema for churn prediction."""
    
    churn_prediction: int = Field(..., description="0=No Churn, 1=Churn")
    churn_probability: float = Field(..., description="Probability of churn (0 to 1)")
    risk_level: str = Field(..., description="Low / Medium / High risk category")


class LTVPredictionResponse(BaseModel):
    """Response schema for LTV prediction."""
    
    predicted_ltv: float = Field(..., description="Predicted Customer Lifetime Value in $")
    ltv_segment: str = Field(..., description="Low Value / Medium Value / High Value")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    
    customers: List[CustomerInput]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[ChurnPredictionResponse]
    total_customers: int
    high_risk_count: int
    avg_churn_probability: float


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    
    status: str = "healthy"
    model_loaded: bool = True
    version: str = "1.0.0"
