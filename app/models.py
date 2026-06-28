"""Pydantic models for API payloads and responses."""
from pydantic import BaseModel
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Single transaction for prediction."""
    features: List[float]


class PredictionResponse(BaseModel):
    """Prediction response."""
    fraud_probability: float
    predicted_class: int
    risk_level: str
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    model_version: str


class MetricsResponse(BaseModel):
    roc_auc: float
    fraud_detection_rate: float
    false_alarm_rate: float
    cost_savings_percent: float
