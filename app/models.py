"""Pydantic models for API payloads and responses."""
from pydantic import BaseModel
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Single transaction for prediction."""
    features: List[float]


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction_id: str
    fraud_probability: float
    predicted_class: int
    risk_level: str
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    model_version: str


class MetricsResponse(BaseModel):
    """Training-time metrics on the held-out test set.

    These are static properties of the shipped model, not live traffic. Live
    numbers come from /monitoring, computed from labeled feedback.
    """
    roc_auc: float
    recall: float
    precision: float
    false_positive_rate: float
    cost_savings_percent: float
    threshold: float
    n_test: int
    n_fraud_test: int


class FeedbackRequest(BaseModel):
    """Ground-truth label for a previous prediction (e.g. a confirmed chargeback)."""
    prediction_id: str
    actual_label: int
