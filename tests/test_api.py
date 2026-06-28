"""Basic API tests for the FastAPI app."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app import main as main_module


client = TestClient(app)


def _expected_feature_len() -> int:
    if main_module.model_data:
        metadata = main_module.model_data.get("metadata", {})
        features = metadata.get("features")
        if isinstance(features, list) and features:
            return len(features)
        model = main_module.model_data.get("model")
        if model is not None and hasattr(model, "n_features_in_"):
            return int(model.n_features_in_)
    return 30


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data


def test_predict_valid():
    # use dummy input sized to the model's expected features
    n_features = _expected_feature_len()
    payload = {"features": [0.0] * n_features}
    r = client.post("/predict", json=payload)
    # if model not loaded, service returns 503; otherwise 200
    assert r.status_code in (200, 503)


def test_predict_invalid():
    n_features = _expected_feature_len()
    payload = {"features": [0.0] * max(1, n_features - 1)}
    r = client.post("/predict", json=payload)
    assert r.status_code in (400, 503)


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code in (200, 503)
