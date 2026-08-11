"""FastAPI app for model serving."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from monitoring.tracker import PerformanceMonitor
from .models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    MetricsResponse,
    FeedbackRequest,
)
from .config import Config
from .registry import ModelRegistry


app = FastAPI(title=Config.API_TITLE, version=Config.API_VERSION)
registry = None
model_data = None
monitor = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global registry, model_data, monitor
    try:
        registry = ModelRegistry(Config.MODELS_DIR)
        model_data = registry.get_model()
        monitor = PerformanceMonitor()
    except Exception as e:
        # Allow app to start but mark health as unhealthy
        model_data = None
        print(f"Warning: model failed to load on startup: {e}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if model_data is None:
        return HealthResponse(status="unhealthy", model_version="unknown")
    return HealthResponse(status="healthy", model_version=registry.current_version)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single transaction prediction."""
    try:
        if model_data is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if len(request.features) == 0:
            raise HTTPException(status_code=400, detail="Empty features")

        model = model_data["model"]
        metadata = model_data.get("metadata", {})

        expected_len = None
        feature_list = metadata.get("features")
        if isinstance(feature_list, list) and feature_list:
            expected_len = len(feature_list)
        elif hasattr(model, "n_features_in_"):
            expected_len = int(model.n_features_in_)

        if expected_len is not None and len(request.features) != expected_len:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_len} features, got {len(request.features)}",
            )

        X = np.array(request.features).reshape(1, -1)

        scaler = model_data.get("scaler")
        if scaler is not None and hasattr(model, "named_steps") and "scaler" in model.named_steps:
            scaler = None
        if scaler is not None:
            X = scaler.transform(X)

        prob = float(model.predict_proba(X)[0][1])

        threshold = metadata.get("optimal_threshold", Config.OPTIMAL_THRESHOLD)
        predicted_class = 1 if prob > threshold else 0

        if prob > 0.7:
            risk = "high"
        elif prob > 0.35:
            risk = "medium"
        else:
            risk = "low"

        prediction_id = ""
        try:
            if monitor is not None:
                prediction_id = monitor.log_prediction(prob, actual_label=None)
        except Exception:
            pass

        return PredictionResponse(
            prediction_id=prediction_id,
            fraud_probability=prob,
            predicted_class=predicted_class,
            risk_level=risk,
            threshold_used=threshold,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict-batch")
async def predict_batch(requests: list[PredictionRequest]):
    """Batch predictions endpoint (returns list of PredictionResponse dicts)."""
    results = []
    for req in requests:
        resp = await predict(req)
        results.append(resp)
    return results


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Return model metrics from metadata."""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    metadata = model_data.get("metadata", {})
    return MetricsResponse(
        roc_auc=float(metadata.get("roc_auc", 0.0)),
        fraud_detection_rate=float(metadata.get("fraud_detection_rate", 0.0)),
        false_alarm_rate=float(metadata.get("false_alarm_rate", 0.0)),
        cost_savings_percent=float(metadata.get("cost_savings_percent", 0.0)),
    )


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Attach a confirmed ground-truth label to a previous /predict call.

    Simulates how fraud labels actually arrive in production: a chargeback
    or manual review confirms whether a transaction was fraud some time
    after the prediction was made. This is what makes /monitoring reflect
    real traffic instead of staying empty forever.
    """
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not available")
    updated = monitor.update_label(request.prediction_id, request.actual_label)
    if not updated:
        raise HTTPException(status_code=404, detail="Unknown prediction_id")
    return {"status": "recorded"}


@app.get("/monitoring")
async def monitoring_summary():
    """Live drift + business-cost metrics computed from labeled traffic so far."""
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not available")
    metadata = (model_data or {}).get("metadata", {})
    threshold = metadata.get("optimal_threshold", Config.OPTIMAL_THRESHOLD)
    return {
        "summary": monitor.get_summary(),
        "business_metrics": monitor.get_business_metrics(threshold=threshold),
    }


@app.get("/models")
async def list_models():
    return {"available_versions": registry.list_versions(), "current_version": registry.current_version}


@app.get("/")
async def root():
    return {"name": "Fraud Detection API", "version": Config.API_VERSION}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
