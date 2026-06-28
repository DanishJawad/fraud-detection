"""Minimal HTTP server fallback so dashboard can run without FastAPI.

Usage:
    python simple_api.py --host 127.0.0.1 --port 8000

This server provides:
  - GET /health
  - GET /metrics
  - POST /predict  (if models/fraud_model.pkl and scaler exist)

It intentionally uses only the Python stdlib so it works even when other
dependencies (like pydantic) fail to install.
"""
import argparse
import io
import json
from wsgiref.simple_server import make_server
from urllib.parse import parse_qs
import os

try:
    import joblib
    import numpy as np
except Exception:
    joblib = None
    np = None


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def load_metadata():
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_model():
    if joblib is None:
        return None, None
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        return model, scaler
    except Exception:
        return None, None


def json_response(start_response, status, payload):
    body = json.dumps(payload).encode("utf-8")
    start_response(status, [("Content-Type", "application/json"), ("Content-Length", str(len(body)) )])
    return [body]


def app(environ, start_response):
    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "GET")

    if path == "/health" and method == "GET":
        metadata = load_metadata()
        if metadata:
            return json_response(start_response, "200 OK", {"status": "healthy", "model_version": metadata.get("model_name", "unknown")})
        else:
            return json_response(start_response, "200 OK", {"status": "unhealthy", "model_version": "none"})

    if path == "/metrics" and method == "GET":
        metadata = load_metadata() or {}
            resp = {
                "roc_auc": float(metadata.get("roc_auc", 0.0)),
                "fraud_detection_rate": float(metadata.get("fraud_detection_rate", 0.0)),
                "false_alarm_rate": float(metadata.get("false_alarm_rate", 0.0)),
                "cost_savings_percent": float(metadata.get("cost_savings_percent", 0.0)),
            }
        return json_response(start_response, "200 OK", resp)

    if path == "/predict" and method == "POST":
        try:
            size = int(environ.get("CONTENT_LENGTH") or 0)
            body = environ["wsgi.input"].read(size) if size else b""
            data = json.loads(body.decode("utf-8") or "{}")
            features = data.get("features")
            if not isinstance(features, list):
                return json_response(start_response, "400 Bad Request", {"error": "features must be a list"})

            model, scaler = load_model()
            if model is None:
                return json_response(start_response, "503 Service Unavailable", {"error": "model not available"})
            if scaler is None and not (hasattr(model, "named_steps") and "scaler" in model.named_steps):
                return json_response(start_response, "503 Service Unavailable", {"error": "scaler not available"})

            if np is None:
                return json_response(start_response, "500 Internal Server Error", {"error": "numpy not available"})

            arr = np.array(features).reshape(1, -1)
            if hasattr(model, "named_steps") and "scaler" in model.named_steps:
                arr_scaled = arr
            else:
                arr_scaled = scaler.transform(arr)
            prob = float(model.predict_proba(arr_scaled)[0][1])
            threshold = load_metadata().get("optimal_threshold", 0.35) if load_metadata() else 0.35
            predicted = 1 if prob > threshold else 0
            return json_response(start_response, "200 OK", {"fraud_probability": prob, "predicted_class": predicted, "threshold_used": threshold})
        except Exception as e:
            return json_response(start_response, "500 Internal Server Error", {"error": str(e)})

    # default
    return json_response(start_response, "404 Not Found", {"error": "not found"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Starting simple API on http://{args.host}:{args.port}")
    with make_server(args.host, args.port, app) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Shutting down")


if __name__ == "__main__":
    main()
