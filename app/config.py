"""Configuration for the fraud detection API."""
import os
from pathlib import Path


class Config:
    """Holds configuration constants and file paths.

    Reads optional environment variables for DB integration.
    """

    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
    SCALER_PATH = MODELS_DIR / "scaler.pkl"
    METADATA_PATH = MODELS_DIR / "metadata.json"

    API_TITLE = "Fraud Detection API"
    API_VERSION = "1.0.0"

    # Default threshold (will be overridden by metadata if present)
    OPTIMAL_THRESHOLD = float(os.getenv("OPTIMAL_THRESHOLD", 0.35))

    # Business costs (defaults, can be changed by CLI/env)
    COST_FALSE_POSITIVE = float(os.getenv("COST_FALSE_POSITIVE", 50))
    COST_FALSE_NEGATIVE = float(os.getenv("COST_FALSE_NEGATIVE", 500))

    # Optional PostgreSQL connection string (psycopg2 style)
    DATABASE_URL = os.getenv("DATABASE_URL", "")
