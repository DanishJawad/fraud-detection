"""Simple model registry for loading model and metadata."""
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from .config import Config


class ModelRegistry:
    """Model versioning and management (lightweight)."""

    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Config.MODELS_DIR
        self.current_version = "1.0.0"
        self.models: Dict[str, Any] = {}
        self._load_current_model()

    def _load_current_model(self):
        """Load current production model from models directory."""
        try:
            model_path = self.models_dir / "fraud_model.pkl"
            metadata_path = self.models_dir / "metadata.json"

            if not model_path.exists():
                raise FileNotFoundError("Model not found in models/ directory")

            model = joblib.load(model_path)

            scaler = None
            scaler_path = self.models_dir / "scaler.pkl"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)

            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

            self.models[self.current_version] = {
                "model": model,
                "scaler": scaler,
                "metadata": metadata,
                "loaded_at": datetime.now().isoformat(),
            }

        except Exception as e:
            # Keep raising so callers know loading failed
            raise RuntimeError(f"Failed to load model: {e}")

    def get_model(self, version: Optional[str] = None):
        """Return model package for a given version (default: current)."""
        version = version or self.current_version
        if version not in self.models:
            raise ValueError(f"Model version {version} not available")
        return self.models[version]

    def list_versions(self):
        return list(self.models.keys())
