"""Performance monitoring utilities."""
from datetime import datetime
from pathlib import Path
import json
import uuid
from typing import Optional, Dict, List
import numpy as np
from app.config import Config

# Try to import optional DB helper
try:
    from .db import init_db, log_prediction as db_log_prediction
except Exception:
    init_db = None
    db_log_prediction = None


class PerformanceMonitor:
    """Track prediction performance and detect drift."""

    def __init__(self, log_file: Path = None):
        self.log_file = log_file or Path("monitoring/predictions.jsonl")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.predictions_log: List[Dict] = []
        self._load_existing_log()

    def _load_existing_log(self):
        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            self.predictions_log.append(json.loads(line))
                        except Exception:
                            continue

        # Initialize DB table if DATABASE_URL configured and helper available
        try:
            if Config.DATABASE_URL and init_db:
                init_db(Config.DATABASE_URL)
        except Exception:
            pass

    def log_prediction(self, fraud_prob: float, actual_label: Optional[int] = None) -> str:
        prediction_id = uuid.uuid4().hex
        entry = {
            "id": prediction_id,
            "timestamp": datetime.now().isoformat(),
            "fraud_probability": float(fraud_prob),
            "actual_label": actual_label,
        }
        self.predictions_log.append(entry)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Also optionally log to PostgreSQL if configured
        try:
            if Config.DATABASE_URL and db_log_prediction:
                db_log_prediction(Config.DATABASE_URL, float(fraud_prob), actual_label, meta=None)
        except Exception:
            # don't let DB issues affect main flow
            pass

        return prediction_id

    def update_label(self, prediction_id: str, actual_label: int) -> bool:
        """Attach a ground-truth label to a previously logged prediction.

        Updates the in-memory log only, so business metrics reflect it for
        the lifetime of this process. It is not replayed back into the JSONL
        file on restart — persisting corrected labels durably would need the
        Postgres path in monitoring/db.py instead of a rewrite-on-every-label
        approach to a flat file.
        """
        for entry in reversed(self.predictions_log):
            if entry.get("id") == prediction_id:
                entry["actual_label"] = actual_label
                return True
        return False

    def get_drift_metrics(self, window_size: int = 1000) -> Optional[Dict]:
        if len(self.predictions_log) < 100:
            return None
        recent = self.predictions_log[-window_size:]
        probs = [p["fraud_probability"] for p in recent]
        mean_prob = float(np.mean(probs))
        std_prob = float(np.std(probs))
        drift_detected = mean_prob > 0.15
        return {
            "mean_fraud_probability": mean_prob,
            "std_fraud_probability": std_prob,
            "sample_size": len(recent),
            "drift_detected": bool(drift_detected),
            "threshold": 0.15,
        }

    def get_business_metrics(self, threshold: float, window_size: int = 1000) -> Optional[Dict]:
        """Compute detection rate / false-positive rate / cost from whichever
        logged predictions have received a label via update_label() so far.

        threshold is required on purpose. A default here would be a second
        copy of the operating threshold that could silently disagree with the
        one the model is actually served at, making these numbers describe a
        model nobody is running.
        """
        labeled = [p for p in self.predictions_log if p.get("actual_label") is not None]
        if not labeled:
            return None
        recent = labeled[-window_size:]
        tp = sum(1 for p in recent if p["fraud_probability"] > threshold and p["actual_label"] == 1)
        fp = sum(1 for p in recent if p["fraud_probability"] > threshold and p["actual_label"] == 0)
        fn = sum(1 for p in recent if p["fraud_probability"] <= threshold and p["actual_label"] == 1)
        tn = sum(1 for p in recent if p["fraud_probability"] <= threshold and p["actual_label"] == 0)
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        cost_fp = float(Config.COST_FALSE_POSITIVE)
        cost_fn = float(Config.COST_FALSE_NEGATIVE)
        return {
            "fraud_detection_rate": float(detection_rate),
            "false_positive_rate": float(fp_rate),
            "cost_false_positives": int(fp * cost_fp),
            "cost_false_negatives": int(fn * cost_fn),
            "total_cost": int((fp * cost_fp) + (fn * cost_fn)),
            "sample_size": len(recent),
        }

    def get_summary(self) -> Dict:
        if not self.predictions_log:
            return {"error": "No predictions logged yet"}
        drift = self.get_drift_metrics()
        return {"total_predictions": len(self.predictions_log), "last_prediction": self.predictions_log[-1]["timestamp"], "drift_metrics": drift}
