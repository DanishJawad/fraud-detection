"""Automated retraining pipeline and promotion logic."""
from pathlib import Path
from datetime import datetime
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


class RetrainingPipeline:
    """Retraining pipeline that promotes a model if it improves ROC-AUC.

    Scaffolding: nothing currently calls this. Before wiring it to a scheduler
    it needs its own validation split for threshold selection. Right now it
    carries the existing threshold forward unchanged, which is only valid if
    the new model's probability distribution matches the old one's — not
    something to assume after retraining on new data.
    """

    def __init__(self, models_dir: Path, data_dir: Path, drift_threshold: float = 0.15):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.drift_threshold = drift_threshold
        self.last_retrain = datetime.now()
        self.current_metadata = None
        self._load_current_metadata()

    def _load_current_metadata(self):
        metadata_path = self.models_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.current_metadata = json.load(f)

    def should_retrain(self, current_metrics: dict = None) -> bool:
        days_since_retrain = (datetime.now() - self.last_retrain).days
        if days_since_retrain > 30:
            return True
        if current_metrics and current_metrics.get("fraud_detection_rate", 0) < 0.85:
            return True
        return False

    def retrain(self, new_data_path: Path, test_data_path: Path) -> bool:
        try:
            df_train = pd.read_csv(new_data_path)
            df_test = pd.read_csv(test_data_path)

            X_train = df_train.drop("Class", axis=1)
            y_train = df_train["Class"]
            X_test = df_test.drop("Class", axis=1)
            y_test = df_test["Class"]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

            new_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            new_model.fit(X_train_bal, y_train_bal)

            y_pred_proba = new_model.predict_proba(X_test_scaled)[:, 1]
            new_roc_auc = roc_auc_score(y_test, y_pred_proba)

            if self.current_metadata:
                # Test-set ROC-AUC lives under "test" in the metadata written
                # by scripts/train.py. Reading a flat "roc_auc" here would
                # silently default to 0 and promote every candidate model.
                current_roc = (self.current_metadata.get("test") or {}).get("roc_auc", 0)
                if new_roc_auc > current_roc:
                    self._promote_model(new_model, scaler, new_roc_auc)
                    return True
                return False
            else:
                self._promote_model(new_model, scaler, new_roc_auc)
                return True

        except Exception as e:
            print(f"Retrain failed: {e}")
            return False

    def _promote_model(self, model, scaler, roc_auc: float):
        joblib.dump(model, self.models_dir / "fraud_model.pkl")
        joblib.dump(scaler, self.models_dir / "scaler.pkl")

        # Must match the shape scripts/train.py writes, or app/main.py cannot
        # find a threshold and refuses to serve. Carried forward rather than
        # hardcoded — a literal here is how the repo previously ended up with
        # several conflicting thresholds.
        previous = self.current_metadata or {}
        threshold = dict(previous.get("threshold") or {})
        threshold["carried_forward_from"] = previous.get("trained_at")
        threshold["selected_on"] = "inherited (not re-selected for this model)"

        metadata = {
            **previous,
            "promoted_at": datetime.now().isoformat(),
            "model": "StandardScaler + SMOTE -> RandomForestClassifier (retrained)",
            "threshold": threshold,
            "test": {**(previous.get("test") or {}), "roc_auc": roc_auc},
        }
        with open(self.models_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        self.last_retrain = datetime.now()
