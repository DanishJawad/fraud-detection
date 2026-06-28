"""A/B testing framework for model comparison."""
from dataclasses import dataclass
from typing import Tuple, List
from datetime import datetime


@dataclass
class ABTestResult:
    model: str
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def total_cost(self) -> float:
        return (self.false_positives * 50) + (self.false_negatives * 500)

    @property
    def detection_rate(self) -> float:
        total_fraud = self.true_positives + self.false_negatives
        return self.true_positives / total_fraud if total_fraud > 0 else 0

    @property
    def false_positive_rate(self) -> float:
        total_legit = self.false_positives + self.true_negatives
        return self.false_positives / total_legit if total_legit > 0 else 0


class ABTest:
    """Simple A/B test runner routing predictions 50/50 by user id hash."""

    def __init__(self, threshold_a: float = 0.35, threshold_b: float = 0.40):
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.results_a = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.results_b = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        self.test_log = []

    def route_prediction(self, fraud_prob: float, user_id: str) -> Tuple[int, str]:
        if hash(user_id) % 2 == 0:
            threshold = self.threshold_a
            model = "A"
        else:
            threshold = self.threshold_b
            model = "B"
        prediction = 1 if fraud_prob > threshold else 0
        return prediction, model

    def record_result(self, model: str, actual_label: int, prediction: int):
        results = self.results_a if model == "A" else self.results_b
        if actual_label == 1:
            if prediction == 1:
                results["tp"] += 1
            else:
                results["fn"] += 1
        else:
            if prediction == 1:
                results["fp"] += 1
            else:
                results["tn"] += 1
        self.test_log.append({"timestamp": datetime.now().isoformat(), "model": model, "actual": actual_label, "prediction": prediction})

    def get_winner(self) -> Tuple[str, float]:
        res_a = ABTestResult("A", self.results_a["tp"], self.results_a["fp"], self.results_a["fn"], self.results_a["tn"])
        res_b = ABTestResult("B", self.results_b["tp"], self.results_b["fp"], self.results_b["fn"], self.results_b["tn"])
        cost_diff = res_a.total_cost - res_b.total_cost
        winner = "B" if cost_diff > 0 else "A"
        return winner, abs(cost_diff)

    def get_summary(self) -> dict:
        winner, cost_diff = self.get_winner()
        return {
            "winner": winner,
            "threshold_a": self.threshold_a,
            "threshold_b": self.threshold_b,
            "cost_a": self.results_a["fp"] * 50 + self.results_a["fn"] * 500,
            "cost_b": self.results_b["fp"] * 50 + self.results_b["fn"] * 500,
            "cost_difference": cost_diff,
            "total_predictions": len(self.test_log),
        }
