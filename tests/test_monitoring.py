import tempfile
from pathlib import Path

from monitoring.tracker import PerformanceMonitor


def _make_monitor():
    tmp = Path(tempfile.mkdtemp()) / "predictions.jsonl"
    return PerformanceMonitor(log_file=tmp)


def test_log_prediction_returns_id_and_feedback_updates_it():
    monitor = _make_monitor()

    pid = monitor.log_prediction(0.9, actual_label=None)
    assert isinstance(pid, str) and pid

    entry = monitor.predictions_log[-1]
    assert entry["actual_label"] is None

    updated = monitor.update_label(pid, actual_label=1)
    assert updated is True
    assert monitor.predictions_log[-1]["actual_label"] == 1


def test_update_label_unknown_id_returns_false():
    monitor = _make_monitor()
    assert monitor.update_label("does-not-exist", actual_label=1) is False


def test_business_metrics_only_use_labeled_predictions():
    monitor = _make_monitor()

    monitor.log_prediction(0.9)  # never labeled — must be excluded
    pid_tp = monitor.log_prediction(0.9)
    pid_fp = monitor.log_prediction(0.9)
    pid_tn = monitor.log_prediction(0.1)

    monitor.update_label(pid_tp, actual_label=1)  # correctly flagged fraud
    monitor.update_label(pid_fp, actual_label=0)  # false alarm
    monitor.update_label(pid_tn, actual_label=0)  # correctly ignored

    metrics = monitor.get_business_metrics(threshold=0.35)

    assert metrics["sample_size"] == 3  # excludes the unlabeled entry
    assert metrics["fraud_detection_rate"] == 1.0
    assert metrics["false_positive_rate"] == 0.5
