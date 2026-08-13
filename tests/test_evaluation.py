import numpy as np

from src.evaluation import (
    compute_cost,
    cost_minimizing_plateau,
    metrics_at_threshold,
    sweep_thresholds,
)


def test_compute_cost_counts_errors_by_type():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])  # 1 false positive, 1 false negative
    cost = compute_cost(y_true, y_pred, cost_fp=10, cost_fn=100)
    assert cost == 10 + 100


def test_sweep_thresholds_finds_known_optimum():
    # Two obviously-fraudulent (high proba) and two obviously-legit (low
    # proba) transactions. Any threshold in (0.3, 0.7) perfectly separates
    # them, so cost should be 0 there and strictly positive outside it.
    proba = np.array([0.05, 0.1, 0.9, 0.95])
    y_true = np.array([0, 0, 1, 1])

    threshold, cost, thresholds, costs = sweep_thresholds(
        proba, y_true, cost_fp=50, cost_fn=500, thresholds=np.linspace(0, 1, 11)
    )

    # argmin returns the first threshold achieving the minimum cost; every
    # threshold strictly between 0.1 and 0.9 perfectly separates the classes.
    assert cost == 0
    assert 0.1 < threshold < 0.9
    assert costs.min() == 0


def test_metrics_at_threshold_computes_precision_recall_and_fpr():
    # 3 frauds (indices 0-2), 5 legitimate. At threshold 0.5 the model flags
    # indices 0, 1 and 3: two true positives, one false positive, one missed
    # fraud (index 2).
    proba = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])

    m = metrics_at_threshold(proba, y_true, 0.5, cost_fp=50, cost_fn=500)

    assert m["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 4}
    assert m["recall"] == 2 / 3          # caught 2 of 3 frauds
    assert m["precision"] == 2 / 3       # 2 of 3 flags were right
    assert m["false_positive_rate"] == 1 / 5
    assert m["cost"] == 50 + 500


def test_metrics_at_threshold_handles_zero_flags_without_dividing_by_zero():
    # A threshold above every score flags nothing, so precision is undefined.
    proba = np.array([0.1, 0.2, 0.3])
    y_true = np.array([1, 0, 0])

    m = metrics_at_threshold(proba, y_true, 0.99, cost_fp=50, cost_fn=500)

    assert m["confusion"]["tp"] == 0 and m["confusion"]["fp"] == 0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_cost_minimizing_plateau_reports_the_full_tie_range():
    # Cost is minimal and equal across the three middle thresholds, so
    # reporting only argmin would imply a sharper optimum than exists.
    thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    costs = np.array([100.0, 10.0, 10.0, 10.0, 90.0])

    low, high, n_tied = cost_minimizing_plateau(thresholds, costs)

    assert (low, high, n_tied) == (0.2, 0.4, 3)
