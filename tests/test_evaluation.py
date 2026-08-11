import numpy as np

from src.evaluation import compute_cost, sweep_thresholds


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
