import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def compute_cost(y_true, y_pred, cost_fp, cost_fn):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * cost_fp + fn * cost_fn


def sweep_thresholds(proba, y_true, cost_fp, cost_fn, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    costs = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        costs.append(compute_cost(y_true, preds, cost_fp, cost_fn))
    costs = np.array(costs)
    idx = costs.argmin()
    return thresholds[idx], costs[idx], thresholds, costs


def cost_minimizing_plateau(thresholds, costs):
    """Range of thresholds that all achieve the minimum cost.

    sweep_thresholds returns argmin, which is the *leftmost* threshold when
    several tie. With only ~57 fraud cases in the validation split the cost
    curve is a step function and ties are wide, so reporting a single value
    implies far more precision than the data supports. Returns (low, high,
    n_tied) so the width of the tie is visible instead of hidden.
    """
    costs = np.asarray(costs)
    thresholds = np.asarray(thresholds)
    tied = thresholds[costs == costs.min()]
    return float(tied.min()), float(tied.max()), int(len(tied))


def metrics_at_threshold(proba, y_true, threshold, cost_fp, cost_fn):
    """Full metric set at one operating threshold.

    Precision is computed here rather than derived by hand from the confusion
    matrix later, so every number that gets published comes out of a tested
    function. Accuracy is deliberately absent: at 0.17% fraud, predicting
    "legitimate" every time scores 99.83% and means nothing.
    """
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "recall": float(recall),
        "precision": float(precision),
        "false_positive_rate": float(false_positive_rate),
        "cost": float(fp * cost_fp + fn * cost_fn),
    }


def auc_score(proba, y_true):
    return roc_auc_score(y_true, proba)
