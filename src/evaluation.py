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


def auc_score(proba, y_true):
    return roc_auc_score(y_true, proba)
