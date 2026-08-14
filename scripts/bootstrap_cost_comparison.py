"""Is the SMOTE pipeline's cost advantage over the baseline real, or noise?

The headline cost comparison rests on a test set containing 75 frauds. A
single point estimate ("13.6% cheaper") hides how much of that is the luck of
which 75 frauds landed in the final time window.

Resamples the test set with replacement and recomputes both models' costs on
each resample, holding the models and their validation-selected thresholds
fixed. Reports how often the SMOTE pipeline is actually cheaper and a
confidence interval on the difference.

Writes outputs/bootstrap_cost.json.

Usage:
    python scripts/bootstrap_cost_comparison.py --n-resamples 5000
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation import sweep_thresholds
from src.models import train_baseline, train_smote_pipeline
from src.preprocessing import load_data, split_xy, train_val_test_split_time
from src.utils import load_config

RANDOM_STATE = 42
BOOTSTRAP_SEED = 0


def _features(X):
    return X.drop(columns=[c for c in ["Time"] if c in X.columns])


def main(data_path: str, out_dir: str, n_resamples: int):
    config = load_config("config.yaml")
    cost_fp, cost_fn = float(config["cost_fp"]), float(config["cost_fn"])

    df = load_data(data_path)
    tr, va, te = train_val_test_split_time(df, val_size=0.2, test_size=0.2)
    X_tr, y_tr = split_xy(tr)
    X_va, y_va = split_xy(va)
    X_te, y_te = split_xy(te)
    X_tr, X_va, X_te = _features(X_tr), _features(X_va), _features(X_te)

    print("Training both models (same protocol as scripts/train.py)...")
    rf = train_smote_pipeline(X_tr, y_tr, random_state=RANDOM_STATE)
    lr = train_baseline(X_tr, y_tr, method="logistic")

    # Thresholds fixed on validation, exactly as in training. They are not
    # re-selected per resample: the question is how the fixed operating points
    # behave on different draws of test data.
    t_rf, _, _, _ = sweep_thresholds(
        rf.predict_proba(X_va)[:, 1], y_va, cost_fp, cost_fn
    )
    t_lr, _, _, _ = sweep_thresholds(
        lr.predict_proba(X_va)[:, 1], y_va, cost_fp, cost_fn
    )

    p_rf = rf.predict_proba(X_te)[:, 1]
    p_lr = lr.predict_proba(X_te)[:, 1]
    flag_rf = (p_rf >= t_rf).astype(int)
    flag_lr = (p_lr >= t_lr).astype(int)
    y = np.asarray(y_te)

    def cost(flag, labels):
        fp = int(((flag == 1) & (labels == 0)).sum())
        fn = int(((flag == 0) & (labels == 1)).sum())
        return fp * cost_fp + fn * cost_fn

    point_rf, point_lr = cost(flag_rf, y), cost(flag_lr, y)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(y)
    diffs = np.empty(n_resamples)
    rf_cheaper = 0
    tied = 0
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        c_rf, c_lr = cost(flag_rf[idx], y[idx]), cost(flag_lr[idx], y[idx])
        diffs[b] = c_lr - c_rf  # positive => SMOTE pipeline cheaper
        if c_rf < c_lr:
            rf_cheaper += 1
        elif c_rf == c_lr:
            tied += 1

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_resamples": n_resamples,
        "n_test": int(n),
        "n_fraud_test": int(y.sum()),
        "costs": {"cost_false_positive": cost_fp, "cost_false_negative": cost_fn},
        "point_estimate": {
            "smote_rf_threshold": float(t_rf),
            "baseline_threshold": float(t_lr),
            "smote_rf_cost": point_rf,
            "baseline_cost": point_lr,
            "smote_rf_cheaper_by_percent": 100.0 * (1 - point_rf / point_lr),
            "smote_rf_roc_auc": float(roc_auc_score(y, p_rf)),
            "baseline_roc_auc": float(roc_auc_score(y, p_lr)),
        },
        "bootstrap": {
            "smote_rf_cheaper_percent": 100.0 * rf_cheaper / n_resamples,
            "tied_percent": 100.0 * tied / n_resamples,
            "baseline_cheaper_percent":
                100.0 * (n_resamples - rf_cheaper - tied) / n_resamples,
            "cost_difference_median": float(np.median(diffs)),
            "cost_difference_ci95_low": float(np.percentile(diffs, 2.5)),
            "cost_difference_ci95_high": float(np.percentile(diffs, 97.5)),
            "ci95_crosses_zero": bool(
                np.percentile(diffs, 2.5) <= 0 <= np.percentile(diffs, 97.5)
            ),
        },
    }

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    with open(out / "bootstrap_cost.json", "w") as f:
        json.dump(results, f, indent=2)

    bs = results["bootstrap"]
    print(f"\npoint estimate: SMOTE+RF ${point_rf:,.0f} vs baseline ${point_lr:,.0f} "
          f"({results['point_estimate']['smote_rf_cheaper_by_percent']:.1f}% cheaper)")
    print(f"{n_resamples} resamples of the {int(n)}-row test set "
          f"({int(y.sum())} frauds):")
    print(f"  SMOTE+RF cheaper: {bs['smote_rf_cheaper_percent']:.1f}%  "
          f"tied: {bs['tied_percent']:.1f}%  "
          f"baseline cheaper: {bs['baseline_cheaper_percent']:.1f}%")
    print(f"  cost difference median ${bs['cost_difference_median']:,.0f}, "
          f"95% CI ${bs['cost_difference_ci95_low']:,.0f} to "
          f"${bs['cost_difference_ci95_high']:,.0f}")
    print(f"  CI crosses zero: {bs['ci95_crosses_zero']}")
    print(f"\nSaved to {out}/bootstrap_cost.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/creditcard.csv")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--n-resamples", type=int, default=5000)
    args = parser.parse_args()
    main(args.data_path, args.out_dir, args.n_resamples)
