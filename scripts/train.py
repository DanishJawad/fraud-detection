"""Canonical, reproducible training pipeline.

This is the single source of truth for models/fraud_model.pkl, the metrics in
outputs/metrics.json, and every number quoted in the README. It supersedes the
exploratory notebooks/*.py scripts, which are kept for their history (they're
what led here) but are no longer what the shipped model traces back to.

Split protocol, which is the point of this script:

    train (oldest 60%) -> fit the model
    val   (next 20%)   -> choose the decision threshold
    test  (newest 20%) -> report final metrics, read exactly once

The threshold is chosen on validation and never on test. An earlier version of
this script swept thresholds against y_test and then reported metrics at that
same threshold on that same data, which biases the reported recall upward
because the threshold is fit to the test split's specific errors.

Usage:
    python scripts/train.py --data-path data/creditcard.csv
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation import (
    cost_minimizing_plateau,
    metrics_at_threshold,
    sweep_thresholds,
)
from src.models import train_baseline, train_smote_pipeline
from src.preprocessing import load_data, split_xy, train_val_test_split_time
from src.utils import load_config, save_model

RANDOM_STATE = 42
VAL_SIZE = 0.2
TEST_SIZE = 0.2

# Reference point only, never used to select anything. Logged so the README's
# claim that 0.5 is the wrong operating point for imbalanced data is backed by
# a number in the repo instead of an assertion.
NAIVE_THRESHOLD = 0.5


def _drop_non_feature_cols(X: pd.DataFrame) -> pd.DataFrame:
    """Drop Time: it's the split axis, so keeping it lets the model key on
    'how late in the dataset is this', which won't generalize to new traffic.
    """
    return X.drop(columns=[c for c in ["Time"] if c in X.columns])


def main(data_path: str, out_dir: str):
    config = load_config("config.yaml")
    cost_fp, cost_fn = float(config["cost_fp"]), float(config["cost_fn"])

    report = []

    def log(line: str = ""):
        """Print and retain, so the run record lands in a tracked file."""
        print(line)
        report.append(line)

    log(f"Loading data: {data_path}")
    df = load_data(data_path)

    train, val, test = train_val_test_split_time(
        df, val_size=VAL_SIZE, test_size=TEST_SIZE
    )
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)
    X_train = _drop_non_feature_cols(X_train)
    X_val = _drop_non_feature_cols(X_val)
    X_test = _drop_non_feature_cols(X_test)
    feature_names = list(X_train.columns)

    n_fraud_total = int(df["Class"].sum())
    log()
    log("Class balance (full dataset)")
    log(f"  rows: {len(df)}  fraud: {n_fraud_total}  "
        f"legitimate: {len(df) - n_fraud_total}")
    log(f"  fraud rate: {100.0 * n_fraud_total / len(df):.6f}%  "
        f"(1 fraud per {len(df) / n_fraud_total:.1f} transactions)")
    log()
    log("Temporal split (train fits the model, val picks the threshold, "
        "test is read once)")
    for name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        log(f"  {name:5s}: {len(X_split):>7} rows  {int(y_split.sum()):>4} fraud  "
            f"({100.0 * y_split.mean():.4f}%)")

    # Baseline for comparison: plain logistic regression, no SMOTE. Gets the
    # identical protocol (threshold picked on val, scored on test) so the
    # cost-savings number compares like with like.
    log()
    log("Training baseline (logistic regression, no SMOTE)...")
    baseline_model = train_baseline(X_train, y_train, method="logistic")
    baseline_val_proba = baseline_model.predict_proba(X_val)[:, 1]
    baseline_threshold, _, _, _ = sweep_thresholds(
        baseline_val_proba, y_val, cost_fp, cost_fn
    )
    baseline_test_proba = baseline_model.predict_proba(X_test)[:, 1]
    baseline_test = metrics_at_threshold(
        baseline_test_proba, y_test, baseline_threshold, cost_fp, cost_fn
    )
    baseline_test_roc_auc = float(roc_auc_score(y_test, baseline_test_proba))

    # Shipped model: StandardScaler -> SMOTE -> RandomForest, SMOTE fit on the
    # training split only (see src/models.train_smote_pipeline).
    log("Training SMOTE pipeline...")
    model = train_smote_pipeline(X_train, y_train, random_state=RANDOM_STATE)

    val_proba = model.predict_proba(X_val)[:, 1]
    threshold, val_cost, val_ths, val_costs = sweep_thresholds(
        val_proba, y_val, cost_fp, cost_fn
    )
    threshold = float(threshold)
    plateau_low, plateau_high, plateau_n = cost_minimizing_plateau(val_ths, val_costs)
    val_metrics = metrics_at_threshold(val_proba, y_val, threshold, cost_fp, cost_fn)
    val_roc_auc = float(roc_auc_score(y_val, val_proba))

    # Test set touched here for the first time, at a threshold already fixed.
    test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at_threshold(test_proba, y_test, threshold, cost_fp, cost_fn)
    test_roc_auc = float(roc_auc_score(y_test, test_proba))
    naive_test = metrics_at_threshold(
        test_proba, y_test, NAIVE_THRESHOLD, cost_fp, cost_fn
    )

    # Point estimate on a single test set holding 75 frauds. Do not quote it
    # as a settled margin: scripts/bootstrap_cost_comparison.py resamples the
    # test set and the 95% interval on the cost difference crosses zero, so
    # the direction is supported but the size is not.
    cost_savings_percent = (
        100.0 * (1 - test_metrics["cost"] / baseline_test["cost"])
        if baseline_test["cost"] > 0
        else 0.0
    )

    log()
    log(f"Cost assumptions: ${cost_fp:.0f} per false positive, "
        f"${cost_fn:.0f} per false negative (ratio 1:{cost_fn / cost_fp:.0f})")
    log()
    log(f"Threshold selected on VALIDATION: {threshold:.2f} "
        f"(val cost ${val_cost:.2f})")
    log(f"  minimum-cost plateau: {plateau_low:.2f}-{plateau_high:.2f} "
        f"({plateau_n} thresholds tie; the sweep keeps the lowest)")
    log(f"  val   ROC-AUC {val_roc_auc:.4f}  recall {val_metrics['recall']:.1%}  "
        f"precision {val_metrics['precision']:.1%}  "
        f"FPR {val_metrics['false_positive_rate']:.4%}")
    log()
    log(f"FINAL — TEST set at threshold {threshold:.2f} (selected on val, "
        f"test read once)")
    log(f"  ROC-AUC:             {test_roc_auc:.4f}")
    log(f"  Recall:              {test_metrics['recall']:.4%}")
    log(f"  Precision:           {test_metrics['precision']:.4%}")
    log(f"  False positive rate: {test_metrics['false_positive_rate']:.6%}")
    log(f"  Confusion:           tp={test_metrics['confusion']['tp']} "
        f"fp={test_metrics['confusion']['fp']} "
        f"fn={test_metrics['confusion']['fn']} "
        f"tn={test_metrics['confusion']['tn']}")
    log(f"  Cost:                ${test_metrics['cost']:.2f}")
    log()
    log(f"Reference — TEST at naive threshold {NAIVE_THRESHOLD} "
        f"(not used, logged for comparison)")
    log(f"  recall {naive_test['recall']:.4%}  "
        f"precision {naive_test['precision']:.4%}  "
        f"FPR {naive_test['false_positive_rate']:.6%}  "
        f"cost ${naive_test['cost']:.2f}")
    log()
    log(f"Baseline — logistic regression, no SMOTE, threshold "
        f"{baseline_threshold:.2f} picked on val")
    log(f"  TEST recall {baseline_test['recall']:.4%}  "
        f"precision {baseline_test['precision']:.4%}  "
        f"FPR {baseline_test['false_positive_rate']:.6%}  "
        f"cost ${baseline_test['cost']:.2f}")
    log(f"  Cost savings, SMOTE pipeline vs. this baseline: "
        f"{cost_savings_percent:.1f}%")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    save_model(model, models_dir / "fraud_model.pkl")

    stale_scaler = models_dir / "scaler.pkl"
    if stale_scaler.exists():
        stale_scaler.unlink()

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "model": "StandardScaler -> SMOTE -> RandomForestClassifier "
                 "(imblearn Pipeline)",
        "features": feature_names,
        "split": {
            "kind": "temporal (chronological by Time)",
            "val_size": VAL_SIZE,
            "test_size": TEST_SIZE,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "n_fraud_train": int(y_train.sum()),
            "n_fraud_val": int(y_val.sum()),
            "n_fraud_test": int(y_test.sum()),
        },
        "class_balance": {
            "n_rows": int(len(df)),
            "n_fraud": n_fraud_total,
            "n_legitimate": int(len(df) - n_fraud_total),
            "fraud_rate": float(n_fraud_total / len(df)),
        },
        "costs": {
            "cost_false_positive": cost_fp,
            "cost_false_negative": cost_fn,
        },
        "threshold": {
            "value": threshold,
            "selected_on": "validation",
            "method": "exhaustive sweep over np.linspace(0, 1, 101), "
                      "minimizing fp*cost_fp + fn*cost_fn",
            "val_cost": float(val_cost),
            # Several thresholds tie for minimum cost. Recorded so the
            # reported value isn't mistaken for a sharp optimum.
            "min_cost_plateau_low": plateau_low,
            "min_cost_plateau_high": plateau_high,
            "min_cost_plateau_n_tied": plateau_n,
        },
        # The headline numbers. Test set scored once, at a threshold fixed on val.
        "test": {
            "roc_auc": test_roc_auc,
            "recall": test_metrics["recall"],
            "precision": test_metrics["precision"],
            "false_positive_rate": test_metrics["false_positive_rate"],
            "confusion": test_metrics["confusion"],
            "cost": test_metrics["cost"],
        },
        "validation": {
            "roc_auc": val_roc_auc,
            "recall": val_metrics["recall"],
            "precision": val_metrics["precision"],
            "false_positive_rate": val_metrics["false_positive_rate"],
            "confusion": val_metrics["confusion"],
            "cost": val_metrics["cost"],
        },
        "test_at_naive_threshold": {
            "threshold": NAIVE_THRESHOLD,
            "recall": naive_test["recall"],
            "precision": naive_test["precision"],
            "false_positive_rate": naive_test["false_positive_rate"],
            "confusion": naive_test["confusion"],
            "cost": naive_test["cost"],
        },
        "baseline_no_smote": {
            "model": "StandardScaler -> LogisticRegression",
            "threshold": float(baseline_threshold),
            "roc_auc": baseline_test_roc_auc,
            "recall": baseline_test["recall"],
            "precision": baseline_test["precision"],
            "false_positive_rate": baseline_test["false_positive_rate"],
            "confusion": baseline_test["confusion"],
            "cost": baseline_test["cost"],
        },
        "cost_savings_percent": float(cost_savings_percent),
    }

    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)

    # outputs/ is tracked in git; models/ is not. The metrics file has to live
    # somewhere a reviewer gets from a plain clone, or the README's numbers are
    # unverifiable without downloading the dataset and retraining.
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"\nSaved metrics to {out_path}/metrics.json")

    # Serving copy. app/main.py reads the threshold from here.
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Saved model + metadata to {models_dir}/")

    # How sensitive is the chosen threshold to the cost assumptions? Threshold
    # still picked on val for every pair; test cost reported at that threshold.
    rows = []
    for cfp in [10, 50, 100]:
        for cfn in [100, 500, 1000]:
            t, val_c, _, _ = sweep_thresholds(val_proba, y_val, cfp, cfn)
            test_c = metrics_at_threshold(test_proba, y_test, t, cfp, cfn)
            rows.append({
                "cost_fp": cfp,
                "cost_fn": cfn,
                "threshold_selected_on_val": float(t),
                "val_cost": float(val_c),
                "test_cost": test_c["cost"],
                "test_recall": test_c["recall"],
                "test_precision": test_c["precision"],
            })
    pd.DataFrame(rows).to_csv(out_path / "cost_sensitivity.csv", index=False)
    log(f"Saved cost sensitivity sweep to {out_path}/cost_sensitivity.csv")

    with open(out_path / "train_log.txt", "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"Saved run log to {out_path}/train_log.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/creditcard.csv")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()
    main(args.data_path, args.out_dir)
