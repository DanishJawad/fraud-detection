"""Canonical, reproducible training pipeline.

This is the single source of truth for models/fraud_model.pkl and
models/metadata.json. It supersedes the exploratory notebooks/*.py scripts,
which are kept for their history (they're what led here) but are no longer
what the shipped model or README numbers are traced back to.

Usage:
    python scripts/train.py --data-path data/creditcard.csv
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.evaluation import sweep_thresholds
from src.models import train_baseline, train_smote_pipeline
from src.preprocessing import load_data, split_xy, train_test_split_time
from src.utils import load_config, save_model

RANDOM_STATE = 42


def _drop_non_feature_cols(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=[c for c in ["Time"] if c in X.columns])


def _metrics_at_threshold(proba, y_true, threshold, cost_fp, cost_fn):
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    cost = fp * cost_fp + fn * cost_fn
    return {
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "fraud_detection_rate": float(detection_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "cost": float(cost),
    }


def main(data_path: str, out_dir: str):
    config = load_config("config.yaml")
    cost_fp, cost_fn = float(config["cost_fp"]), float(config["cost_fn"])

    print(f"Loading data: {data_path}")
    df = load_data(data_path)

    # Temporal split: train on earlier transactions, test on later ones, so
    # evaluation mirrors how the model would actually be used (scoring
    # transactions that happen after the ones it was trained on).
    train, test = train_test_split_time(df)
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)
    X_train = _drop_non_feature_cols(X_train)
    X_test = _drop_non_feature_cols(X_test)
    feature_names = list(X_train.columns)

    print(f"Train: {len(X_train)} rows ({y_train.sum()} fraud) | "
          f"Test: {len(X_test)} rows ({y_test.sum()} fraud)")

    # Baseline for comparison: plain logistic regression, no SMOTE. Trained
    # and scored the same way, so the cost-savings number below has a floor
    # to compare against instead of being a bare number.
    print("Training baseline (no SMOTE)...")
    baseline_model = train_baseline(X_train, y_train, method="logistic")
    baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
    baseline_t, baseline_cost, _, _ = sweep_thresholds(baseline_proba, y_test, cost_fp, cost_fn)

    # Shipped model: StandardScaler -> SMOTE -> RandomForest, SMOTE fit on
    # the training split only (see src/models.train_smote_pipeline).
    print("Training SMOTE pipeline...")
    model = train_smote_pipeline(X_train, y_train, random_state=RANDOM_STATE)
    proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, proba)
    threshold, cost, thresholds, costs = sweep_thresholds(proba, y_test, cost_fp, cost_fn)
    at_threshold = _metrics_at_threshold(proba, y_test, threshold, cost_fp, cost_fn)

    cost_savings_percent = 100.0 * (1 - cost / baseline_cost) if baseline_cost > 0 else 0.0

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Optimal threshold: {threshold:.3f} (cost=${cost:.2f})")
    print(f"Baseline (no SMOTE) threshold: {baseline_t:.3f} (cost=${baseline_cost:.2f})")
    print(f"Cost savings vs. baseline: {cost_savings_percent:.1f}%")
    print(f"Fraud detection rate: {at_threshold['fraud_detection_rate']:.1%}")
    print(f"False alarm rate: {at_threshold['false_alarm_rate']:.4%}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    save_model(model, models_dir / "fraud_model.pkl")

    stale_scaler = models_dir / "scaler.pkl"
    if stale_scaler.exists():
        stale_scaler.unlink()

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "model": "StandardScaler -> SMOTE -> RandomForestClassifier (imblearn Pipeline)",
        "features": feature_names,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "optimal_threshold": float(threshold),
        "optimal_cost": float(cost),
        "roc_auc": float(roc_auc),
        "fraud_detection_rate": at_threshold["fraud_detection_rate"],
        "false_alarm_rate": at_threshold["false_alarm_rate"],
        "confusion": at_threshold["confusion"],
        "cost_false_positive": cost_fp,
        "cost_false_negative": cost_fn,
        "baseline_no_smote": {
            "threshold": float(baseline_t),
            "cost": float(baseline_cost),
        },
        "cost_savings_percent": float(cost_savings_percent),
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model + metadata to {models_dir}/")

    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    cost_fp_vals = [10, 50, 100]
    cost_fn_vals = [100, 500, 1000]
    rows = []
    for cfp in cost_fp_vals:
        for cfn in cost_fn_vals:
            t, c, _, _ = sweep_thresholds(proba, y_test, cfp, cfn)
            rows.append({"cost_fp": cfp, "cost_fn": cfn, "opt_threshold": t, "opt_cost": c})
    pd.DataFrame(rows).to_csv(out_path / "cost_sensitivity.csv", index=False)
    print(f"Saved cost sensitivity sweep to {out_path}/cost_sensitivity.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/creditcard.csv")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()
    main(args.data_path, args.out_dir)
