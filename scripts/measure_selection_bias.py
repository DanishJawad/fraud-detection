"""Measure how much threshold-on-test selection inflates the reported score.

This repo previously swept the decision threshold against y_test and then
reported metrics at the winning threshold on that same test set. That is
optimistically biased: the threshold is fit to the test split's particular
errors. The README claims a specific size for that bias, so this script
regenerates both sides of the comparison instead of quoting a run whose
artifact no longer exists.

Runs three configurations on the same data:

  old_biased    80/20 split, threshold swept on test, scored on test
                (the original, wrong protocol)
  old_honest    the same 80/20 model, but scored at a threshold taken from
                the last 20% of its own training data, isolating how much of
                the difference is selection bias rather than training size
  current       60/20/20 split, threshold selected on val, scored on test
                (what scripts/train.py does now)

Writes outputs/selection_bias.json.

Usage:
    python scripts/measure_selection_bias.py
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import roc_auc_score

from src.evaluation import metrics_at_threshold, sweep_thresholds
from src.models import train_smote_pipeline
from src.preprocessing import load_data, split_xy, train_val_test_split_time
from src.utils import load_config

RANDOM_STATE = 42


def _features(X):
    return X.drop(columns=[c for c in ["Time"] if c in X.columns])


def _old_two_way_split(df, test_size=0.2, time_col="Time"):
    """The original, wrong split: 80/20, no validation carve-out.

    Kept only here, not in src/preprocessing.py, because this script is the
    one place that deliberately reproduces the old protocol for comparison.
    Nothing that trains or serves the current model should call this.
    """
    cutoff = df[time_col].quantile(1 - test_size)
    train = df[df[time_col] <= cutoff]
    test = df[df[time_col] > cutoff]
    return train, test


def main(data_path: str, out_dir: str):
    config = load_config("config.yaml")
    cost_fp, cost_fn = float(config["cost_fp"]), float(config["cost_fn"])

    df = load_data(data_path)
    results = {}

    # --- The original protocol: 80/20, threshold chosen on the test set -----
    train80, test20 = _old_two_way_split(df, test_size=0.2)
    X80, y80 = split_xy(train80)
    Xt20, yt20 = split_xy(test20)
    X80, Xt20 = _features(X80), _features(Xt20)

    print(f"old 80/20 split: train {len(X80)} rows ({int(y80.sum())} fraud), "
          f"test {len(Xt20)} rows ({int(yt20.sum())} fraud)")
    model80 = train_smote_pipeline(X80, y80, random_state=RANDOM_STATE)
    proba20 = model80.predict_proba(Xt20)[:, 1]

    biased_t, _, _, _ = sweep_thresholds(proba20, yt20, cost_fp, cost_fn)
    biased = metrics_at_threshold(proba20, yt20, biased_t, cost_fp, cost_fn)
    results["old_biased"] = {
        "protocol": "80/20 temporal split; threshold swept on TEST and "
                    "reported on TEST (biased)",
        "n_train": int(len(X80)),
        "n_fraud_train": int(y80.sum()),
        "n_test": int(len(Xt20)),
        "n_fraud_test": int(yt20.sum()),
        "threshold": float(biased_t),
        "roc_auc": float(roc_auc_score(yt20, proba20)),
        "recall": biased["recall"],
        "precision": biased["precision"],
        "false_positive_rate": biased["false_positive_rate"],
        "confusion": biased["confusion"],
        "cost": biased["cost"],
    }

    # --- Same model, threshold taken from held-out data instead ------------
    # The last 20% of the 80% training window, so the model is unchanged and
    # only the threshold's provenance differs.
    cutoff = X80.index[int(len(X80) * 0.75)]
    holdout_mask = X80.index >= cutoff
    proba_holdout = model80.predict_proba(X80[holdout_mask])[:, 1]
    honest_t, _, _, _ = sweep_thresholds(
        proba_holdout, y80[holdout_mask], cost_fp, cost_fn
    )
    honest = metrics_at_threshold(proba20, yt20, honest_t, cost_fp, cost_fn)
    results["old_honest"] = {
        "protocol": "same 80/20 model; threshold selected on a slice of its "
                    "own training window, then scored on TEST",
        "threshold": float(honest_t),
        "roc_auc": results["old_biased"]["roc_auc"],
        "recall": honest["recall"],
        "precision": honest["precision"],
        "false_positive_rate": honest["false_positive_rate"],
        "confusion": honest["confusion"],
        "cost": honest["cost"],
    }

    # --- Current protocol: 60/20/20, threshold on val ----------------------
    tr, va, te = train_val_test_split_time(df, val_size=0.2, test_size=0.2)
    Xtr, ytr = split_xy(tr)
    Xva, yva = split_xy(va)
    Xte, yte = split_xy(te)
    Xtr, Xva, Xte = _features(Xtr), _features(Xva), _features(Xte)

    print(f"current 60/20/20 split: train {len(Xtr)} rows "
          f"({int(ytr.sum())} fraud)")
    model60 = train_smote_pipeline(Xtr, ytr, random_state=RANDOM_STATE)
    val_t, _, _, _ = sweep_thresholds(
        model60.predict_proba(Xva)[:, 1], yva, cost_fp, cost_fn
    )
    proba_te = model60.predict_proba(Xte)[:, 1]
    cur = metrics_at_threshold(proba_te, yte, val_t, cost_fp, cost_fn)
    results["current"] = {
        "protocol": "60/20/20 temporal split; threshold selected on VAL, "
                    "scored once on TEST",
        "n_train": int(len(Xtr)),
        "n_fraud_train": int(ytr.sum()),
        "n_test": int(len(Xte)),
        "n_fraud_test": int(yte.sum()),
        "threshold": float(val_t),
        "roc_auc": float(roc_auc_score(yte, proba_te)),
        "recall": cur["recall"],
        "precision": cur["precision"],
        "false_positive_rate": cur["false_positive_rate"],
        "confusion": cur["confusion"],
        "cost": cur["cost"],
    }

    results["deltas"] = {
        "recall_old_biased_minus_current":
            results["old_biased"]["recall"] - results["current"]["recall"],
        "roc_auc_old_biased_minus_current":
            results["old_biased"]["roc_auc"] - results["current"]["roc_auc"],
        # Same model, same test set, only the threshold's source differs.
        # This is selection bias with training-set size held constant.
        "recall_selection_bias_only":
            results["old_biased"]["recall"] - results["old_honest"]["recall"],
    }
    results["generated_at"] = datetime.now(timezone.utc).isoformat()
    results["random_state"] = RANDOM_STATE

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    with open(out / "selection_bias.json", "w") as f:
        json.dump(results, f, indent=2)

    for key in ["old_biased", "old_honest", "current"]:
        r = results[key]
        print(f"\n{key}: threshold {r['threshold']:.2f}")
        print(f"  ROC-AUC {r['roc_auc']:.4f}  recall {r['recall']:.4%}  "
              f"precision {r['precision']:.4%}  cost ${r['cost']:,.0f}")
    d = results["deltas"]
    print(f"\nrecall inflation, old-biased vs current: "
          f"{d['recall_old_biased_minus_current']:+.4%}")
    print(f"recall inflation from threshold selection alone (same model): "
          f"{d['recall_selection_bias_only']:+.4%}")
    print(f"\nSaved to {out}/selection_bias.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/creditcard.csv")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()
    main(args.data_path, args.out_dir)
