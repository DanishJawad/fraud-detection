"""Replay held-out transactions through a running API to prove monitoring is
live, not dead code.

For each row it calls POST /predict, then immediately POST /feedback with
the row's real label (standing in for a delayed confirmation like a
chargeback), and every --report-every requests prints GET /monitoring so you
can watch detection rate / false-positive rate / cost move as real traffic
is scored.

Usage:
    python scripts/simulate_traffic.py --n 200
"""
import argparse
import time

import requests

from src.preprocessing import load_data, split_xy, train_val_test_split_time


def main(api_url: str, data_path: str, n: int, report_every: int, delay: float):
    df = load_data(data_path)
    # Same temporal split as scripts/train.py, so this is genuinely the
    # held-out slice: never trained on, and never used to pick the threshold.
    _, _, test = train_val_test_split_time(df)
    X_test, y_test = split_xy(test)
    X_test = X_test.drop(columns=[c for c in ["Time"] if c in X_test.columns])

    if n:
        X_test, y_test = X_test.iloc[:n], y_test.iloc[:n]

    print(f"Replaying {len(X_test)} transactions against {api_url}")
    sent = 0
    for i in range(len(X_test)):
        features = X_test.iloc[i].tolist()
        label = int(y_test.iloc[i])

        r = requests.post(f"{api_url}/predict", json={"features": features}, timeout=5)
        r.raise_for_status()
        prediction_id = r.json()["prediction_id"]

        requests.post(
            f"{api_url}/feedback",
            json={"prediction_id": prediction_id, "actual_label": label},
            timeout=5,
        ).raise_for_status()

        sent += 1
        if delay:
            time.sleep(delay)

        if sent % report_every == 0 or sent == len(X_test):
            metrics = requests.get(f"{api_url}/monitoring", timeout=5).json()
            bm = metrics.get("business_metrics") or {}
            print(
                f"[{sent}/{len(X_test)}] detection_rate={bm.get('fraud_detection_rate', 0):.1%} "
                f"false_positive_rate={bm.get('false_positive_rate', 0):.3%} "
                f"total_cost=${bm.get('total_cost', 0)}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--data-path", default="data/creditcard.csv")
    parser.add_argument("--n", type=int, default=200, help="number of transactions to replay")
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to sleep between requests")
    args = parser.parse_args()
    main(args.api_url, args.data_path, args.n, args.report_every, args.delay)
