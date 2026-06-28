"""Baseline experiment script
Usage: python notebooks/01_baseline.py --data-path data/creditcard.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
from src.preprocessing import load_data, train_test_split_time, split_xy
from src.models import train_baseline
from src.evaluation import sweep_thresholds, auc_score
from src.utils import save_model


def main(data_path, cost_fp, cost_fn):
    print('Loading data:', data_path)
    df = load_data(data_path)
    train, test = train_test_split_time(df)
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    # For baseline, drop Time and Class columns
    drop_cols = ['Time','Class'] if 'Time' in X_train.columns else ['Class']
    X_train_proc = X_train.drop(columns=drop_cols, errors='ignore')
    X_test_proc = X_test.drop(columns=drop_cols, errors='ignore')

    model = train_baseline(X_train_proc, y_train, method='logistic')
    proba = model.predict_proba(X_test_proc)[:,1]
    auc = auc_score(proba, y_test)
    print(f'ROC-AUC: {auc:.4f}')

    opt_t, opt_cost, thresholds, costs = sweep_thresholds(proba, y_test, cost_fp, cost_fn)
    print(f'Optimal threshold: {opt_t:.3f} with expected cost = ${opt_cost:.2f}')

    # Save model
    save_model(model, 'models/fraud_model.pkl')
    print('Saved model to models/fraud_model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/creditcard.csv')
    parser.add_argument('--cost-fp', type=float, default=50.0)
    parser.add_argument('--cost-fn', type=float, default=500.0)
    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f'Please place the dataset at {args.data_path}')
    main(args.data_path, args.cost_fp, args.cost_fn)
