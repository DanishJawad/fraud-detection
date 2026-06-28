"""Cost sensitivity and threshold optimization
Computes optimal threshold on test set and sweeps cost ratios.
"""
import argparse
import pandas as pd
import numpy as np
import os
from src.utils import load_model
from src.evaluation import sweep_thresholds


def main(processed_path, model_path, out_dir):
    df = pd.read_csv(processed_path)
    if 'Class' not in df.columns:
        raise ValueError('Processed data must contain Class column for evaluation')
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Load model
    model = load_model(model_path)
    proba = model.predict_proba(X)[:,1]

    os.makedirs(out_dir, exist_ok=True)

    # Single default sweep
    t, cost, thresholds, costs = sweep_thresholds(proba, y, cost_fp=50, cost_fn=500)
    print(f'Default optimal threshold: {t:.3f} cost=${cost:.2f}')

    # Sensitivity over cost ratios
    cost_fp_vals = [10,50,100]
    cost_fn_vals = [100,500,1000]
    rows = []
    for cfp in cost_fp_vals:
        for cfn in cost_fn_vals:
            t, cost, _, _ = sweep_thresholds(proba, y, cfp, cfn)
            rows.append({'cost_fp': cfp, 'cost_fn': cfn, 'opt_threshold': t, 'opt_cost': cost})
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, 'cost_sensitivity.csv')
    df_out.to_csv(out_path, index=False)
    print('Saved sensitivity results to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-path', default='data/processed.csv')
    parser.add_argument('--model-path', default='models/fraud_model.pkl')
    parser.add_argument('--out-dir', default='outputs')
    args = parser.parse_args()
    main(args.processed_path, args.model_path, args.out_dir)
