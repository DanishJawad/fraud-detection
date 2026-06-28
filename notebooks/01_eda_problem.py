"""EDA script: notebooks/01_eda_problem.py
Run: PYTHONPATH=. python notebooks/01_eda_problem.py --data-path data/creditcard.csv
"""
import argparse
import pandas as pd


def main(data_path):
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")
    class_counts = df['Class'].value_counts()
    print(f"Class distribution:\n{class_counts.to_dict()}")
    fraud_pct = df['Class'].sum() / len(df) * 100
    print(f"Fraud %: {fraud_pct:.6f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/creditcard.csv')
    args = parser.parse_args()
    main(args.data_path)
