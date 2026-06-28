"""Feature engineering script
Saves processed data to data/processed.csv
"""
import os
import argparse
import pandas as pd
from src.features import create_features


def main(data_path, out_path):
    df = pd.read_csv(data_path)
    print('Raw rows:', len(df))
    df_proc = create_features(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_proc.to_csv(out_path, index=False)
    print('Saved processed data to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='data/creditcard.csv')
    parser.add_argument('--out-path', default='data/processed.csv')
    args = parser.parse_args()
    main(args.data_path, args.out_path)
