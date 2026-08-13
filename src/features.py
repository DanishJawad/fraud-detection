import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_features(df):
    """Add engineered amount/time columns used by the exploration notebooks.

    Exploration only. The shipped model (scripts/train.py) does not use this:
    it trains on the raw V1-V28 + Amount columns and does its scaling inside
    the pipeline, where the scaler is fit on the training split alone.

    This function fits a StandardScaler on whatever frame it is handed, so
    calling it on a full dataset before splitting leaks test-set statistics
    into the training features. Call it per-split or not at all.
    """
    df = df.copy()

    if 'Amount' in df.columns:
        scaler = StandardScaler()
        df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_z'] = (
            (df['Amount'] - df['Amount'].mean()) / (df['Amount'].std() + 1e-9)
        )
    else:
        df['Amount_scaled'] = 0.0
        df['Amount_log'] = 0.0
        df['Amount_z'] = 0.0

    if 'Time' in df.columns:
        df['Hour'] = ((df['Time'] // 3600) % 24).astype(int)
    else:
        df['Hour'] = 0

    return df
