import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_features(df):
    df = df.copy()
        # Ensure required columns exist
        if 'Amount' in df.columns:
            scaler = StandardScaler()
            # fit_transform expects 2D array
            df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
            df['Amount_log'] = np.log1p(df['Amount'])
        else:
            df['Amount_scaled'] = 0.0
            df['Amount_log'] = 0.0
        if 'Time' in df.columns:
            df['Hour'] = ((df['Time'] // 3600) % 24).astype(int)
        else:
            df['Hour'] = 0
        # Example synthetic feature: amount z-score within dataset
        try:
            df['Amount_z'] = (df['Amount'] - df['Amount'].mean()) / (df['Amount'].std() + 1e-9)
        except Exception:
            df['Amount_z'] = 0.0
    # Drop original Amount if desired
    return df
