import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    return df


def train_test_split_time(df, test_size=0.2, time_col='Time'):
    # simple temporal split based on Time percentile to simulate streaming
    cutoff = df[time_col].quantile(1 - test_size)
    train = df[df[time_col] <= cutoff]
    test = df[df[time_col] > cutoff]
    return train, test


def split_xy(df, target='Class'):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
