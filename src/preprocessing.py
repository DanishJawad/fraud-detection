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


def train_val_test_split_time(df, val_size=0.2, test_size=0.2, time_col='Time'):
    """Split chronologically into train / validation / test.

    Validation exists so the decision threshold can be chosen without ever
    looking at the test set. Selecting a threshold on the same data the final
    metrics are reported on biases those metrics upward, because the threshold
    is fit to that split's specific errors.

    Splits on time rather than at random: fraud patterns shift, so training on
    later transactions to predict earlier ones would leak future information
    and overstate performance.

    Returns (train, val, test) with train earliest and test latest.
    """
    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size + test_size must be < 1.0, got {val_size + test_size}"
        )

    val_cutoff = df[time_col].quantile(1 - val_size - test_size)
    test_cutoff = df[time_col].quantile(1 - test_size)

    train = df[df[time_col] <= val_cutoff]
    val = df[(df[time_col] > val_cutoff) & (df[time_col] <= test_cutoff)]
    test = df[df[time_col] > test_cutoff]
    return train, val, test


def split_xy(df, target='Class'):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
