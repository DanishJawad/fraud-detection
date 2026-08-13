import pandas as pd
import pytest

from src.preprocessing import train_val_test_split_time


def _frame(n=1000):
    """Rows ordered in time, with an id so overlap is detectable."""
    return pd.DataFrame({
        "Time": range(n),
        "row_id": range(n),
        "Class": [0] * n,
    })


def test_splits_are_chronological_and_do_not_overlap():
    train, val, test = train_val_test_split_time(_frame(), val_size=0.2, test_size=0.2)

    # The whole point of the split: the threshold is chosen on val and the
    # final metrics come from test, so a single shared row would invalidate
    # the claim that test was never used for selection.
    assert set(train["row_id"]) & set(val["row_id"]) == set()
    assert set(train["row_id"]) & set(test["row_id"]) == set()
    assert set(val["row_id"]) & set(test["row_id"]) == set()

    assert train["Time"].max() < val["Time"].min()
    assert val["Time"].max() < test["Time"].min()


def test_split_covers_every_row():
    df = _frame()
    train, val, test = train_val_test_split_time(df, val_size=0.2, test_size=0.2)
    assert len(train) + len(val) + len(test) == len(df)


def test_split_sizes_are_approximately_as_requested():
    train, val, test = train_val_test_split_time(_frame(1000), val_size=0.2, test_size=0.2)
    assert len(val) == pytest.approx(200, abs=5)
    assert len(test) == pytest.approx(200, abs=5)
    assert len(train) == pytest.approx(600, abs=5)


def test_rejects_split_leaving_no_training_data():
    with pytest.raises(ValueError):
        train_val_test_split_time(_frame(), val_size=0.6, test_size=0.5)
