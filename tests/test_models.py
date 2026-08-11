import numpy as np
import pandas as pd

from src.models import train_smote_pipeline


def _make_imbalanced_data(n=300, n_features=4, fraud_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series((rng.random(n) < fraud_rate).astype(int))
    # make the minority class separable so the classifier has something real to learn
    X.loc[y == 1, "f0"] += 4.0
    return X, y


def test_train_smote_pipeline_does_not_touch_held_out_data():
    X_train, y_train = _make_imbalanced_data(seed=1)
    X_holdout, y_holdout = _make_imbalanced_data(seed=2)

    original_train_len = len(X_train)
    original_holdout_len = len(X_holdout)

    pipeline = train_smote_pipeline(X_train, y_train, random_state=42)

    # Fitting must never mutate the caller's frames — SMOTE only resamples
    # an internal copy during .fit(), and holdout is never passed to fit at all.
    assert len(X_train) == original_train_len
    assert len(X_holdout) == original_holdout_len

    proba = pipeline.predict_proba(X_holdout)[:, 1]
    assert len(proba) == len(y_holdout)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_train_smote_pipeline_is_reproducible():
    X_train, y_train = _make_imbalanced_data(seed=1)
    X_test, _ = _make_imbalanced_data(seed=2)

    pipeline_a = train_smote_pipeline(X_train, y_train, random_state=42)
    pipeline_b = train_smote_pipeline(X_train, y_train, random_state=42)

    proba_a = pipeline_a.predict_proba(X_test)[:, 1]
    proba_b = pipeline_b.predict_proba(X_test)[:, 1]
    np.testing.assert_array_equal(proba_a, proba_b)
