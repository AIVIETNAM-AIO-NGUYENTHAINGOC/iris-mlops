"""Feature checks: the scaler in the training pipeline is correct and consistent."""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from src.model.train import load_config, split_data, train


@pytest.fixture(scope="module")
def scaler_and_train_data():
    cfg = load_config()
    X_train = split_data(cfg)[0]
    return train(cfg).named_steps["scaler"], X_train


def test_scaled_features_are_standardized(scaler_and_train_data):
    # Correct: after scaling, each training feature has mean ~0 and std ~1.
    scaler, X_train = scaler_and_train_data
    Z = scaler.transform(X_train)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-6)


def test_scaler_fitted_on_train_split_only(scaler_and_train_data):
    # Consistent: scaler statistics come from the train split, not the full dataset,
    # so no test data leaks into training.
    scaler, X_train = scaler_and_train_data
    X_all, _ = load_iris(return_X_y=True)
    assert np.allclose(scaler.mean_, X_train.mean(axis=0))
    assert not np.allclose(scaler.mean_, X_all.mean(axis=0))


def test_single_row_matches_batch(scaler_and_train_data):
    # Consistent: the API scales one row at a time, which must match the batch transform used in training.
    scaler, X_train = scaler_and_train_data
    assert np.allclose(scaler.transform(X_train[:1]), scaler.transform(X_train)[:1])
