"""Model predicts fixed samples correctly. Run src/model/train.py first."""

import joblib
import pytest

SAMPLES = [
    ([5.1, 3.5, 1.4, 0.2], 0),  # setosa
    ([6.4, 3.2, 4.5, 1.5], 1),  # versicolor
    ([6.3, 3.3, 6.0, 2.5], 2),  # virginica
]


@pytest.fixture(scope="module")
def model():
    return joblib.load("artifacts/model.joblib")


@pytest.mark.parametrize("features, expected", SAMPLES)
def test_predicts_fixed_samples(model, features, expected):
    assert model.predict([features])[0] == expected
