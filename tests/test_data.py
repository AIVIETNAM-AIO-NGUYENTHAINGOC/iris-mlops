"""Data checks: schema, no nulls, all 3 classes."""

from sklearn.datasets import load_iris

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

df = load_iris(as_frame=True).frame


def test_schema():
    assert list(df.columns) == FEATURES + ["target"]


def test_no_nulls():
    assert not df.isna().any().any()


def test_three_classes():
    assert sorted(df["target"].unique()) == [0, 1, 2]
