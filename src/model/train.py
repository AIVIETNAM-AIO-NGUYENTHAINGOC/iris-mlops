"""Train logistic regression on Iris and save artifacts/model.joblib."""

from pathlib import Path

import joblib
import yaml
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ARTIFACTS = Path("artifacts")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_data(cfg: dict):
    """Fixed split (random_state + stratify) so train and evaluate share the same test set."""
    X, y = load_iris(return_X_y=True)
    return train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )


def main() -> None:
    cfg = load_config()
    X_train, _, y_train, _ = split_data(cfg)
    model = LogisticRegression(**cfg["model"], random_state=cfg["data"]["random_state"])
    model.fit(X_train, y_train)
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "model.joblib")
    print(f"Saved {ARTIFACTS / 'model.joblib'}")


if __name__ == "__main__":
    main()
