"""Compute test-set metrics and write artifacts/metrics.json."""

import json

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from train import ARTIFACTS, load_config, split_data


def main() -> None:
    cfg = load_config()
    _, X_test, _, y_test = split_data(cfg)
    model = joblib.load(ARTIFACTS / "model.joblib")
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
