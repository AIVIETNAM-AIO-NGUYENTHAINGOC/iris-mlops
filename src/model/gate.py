"""Quality gate: exit 1 if accuracy is below the threshold, so CI stops before shipping."""

import json
import sys

from train import ARTIFACTS, load_config


def main() -> int:
    threshold = load_config()["gate"]["min_accuracy"]
    accuracy = json.loads((ARTIFACTS / "metrics.json").read_text())["accuracy"]
    if accuracy < threshold:
        print(f"FAIL: accuracy {accuracy:.4f} < {threshold}")
        return 1
    print(f"PASS: accuracy {accuracy:.4f} >= {threshold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
