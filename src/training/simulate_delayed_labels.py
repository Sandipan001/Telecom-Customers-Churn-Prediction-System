
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STREAM_PATH = PROJECT_ROOT / "data" / "processed" / "delayed_label_stream.csv"
CONFIG_PATH = PROJECT_ROOT / "reports" / "delayed_label_simulation.json"


def main():
    data = pd.read_csv(STREAM_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    required = {
        "Churn",
        "arrival_batch",
        "label_available_batch",
    }
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    expected = int(config["stream_customers"])
    if len(data) != expected:
        raise ValueError(
            f"Expected {expected} training-stream rows, found {len(data)}. "
            "Run prepare_delayed_dataset.py first."
        )

    # Reset the simulated label state on every run.
    data["label_revealed"] = False
    data["revealed_churn_label"] = pd.NA

    total_arrival_batches = int(config["total_arrival_batches"])
    delay = int(config["label_delay_batches"])

    # Reveal labels only when their availability batch is reached.
    for current_batch in range(1, total_arrival_batches + delay + 1):
        eligible = data["label_available_batch"] <= current_batch

        data.loc[eligible, "label_revealed"] = True
        data.loc[eligible, "revealed_churn_label"] = data.loc[
            eligible, "Churn"
        ]

    data.to_csv(STREAM_PATH, index=False)

    print("Delayed-label simulation completed.")
    print(f"Training-stream customers: {len(data)}")
    print(f"Arrival batches: {total_arrival_batches}")
    print(f"Labels eventually revealed: {int(data['label_revealed'].sum())}")
    print(f"Stream saved to: {STREAM_PATH}")


if __name__ == "__main__":
    main()
