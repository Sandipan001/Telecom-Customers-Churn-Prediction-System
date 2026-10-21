
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
STREAM_PATH = PROJECT_ROOT / "data" / "processed" / "delayed_label_stream.csv"
HOLDOUT_PATH = PROJECT_ROOT / "data" / "processed" / "delayed_label_holdout.csv"
CONFIG_PATH = PROJECT_ROOT / "reports" / "delayed_label_simulation.json"

RANDOM_SEED = 42
HOLDOUT_SIZE = 0.2
BATCH_SIZE = 500
LABEL_DELAY_BATCHES = 2


def main():
    data = pd.read_csv(DATA_PATH)

    if "Churn" not in data.columns:
        raise ValueError("The dataset must contain a Churn column.")

    stream, holdout = train_test_split(
        data,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_SEED,
        stratify=data["Churn"],
    )

    stream = stream.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    holdout = holdout.reset_index(drop=True)

    stream["customer_record_id"] = range(1, len(stream) + 1)
    stream["arrival_batch"] = (
        (stream["customer_record_id"] - 1) // BATCH_SIZE
    ) + 1
    stream["label_available_batch"] = (
        stream["arrival_batch"] + LABEL_DELAY_BATCHES
    )

    stream["label_revealed"] = False
    stream["revealed_churn_label"] = pd.NA

    STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    stream.to_csv(STREAM_PATH, index=False)
    holdout.to_csv(HOLDOUT_PATH, index=False)

    total_batches = int(stream["arrival_batch"].max())

    config = {
        "simulation_type": "synthetic_delayed_label_simulation",
        "random_seed": RANDOM_SEED,
        "holdout_size": HOLDOUT_SIZE,
        "batch_size": BATCH_SIZE,
        "label_delay_batches": LABEL_DELAY_BATCHES,
        "stream_customers": int(len(stream)),
        "holdout_customers": int(len(holdout)),
        "total_arrival_batches": total_batches,
        "note": (
            "Customer arrival and label delays are simulated. "
            "The fixed holdout is excluded from the training stream. "
            "The source dataset has no real event timestamps."
        ),
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"Training stream customers: {len(stream)}")
    print(f"Fixed holdout customers: {len(holdout)}")
    print(f"Arrival batches: {total_batches}")
    print(f"Label delay: {LABEL_DELAY_BATCHES} batches")
    print(f"Training stream saved to: {STREAM_PATH}")
    print(f"Fixed holdout saved to: {HOLDOUT_PATH}")
    print(f"Simulation config saved to: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
