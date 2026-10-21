
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STREAM_PATH = PROJECT_ROOT / "data" / "processed" / "delayed_label_stream.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "label_replay"
CONFIG_PATH = PROJECT_ROOT / "reports" / "delayed_label_simulation.json"


def main():
    data = pd.read_csv(STREAM_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    total_batches = int(config["total_arrival_batches"])
    delay = int(config["label_delay_batches"])

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    replay_summary = []

    for current_batch in range(1, total_batches + delay + 1):
        arrived = data[data["arrival_batch"] <= current_batch].copy()

        arrived["label_available"] = (
            arrived["label_available_batch"] <= current_batch
        )

        arrived["available_churn_label"] = arrived[
            "revealed_churn_label"
        ].where(arrived["label_available"], pd.NA)

        output_path = OUTPUT_PATH / f"batch_{current_batch:02d}.csv"
        arrived.to_csv(output_path, index=False)

        available_count = int(arrived["label_available"].sum())

        replay_summary.append(
            {
                "current_batch": current_batch,
                "customers_arrived": int(len(arrived)),
                "labels_available": available_count,
                "labels_pending": int(len(arrived) - available_count),
                "snapshot": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )

    summary_path = OUTPUT_PATH / "replay_summary.csv"
    pd.DataFrame(replay_summary).to_csv(summary_path, index=False)

    print(f"Replay snapshots saved to: {OUTPUT_PATH}")
    print(f"Replay summary saved to: {summary_path}")
    print(pd.DataFrame(replay_summary).to_string(index=False))


if __name__ == "__main__":
    main()
