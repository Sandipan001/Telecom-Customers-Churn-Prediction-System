
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
SELECTION_PATH = PROJECT_ROOT / "reports" / "threshold_selection.json"
HISTORY_PATH = PROJECT_ROOT / "reports" / "threshold_history.json"


def main():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    with open(SELECTION_PATH, "r", encoding="utf-8") as file:
        selection = json.load(file)

    current_threshold = float(metadata["decision_threshold"])
    proposed_threshold = float(selection["selected_threshold"])

    print(f"Current threshold: {current_threshold:.2f}")
    print(f"Proposed threshold: {proposed_threshold:.2f}")
    print(f"Validation business cost: {selection['validation_total_cost']:.2f}")

    if proposed_threshold == current_threshold:
        print("No change required. The active threshold is already selected.")
        return

    history = []

    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            history = json.load(file)

    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "previous_threshold": current_threshold,
            "proposed_threshold": proposed_threshold,
            "validation_business_cost": selection["validation_total_cost"],
            "status": "pending_approval"
        }
    )

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    print("Promotion has NOT been applied.")
    print(f"Review record saved to: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
