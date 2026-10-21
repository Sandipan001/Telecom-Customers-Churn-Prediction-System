
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
VERSIONS_PATH = PROJECT_ROOT / "models" / "versions"


def main():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Active model or metadata is missing.")

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    version_name = (
        "initial_champion_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    version_dir = VERSIONS_PATH / version_name
    version_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(MODEL_PATH, version_dir / "churn_model.joblib")
    shutil.copy2(METADATA_PATH, version_dir / "model_metadata.json")

    print(f"Initial champion saved: {version_dir}")
    print(f"Model: {metadata.get('model_name', 'Unknown')}")
    print(f"Threshold: {metadata['decision_threshold']}")
    print("Active model and metadata were not modified.")


if __name__ == "__main__":
    main()
