
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSIONS_PATH = PROJECT_ROOT / "models" / "versions"
SNAPSHOTS_PATH = PROJECT_ROOT / "models" / "rollback_snapshots"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def main():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        print("Active model or metadata is missing. Nothing was changed.")
        return

    if not VERSIONS_PATH.exists():
        print("No model versions directory found.")
        return

    versions = sorted(
        [
            path
            for path in VERSIONS_PATH.iterdir()
            if path.is_dir()
            and (path / "churn_model.joblib").exists()
            and (path / "model_metadata.json").exists()
        ],
        reverse=True,
    )

    if not versions:
        print("No restorable model versions found.")
        return

    print("\nAvailable model versions:")
    for index, version in enumerate(versions, start=1):
        print(f"{index}. {version.name}")

    choice = input("\nEnter the number of the version to restore: ").strip()

    if not choice.isdigit() or not 1 <= int(choice) <= len(versions):
        print("Invalid selection. Nothing was changed.")
        return

    selected_version = versions[int(choice) - 1]

    confirmation = input(
        f"Restore version {selected_version.name}? Type YES to confirm: "
    ).strip()

    if confirmation != "YES":
        print("Rollback cancelled. Nothing was changed.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_dir = SNAPSHOTS_PATH / timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(MODEL_PATH, snapshot_dir / "churn_model.joblib")
    shutil.copy2(METADATA_PATH, snapshot_dir / "model_metadata.json")

    try:
        with open(
            selected_version / "model_metadata.json",
            "r",
            encoding="utf-8",
        ) as file:
            restored_metadata = json.load(file)

        if not 0 < float(restored_metadata["decision_threshold"]) < 1:
            raise ValueError("Restored threshold is invalid.")

        shutil.copy2(
            selected_version / "churn_model.joblib",
            MODEL_PATH,
        )
        shutil.copy2(
            selected_version / "model_metadata.json",
            METADATA_PATH,
        )
    except Exception:
        shutil.copy2(snapshot_dir / "churn_model.joblib", MODEL_PATH)
        shutil.copy2(snapshot_dir / "model_metadata.json", METADATA_PATH)
        raise

    print(f"Restored version: {selected_version.name}")
    print(f"Previous active state backed up to: {snapshot_dir}")
    print("Restart the API process or container to load the restored model.")


if __name__ == "__main__":
    main()
