
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.cost_model import calculate_business_cost

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
RETRAIN_PATH = PROJECT_ROOT / "src" / "training" / "retrain_model.py"
ROLLBACK_PATH = PROJECT_ROOT / "src" / "training" / "rollback_model.py"

required_paths = [
    DATA_PATH,
    MODEL_PATH,
    METADATA_PATH,
    RETRAIN_PATH,
    ROLLBACK_PATH,
]

missing_paths = [path for path in required_paths if not path.exists()]

if missing_paths:
    for path in missing_paths:
        print(f"MISSING: {path}")
    raise SystemExit("Preflight failed. Fix missing files before proceeding.")

data = pd.read_csv(DATA_PATH)

if "Churn" not in data.columns:
    raise SystemExit("Preflight failed: Churn column is missing.")

if not set(data["Churn"].dropna().unique()).issubset({"Yes", "No"}):
    raise SystemExit("Preflight failed: unexpected Churn target values.")

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

if "decision_threshold" not in metadata:
    raise SystemExit("Preflight failed: decision_threshold is missing.")

if not 0 < float(metadata["decision_threshold"]) < 1:
    raise SystemExit("Preflight failed: decision_threshold must be between 0 and 1.")

if "features" not in metadata:
    raise SystemExit("Preflight failed: feature metadata is missing.")

missing_features = set(metadata["features"]) - set(data.columns)

if missing_features:
    raise SystemExit(
        f"Preflight failed: metadata features missing from dataset: {missing_features}"
    )

X = data[metadata["features"]].head(5)
probabilities = model.predict_proba(X)[:, 1]

if len(probabilities) != 5:
    raise SystemExit("Preflight failed: expected five prediction probabilities.")

if not ((probabilities >= 0) & (probabilities <= 1)).all():
    raise SystemExit("Preflight failed: invalid prediction probabilities.")

print("Preflight passed.")
print(f"Dataset rows: {len(data)}")
print(f"Model: {metadata.get('model_name', 'Unknown')}")
print(f"Active threshold: {metadata['decision_threshold']}")
print(f"Feature count: {len(metadata['features'])}")
print("Saved model loaded and five predictions verified.")
print("No model or metadata files were modified.")
