
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.cost_model import calculate_business_cost

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

TARGET = "Churn"
RANDOM_STATE = 42

data = pd.read_csv(DATA_PATH)

X = data.drop(columns=[TARGET, "customerID"], errors="ignore")
y = data[TARGET].map({"No": 0, "Yes": 1})

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

feature_names = metadata["features"]
X_val = X_val[feature_names]

probabilities = model.predict_proba(X_val)[:, 1]
thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

results = []

for threshold in thresholds:
    result = calculate_business_cost(y_val, probabilities, threshold)
    results.append(result)

results_df = pd.DataFrame(results).sort_values("total_cost")
cost_report_path = PROJECT_ROOT / "reports" / "threshold_selection.csv"
cost_report_path.parent.mkdir(parents=True, exist_ok=True)

results_df.sort_values("threshold").to_csv(
    cost_report_path,
    index=False,
)
current_threshold = float(metadata["decision_threshold"])
current_result = calculate_business_cost(
    y_val,
    probabilities,
    current_threshold,
)

best_result = results_df.iloc[0]

selection_report = {
    "selection_method": "minimum_validation_business_cost",
    "selected_threshold": float(best_result["threshold"]),
    "validation_total_cost": float(best_result["total_cost"]),
    "validation_cost_per_customer": float(best_result["cost_per_customer"]),
    "business_assumptions": {
        "false_negative_cost": 200,
        "retention_offer_cost": 25,
        "successful_retention_benefit": 100,
        "retention_success_rate": 0.50
    },
    "validation_customers": int(len(y_val)),
    "active_model_threshold": float(metadata["decision_threshold"]),
    "active_threshold_changed": False
}

selection_report_path = PROJECT_ROOT / "reports" / "threshold_selection.json"
selection_report_path.parent.mkdir(parents=True, exist_ok=True)

with open(selection_report_path, "w", encoding="utf-8") as file:
    json.dump(selection_report, file, indent=4)

print(f"\nThreshold selection report saved to: {selection_report_path}")


print("\nCURRENT MODEL THRESHOLD")
print(f"Threshold: {current_threshold:.2f}")
print(f"Total business cost: {current_result['total_cost']:.2f}")
print(f"Cost per validation customer: {current_result['cost_per_customer']:.2f}")

print("\nLOWEST-COST THRESHOLD")
print(f"Threshold: {best_result['threshold']:.2f}")
print(f"Total business cost: {best_result['total_cost']:.2f}")
print(f"Cost per validation customer: {best_result['cost_per_customer']:.2f}")
print(f"False negatives: {int(best_result['false_negatives'])}")
print(f"False positives: {int(best_result['false_positives'])}")
print(f"True positives: {int(best_result['true_positives'])}")

print("\nTOP 10 THRESHOLDS BY BUSINESS COST")
print(
    results_df[
        [
            "threshold",
            "total_cost",
            "cost_per_customer",
            "false_negatives",
            "false_positives",
            "true_positives",
        ]
    ]
    .head(10)
    .to_string(index=False)
)
