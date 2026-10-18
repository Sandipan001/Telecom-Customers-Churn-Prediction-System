
import pandas as pd
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
training_data_path = project_root / "data" / "processed" / "cleaned_data.csv"
prediction_log_path = project_root / "logs" / "predictions.csv"

training_data = pd.read_csv(training_data_path)
prediction_data = pd.read_csv(prediction_log_path)

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

if prediction_data.empty:
    print("No prediction records available.")
    raise SystemExit(0)

print("Data drift monitoring")
print("Prediction records:", len(prediction_data))

for feature in numeric_features:
    if feature not in prediction_data.columns:
        print(f"{feature}: unavailable in prediction log")
        continue

    reference = pd.to_numeric(training_data[feature], errors="coerce").dropna()
    current = pd.to_numeric(prediction_data[feature], errors="coerce").dropna()

    if current.empty:
        print(f"{feature}: no valid monitoring data")
        continue

    edges = reference.quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).to_numpy()
    edges = sorted(set(edges))

    if len(edges) < 2:
        print(f"{feature}: cannot calculate PSI")
        continue

    reference_counts = pd.cut(
        reference, bins=edges, include_lowest=True
    ).value_counts(sort=False)

    current_counts = pd.cut(
        current, bins=edges, include_lowest=True
    ).value_counts(sort=False)

    reference_pct = (reference_counts / len(reference)).clip(lower=0.0001)
    current_pct = (current_counts / len(current)).clip(lower=0.0001)

    psi = ((current_pct - reference_pct) *
           (current_pct / reference_pct).apply(lambda x: __import__("math").log(x))).sum()

    print(f"{feature}: PSI = {psi:.4f}")

    if len(current) < 30:
        print("  Warning: very few records; drift result is unreliable.")
    elif psi < 0.1:
        print("  Interpretation: little detected distribution change.")
    elif psi < 0.2:
        print("  Interpretation: moderate distribution change; investigate.")
    else:
        print("  Interpretation: substantial distribution change; investigate.")
