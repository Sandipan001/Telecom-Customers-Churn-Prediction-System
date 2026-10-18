
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timezone


project_root = Path(__file__).resolve().parents[2]
training_data_path = project_root / "data" / "processed" / "cleaned_data.csv"
prediction_log_path = project_root / "logs" / "predictions.csv"

reports_dir = project_root / "reports" / "drift"
reports_dir.mkdir(parents=True, exist_ok=True)
report_path = reports_dir / "drift_history.csv"

training_data = pd.read_csv(training_data_path)
prediction_data = pd.read_csv(prediction_log_path)

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

minimum_records = 100
epsilon = 0.0001

drift_results = []

print("Data drift monitoring")
print("Prediction records:", len(prediction_data))

if len(prediction_data) < minimum_records:
    print(
        f"Not enough prediction records for drift assessment. "
        f"Collect at least {minimum_records} records first."
    )
    raise SystemExit(0)

for feature in numeric_features:
    reference = pd.to_numeric(
        training_data[feature], errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        prediction_data[feature], errors="coerce"
    ).dropna()

    if current.empty or reference.empty:
        print(f"{feature}: insufficient valid data")
        continue

    edges = np.unique(
        np.quantile(reference, [0, 0.2, 0.4, 0.6, 0.8, 1])
    )

    if len(edges) < 2:
        print(f"{feature}: insufficient variation for PSI")
        continue

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_bins = pd.cut(
        reference, bins=edges, include_lowest=True
    )
    current_bins = pd.cut(
        current, bins=edges, include_lowest=True
    )

    reference_pct = (
        reference_bins.value_counts(sort=False) / len(reference)
    ).to_numpy()

    current_pct = (
        current_bins.value_counts(sort=False) / len(current)
    ).to_numpy()

    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    reference_pct /= reference_pct.sum()
    current_pct /= current_pct.sum()

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    print(f"{feature}: PSI = {psi:.4f}")
    drift_results.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature": feature,
        "feature_type": "numeric" if feature in numeric_features else "categorical",
        "psi": float(psi),
        "status": (
            "low" if psi < 0.1
            else "moderate" if psi < 0.2
            else "high"
        )
    })
    
    if psi < 0.1:
        status = "low"
        print("  Interpretation: little detected distribution change.")
    elif psi < 0.2:
        status = "moderate"
        print("  Interpretation: moderate distribution change; investigate.")
    else:
        status = "high"
        print("  ALERT: substantial distribution change detected.")

        alert_path = reports_dir / "drift_alerts.csv"

        alert = pd.DataFrame([{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature": feature,
            "feature_type": (
                "numeric" if feature in numeric_features else "categorical"
            ),
            "psi": float(psi),
            "status": status
        }])

        if alert_path.exists():
            alert.to_csv(alert_path, mode="a", header=False, index=False)
        else:
            alert.to_csv(alert_path, index=False)



print("\nCategorical drift monitoring")

for feature in categorical_features:
    if feature not in prediction_data.columns:
        print(f"{feature}: unavailable in prediction log")
        continue

    reference = training_data[feature].fillna("Missing").astype(str)
    current = prediction_data[feature].fillna("Missing").astype(str)

    categories = sorted(set(reference) | set(current))

    reference_pct = (
        reference.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy()
    )

    current_pct = (
        current.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .to_numpy()
    )

    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    reference_pct /= reference_pct.sum()
    current_pct /= current_pct.sum()

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    print(f"{feature}: PSI = {psi:.4f}")
    drift_results.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature": feature,
        "feature_type": "numeric" if feature in numeric_features else "categorical",
        "psi": float(psi),
        "status": (
            "low" if psi < 0.1
            else "moderate" if psi < 0.2
            else "high"
        )
    })
    
    if psi < 0.1:
        status = "low"
        print("  Interpretation: little detected distribution change.")
    elif psi < 0.2:
        status = "moderate"
        print("  Interpretation: moderate distribution change; investigate.")
    else:
        status = "high"
        print("  ALERT: substantial distribution change detected.")

        alert_path = reports_dir / "drift_alerts.csv"

        alert = pd.DataFrame([{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature": feature,
            "feature_type": (
                "numeric" if feature in numeric_features else "categorical"
            ),
            "psi": float(psi),
            "status": status
        }])

        if alert_path.exists():
            alert.to_csv(alert_path, mode="a", header=False, index=False)
        else:
            alert.to_csv(alert_path, index=False)



if drift_results:
    report = pd.DataFrame(drift_results)

    if report_path.exists():
        report.to_csv(
            report_path,
            mode="a",
            header=False,
            index=False
        )
    else:
        report.to_csv(report_path, index=False)

    print(f"\nHistorical drift report saved to: {report_path}")
