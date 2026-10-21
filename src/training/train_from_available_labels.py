
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STREAM_PATH = (
    PROJECT_ROOT / "data" / "processed" / "delayed_label_stream.csv"
)
CONFIG_PATH = PROJECT_ROOT / "reports" / "delayed_label_simulation.json"
MODEL_PATH = PROJECT_ROOT / "models" / "delayed_label_candidate.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "delayed_label_training.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        help="Current batch number for the delayed-label simulation.",
    )
    args = parser.parse_args()

    if not STREAM_PATH.exists() or not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Delayed-label stream or configuration is missing. "
            "Prepare and simulate the delayed-label dataset first."
        )

    data = pd.read_csv(STREAM_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    if args.batch < 1:
        raise ValueError("Batch must be at least 1.")

    if args.batch > int(config["total_arrival_batches"]) + int(
        config["label_delay_batches"]
    ):
        raise ValueError("Batch exceeds the configured simulation period.")

    # Only customers who have arrived AND whose labels are available.
    available = data[
        (data["arrival_batch"] <= args.batch)
        & (data["label_available_batch"] <= args.batch)
        & (data["label_revealed"].astype(str).str.lower() == "true")
    ].copy()

    if available.empty:
        raise ValueError(
            f"No labels are available at batch {args.batch}. "
            "Choose a later batch."
        )

    # Use the revealed label, not the original Churn column.
    y = available["revealed_churn_label"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError(
            "Some available customers have missing or invalid revealed labels."
        )

    excluded_columns = [
        "Churn",
        "customerID",
        "customer_record_id",
        "arrival_batch",
        "label_available_batch",
        "label_revealed",
        "revealed_churn_label",
    ]

    X = available.drop(columns=excluded_columns, errors="ignore")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in X.columns
        if column not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    candidate = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    candidate.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(candidate, MODEL_PATH)

    report = {
        "simulation_type": config.get("simulation_type"),
        "training_batch": args.batch,
        "customers_arrived_by_batch": int(
            (data["arrival_batch"] <= args.batch).sum()
        ),
        "customers_with_available_labels": int(len(available)),
        "training_positive_labels": int(y.sum()),
        "training_negative_labels": int((y == 0).sum()),
        "feature_count": int(X.shape[1]),
        "holdout_path": "data/processed/delayed_label_holdout.csv",
        "holdout_used_for_training": False,
        "label_source": "revealed_churn_label",
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Delayed-label candidate trained successfully.")
    print(f"Batch: {args.batch}")
    print(f"Customers arrived: {report['customers_arrived_by_batch']}")
    print(f"Labels available for training: {len(available)}")
    print(f"Features: {X.shape[1]}")
    print(f"Holdout used for training: {report['holdout_used_for_training']}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
