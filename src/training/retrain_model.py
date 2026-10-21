
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cost_model import calculate_business_cost


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
VERSIONS_PATH = PROJECT_ROOT / "models" / "versions"
REPORTS_PATH = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
ROC_AUC_TOLERANCE = 0.01


def evaluate_model(model, X_eval, y_eval, threshold):
    probabilities = model.predict_proba(X_eval)[:, 1]
    business_cost = calculate_business_cost(y_eval, probabilities, threshold)

    return {
        "roc_auc": float(roc_auc_score(y_eval, probabilities)),
        "business_cost": business_cost,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve promotion after the evaluation gate passes",
    )
    args = parser.parse_args()

    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=["Churn", "customerID"], errors="ignore")
    y = data["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected values found in the Churn target.")

    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
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

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

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
        ("preprocessor", clone(preprocessor)),
        (
            "classifier",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
    ])

    champion = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    champion_threshold = float(metadata["decision_threshold"])

    candidate.fit(X_train, y_train)

    validation_probabilities = candidate.predict_proba(X_val)[:, 1]
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

    validation_results = [
        calculate_business_cost(y_val, validation_probabilities, threshold)
        for threshold in thresholds
    ]

    best_validation_result = min(
        validation_results,
        key=lambda result: result["total_cost"],
    )
    candidate_threshold = float(best_validation_result["threshold"])

    champion_holdout = evaluate_model(
        champion,
        X_holdout[metadata["features"]],
        y_holdout,
        champion_threshold,
    )

    candidate_holdout = evaluate_model(
        candidate,
        X_holdout[metadata["features"]],
        y_holdout,
        candidate_threshold,
    )

    champion_cost = champion_holdout["business_cost"]["total_cost"]
    candidate_cost = candidate_holdout["business_cost"]["total_cost"]

    candidate_cost_improved = candidate_cost < champion_cost
    candidate_auc_acceptable = (
        candidate_holdout["roc_auc"]
        >= champion_holdout["roc_auc"] - ROC_AUC_TOLERANCE
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_model": "Logistic Regression",
        "champion_threshold": champion_threshold,
        "candidate_threshold": candidate_threshold,
        "validation_business_cost": float(
            best_validation_result["total_cost"]
        ),
        "champion_holdout_business_cost": float(champion_cost),
        "candidate_holdout_business_cost": float(candidate_cost),
        "champion_holdout_roc_auc": champion_holdout["roc_auc"],
        "candidate_holdout_roc_auc": candidate_holdout["roc_auc"],
        "cost_improved": candidate_cost_improved,
        "roc_auc_guardrail_passed": candidate_auc_acceptable,
        "promotion_approved": False,
        "status": "rejected",
    }

    print("\nChampion holdout results")
    print(f"Threshold: {champion_threshold:.2f}")
    print(f"Business cost: {champion_cost:.2f}")
    print(f"ROC-AUC: {champion_holdout['roc_auc']:.4f}")

    print("\nCandidate holdout results")
    print(f"Threshold: {candidate_threshold:.2f}")
    print(f"Business cost: {candidate_cost:.2f}")
    print(f"ROC-AUC: {candidate_holdout['roc_auc']:.4f}")

    # if not (candidate_cost_improved and candidate_auc_acceptable):
    #     print("\nCandidate rejected. Champion remains unchanged.")
    # else:
    #     timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    #     version_dir = VERSIONS_PATH / timestamp
    #     version_dir.mkdir(parents=True, exist_ok=False)

    #     shutil.copy2(MODEL_PATH, version_dir / "churn_model.joblib")

    #     with open(version_dir / "model_metadata.json", "w", encoding="utf-8") as file:
    #         json.dump(metadata, file, indent=2)

    #     joblib.dump(candidate, MODEL_PATH)

    #     metadata["decision_threshold"] = candidate_threshold
    #     metadata["validation_business_cost"] = float(
    #         best_validation_result["total_cost"]
    #     )
    #     metadata["model_name"] = "Logistic Regression"
    #     metadata["promotion_reason"] = (
    #         "Lower holdout business cost while satisfying the ROC-AUC guardrail."
    #     )
    #     metadata["promoted_at"] = datetime.now(timezone.utc).isoformat()
    #     metadata["model_version"] = timestamp

    #     with open(METADATA_PATH, "w", encoding="utf-8") as file:
    #         json.dump(metadata, file, indent=2)

    #     report["promotion_approved"] = True
    #     report["status"] = "promoted"
    #     report["model_version"] = timestamp

    #     print("\nCandidate promoted.")
    #     print(f"Previous champion backed up to: {version_dir}")
    # if not (candidate_cost_improved and candidate_auc_acceptable):
    #     print("\nCandidate rejected. Champion remains unchanged.")
    # elif not args.approve:
    #     report["status"] = "pending_approval"
    #     print("\nCandidate passed the evaluation gate.")
    #     print("Promotion is pending explicit approval.")
    #     print("Review reports/promotion_report.json before approving.")
    # else:
    #     timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")



    if not (candidate_cost_improved and candidate_auc_acceptable):
        report["status"] = "rejected"
        report["promotion_approved"] = False
        print("\nCandidate rejected. Champion remains unchanged.")

    elif not args.approve:
        report["status"] = "pending_approval"
        report["promotion_approved"] = False
        print("\nCandidate passed the evaluation gate.")
        print("Promotion is pending explicit approval.")

    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_dir = VERSIONS_PATH / timestamp
        version_dir.mkdir(parents=True, exist_ok=False)

        shutil.copy2(MODEL_PATH, version_dir / "churn_model.joblib")

        with open(
            METADATA_PATH, "r", encoding="utf-8"
        ) as file:
            previous_metadata = json.load(file)

        with open(
            version_dir / "model_metadata.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(previous_metadata, file, indent=2)

        joblib.dump(candidate, MODEL_PATH)

        metadata["decision_threshold"] = candidate_threshold
        metadata["validation_business_cost"] = float(
            best_validation_result["total_cost"]
        )
        metadata["model_name"] = "Logistic Regression"
        metadata["promotion_reason"] = (
            "Lower holdout business cost while satisfying the ROC-AUC guardrail."
        )
        metadata["promoted_at"] = datetime.now(timezone.utc).isoformat()
        metadata["model_version"] = timestamp

        with open(METADATA_PATH, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        report["promotion_approved"] = True
        report["status"] = "promoted"
        report["model_version"] = timestamp

        print("\nCandidate promoted.")
        print(f"Previous champion backed up to: {version_dir}")

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_PATH / "promotion_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Promotion report saved to: {report_path}")


if __name__ == "__main__":
    main()
