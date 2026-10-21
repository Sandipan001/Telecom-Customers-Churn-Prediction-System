
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HOLDOUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "delayed_label_holdout.csv"
)
CANDIDATE_PATH = PROJECT_ROOT / "models" / "delayed_label_candidate.joblib"
CHAMPION_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "delayed_candidate_evaluation.json"


def evaluate_model(model, X, y, threshold):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "f1_score": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def main():
    for path in [
        HOLDOUT_PATH,
        CANDIDATE_PATH,
        CHAMPION_PATH,
        METADATA_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    holdout = pd.read_csv(HOLDOUT_PATH)

    if "Churn" not in holdout.columns:
        raise ValueError("The fixed holdout must contain the Churn column.")

    y = holdout["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Holdout contains missing or invalid Churn labels.")

    # Exclude the target and identifier from model inputs.
    X = holdout.drop(columns=["Churn", "customerID"], errors="ignore")

    candidate = joblib.load(CANDIDATE_PATH)
    champion = joblib.load(CHAMPION_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    candidate_results = evaluate_model(candidate, X, y, threshold=0.5)
    champion_threshold = float(metadata.get("decision_threshold", 0.5))
    champion_results = evaluate_model(
        champion, X, y, threshold=champion_threshold
    )

    report = {
        "evaluation_dataset": "data/processed/delayed_label_holdout.csv",
        "holdout_customers": int(len(holdout)),
        "holdout_used_for_training": False,
        "candidate_training_batch": 5,
        "candidate_training_labels": 1500,
        "candidate": candidate_results,
        "champion": champion_results,
        "comparison": {
            "candidate_f1_minus_champion_f1": float(
                candidate_results["f1_score"]
                - champion_results["f1_score"]
            ),
            "candidate_auc_minus_champion_auc": float(
                candidate_results["roc_auc"]
                - champion_results["roc_auc"]
            ),
        },
        "promotion_performed": False,
        "note": (
            "This is a fixed random holdout, not a future-time holdout. "
            "The delayed arrival and label timing are simulated."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Delayed-label candidate evaluation completed.")
    print(f"Fixed holdout customers: {len(holdout)}")
    print("Holdout used for training: False")
    print("\nChampion:")
    print(json.dumps(champion_results, indent=2))
    print("\nDelayed-label candidate:")
    print(json.dumps(candidate_results, indent=2))
    print(f"\nPromotion performed: {report['promotion_performed']}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
