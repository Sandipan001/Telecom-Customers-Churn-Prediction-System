
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "customer_explanation.json"


def main():
    for path in [MODEL_PATH, METADATA_PATH, DATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # Use a real customer row, but exclude its identifier and target.
    feature_columns = [
        column for column in data.columns
        if column not in {"Churn", "customerID"}
    ]

    X = data[feature_columns].copy()

    # Choose one customer for a reproducible demonstration.
    customer_index = 0
    customer = X.iloc[[customer_index]]

    probability = float(model.predict_proba(customer)[0, 1])
    threshold = float(metadata.get("decision_threshold", 0.5))
    prediction = int(probability >= threshold)

    # Explain the model's prediction in the transformed feature space.
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed_background = preprocessor.transform(X.sample(
        n=min(100, len(X)),
        random_state=42,
    ))
    transformed_customer = preprocessor.transform(customer)

    feature_names = preprocessor.get_feature_names_out()

    # Logistic regression is linear, so LinearExplainer is appropriate.
    explainer = shap.LinearExplainer(
        classifier,
        transformed_background,
    )
    shap_values = np.asarray(explainer.shap_values(transformed_customer))
    contributions = shap_values.reshape(-1)

    explanation = pd.DataFrame({
        "feature": feature_names,
        "shap_value": contributions,
    })

    explanation["direction"] = np.where(
        explanation["shap_value"] >= 0,
        "pushes toward churn",
        "pushes away from churn",
    )

    explanation["absolute_impact"] = explanation["shap_value"].abs()
    explanation = explanation.sort_values(
        "absolute_impact",
        ascending=False,
    )

    result = {
        "customer_row_index": customer_index,
        "actual_churn_label": (
            str(data.iloc[customer_index]["Churn"])
            if "Churn" in data.columns
            else None
        ),
        "predicted_churn_probability": probability,
        "decision_threshold": threshold,
        "predicted_churn": bool(prediction),
        "model_version": metadata.get("model_version", "unversioned"),
        "explanation_method": "SHAP LinearExplainer",
        "feature_space": "Preprocessed model features",
        "top_contributors": explanation.head(10)[
            ["feature", "shap_value", "direction"]
        ].to_dict(orient="records"),
        "note": (
            "SHAP values explain the model output in its transformed feature "
            "space. They indicate model contribution, not causation."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("Customer-level explanation generated.")
    print(f"Customer row: {customer_index}")
    print(f"Churn probability: {probability:.4f}")
    print(f"Decision threshold: {threshold:.2f}")
    print(f"Predicted churn: {bool(prediction)}")
    print("\nTop feature contributions:")
    print(explanation.head(10)[
        ["feature", "shap_value", "direction"]
    ].to_string(index=False))
    print(f"\nExplanation saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
