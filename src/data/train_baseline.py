
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "processed" / "cleaned_data.csv"

df = pd.read_csv(data_path)

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full,
)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

# model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier

candidates = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

for name, classifier in candidates.items():
    candidate = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    candidate.fit(X_train, y_train)
    val_proba = candidate.predict_proba(X_val)[:, 1]

    threshold_results = []

    for threshold in [
        0.20, 0.25, 0.30, 0.35, 0.40,
        0.45, 0.50, 0.55, 0.60
    ]:
        val_pred = (val_proba >= threshold).astype(int)

        precision = precision_score(
            y_val, val_pred, zero_division=0
        )
        recall = recall_score(
            y_val, val_pred, zero_division=0
        )
        f1 = f1_score(
            y_val, val_pred, zero_division=0
        )

        threshold_results.append(
            (threshold, precision, recall, f1)
        )

    best = max(threshold_results, key=lambda result: result[3])

    results.append({
        "name": name,
        "model": candidate,
        "threshold": best[0],
        "precision": best[1],
        "recall": best[2],
        "f1": best[3],
        "roc_auc": roc_auc_score(y_val, val_proba),
    })

print("Validation comparison:")

for result in results:
    print(f"\nModel: {result['name']}")
    print(f"Selected threshold: {result['threshold']:.2f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1-score: {result['f1']:.4f}")
    print(f"ROC-AUC: {result['roc_auc']:.4f}")

champion = max(results, key=lambda result: result["f1"])

print("\nSelected champion:", champion["name"])
print("Validation F1-score:", round(champion["f1"], 4))
print("Selected threshold:", champion["threshold"])


import json

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

champion_model = champion["model"]
champion_threshold = champion["threshold"]

test_proba = champion_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= champion_threshold).astype(int)

print("\nFinal champion test evaluation")
print("Model:", champion["name"])
print("Threshold:", champion_threshold)
print("Accuracy:", round(accuracy_score(y_test, test_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, test_proba), 4))
print("Precision:", round(precision_score(y_test, test_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, test_pred, zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, test_pred, zero_division=0), 4))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=["Stayed", "Churned"],
        zero_division=0,
    )
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_pred, labels=[0, 1]))

models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(champion_model, models_dir / "churn_model.joblib")

metadata = {
    "model_name": champion["name"],
    "decision_threshold": float(champion_threshold),
    "validation_f1": float(champion["f1"]),
    "validation_roc_auc": float(champion["roc_auc"]),
    "features": X.columns.tolist(),
    "target_mapping": {"No": 0, "Yes": 1},
}

with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\nSaved model:", models_dir / "churn_model.joblib")
print("Saved metadata:", models_dir / "model_metadata.json")
