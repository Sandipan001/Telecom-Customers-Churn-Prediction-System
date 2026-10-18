
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "processed" / "cleaned_data.csv"
model_path = project_root / "models" / "churn_model.joblib"
metadata_path = project_root / "models" / "model_metadata.json"

data = pd.read_csv(data_path)
X = data.drop(columns=["Churn", "customerID"], errors="ignore")
y = data["Churn"].map({"No": 0, "Yes": 1})

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

candidate = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

champion = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

threshold = metadata["decision_threshold"]

candidate.fit(X_train, y_train)

def evaluate(model, X_eval, y_eval, decision_threshold):
    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions = probabilities >= decision_threshold

    return {
        "f1": f1_score(y_eval, predictions),
        "roc_auc": roc_auc_score(y_eval, probabilities)
    }

champion_metrics = evaluate(champion, X_val, y_val, threshold)
candidate_metrics = evaluate(candidate, X_val, y_val, threshold)

print("Champion validation results")
print(f"F1-score: {champion_metrics['f1']:.4f}")
print(f"ROC-AUC: {champion_metrics['roc_auc']:.4f}")

print("\nCandidate validation results")
print(f"F1-score: {candidate_metrics['f1']:.4f}")
print(f"ROC-AUC: {candidate_metrics['roc_auc']:.4f}")

f1_improved = candidate_metrics["f1"] > champion_metrics["f1"]
auc_acceptable = (
    candidate_metrics["roc_auc"] >= champion_metrics["roc_auc"] - 0.01
)

if f1_improved and auc_acceptable:
    backup_path = project_root / "models" / "churn_model_previous.joblib"
    joblib.dump(champion, backup_path)
    joblib.dump(candidate, model_path)

    metadata["validation_f1"] = candidate_metrics["f1"]
    metadata["validation_roc_auc"] = candidate_metrics["roc_auc"]
    metadata["model_name"] = "Logistic Regression"
    metadata["promotion_reason"] = (
        "Candidate improved validation F1 while meeting the ROC-AUC guardrail."
    )

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nCandidate promoted. Previous model backed up.")
else:
    print("\nCandidate rejected. Existing model remains unchanged.")
