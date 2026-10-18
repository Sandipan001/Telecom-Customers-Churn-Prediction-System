
import json
import joblib
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "models" / "churn_model.joblib"
metadata_path = project_root / "models" / "model_metadata.json"
data_path = project_root / "data" / "processed" / "cleaned_data.csv"

model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

data = pd.read_csv(data_path)
X = data.drop(columns=["Churn", "customerID"], errors="ignore")
sample = X.head(5)

probabilities = model.predict_proba(sample)[:, 1]
threshold = metadata["decision_threshold"]
predictions = probabilities >= threshold

results = pd.DataFrame({
    "Churn Probability": probabilities.round(4),
    "Predicted Churn": predictions
})

results["Predicted Churn"] = results["Predicted Churn"].map({
    True: "Yes",
    False: "No"
})

print("Saved model loaded successfully.")
print("Model:", metadata["model_name"])
print("Decision threshold:", threshold)
print("\nSample predictions:")
print(results.to_string(index=False))
