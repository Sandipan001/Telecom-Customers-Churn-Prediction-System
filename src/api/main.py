
import json
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
import csv
from datetime import datetime, timezone

project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "models" / "churn_model.joblib"
metadata_path = project_root / "models" / "model_metadata.json"

model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

threshold = metadata["decision_threshold"]
feature_names = metadata["features"]

app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    description="Predicts customer churn using a trained machine learning model.",
    version="1.0.0"
)


class CustomerData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=1000)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0, le=10000)
    TotalCharges: float = Field(ge=0, le=1000000)


@app.get("/")
def home():
    return {
        "message": "Telecom Churn Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": metadata["model_name"]
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):
    customer_data = customer.model_dump()
    input_data = pd.DataFrame([customer_data])
    input_data = input_data[feature_names]

    probability = float(model.predict_proba(input_data)[0][1])
    prediction = probability >= threshold
        
    with open(prediction_log_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow([
                "timestamp",
                *feature_names,
                "churn_probability",
                "decision_threshold",
                "predicted_churn"
            ])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            *[customer_data[name] for name in feature_names],
            round(probability, 4),
            threshold,
            "Yes" if prediction else "No"
        ])
    return {
        "churn_prediction": "Yes" if prediction else "No",
        "churn_probability": round(probability, 4),
        "decision_threshold": threshold
    }

logs_dir = project_root / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
prediction_log_path = logs_dir / "predictions.csv"