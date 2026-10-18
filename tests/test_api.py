
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

valid_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_prediction():
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 200

    result = response.json()
    assert result["churn_prediction"] in ["Yes", "No"]
    assert 0 <= result["churn_probability"] <= 1
    assert result["decision_threshold"] == 0.35


def test_invalid_customer_data():
    invalid_customer = valid_customer.copy()
    invalid_customer["SeniorCitizen"] = 5

    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422


def test_extra_field_rejected():
    customer = valid_customer.copy()
    customer["customerID"] = "TEST-001"

    response = client.post("/predict", json=customer)
    assert response.status_code == 422
