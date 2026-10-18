# Self-Retraining Customer Churn Prediction System

An end-to-end machine learning project that predicts customer churn, serves predictions through a REST API, monitors incoming prediction data for distribution changes, and provides a dashboard for model and prediction monitoring.

## Features

- **Data pipeline:** Dataset inspection, validation, cleaning, and exploratory data analysis.
- **Machine learning:** Logistic Regression and Random Forest comparison, validation-based threshold selection, and model evaluation.
- **Prediction API:** FastAPI endpoints for health checks and customer churn predictions.
- **Input validation:** Pydantic validation for customer data.
- **Containerization:** Docker image for reproducible API execution.
- **Prediction logging:** Persistent CSV logging of predictions and model inputs.
- **Drift monitoring:** Population Stability Index (PSI) checks for numerical features.
- **Retraining prototype:** Candidate evaluation and guarded model promotion.
- **Dashboard:** Streamlit visualization of prediction activity, churn probabilities, and model metadata.
- **Automated tests:** API endpoint and input-validation tests using pytest.

## Technology Stack

Python, Pandas, NumPy, Scikit-learn, FastAPI, Uvicorn, Streamlit, Docker, Pytest, and HTTPX.

## Project Structure

```text
telecom-churn-system/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── churn_model.joblib
│   └── model_metadata.json
├── logs/
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── data/
│   ├── monitoring/
│   └── training/
├── tests/
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Local Setup

Create and activate a Python virtual environment, then install the dependencies listed in `requirements.txt`.

Place the source dataset at `data/raw/data.csv` and run the data preparation scripts before training.

Start the prediction API:

```bash
python -m uvicorn src.api.main:app --reload
```

Open `http://localhost:8000/docs` to explore and test the API.

Start the dashboard in a separate terminal:

```bash
python -m streamlit run src/dashboard/app.py
```

Run automated tests:

```bash
python -m pytest tests/test_api.py -v
```

## Docker

Build the API image:

```bash
docker build -t telecom-churn-api .
```

Run the API with persistent prediction logs:

```bash
docker run -d --name telecom-churn-container -p 8000:8000 -v "./logs:/app/logs" telecom-churn-api
```

## Current Limitations

- Drift monitoring requires a sufficient number of prediction records.
- Distribution drift does not necessarily mean model performance has degraded.
- The retraining workflow is a prototype; scheduled, drift-triggered retraining is not yet fully automated.
- Model promotion requires careful validation and should be evaluated on representative data.
- The dashboard and API currently run as separate processes.

## Future Improvements

- Add categorical drift monitoring and historical drift reports.
- Automate scheduled monitoring and controlled retraining.
- Track actual churn outcomes to measure model performance over time.
- Add model versioning, experiment tracking, and deployment rollback.
- Add a database-backed prediction history and dashboard filters.
- Deploy the API and dashboard to a suitable hosting environment.
