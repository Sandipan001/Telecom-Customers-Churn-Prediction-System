
import json
import pandas as pd
import streamlit as st
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
metadata_path = project_root / "models" / "model_metadata.json"
prediction_log_path = project_root / "logs" / "predictions.csv"

st.set_page_config(
    page_title="Customer Churn Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")
st.caption("Monitor prediction activity and the current model.")

if metadata_path.exists():
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
else:
    metadata = {}

if prediction_log_path.exists():
    predictions = pd.read_csv(prediction_log_path)
else:
    predictions = pd.DataFrame()

total_predictions = len(predictions)

if not predictions.empty and "predicted_churn" in predictions.columns:
    churn_count = (
        predictions["predicted_churn"].astype(str).str.lower() == "yes"
    ).sum()
    churn_rate = churn_count / total_predictions * 100
else:
    churn_count = 0
    churn_rate = 0.0

col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", total_predictions)
col2.metric("Predicted Churn", int(churn_count))
col3.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")

st.subheader("Current Model")

model_name = metadata.get("model_name", "Unknown")
threshold = metadata.get("decision_threshold", "Unknown")

st.write(f"**Model:** {model_name}")
st.write(f"**Decision threshold:** {threshold}")

if "validation_f1" in metadata:
    st.write(f"**Recorded validation F1:** {metadata['validation_f1']:.4f}")

if "validation_roc_auc" in metadata:
    st.write(
        f"**Recorded validation ROC-AUC:** "
        f"{metadata['validation_roc_auc']:.4f}"
    )

st.subheader("Prediction Activity")

if predictions.empty:
    st.info("No prediction records found yet. Make predictions through the API.")
else:
    if "timestamp" in predictions.columns:
        predictions["timestamp"] = pd.to_datetime(
            predictions["timestamp"], errors="coerce", utc=True
        )
        predictions = predictions.sort_values("timestamp")

        activity = predictions.dropna(subset=["timestamp"]).copy()
        if not activity.empty:
            activity["prediction_count"] = 1
            activity = activity.set_index("timestamp")
            st.line_chart(activity["prediction_count"].cumsum())

    
    st.subheader("Churn Probability Distribution")

    if "churn_probability" in predictions.columns:
        predictions["churn_probability"] = pd.to_numeric(
            predictions["churn_probability"], errors="coerce"
        )

        probability_data = predictions["churn_probability"].dropna()

        if not probability_data.empty:
            st.bar_chart(
                probability_data.value_counts(
                    bins=10, sort=False
                ).sort_index()
            )
        else:
            st.info("No valid churn probabilities available.")
    else:
        st.info("Churn probabilities are not available in the prediction log.")

    st.subheader("Recent Predictions")
    st.dataframe(predictions.tail(20), use_container_width=True)



if st.button("Refresh dashboard"):
    st.rerun()



st.subheader("Data Drift Monitoring")

drift_report_path = project_root / "reports" / "drift" / "drift_history.csv"
drift_alert_path = project_root / "reports" / "drift" / "drift_alerts.csv"

if drift_report_path.exists():
    drift_history = pd.read_csv(drift_report_path)

    if not drift_history.empty:
        st.write("Latest drift measurements")

        latest_timestamp = drift_history["timestamp"].max()
        latest_drift = drift_history[
            drift_history["timestamp"] == latest_timestamp
        ]

        st.dataframe(latest_drift, use_container_width=True)

        st.write("Historical PSI by feature")

        selected_feature = st.selectbox(
            "Choose a feature",
            sorted(drift_history["feature"].unique())
        )

        feature_history = drift_history[
            drift_history["feature"] == selected_feature
        ].copy()

        feature_history["timestamp"] = pd.to_datetime(
            feature_history["timestamp"], errors="coerce", utc=True
        )

        feature_history = feature_history.dropna(
            subset=["timestamp"]
        ).sort_values("timestamp")

        if not feature_history.empty:
            chart_data = feature_history.set_index("timestamp")
            st.line_chart(chart_data["psi"])

            st.caption(
                "Reference lines: PSI below 0.1 is generally treated as low, "
                "0.1–0.2 as moderate, and 0.2 or above as high."
            )
    else:
        st.info("The drift history report is empty.")
else:
    st.info(
        "No drift history yet. The monitor will create it after "
        "at least 100 prediction records are available."
    )

st.subheader("Drift Alerts")

if drift_alert_path.exists():
    drift_alerts = pd.read_csv(drift_alert_path)

    if not drift_alerts.empty:
        st.warning(f"{len(drift_alerts)} high-PSI alert records found.")
        st.dataframe(
            drift_alerts.tail(20),
            use_container_width=True
        )
    else:
        st.success("No high-PSI alerts recorded.")
else:
    st.info("No drift alerts have been recorded.")
