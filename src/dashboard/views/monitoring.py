import joblib
import numpy as np
import shap
import json
import pandas as pd
import streamlit as st
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root = Path(__file__).resolve().parents[3]
metadata_path = project_root / "models" / "model_metadata.json"
prediction_log_path = project_root / "logs" / "predictions.csv"

st.title("Under the hood: model monitoring")
st.caption("For engineers and interviewers: prediction activity, drift checks, cost curve and model explanations.")

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



cost_report_path = PROJECT_ROOT / "reports" / "cost_thresholds.csv"

st.subheader("Business Cost vs Decision Threshold")

if cost_report_path.exists():
    cost_df = pd.read_csv(cost_report_path)

    if not cost_df.empty:
        cost_df = cost_df.sort_values("threshold")
        best_row = cost_df.loc[cost_df["total_cost"].idxmin()]

        st.metric(
            "Lowest-Cost Threshold",
            f"{best_row['threshold']:.2f}",
            f"Modeled cost: {best_row['total_cost']:,.0f}",
        )

        chart_df = cost_df.set_index("threshold")[["total_cost"]]
        st.line_chart(chart_df)

        st.caption(
            "Costs are estimated using the configured business assumptions "
            "and validation data. The selected threshold is not automatically "
            "applied to the active API model."
        )

        st.dataframe(
            cost_df[
                [
                    "threshold",
                    "total_cost",
                    "cost_per_customer",
                    "false_negatives",
                    "false_positives",
                    "true_positives",
                    "offers_sent",
                ]
            ].sort_values("total_cost").head(10),
            use_container_width=True,
        )
    else:
        st.info("The cost-threshold report is empty. Run the evaluation script first.")
else:
    st.info("Cost-threshold report not found. Run the evaluation script first.")



# ---------------------------------------------------------
# Customer-level SHAP Explainability
# ---------------------------------------------------------

st.divider()
st.header("Customer-Level Churn Explanation")

model_path = PROJECT_ROOT / "models" / "churn_model.joblib"
customer_data_path = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
)

if not model_path.exists() or not customer_data_path.exists():
    st.info(
        "The model or cleaned dataset is missing. "
        "Train the model and prepare the dataset first."
    )
else:
    @st.cache_resource
    def load_explanation_model(path):
        return joblib.load(path)

    @st.cache_data
    def load_customer_data(path):
        return pd.read_csv(path)

    model = load_explanation_model(str(model_path))
    customer_data = load_customer_data(str(customer_data_path))

    feature_columns = [
        column
        for column in customer_data.columns
        if column not in {"Churn", "customerID"}
    ]

    X_customers = customer_data[feature_columns].copy()

    selected_row = st.number_input(
        "Select customer row",
        min_value=0,
        max_value=len(X_customers) - 1,
        value=0,
        step=1,
        help="Select a row from the cleaned dataset for demonstration.",
    )

    customer = X_customers.iloc[[int(selected_row)]]

    probability = float(model.predict_proba(customer)[0, 1])
    active_threshold = float(
        metadata.get("decision_threshold", 0.5)
    )
    predicted_churn = probability >= active_threshold

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric(
        "Churn Probability",
        f"{probability:.1%}",
    )
    metric_col2.metric(
        "Active Threshold",
        f"{active_threshold:.1%}",
    )
    metric_col3.metric(
        "Prediction",
        "Likely to churn" if predicted_churn else "Unlikely to churn",
    )

    st.caption(
        "This prediction uses the active champion model. "
        "The selected row is a demonstration customer from the cleaned dataset."
    )

    if st.button("Explain selected customer"):
        try:
            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            background = X_customers.sample(
                n=min(100, len(X_customers)),
                random_state=42,
            )

            transformed_background = preprocessor.transform(background)
            transformed_customer = preprocessor.transform(customer)

            feature_names = preprocessor.get_feature_names_out()

            explainer = shap.LinearExplainer(
                classifier,
                transformed_background,
            )

            shap_values = np.asarray(
                explainer.shap_values(transformed_customer)
            ).reshape(-1)

            explanation_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP value (log-odds)": shap_values,
            })

            explanation_df["Direction"] = np.where(
                explanation_df["SHAP value (log-odds)"] >= 0,
                "Pushes toward churn",
                "Pushes away from churn",
            )

            explanation_df["Absolute impact"] = (
                explanation_df["SHAP value (log-odds)"].abs()
            )

            explanation_df = explanation_df.sort_values(
                "Absolute impact",
                ascending=False,
            ).head(10)

            st.subheader("Top 10 Feature Contributions")

            chart_df = explanation_df.set_index("Feature")[
                ["SHAP value (log-odds)"]
            ]

            st.bar_chart(chart_df)

            st.dataframe(
                explanation_df[
                    ["Feature", "SHAP value (log-odds)", "Direction"]
                ],
                use_container_width=True,
            )

            st.caption(
                "Positive SHAP values push the model's log-odds toward "
                "churn; negative values push away from churn. These are "
                "model contributions, not proof of causation."
            )

        except Exception as exc:
            st.error(f"Could not generate the SHAP explanation: {exc}")
