"""'Try it' page: a plain-language demo of the churn model for non-technical users."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"

CONTRACT_LABELS = {
    "Month-to-month": "Month to month (can leave any time)",
    "One year": "One year",
    "Two year": "Two years",
}
INTERNET_LABELS = {
    "Fiber optic": "Fiber (fastest, costs more)",
    "DSL": "DSL (standard)",
    "No": "No internet at all",
}
PAYMENT_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

PRESETS = {
    "new": {
        "label": "🆕 New customer, flexible plan",
        "values": dict(
            t_tenure=2, t_contract="Month-to-month", t_internet="Fiber optic",
            t_monthly=85, t_pay="Electronic check", t_support=False,
            t_security=False, t_stream=True, t_paperless=True,
            t_senior=False, t_partner=False, t_dep=False,
        ),
    },
    "loyal": {
        "label": "🏅 Loyal long-term customer",
        "values": dict(
            t_tenure=60, t_contract="Two year", t_internet="DSL",
            t_monthly=55, t_pay="Credit card (automatic)", t_support=True,
            t_security=True, t_stream=False, t_paperless=False,
            t_senior=False, t_partner=True, t_dep=True,
        ),
    },
    "budget": {
        "label": "💸 Budget customer, phone only",
        "values": dict(
            t_tenure=24, t_contract="One year", t_internet="No",
            t_monthly=20, t_pay="Mailed check", t_support=False,
            t_security=False, t_stream=False, t_paperless=False,
            t_senior=False, t_partner=False, t_dep=False,
        ),
    },
}

OPTIONAL_DEFAULTS = dict(
    t_gender="Female", t_phone="Yes", t_lines="No",
    t_backup=False, t_device=False,
)

# Plain-English sentences for each model feature value.
CATEGORY_TEXT = {
    ("Contract", "Month-to-month"): "Month-to-month contract: easy to leave",
    ("Contract", "One year"): "One-year contract",
    ("Contract", "Two year"): "Two-year contract",
    ("InternetService", "Fiber optic"): "Fiber internet plan",
    ("InternetService", "DSL"): "DSL internet plan",
    ("InternetService", "No"): "No internet service",
    ("OnlineSecurity", "Yes"): "Has the online security add-on",
    ("OnlineSecurity", "No"): "No online security add-on",
    ("TechSupport", "Yes"): "Has the tech support add-on",
    ("TechSupport", "No"): "No tech support add-on",
    ("OnlineBackup", "Yes"): "Has the online backup add-on",
    ("OnlineBackup", "No"): "No online backup add-on",
    ("DeviceProtection", "Yes"): "Has device protection",
    ("DeviceProtection", "No"): "No device protection",
    ("StreamingTV", "Yes"): "Streams TV through the company",
    ("StreamingTV", "No"): "Does not stream TV through the company",
    ("StreamingMovies", "Yes"): "Streams movies through the company",
    ("StreamingMovies", "No"): "Does not stream movies through the company",
    ("PaperlessBilling", "Yes"): "Gets paperless bills",
    ("PaperlessBilling", "No"): "Gets paper bills",
    ("PaymentMethod", "Electronic check"): "Pays by electronic check",
    ("PaymentMethod", "Mailed check"): "Pays by mailed check",
    ("PaymentMethod", "Bank transfer (automatic)"): "Pays by automatic bank transfer",
    ("PaymentMethod", "Credit card (automatic)"): "Pays by automatic credit card",
    ("Partner", "Yes"): "Has a partner",
    ("Partner", "No"): "No partner",
    ("Dependents", "Yes"): "Has dependents",
    ("Dependents", "No"): "No dependents",
    ("PhoneService", "Yes"): "Has phone service",
    ("PhoneService", "No"): "No phone service",
    ("MultipleLines", "Yes"): "Has multiple phone lines",
    ("MultipleLines", "No"): "Single phone line",
}
# Gender is not shown as a reason. Total charges overlaps with months and monthly bill,
# so showing it separately would confuse more than it explains.
HIDDEN_FROM_REASONS = {"gender", "TotalCharges"}


# ----------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_explainer(_model):
    """SHAP explainer for the linear model, using 100 real customers as background."""
    data = pd.read_csv(DATA_PATH)
    features = load_metadata()["features"]
    background = data[features].sample(n=min(100, len(data)), random_state=42)
    preprocessor = _model.named_steps["preprocessor"]
    transformed = _to_dense(preprocessor.transform(background))
    return shap.LinearExplainer(_model.named_steps["classifier"], transformed)


def _to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


# ---------------------------------------------------------------- helpers
def apply_values(values):
    for key, value in values.items():
        st.session_state[key] = value


def set_preset(name):
    apply_values(PRESETS[name]["values"])
    apply_values(OPTIONAL_DEFAULTS)


def build_customer(state):
    """Turn the simple form answers into the exact columns the model expects."""
    internet = state["t_internet"]
    has_internet = internet != "No"

    def addon(flag):
        if not has_internet:
            return "No internet service"
        return "Yes" if flag else "No"

    phone = state["t_phone"]
    lines = state["t_lines"] if phone == "Yes" else "No phone service"
    monthly = float(state["t_monthly"])
    tenure = int(state["t_tenure"])

    return {
        "gender": state["t_gender"],
        "SeniorCitizen": int(state["t_senior"]),
        "Partner": "Yes" if state["t_partner"] else "No",
        "Dependents": "Yes" if state["t_dep"] else "No",
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        "OnlineSecurity": addon(state["t_security"]),
        "OnlineBackup": addon(state["t_backup"]),
        "DeviceProtection": addon(state["t_device"]),
        "TechSupport": addon(state["t_support"]),
        "StreamingTV": addon(state["t_stream"]),
        "StreamingMovies": addon(state["t_stream"]),
        "Contract": state["t_contract"],
        "PaperlessBilling": "Yes" if state["t_paperless"] else "No",
        "PaymentMethod": state["t_pay"],
        "MonthlyCharges": monthly,
        "TotalCharges": round(monthly * tenure, 2),
    }


def score(model, features, customer):
    frame = pd.DataFrame([customer])[features]
    return float(model.predict_proba(frame)[0, 1])


def pct(value, signed=False):
    if signed:
        return f"{value:+.0%}"
    return "<1%" if value < 0.01 else f"{value:.0%}"


def risk_level(probability, threshold):
    if probability >= threshold:
        return "high"
    if probability >= threshold / 2:
        return "medium"
    return "low"


def split_feature_name(name, feature_list):
    """'categorical__Contract_Two year' -> ('Contract', 'Two year')."""
    if name.startswith("numeric__"):
        return name.replace("numeric__", "", 1), None
    rest = name.replace("categorical__", "", 1)
    for column in sorted(feature_list, key=len, reverse=True):
        if rest.startswith(column + "_"):
            return column, rest[len(column) + 1:]
    return rest, None


def numeric_sentence(column, pushes_up, customer):
    tenure = customer["tenure"]
    monthly = customer["MonthlyCharges"]
    if column == "tenure":
        return (f"Has been a customer for only {tenure} month(s)" if pushes_up
                else f"Has stayed {tenure} months, a sign of loyalty")
    if column == "MonthlyCharges":
        return f"Monthly bill of ${monthly:,.0f}"
    if column == "SeniorCitizen":
        return ("Customer is a senior citizen" if customer["SeniorCitizen"]
                else "Customer is not a senior citizen")
    return column


def explain(model, features, customer):
    """Return (reasons_up, reasons_down) as plain-English sentences."""
    explainer = load_explainer(model)
    preprocessor = model.named_steps["preprocessor"]
    frame = pd.DataFrame([customer])[features]
    transformed = _to_dense(preprocessor.transform(frame))
    contributions = np.asarray(explainer.shap_values(transformed)).reshape(-1)
    names = preprocessor.get_feature_names_out()

    rows = []
    for index, name in enumerate(names):
        column, value = split_feature_name(name, features)
        if column in HIDDEN_FROM_REASONS:
            continue
        if value is not None and transformed[0, index] < 0.5:
            continue  # skip "this customer does NOT have X" noise
        pushes_up = contributions[index] > 0
        if value is None:
            sentence = numeric_sentence(column, pushes_up, customer)
        else:
            sentence = CATEGORY_TEXT.get((column, value), f"{column}: {value}")
        rows.append((abs(contributions[index]), pushes_up, sentence))

    rows.sort(reverse=True)
    up = [text for _, is_up, text in rows if is_up][:3]
    down = [text for _, is_up, text in rows if not is_up][:3]
    return up, down


def what_if_options(model, features, customer, base_score):
    """Try a few realistic changes and see how the score moves."""
    options = []

    def try_change(description, changes):
        changed = dict(customer, **changes)
        new_score = score(model, features, changed)
        options.append((description, new_score))

    if customer["Contract"] == "Month-to-month":
        try_change("Moves to a one-year contract", {"Contract": "One year"})
        try_change("Moves to a two-year contract", {"Contract": "Two year"})
    elif customer["Contract"] == "One year":
        try_change("Moves to a two-year contract", {"Contract": "Two year"})
    if customer["InternetService"] != "No":
        if customer["TechSupport"] != "Yes":
            try_change("Adds the tech support add-on", {"TechSupport": "Yes"})
        if customer["OnlineSecurity"] != "Yes":
            try_change("Adds the online security add-on", {"OnlineSecurity": "Yes"})
    if customer["PaymentMethod"] == "Electronic check":
        try_change("Switches to automatic credit card payment",
                   {"PaymentMethod": "Credit card (automatic)"})

    better = [(d, s) for d, s in options if s <= base_score - 0.01]
    better.sort(key=lambda item: item[1])
    return better[:3]


# ------------------------------------------------------------------- page
if not (MODEL_PATH.exists() and METADATA_PATH.exists()):
    st.error("Model files not found. Train the model first (see README).")
    st.stop()

model = load_model()
metadata = load_metadata()
features = metadata["features"]
threshold = float(metadata.get("decision_threshold", 0.5))

if "t_tenure" not in st.session_state:
    set_preset("new")

st.title("Will this customer leave? 📞")
st.markdown(
    "Phone and internet companies lose customers every month. "
    "This is called **churn**. It is much cheaper to keep a customer than to win a new one, "
    "so companies try to spot who might leave **before** they do. "
    "This tool does exactly that: describe a customer below and it gives a **risk score**."
)

with st.container(border=True):
    st.markdown("**How to use it:** 1) Pick an example or fill in the details. "
                "2) Press **Check this customer**. 3) Read the result and see what might change it.")
    st.caption("Demo built on a public sample dataset of telecom customers. "
               "These are not real people.")

st.subheader("Step 1: Describe a customer")

st.write("Quick start, pick an example:")
preset_columns = st.columns(len(PRESETS))
for column, (key, preset) in zip(preset_columns, PRESETS.items()):
    column.button(preset["label"], key=f"preset_{key}",
                  on_click=set_preset, args=(key,))

left, right = st.columns(2)
with left:
    st.slider("How many months has this person been a customer?", 0, 72, key="t_tenure",
              help="Longer relationships usually mean lower risk.")
    st.selectbox("Contract type", list(CONTRACT_LABELS), key="t_contract",
                 format_func=CONTRACT_LABELS.get,
                 help="Short or flexible contracts make it easier to leave.")
    st.selectbox("Internet plan", list(INTERNET_LABELS), key="t_internet",
                 format_func=INTERNET_LABELS.get)
    st.slider("Monthly bill ($)", 18, 120, key="t_monthly")
    st.selectbox("How do they pay?", PAYMENT_OPTIONS, key="t_pay")

with right:
    st.write("Extras this customer has:")
    has_internet = st.session_state["t_internet"] != "No"
    st.checkbox("Tech support", key="t_support", disabled=not has_internet)
    st.checkbox("Online security", key="t_security", disabled=not has_internet)
    st.checkbox("Streams TV or movies", key="t_stream", disabled=not has_internet)
    st.checkbox("Paperless billing", key="t_paperless")
    st.write("About the customer:")
    st.checkbox("Senior citizen", key="t_senior")
    st.checkbox("Has a partner", key="t_partner")
    st.checkbox("Has dependents", key="t_dep")

with st.expander("More details (optional)"):
    for key, value in OPTIONAL_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    more_a, more_b = st.columns(2)
    with more_a:
        st.selectbox("Gender", ["Female", "Male"], key="t_gender")
        st.selectbox("Phone service", ["Yes", "No"], key="t_phone")
        st.selectbox("Multiple phone lines", ["No", "Yes"], key="t_lines",
                     disabled=st.session_state["t_phone"] != "Yes")
    with more_b:
        st.checkbox("Online backup", key="t_backup", disabled=not has_internet)
        st.checkbox("Device protection", key="t_device", disabled=not has_internet)
    st.caption("Total amount paid so far is estimated as monthly bill x months.")

st.subheader("Step 2: See the result")
check = st.button("Check this customer", type="primary")

if check or st.session_state.get("t_checked"):
    st.session_state["t_checked"] = True
    customer = build_customer(st.session_state)
    probability = score(model, features, customer)
    level = risk_level(probability, threshold)

    result_col, detail_col = st.columns([1, 2])
    with result_col:
        st.metric("Risk score", pct(probability))
        st.progress(min(max(probability, 0.0), 1.0))
    with detail_col:
        if level == "high":
            st.error("🔴 **High risk.** A retention call or offer is worth considering.")
        elif level == "medium":
            st.warning("🟠 **Medium risk.** Keep an eye on this customer.")
        else:
            st.success("🟢 **Low risk.** No action needed.")
        st.caption(
            f"The company acts when the score reaches {threshold:.0%}. This line was chosen to "
            "keep the total cost of lost customers and retention offers as low as possible. "
            "The score is a ranking tool, not an exact chance of leaving."
        )

    try:
        up, down = explain(model, features, customer)
        col_up, col_down = st.columns(2)
        with col_up:
            st.markdown("**What raises the risk** ⬆️")
            for sentence in up or ["Nothing stands out"]:
                st.write(f"- {sentence}")
        with col_down:
            st.markdown("**What lowers the risk** ⬇️")
            for sentence in down or ["Nothing stands out"]:
                st.write(f"- {sentence}")
    except Exception as exc:  # explanation is a bonus; never block the score
        st.info(f"Reasons are unavailable right now ({exc}).")

    if level != "low":
        ideas = what_if_options(model, features, customer, probability)
        if ideas:
            st.markdown("**What if...** the company could change one thing?")
            what_if = pd.DataFrame({
                "If the customer...": [d for d, _ in ideas],
                "New risk score": [pct(s) for _, s in ideas],
                "Change": [pct(s - probability, signed=True) for _, s in ideas],
            })
            st.dataframe(what_if, hide_index=True)
            st.caption("These are patterns the model learned from past customers. "
                       "They show association, not a guarantee that an offer would work.")

st.divider()
st.caption("Curious how this was built, tested and monitored? "
           "Open **How it works** or **Under the hood** in the sidebar.")
