"""'How it works' page: the project explained in plain language."""

import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

metadata = {}
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

st.title("How it works 💡")
st.write("No technical background needed. Here is the idea in four steps.")

steps = st.columns(4)
cards = [
    ("1️⃣ Learn", "The system studied about 7,000 past telecom customers: who stayed, who left, "
                 "what plans they had and how long they had been customers."),
    ("2️⃣ Score", "For a new customer it gives a risk score. A higher score means the customer "
                 "looks more like people who left in the past."),
    ("3️⃣ Decide", "If the score crosses a chosen line, the company reaches out with a call or an offer."),
    ("4️⃣ Stay fresh", "Customers change over time. The system watches for changes and "
                      "tests a newer model before letting it replace the current one."),
]
for column, (title, text) in zip(steps, cards):
    with column.container(border=True):
        st.markdown(f"**{title}**")
        st.write(text)

st.subheader("Words you may see")
glossary = {
    "Churn": "When a customer stops using a service.",
    "Risk score": "A number from 0% to 100%. Higher means more likely to leave. "
                  "It ranks customers; it is not an exact chance.",
    "Action line (threshold)": "The score at which the company acts. "
                               "Set too low, many offers are wasted. Set too high, leavers are missed.",
    "Drift": "When new customers look different from the ones the model learned from, "
             "for example higher bills. It is a warning sign, not proof the model is wrong.",
    "Retraining": "Teaching the model again with newer data.",
    "Promotion gate": "A safety check. A new model replaces the current one only if it is "
                      "cheaper to run for the business and its accuracy does not drop.",
}
for term, meaning in glossary.items():
    st.markdown(f"**{term}:** {meaning}")

st.subheader("What was actually decided")
st.write(
    "Missing a customer who leaves usually costs more than sending an offer to someone who "
    "would have stayed. So instead of picking the line that looks best on a textbook score, "
    "the action line was chosen to **minimise estimated business cost**."
)
if metadata:
    st.caption(
        f"Current model: {metadata.get('model_name', 'unknown')}. "
        f"Current action line: {metadata.get('decision_threshold', 'unknown')}."
    )

st.subheader("Honest notes")
st.warning(
    "- The data is a public sample dataset, not real customers.\n"
    "- The cost numbers (cost of a lost customer, cost of an offer) are assumptions for the demo.\n"
    "- Customer arrival times and the delay before labels appear are simulated, "
    "because the dataset has no dates.\n"
    "- The risk score is for ranking. It is not a calibrated probability."
)
# st.info("Engineers and interviewers: open **Under the hood** in the sidebar for drift checks, "
#         "the cost curve, model explanations and prediction logs.")
