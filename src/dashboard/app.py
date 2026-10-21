"""Entry point for the dashboard.

Run from the project root:
    python -m streamlit run src/dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📞",
    layout="wide",
)

pages = [
    st.Page("views/try_it.py", title="Try it", icon="🎯", default=True),
    st.Page("views/how_it_works.py", title="How it works", icon="💡"),
    st.Page("views/monitoring.py", title="Under the hood", icon="🛠️"),
]

st.navigation(pages).run()
