"""
app/streamlit_app.py

Phase A: Goal
Provide a single Streamlit entrypoint. Streamlit automatically discovers files in
app/pages/ as separate pages, so this file focuses on:
- global page config
- a short landing header
- basic navigation guidance
"""

import streamlit as st


st.set_page_config(
    page_title="Churn + Uplift Retention Console",
    page_icon="📈",
    layout="wide",
)

st.title("Churn + Uplift Retention Console 📈")
st.caption(
    "Use the pages in the left sidebar to explore risk, uplift, and targeting policies."
)

st.markdown(
    """
**Pages**
- **Overview**: dataset summary + saved artifacts
- **Churn Risk**: probability model metrics and top-risk users
- **Uplift Modeling**: uplift distributions and AUUC/Qini-style summaries
- **Targeting Simulator**: budget vs expected impact
"""
)
