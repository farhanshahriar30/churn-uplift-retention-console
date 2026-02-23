"""
Phase A: Goal
Provide a single Streamlit entrypoint. Streamlit automatically discovers files in
app/pages/ as separate pages, so this file focuses on:
- global page config
- a short landing header
- basic navigation guidance
"""

import streamlit as st

# Phase B: Global config must be the FIRST Streamlit command and called only once per app
st.set_page_config(
    page_title="Retention Decision Console",
    page_icon="📈",
    layout="wide",
)

# Optional: make sidebar feel branded (this sits above the page navigation)
st.sidebar.title("Retention Decision Console 🎯")
st.sidebar.caption("Navigate pages below")

st.title("Churn + Uplift Retention Console 📈")
st.caption(
    "Use the pages in the left sidebar to explore risk, uplift, and targeting policies."
)

st.info(
    """
### In plain English
We’re choosing **who to email** and **which email to send** so we get the most **extra purchases**, using the same budget.

### What this is
This is a **Retention Decision Console** that helps a business run smarter customer campaigns.

Instead of only asking “Who is most likely to churn?”, we answer the more actionable question:
> **“Who will actually change their behavior if we intervene, and what’s the best intervention?”**

### What it does
- **Outcome / risk modeling:** estimates each customer’s likelihood to convert (buy) in the next window.
- **Uplift (causal) modeling:** estimates the *extra impact* of emailing a customer (Mens vs Womens vs No E-Mail).
- **Targeting optimization:** given a **budget and capacity**, recommends who to target and with which campaign to maximize **incremental conversions**.

### Why it matters
A budget is wasted if we message:
- people who would convert anyway, or
- people who won’t respond regardless.

This console aims to maximize **ROI per message** by targeting customers with the highest expected incremental benefit.

### How to use this dashboard (quick path)
1) **Overview**: confirms data + experiment setup.  
2) **Uplift Modeling**: shows who benefits from each email campaign.  
3) **Targeting Simulator**: set a budget and see which strategy produces more extra purchases.
"""
)

st.markdown(
    """
**Pages**
- **Overview**: dataset summary + saved artifacts
- **Churn Risk**: probability model metrics and top-risk users
- **Uplift Modeling**: uplift distributions and AUUC/Qini-style summaries
- **Targeting Simulator**: budget vs expected impact
- **Model Monitoring**: drift/base-rate checks
"""
)
