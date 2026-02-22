import sys
from pathlib import Path

# Phase A: Ensure repo root is on PYTHONPATH so `import src...` works under Streamlit
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from src.config import FEATURE_COLS, OUTCOME_CONVERSION, TREATMENT_COL

st.set_page_config(page_title="Model Monitoring", page_icon="🩺", layout="wide")

# Phase B: Page purpose (lightweight monitoring checks)
# - base conversion rate by split
# - treatment mix by split
# - quick per-feature summaries to spot large distribution shifts

st.header("Model Monitoring 🩺")

st.info(
    """
**What you’re seeing**

This page checks whether the data and campaign setup look stable.

- **Base rates**: conversion rate across train/val/test. Big differences can mean the environment changed.
- **Treatment mix**: % receiving No E-Mail vs Mens vs Womens. This should stay consistent because assignment was randomized.
- **Feature summary**: quick scan to spot obvious shifts (for example, customers suddenly becoming “more urban” or spending patterns changing).

Business translation: this helps answer “Can we trust the model outputs today the way we trusted them when we trained it?”
"""
)

# Phase C: Load the dataset splits
train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

# Phase D: Base rates (conversion rate per split)
st.subheader("Base rates")
rates = pd.DataFrame(
    {
        "split": ["train", "val", "test"],
        "conversion_rate": [
            float(train[OUTCOME_CONVERSION].mean()),
            float(val[OUTCOME_CONVERSION].mean()),
            float(test[OUTCOME_CONVERSION].mean()),
        ],
    }
)
st.caption(
    "If base conversion rate shifts a lot over time/splits, model performance can drift because customer behavior changed."
)

st.dataframe(rates)

# Phase E: Treatment mix (sanity check randomization stayed stable across splits)
st.subheader("Treatment mix")
mix = pd.DataFrame(
    {
        "split": ["train", "val", "test"],
        "No E-Mail": [
            float((train[TREATMENT_COL] == "No E-Mail").mean()),
            float((val[TREATMENT_COL] == "No E-Mail").mean()),
            float((test[TREATMENT_COL] == "No E-Mail").mean()),
        ],
        "Mens E-Mail": [
            float((train[TREATMENT_COL] == "Mens E-Mail").mean()),
            float((val[TREATMENT_COL] == "Mens E-Mail").mean()),
            float((test[TREATMENT_COL] == "Mens E-Mail").mean()),
        ],
        "Womens E-Mail": [
            float((train[TREATMENT_COL] == "Womens E-Mail").mean()),
            float((val[TREATMENT_COL] == "Womens E-Mail").mean()),
            float((test[TREATMENT_COL] == "Womens E-Mail").mean()),
        ],
    }
)
st.caption(
    "Treatment assignment should stay close to ~1/3 each. Large deviations can indicate the experiment or targeting logic changed."
)

st.dataframe(mix)

st.divider()

# Phase F: Feature summary (quick drift scan)
# We show simple summaries rather than heavy drift tooling:
# - numeric: mean/std and quartiles
# - categorical: top values
st.subheader("Feature summary (quick drift scan)")

feature = st.selectbox("Feature", FEATURE_COLS)


def _summ(df: pd.DataFrame) -> dict:
    s = df[feature]
    if s.dtype == "object":
        top = s.value_counts().head(5).to_dict()
        return {"type": "categorical", "top_values": top}

    return {
        "type": "numeric",
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
    }


st.caption(
    "Quick drift scan: compare train vs val vs test summaries. Big differences can signal a data shift worth investigating."
)
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Train")
    st.json(_summ(train))
with col2:
    st.write("Val")
    st.json(_summ(val))
with col3:
    st.write("Test")
    st.json(_summ(test))
