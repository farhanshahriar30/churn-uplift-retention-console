import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
import pandas as pd

from src.config import FEATURE_COLS, OUTCOME_CONVERSION, TREATMENT_COL

st.set_page_config(page_title="Model Monitoring", page_icon="🩺", layout="wide")

"""
Phase A: Page purpose
Basic monitoring-style checks (lightweight):
- base conversion rate by split
- treatment mix by split
- simple feature summaries to spot big shifts
"""

st.header("Model Monitoring 🩺")

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

# Phase B: Base rates
st.subheader("Base rates")
rates = pd.DataFrame(
    {
        "split": ["train", "val", "test"],
        "conversion_rate": [
            train[OUTCOME_CONVERSION].mean(),
            val[OUTCOME_CONVERSION].mean(),
            test[OUTCOME_CONVERSION].mean(),
        ],
    }
)
st.dataframe(rates)

st.subheader("Treatment mix")
mix = pd.DataFrame(
    {
        "split": ["train", "val", "test"],
        "No E-Mail": [
            (train[TREATMENT_COL] == "No E-Mail").mean(),
            (val[TREATMENT_COL] == "No E-Mail").mean(),
            (test[TREATMENT_COL] == "No E-Mail").mean(),
        ],
        "Mens E-Mail": [
            (train[TREATMENT_COL] == "Mens E-Mail").mean(),
            (val[TREATMENT_COL] == "Mens E-Mail").mean(),
            (test[TREATMENT_COL] == "Mens E-Mail").mean(),
        ],
        "Womens E-Mail": [
            (train[TREATMENT_COL] == "Womens E-Mail").mean(),
            (val[TREATMENT_COL] == "Womens E-Mail").mean(),
            (test[TREATMENT_COL] == "Womens E-Mail").mean(),
        ],
    }
)
st.dataframe(mix)

st.divider()

# Phase C: Feature summaries
st.subheader("Feature summary (quick drift scan)")

feature = st.selectbox("Feature", FEATURE_COLS)


def _summ(df: pd.DataFrame) -> dict:
    s = df[feature]
    if s.dtype == "object":
        top = s.value_counts().head(5).to_dict()
        return {"type": "categorical", "top_values": top}
    else:
        return {
            "type": "numeric",
            "mean": float(s.mean()),
            "std": float(s.std()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
        }


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
