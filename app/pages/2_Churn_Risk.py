"""
app/pages/2_Churn_Risk.py

Phase A: Goal
Show the conversion probability model outputs:
- distribution of predicted probabilities
- top users by predicted "risk" (1 - p_conv)
- metrics snapshot (AUC/PR-AUC + lift)

We use validation split by default so the page loads quickly and avoids leakage.
"""

import sys
from pathlib import Path

# Add repo root to PYTHONPATH so `import src...` works when Streamlit runs pages
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import pandas as pd
import streamlit as st

from src.config import FEATURE_COLS, OUTCOME_CONVERSION
from src.utils.io import load_joblib

import streamlit as st

st.set_page_config(
    page_title="Churn + Uplift Retention Console", page_icon="📈", layout="wide"
)


ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


st.header("Churn Risk (Outcome Model)")

# Phase B: Load artifacts
preprocess_payload = load_joblib(ARTIFACTS_DIR / "preprocess.joblib")
model = load_joblib(ARTIFACTS_DIR / "churn_model.joblib")
preprocessor = preprocess_payload["preprocessor"]

# Phase C: Choose split (val/test)
split = st.radio("Split", ["val", "test"], horizontal=True)
df = pd.read_csv(f"data/processed/{split}.csv")

# Phase D: Score
X = preprocessor.transform(df[FEATURE_COLS])
p_conv = model.predict_proba(X)[:, 1]
risk = 1.0 - p_conv

# Attach scores for display
out = df.copy()
out["p_conversion"] = p_conv
out["risk"] = risk

# Phase E: Metrics panels
col1, col2, col3 = st.columns(3)

# AUC/PR-AUC
metrics_path = ARTIFACTS_DIR / "churn_metrics.json"
if metrics_path.exists():
    m = _load_json(metrics_path)
    with col1:
        st.subheader("Discrimination")
        st.metric("ROC-AUC", f"{m[split]['roc_auc']:.3f}")
        st.metric("PR-AUC", f"{m[split]['pr_auc']:.4f}")
else:
    with col1:
        st.warning("Missing churn_metrics.json")

# Lift + Brier from churn_metrics module output (we didn't save it yet, so we compute quickly here)
with col2:
    st.subheader("Base rate")
    st.metric("Conversion rate", f"{out[OUTCOME_CONVERSION].mean():.4f}")
    st.metric("Avg predicted p(conv)", f"{out['p_conversion'].mean():.4f}")

with col3:
    st.subheader("Top risk slice")
    top_k = st.slider("Top % by risk", min_value=1, max_value=50, value=10, step=1)
    n_top = max(1, int(len(out) * (top_k / 100)))
    top = out.sort_values("risk", ascending=False).head(n_top)
    top_rate = float(top[OUTCOME_CONVERSION].mean())
    base_rate = float(out[OUTCOME_CONVERSION].mean())
    lift = (top_rate / base_rate) if base_rate > 0 else 0.0
    st.metric(f"Lift @ {top_k}%", f"{lift:.2f}x")
    st.caption(f"Top conv rate: {top_rate:.4f} vs base: {base_rate:.4f}")

st.divider()

# Phase F: Distribution
st.subheader("Predicted probability distribution")
st.bar_chart(
    pd.Series(p_conv).clip(0, 1).round(3).value_counts().sort_index(), height=250
)

st.divider()

# Phase G: Top users table
st.subheader("Top users by risk")
show_n = st.slider("Rows", 10, 200, 50, 10)

cols_to_show = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    OUTCOME_CONVERSION,
    "p_conversion",
    "risk",
]

st.dataframe(
    out.sort_values("risk", ascending=False)[cols_to_show]
    .head(show_n)
    .reset_index(drop=True)
)
