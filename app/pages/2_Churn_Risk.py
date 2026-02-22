"""
app/pages/2_Churn_Risk.py

Phase A: Goal
Show the conversion probability model outputs:
- distribution of predicted probabilities
- top users by predicted conversion probability
- metrics snapshot (AUC/PR-AUC + lift)

We use validation split by default so the page loads quickly and avoids leakage.
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

# Add repo root to PYTHONPATH so `import src...` works when Streamlit runs pages
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FEATURE_COLS, OUTCOME_CONVERSION
from src.utils.io import load_joblib

st.set_page_config(
    page_title="Churn + Uplift Retention Console",
    page_icon="📈",
    layout="wide",
)

ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


st.header("Outcome Model (Conversion Probability)")

st.info(
    """
**What you’re seeing**

This page predicts **who is likely to convert** (or “stay engaged”) in the next window.

- **Conversion probability** is the model’s estimate of “chance this customer buys in the next period.”
- **Lift** answers: “If we target the top X% most likely to convert, how much better is that than targeting randomly?”
- We use **calibrated probabilities**, which means the numbers behave like real probabilities (they average out close to the true base rate).

This page is about **ranking and understanding baseline likelihood**. It does *not* tell you who to target with an offer (that’s uplift).
"""
)

# Phase B: Load artifacts
preprocess_payload = load_joblib(ARTIFACTS_DIR / "preprocess.joblib")
cal_path = ARTIFACTS_DIR / "churn_model_calibrated.joblib"
if cal_path.exists():
    model = load_joblib(cal_path)
else:
    model = load_joblib(ARTIFACTS_DIR / "churn_model.joblib")

preprocessor = preprocess_payload["preprocessor"]

# Phase C: Choose split (val/test)
split = st.radio("Split", ["val", "test"], horizontal=True)
df = pd.read_csv(f"data/processed/{split}.csv")

# Phase D: Score
X = preprocessor.transform(df[FEATURE_COLS])
p_conv = model.predict_proba(X)[:, 1]  # P(conversion=1 | X)
risk = 1.0 - p_conv  # "churn risk" proxy

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
        st.metric("ROC-AUC", f"{m[split]['calibrated']['roc_auc']:.3f}")
        st.metric("PR-AUC", f"{m[split]['calibrated']['pr_auc']:.4f}")

else:
    with col1:
        st.warning("Missing artifacts/churn_metrics.json")

with col2:
    st.subheader("Base rate")
    base_rate = float(out[OUTCOME_CONVERSION].mean())
    st.metric("Conversion rate", f"{base_rate:.4f}")
    st.metric("Avg predicted p(conv)", f"{float(p_conv.mean()):.4f}")

with col3:
    st.subheader("Top probability slice")
    top_k = st.slider(
        "Top % by predicted p(conv)", min_value=1, max_value=50, value=10, step=1
    )

    n_top = max(1, int(len(out) * (top_k / 100)))
    top = out.sort_values("p_conversion", ascending=False).head(n_top)

    top_rate = float(top[OUTCOME_CONVERSION].mean())
    lift = (top_rate / base_rate) if base_rate > 0 else 0.0

    st.metric(f"Lift @ {top_k}%", f"{lift:.2f}x")
    st.caption(f"Top conv rate: {top_rate:.4f} vs base: {base_rate:.4f}")

st.divider()

# Phase F: Distribution (bin into a histogram so it stays readable)
st.subheader("Predicted probability distribution")

bins = st.slider("Histogram bins", min_value=10, max_value=80, value=40, step=5)
hist, bin_edges = np.histogram(p_conv.clip(0, 1), bins=bins, range=(0.0, 1.0))

hist_df = pd.DataFrame(
    {
        "bin_left": bin_edges[:-1],
        "count": hist,
    }
).set_index("bin_left")

st.bar_chart(hist_df["count"], height=260)

st.divider()

# Phase G: Top users table
st.subheader("Top users by predicted conversion probability")
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
    out.sort_values("p_conversion", ascending=False)[cols_to_show]
    .head(show_n)
    .reset_index(drop=True)
)
