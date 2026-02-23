"""
Phase A: Goal
Show the conversion probability model outputs:
- how well the model ranks customers (deciles)
- metrics snapshot (AUC/PR-AUC + lift)
- top users by predicted conversion probability

We use validation split by default so the page loads quickly and avoids leakage.
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

if "page_config_set" not in st.session_state:
    st.set_page_config(
        page_title="Retention Decision Console", page_icon="📈", layout="wide"
    )
    st.session_state["page_config_set"] = True

# Add repo root to PYTHONPATH so `import src...` works when Streamlit runs pages
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FEATURE_COLS, OUTCOME_CONVERSION
from src.utils.io import load_joblib


ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


st.header("Outcome Model (Conversion Probability)")

st.info(
    """
**What you’re seeing**

This page estimates **who is most likely to buy** in the next window.

- **Conversion probability** = the model’s estimate of “chance this customer buys soon.”
- **Lift** answers: “If we focus on the top X% most likely to buy, how much better is that than picking customers at random?”
- **Deciles** are simply **10 buckets** from lowest → highest predicted probability. If the model ranks well, the highest bucket should usually buy more often than the lowest bucket.

This page helps with **ranking baseline likelihood**. It does *not* decide who should get an offer (that’s handled on the uplift and targeting pages).
"""
)


# Phase B: Load artifacts (use calibrated model if present for probability display)
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

# Phase E: Metrics panels (model quality + lift)
col1, col2, col3 = st.columns(3)

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

# Phase F: Decile ranking view (more interpretable than histogram for rare events)
st.subheader("Ranking quality (deciles)")

# Create deciles where Decile 10 = highest predicted probability
tmp = out[["p_conversion", OUTCOME_CONVERSION]].copy()
tmp["decile"] = pd.qcut(tmp["p_conversion"], q=10, labels=False, duplicates="drop") + 1

decile_table = tmp.groupby("decile", as_index=False).agg(
    n=("p_conversion", "size"),
    avg_pred=("p_conversion", "mean"),
    actual_rate=(OUTCOME_CONVERSION, "mean"),
)

# Make it read naturally: Decile 10 at top (best predicted)
decile_table = decile_table.sort_values("decile", ascending=False).reset_index(
    drop=True
)

# Add lift per decile vs base
decile_table["lift_vs_base"] = (
    decile_table["actual_rate"] / base_rate if base_rate > 0 else np.nan
)

st.caption(
    "Decile 10 = the customers the model thinks are most likely to buy; Decile 1 = least likely."
)

st.dataframe(decile_table)

# Chart: actual vs predicted by decile
chart_df = decile_table.set_index("decile")[["avg_pred", "actual_rate"]]
st.caption(
    "Predicted vs actual buying rate by bucket. Small ups and downs are normal because purchases are rare."
)


st.line_chart(chart_df)

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
st.caption(
    "Top-ranked customers by predicted likelihood to buy. This list is not yet the ‘who to target’ list."
)


st.dataframe(
    out.sort_values("p_conversion", ascending=False)[cols_to_show]
    .head(show_n)
    .reset_index(drop=True)
)
