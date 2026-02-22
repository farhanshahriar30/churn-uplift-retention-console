import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH so `import src...` works under Streamlit
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.config import FEATURE_COLS, CONTROL_LABEL, TREATMENT_LABELS, OUTCOME_CONVERSION
from src.utils.io import load_joblib
from src.data.treatment import filter_binary_task

st.set_page_config(
    page_title="Churn + Uplift Retention Console", page_icon="📈", layout="wide"
)

ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def compute_uplifts(df: pd.DataFrame, payload: dict) -> dict[str, np.ndarray]:
    """
    Phase B: Compute per-treatment uplift arrays.

    Steps:
    1) Transform X using the saved preprocessor
    2) For each treatment label:
       - p1(x) = treated_model.predict_proba(X)[:,1]
       - p0(x) = control_model.predict_proba(X)[:,1]
       - uplift = p1(x) - p0(x)
    """
    preprocessor = payload["preprocessor"]
    treatment_models = payload["treatment_models"]

    X = preprocessor.transform(df[FEATURE_COLS])

    uplifts = {}
    for label in TREATMENT_LABELS:
        m_treated = treatment_models[label]["treated"]
        m_control = treatment_models[label]["control"]
        p1 = m_treated.predict_proba(X)[:, 1]
        p0 = m_control.predict_proba(X)[:, 1]
        uplifts[label] = p1 - p0

    return uplifts


def recommend_best_action(uplifts: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Phase C: Choose the best action per user.

    Logic:
    - Stack uplift scores (n_users x n_treatments)
    - best_action = argmax uplift
    - If best_uplift <= 0 -> recommend CONTROL_LABEL (No E-Mail)
    """
    tau_stack = np.vstack([uplifts[label] for label in TREATMENT_LABELS]).T
    best_idx = np.argmax(tau_stack, axis=1)
    best_uplift = tau_stack[np.arange(tau_stack.shape[0]), best_idx]

    best_action = np.array([TREATMENT_LABELS[i] for i in best_idx], dtype=object)
    best_action = np.where(best_uplift > 0, best_action, CONTROL_LABEL)

    return pd.DataFrame({"best_action": best_action, "best_uplift": best_uplift})


def qini_curve(df_binary: pd.DataFrame, uplift_scores: np.ndarray) -> pd.DataFrame:
    """
    Phase D: Build a Qini-style curve for a binary task.

    df_binary must contain:
    - T: 0/1 treatment indicator
    - conversion: 0/1 outcome

    Curve definition (one common estimator):
    - Sort by predicted uplift descending.
    - As we include more users, track:
      incremental = n_treated_so_far * (rate_treated_so_far - rate_control_so_far)
    """
    tmp = df_binary.copy()
    tmp["uplift"] = uplift_scores
    tmp = tmp.sort_values("uplift", ascending=False).reset_index(drop=True)

    t = tmp["T"].to_numpy(dtype=int)
    y = tmp[OUTCOME_CONVERSION].to_numpy(dtype=int)

    inc, frac = [], []

    n = len(tmp)
    treated_count = control_count = 0
    treated_y = control_y = 0

    for i in range(n):
        if t[i] == 1:
            treated_count += 1
            treated_y += y[i]
        else:
            control_count += 1
            control_y += y[i]

        rt = treated_y / treated_count if treated_count > 0 else 0.0
        rc = control_y / control_count if control_count > 0 else 0.0
        incremental = treated_count * (rt - rc)

        inc.append(incremental)
        frac.append((i + 1) / n)

    return pd.DataFrame({"frac": frac, "incremental": inc})


def auuc(curve: pd.DataFrame) -> float:
    """
    Phase E: Summarize the curve area (AUUC).
    Higher means “better uplift ordering” under this estimator.
    """
    x = curve["frac"].to_numpy()
    y = curve["incremental"].to_numpy()
    return float(np.trapezoid(y, x))


# ---------------- UI ----------------

st.header("Uplift Modeling")

split = st.radio("Split", ["val", "test"], horizontal=True)
df = pd.read_csv(f"data/processed/{split}.csv")

payload = load_joblib(ARTIFACTS_DIR / "uplift_model.joblib")

uplifts = compute_uplifts(df, payload)
rec = recommend_best_action(uplifts)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Best action mix")
    st.bar_chart(rec["best_action"].value_counts())

with col2:
    st.subheader("Best uplift summary")
    st.metric("Mean", f"{rec['best_uplift'].mean():.4f}")
    st.metric("Min", f"{rec['best_uplift'].min():.4f}")
    st.metric("Max", f"{rec['best_uplift'].max():.4f}")

st.divider()
st.subheader("Per-treatment uplift distributions")

# Histogram comparison (cleaner than the noisy time-series style line plot)
bins = st.slider("Histogram bins", min_value=20, max_value=120, value=60, step=10)

hist_df = pd.DataFrame(
    {
        label: np.histogram(uplifts[label], bins=bins, range=(-1.0, 1.0))[0]
        for label in TREATMENT_LABELS
    }
)
st.bar_chart(hist_df, height=260)

# Percentiles table (p5/p50/p95)
pct_rows = []
for label in TREATMENT_LABELS:
    tau = uplifts[label]
    pct_rows.append(
        {
            "treatment": label,
            "p05": float(np.percentile(tau, 5)),
            "p50": float(np.percentile(tau, 50)),
            "p95": float(np.percentile(tau, 95)),
        }
    )

st.caption("Uplift percentiles (helps summarize spread without noise).")
st.dataframe(pd.DataFrame(pct_rows))

st.divider()
st.subheader("AUUC (one-vs-control evaluation)")

rows = []
for label in TREATMENT_LABELS:
    task = filter_binary_task(df, label)
    tau = compute_uplifts(task, payload)[label]
    curve = qini_curve(task, tau)
    rows.append({"treatment": label, "auuc": auuc(curve)})

st.dataframe(pd.DataFrame(rows))
