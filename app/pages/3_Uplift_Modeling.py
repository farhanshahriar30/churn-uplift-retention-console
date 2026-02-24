import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

if "page_config_set" not in st.session_state:
    st.set_page_config(
        page_title="Retention Decision Console", page_icon="📈", layout="wide"
    )
    st.session_state["page_config_set"] = True

from src.config import (
    FEATURE_COLS,
    CONTROL_LABEL,
    TREATMENT_LABELS,
    OUTCOME_CONVERSION,
)
from src.data.treatment import filter_binary_task
from src.utils.io import load_joblib

ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def compute_uplifts(df: pd.DataFrame, payload: dict) -> dict[str, np.ndarray]:
    """
    Phase B: Compute per-treatment uplift arrays.

    For each treatment label:
    - p1(x) = P(Y=1 | treated, X=x)
    - p0(x) = P(Y=1 | control, X=x)
    - uplift = p1(x) - p0(x)
    """
    preprocessor = payload["preprocessor"]
    treatment_models = payload["treatment_models"]

    X = preprocessor.transform(df[FEATURE_COLS])

    uplifts: dict[str, np.ndarray] = {}
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

    - Choose the treatment with the highest uplift.
    - If best uplift <= 0, recommend control (No E-Mail).
    """
    tau_stack = np.vstack([uplifts[label] for label in TREATMENT_LABELS]).T
    best_idx = np.argmax(tau_stack, axis=1)
    best_uplift = tau_stack[np.arange(tau_stack.shape[0]), best_idx]

    best_action = np.array([TREATMENT_LABELS[i] for i in best_idx], dtype=object)
    best_action = np.where(best_uplift > 0, best_action, CONTROL_LABEL)

    return pd.DataFrame({"best_action": best_action, "best_uplift": best_uplift})


def qini_curve(df_binary: pd.DataFrame, uplift_scores: np.ndarray) -> pd.DataFrame:
    """
    Phase D: Build a Qini-style curve for one-vs-control evaluation.

    We estimate cumulative incremental conversions as we target more users
    (sorted by predicted uplift).
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
    Phase E: Summarize the Qini curve by AUUC.

    Higher AUUC means better uplift ranking under this estimator.
    """
    x = curve["frac"].to_numpy()
    y = curve["incremental"].to_numpy()
    return float(np.trapezoid(y, x))


# ---------------- UI ----------------

st.header("Uplift Modeling")

st.info(
    """
**What you’re seeing**

Uplift answers a different question than “risk”:

> **“If we email this customer, how much does it change their chance of buying compared to doing nothing?”**

- **Mens uplift**: expected change in purchase rate if we send the Mens email vs **No E-Mail**.
- **Womens uplift**: expected change in purchase rate if we send the Womens email vs **No E-Mail**.
- **Best action mix**: which option the model recommends most often (Mens / Womens / No E-Mail).
- **Impact ranking score (AUUC)**: checks whether the model is good at putting the “most helped by email” customers near the top.

Why this matters: it helps avoid emailing
- people who would buy anyway, and
- people who won’t respond even if we email them.
"""
)

split = st.radio("Split", ["val", "test"], horizontal=True)
df = pd.read_csv(f"data/processed/{split}.csv")

payload = load_joblib(ARTIFACTS_DIR / "uplift_model.joblib")
uplifts = compute_uplifts(df, payload)
rec = recommend_best_action(uplifts)

# Phase G: Ground-truth RCT summary (makes Womens meaningful even if Mens wins more often)
st.subheader("What actually happened in the experiment (ground truth)")

st.caption(
    "Because the campaign assignment was randomized, comparing conversion/spend across groups gives a clean estimate of each campaign’s average impact."
)

# We compute this directly from the split you selected (val/test). You can switch splits above.
seg_col = "segment"
spend_col = "spend"

if seg_col in df.columns and OUTCOME_CONVERSION in df.columns:
    grp = df.groupby(seg_col).agg(
        n=(OUTCOME_CONVERSION, "size"),
        conversion_rate=(OUTCOME_CONVERSION, "mean"),
    )

    # Add spend summaries if available
    if spend_col in df.columns:
        grp["avg_spend_per_customer"] = df.groupby(seg_col)[spend_col].mean()
        # Spend among converters only (often where Womens can look stronger)
        conv_only = df[df[OUTCOME_CONVERSION] == 1]
        if len(conv_only) > 0:
            grp["avg_spend_if_converted"] = conv_only.groupby(seg_col)[spend_col].mean()
        else:
            grp["avg_spend_if_converted"] = np.nan

    # Lift vs control (absolute + relative)
    if CONTROL_LABEL in grp.index:
        control_rate = float(grp.loc[CONTROL_LABEL, "conversion_rate"])
        grp["abs_lift_vs_control"] = grp["conversion_rate"] - control_rate
        grp["rel_lift_vs_control"] = np.where(
            control_rate > 0,
            (grp["conversion_rate"] - control_rate) / control_rate,
            np.nan,
        )
    else:
        grp["abs_lift_vs_control"] = np.nan
        grp["rel_lift_vs_control"] = np.nan

    # Sort by conversion rate to make it easy to scan
    grp = grp.sort_values("conversion_rate", ascending=False)

    # Formatting for readability
    pretty = grp.copy()
    pretty["conversion_rate"] = pretty["conversion_rate"].map(lambda x: f"{x:.4f}")
    pretty["abs_lift_vs_control"] = pretty["abs_lift_vs_control"].map(
        lambda x: f"{x:.4f}" if pd.notna(x) else "—"
    )
    pretty["rel_lift_vs_control"] = pretty["rel_lift_vs_control"].map(
        lambda x: f"{x:.0%}" if pd.notna(x) else "—"
    )

    if "avg_spend_per_customer" in pretty.columns:
        pretty["avg_spend_per_customer"] = pretty["avg_spend_per_customer"].map(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
        )
    if "avg_spend_if_converted" in pretty.columns:
        pretty["avg_spend_if_converted"] = pretty["avg_spend_if_converted"].map(
            lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
        )

    st.dataframe(pretty)

    st.caption(
        "Takeaway: both campaigns beat **No E-Mail** on conversion; Mens is stronger on conversion in this dataset, while Womens can still be meaningful (especially to check on spend)."
    )
else:
    st.warning(
        "Missing required columns to compute experiment summary (segment/conversion)."
    )

st.divider()

# Phase H: High-level model summaries
col1, col2 = st.columns(2)
with col1:
    st.subheader("Best action mix (model recommendation)")
    st.caption(
        "Recommended action per customer. **No E-Mail** means the model expects emailing won’t help this customer."
    )
    st.bar_chart(rec["best_action"].value_counts())

with col2:
    st.subheader("Best uplift summary (model)")
    st.metric("Mean", f"{rec['best_uplift'].mean():.4f}")
    st.metric("Min", f"{rec['best_uplift'].min():.4f}")
    st.metric("Max", f"{rec['best_uplift'].max():.4f}")

st.divider()

# Phase I: Per-treatment uplift distribution (best demo-friendly view)
st.subheader("Per-treatment uplift distributions")

st.caption(
    "This chart shows *how many customers* get at least a given uplift. "
    "Further right = bigger lift (more additional purchases caused by emailing)."
)

# Phase I1: Combine uplifts to pick sensible x-range (robust to outliers)
all_tau = np.concatenate([uplifts[label] for label in TREATMENT_LABELS])
lo = float(np.percentile(all_tau, 1))
hi = float(np.percentile(all_tau, 99))
pad = 0.15 * (hi - lo) if hi > lo else 0.001
lo, hi = lo - pad, hi + pad

n_points = 150

# Phase I3: Build survival curves: P(uplift >= x)
x_grid = np.linspace(lo, hi, n_points)

cdf_df = pd.DataFrame(index=x_grid)
for label in TREATMENT_LABELS:
    tau = uplifts[label]
    # Survival curve: for each threshold x, share of customers with uplift >= x
    cdf_df[label] = [(tau >= x).mean() for x in x_grid]

# Streamlit line_chart expects index as x-axis
st.line_chart(cdf_df, height=280)

st.caption(
    "How to read: at x=0, the y-value is the **% of customers helped** (uplift > 0). "
    "At x=0.002, it’s the % expected to gain at least +0.2 percentage points conversion probability."
)

# Phase I4: Simple table that non-technical users understand instantly
thresholds = [0.0, 0.002, 0.005]  # tweak these if you want
rows = []
for label in TREATMENT_LABELS:
    tau = uplifts[label]
    row = {
        "treatment": label,
        "% helped (uplift > 0)": float((tau > 0).mean()),
        "median uplift": float(np.median(tau)),
        "p95 uplift": float(np.percentile(tau, 95)),
    }
    for thr in thresholds:
        row[f"% uplift ≥ {thr:+.3f}"] = float((tau >= thr).mean())
    rows.append(row)

summary = pd.DataFrame(rows)

# Pretty formatting (keeps it readable)
pct_cols = [c for c in summary.columns if c.startswith("%")]
for c in pct_cols:
    summary[c] = summary[c].map(lambda v: f"{v:.1%}")
summary["median uplift"] = summary["median uplift"].map(lambda v: f"{v:+.4f}")
summary["p95 uplift"] = summary["p95 uplift"].map(lambda v: f"{v:+.4f}")

st.caption(
    "Quick summary: how often each campaign helps, and how big the uplift is for strong responders."
)
st.dataframe(summary)

st.divider()

# Phase J: AUUC evaluation (one-vs-control)
st.subheader("AUUC (one-vs-control evaluation)")
st.caption(
    "AUUC (impact ranking score): higher means better at ranking customers who truly benefit from outreach."
)

rows = []
for label in TREATMENT_LABELS:
    task = filter_binary_task(df, label)
    tau = compute_uplifts(task, payload)[label]
    curve = qini_curve(task, tau)
    rows.append({"treatment": label, "auuc": auuc(curve)})

st.dataframe(pd.DataFrame(rows))
