import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
import pandas as pd

from src.policy.targeting import PolicyInputs
from src.eval.policy_metrics import simulate_strategies

st.set_page_config(page_title="Targeting Simulator", page_icon="🎯", layout="wide")

"""
Phase A: Page purpose
Interactive “what-if” console:
- choose constraints (budget, max volume, costs)
- compare uplift vs risk vs random under those constraints
- show expected impact metrics
"""

st.header("Targeting Simulator 🎯")

# Phase B: Inputs
colA, colB, colC = st.columns(3)

with colA:
    split = st.radio("Split", ["val", "test"], horizontal=True)
    max_volume = st.number_input(
        "Max outreach volume", min_value=0, value=5000, step=500
    )

with colB:
    budget = st.number_input("Budget ($)", min_value=0.0, value=200.0, step=50.0)
    cost_mens = st.number_input(
        "Cost per Mens email ($)", min_value=0.0, value=0.02, step=0.01
    )

with colC:
    cost_womens = st.number_input(
        "Cost per Womens email ($)", min_value=0.0, value=0.02, step=0.01
    )
    st.caption("Objective is retained_customers for now (profit comes next).")

df = pd.read_csv(f"data/processed/{split}.csv")

inputs = PolicyInputs(
    budget=float(budget),
    cost_mens=float(cost_mens),
    cost_womens=float(cost_womens),
    ltv_value=1.0,
    max_volume=int(max_volume) if max_volume > 0 else None,
    objective="retained_customers",
)

# Phase C: Run simulation
out = simulate_strategies(df, inputs)

st.divider()
st.subheader("Results")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        "Uplift: expected incremental conversions",
        f"{out['uplift']['expected_incremental_conversions']:.1f}",
    )
    st.metric(
        "Uplift: cost per incremental",
        f"{out['uplift']['cost_per_incremental']:.4f}"
        if out["uplift"]["cost_per_incremental"]
        else "—",
    )

with c2:
    st.metric(
        "Risk: expected incremental conversions",
        f"{out['risk']['expected_incremental_conversions']:.1f}",
    )
    st.metric(
        "Risk: cost per incremental",
        f"{out['risk']['cost_per_incremental']:.4f}"
        if out["risk"]["cost_per_incremental"]
        else "—",
    )

with c3:
    st.metric(
        "Random: expected incremental conversions",
        f"{out['random']['expected_incremental_conversions']:.1f}",
    )
    st.metric(
        "Random: cost per incremental",
        f"{out['random']['cost_per_incremental']:.4f}"
        if out["random"]["cost_per_incremental"]
        else "—",
    )

st.divider()
st.subheader("Action mix (uplift strategy)")
if out["uplift"]["action_mix"]:
    st.bar_chart(out["uplift"]["action_mix"])
else:
    st.info("No users selected under current constraints.")

st.caption(f"Base conversion rate ({split}): {out['base_conversion_rate']:.4f}")
