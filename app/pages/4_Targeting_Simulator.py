import sys
from pathlib import Path

# Phase A: Make sure Streamlit can import from `src/` when running multipage apps
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

if "page_config_set" not in st.session_state:
    st.set_page_config(
        page_title="Retention Decision Console", page_icon="📈", layout="wide"
    )
    st.session_state["page_config_set"] = True

from src.eval.policy_metrics import simulate_strategies
from src.policy.targeting import PolicyInputs


# Phase B: Page purpose
# This is an interactive "what-if" console:
# - user sets constraints (budget, max outreach volume, per-email costs)
# - we compare three strategies under identical constraints:
#   1) uplift targeting
#   2) risk targeting
#   3) random targeting
st.header("Targeting Simulator 🎯")

st.info(
    """
**What you’re seeing**

This page turns the models into a **decision**: who to email, and which email to send.

You set real-world constraints:
- **Budget**: how much you can spend
- **Cost per email**: Mens / Womens
- **Max outreach volume**: how many people you can contact

Then we compare three strategies under the same constraints:
1) **Uplift targeting**: email the people most likely to be helped by outreach.
2) **Risk targeting**: email based on “likelihood/risk” scores (not necessarily impact).
3) **Random**: a baseline for comparison.

Key outputs:
- **Expected extra purchases** (incremental conversions): how many additional purchases we expect *because of* the targeting choice.
- **Cost per extra purchase**: how much spend per additional purchase (lower is better).

This is the most “actionable” page: it shows the expected payoff of your targeting policy.
"""
)


# Phase C: Collect user inputs that define the policy constraints
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
    st.caption(
        "Objective is ‘retained_customers’ for now. Profit optimization can be added next."
    )


# Phase D: Load the selected split and build a PolicyInputs object
# PolicyInputs is the single config object used by the decision layer.
df = pd.read_csv(f"data/processed/{split}.csv")

inputs = PolicyInputs(
    budget=float(budget),
    cost_mens=float(cost_mens),
    cost_womens=float(cost_womens),
    ltv_value=1.0,
    max_volume=int(max_volume) if max_volume > 0 else None,
    objective="retained_customers",
)

# Phase E: Run the simulator
# simulate_strategies() applies the same constraints for each strategy and returns
# expected incremental conversions and cost-efficiency metrics.
out = simulate_strategies(df, inputs)

st.divider()
st.subheader("Results")

# Phase F: Visual comparison first (fast to interpret)
st.caption(
    "Same budget and capacity for all strategies. Higher bar = more expected **extra purchases** caused by the targeting strategy."
)

st.bar_chart(
    {
        "uplift": out["uplift"]["expected_incremental_conversions"],
        "risk": out["risk"]["expected_incremental_conversions"],
        "random": out["random"]["expected_incremental_conversions"],
    },
    height=220,
)

# Phase G: Detailed metrics per strategy
st.caption(
    "Cost per extra purchase = total spend ÷ expected extra purchases (lower is better)."
)


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

# Phase H: Show which email campaign gets chosen under uplift targeting
if out["uplift"]["action_mix"]:
    st.caption(
        "Under these constraints, this shows which campaign the uplift strategy chooses most often."
    )

    st.bar_chart(out["uplift"]["action_mix"])
else:
    st.info("No users selected under current constraints.")

st.caption(f"Baseline purchase rate in this split: {out['base_conversion_rate']:.4f}")
