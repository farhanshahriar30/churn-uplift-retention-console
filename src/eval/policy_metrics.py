"""
src/eval/policy_metrics.py

Phase A: Goal
Compare targeting strategies under the same constraints:
- uplift: target users with highest predicted uplift
- risk: target users with highest "risk" proxy (1 - predicted conversion prob)
- random: target random users

We keep the action choice consistent across strategies:
- for each user, action = best_action derived from uplift payload
This way, the comparison focuses on *who you pick*, not *which email you send*.

Phase B: What we report (expected values)
Given a selected set S:
- expected_incremental_conversions = sum(best_uplift_i for i in S)
- total_cost = sum(cost(best_action_i) for i in S)
- cost_per_incremental = total_cost / expected_incremental_conversions  (if > 0)

These are model-based expectations (good for simulation + UI).

Phase C: Realism diagnostics (added)
We also report:
- avg_uplift_selected, median_uplift_selected
- implied_uplift_rate = expected_incremental_conversions / n_selected
These help sanity-check magnitude while keeping comparisons identical.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    CONTROL_LABEL,
    FEATURE_COLS,
    OUTCOME_CONVERSION,
    TREATMENT_LABELS,
)
from src.utils.io import load_joblib
from src.policy.targeting import PolicyInputs, _action_cost


def recommend_from_uplift_payload(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    """
    Phase D: Build per-user recommended action and best uplift using only payload contents.
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

    tau_stack = np.vstack([uplifts[label] for label in TREATMENT_LABELS]).T
    best_idx = np.argmax(tau_stack, axis=1)
    best_uplift = tau_stack[np.arange(len(df)), best_idx]

    best_action = np.array([TREATMENT_LABELS[i] for i in best_idx], dtype=object)
    best_action = np.where(best_uplift > 0, best_action, CONTROL_LABEL)

    return pd.DataFrame(
        {"best_action": best_action, "best_uplift": best_uplift}, index=df.index
    )


def churn_probs(df: pd.DataFrame) -> np.ndarray:
    """
    Phase E: Get baseline conversion probabilities from the outcome model.

    We prefer the calibrated model for probability-based views and ordering.
    If it's missing, we fall back to the base model.
    """
    preprocess_payload = load_joblib(ARTIFACTS_DIR / "preprocess.joblib")
    preprocessor = preprocess_payload["preprocessor"]

    cal_path = ARTIFACTS_DIR / "churn_model_calibrated.joblib"
    if cal_path.exists():
        model = load_joblib(cal_path)
    else:
        model = load_joblib(ARTIFACTS_DIR / "churn_model.joblib")

    X = preprocessor.transform(df[FEATURE_COLS])
    return model.predict_proba(X)[:, 1]


def _apply_constraints(
    df: pd.DataFrame,
    rec: pd.DataFrame,
    ordered_idx: np.ndarray,
    inputs: PolicyInputs,
) -> pd.DataFrame:
    """
    Phase F: Apply max_volume + budget constraints in the given order.

    We only consider actionable users:
    - best_action != control
    - best_uplift > 0
    """
    tmp = rec.loc[ordered_idx].copy()
    tmp = tmp[(tmp["best_action"] != CONTROL_LABEL) & (tmp["best_uplift"] > 0)].copy()

    if inputs.max_volume is not None:
        tmp = tmp.head(int(inputs.max_volume)).copy()

    if tmp.empty:
        tmp["cost"] = []
        return tmp

    tmp["cost"] = tmp["best_action"].apply(
        lambda a: _action_cost(a, inputs.cost_mens, inputs.cost_womens)
    )

    tmp["cum_cost"] = tmp["cost"].cumsum()
    tmp = tmp[tmp["cum_cost"] <= float(inputs.budget)].copy()

    if "customer_id" in df.columns:
        tmp = tmp.join(df["customer_id"], how="left")

    return tmp


def simulate_strategies(df: pd.DataFrame, inputs: PolicyInputs) -> dict:
    """
    Phase G: Run the three strategies and return comparable metrics.
    """
    uplift_payload = load_joblib(ARTIFACTS_DIR / "uplift_model.joblib")
    rec = recommend_from_uplift_payload(df, uplift_payload)

    # Strategy 1: uplift ordering (descending best_uplift)
    uplift_order = rec["best_uplift"].sort_values(ascending=False).index.to_numpy()
    uplift_sel = _apply_constraints(df, rec, uplift_order, inputs)

    # Strategy 2: risk ordering (descending risk = 1 - p_conversion)
    p_conv = churn_probs(df)
    risk = 1.0 - p_conv
    risk_order = df.index.to_numpy()[np.argsort(-risk)]
    risk_sel = _apply_constraints(df, rec, risk_order, inputs)

    # Strategy 3: random ordering
    rng = np.random.default_rng(42)
    rand_order = df.index.to_numpy().copy()
    rng.shuffle(rand_order)
    rand_sel = _apply_constraints(df, rec, rand_order, inputs)

    def _summarize(selected: pd.DataFrame) -> dict:
        if selected.empty:
            return {
                "n_selected": 0,
                "total_cost": 0.0,
                "expected_incremental_conversions": 0.0,
                "implied_uplift_rate": 0.0,
                "avg_uplift_selected": 0.0,
                "median_uplift_selected": 0.0,
                "cost_per_incremental": None,
                "action_mix": {},
            }

        inc = float(selected["best_uplift"].sum())
        n = int(len(selected))
        cost = float(selected["cost"].sum())
        cpi = (cost / inc) if inc > 0 else None

        return {
            "n_selected": n,
            "total_cost": cost,
            "expected_incremental_conversions": inc,
            "implied_uplift_rate": (inc / n) if n > 0 else 0.0,
            "avg_uplift_selected": float(selected["best_uplift"].mean()),
            "median_uplift_selected": float(selected["best_uplift"].median()),
            "cost_per_incremental": cpi,
            "action_mix": selected["best_action"].value_counts().to_dict(),
        }

    return {
        "inputs": {
            "budget": float(inputs.budget),
            "cost_mens": float(inputs.cost_mens),
            "cost_womens": float(inputs.cost_womens),
            "max_volume": int(inputs.max_volume)
            if inputs.max_volume is not None
            else None,
            "objective": inputs.objective,
        },
        "uplift": _summarize(uplift_sel),
        "risk": _summarize(risk_sel),
        "random": _summarize(rand_sel),
        "base_conversion_rate": float(df[OUTCOME_CONVERSION].mean())
        if OUTCOME_CONVERSION in df.columns
        else None,
    }


if __name__ == "__main__":
    val = pd.read_csv("data/processed/val.csv")

    inputs = PolicyInputs(
        budget=200.0,
        cost_mens=0.02,
        cost_womens=0.02,
        ltv_value=1.0,
        max_volume=5000,
        objective="retained_customers",
    )

    out = simulate_strategies(val, inputs)
    print(json.dumps(out, indent=2))

    (ARTIFACTS_DIR / "policy_simulation_val.json").write_text(json.dumps(out, indent=2))
    print("Saved:", ARTIFACTS_DIR / "policy_simulation_val.json")
