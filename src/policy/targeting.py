"""
Phase A: Goal
Choose who to target under constraints (budget and/or max outreach volume)
using uplift estimates.

We support two objectives:
1) retained_customers:
   - score = best_uplift
2) profit:
   - score = best_uplift * LTV - cost(best_action)

Phase B: Important implementation choice (artifact loading)
We load a plain dict payload from artifacts/uplift_model.joblib and compute
uplift directly from the stored treated/control sklearn models. No custom class
unpickling occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.config import CONTROL_LABEL, TREATMENT_LABELS


@dataclass
class PolicyInputs:
    budget: float
    cost_mens: float
    cost_womens: float
    ltv_value: float
    max_volume: Optional[int] = None
    objective: str = "retained_customers"  # or "profit"


def _action_cost(action: str, cost_mens: float, cost_womens: float) -> float:
    if action == "Mens E-Mail":
        return cost_mens
    if action == "Womens E-Mail":
        return cost_womens
    return 0.0


def score_users(recommendations: pd.DataFrame, inputs: PolicyInputs) -> pd.Series:
    if inputs.objective not in {"retained_customers", "profit"}:
        raise ValueError("objective must be 'retained_customers' or 'profit'")

    best_uplift = recommendations["best_uplift"].to_numpy(dtype=float)
    best_action = recommendations["best_action"].astype(str).to_numpy()

    costs = np.array(
        [_action_cost(a, inputs.cost_mens, inputs.cost_womens) for a in best_action],
        dtype=float,
    )

    if inputs.objective == "retained_customers":
        score = best_uplift
    else:
        score = best_uplift * float(inputs.ltv_value) - costs

    return pd.Series(score, index=recommendations.index, name="score")


def select_targets(
    df_users: pd.DataFrame, recommendations: pd.DataFrame, inputs: PolicyInputs
) -> pd.DataFrame:
    """
    Phase C: Selection logic
    1) compute score
    2) drop control + non-positive score
    3) sort by score
    4) apply max_volume
    5) apply budget via cumulative cost
    """
    rec = recommendations.copy()
    rec["score"] = score_users(rec, inputs)

    actionable = rec[(rec["best_action"] != CONTROL_LABEL) & (rec["score"] > 0)].copy()
    if actionable.empty:
        return actionable

    actionable["cost"] = actionable["best_action"].apply(
        lambda a: _action_cost(a, inputs.cost_mens, inputs.cost_womens)
    )

    actionable = actionable.sort_values("score", ascending=False)

    if inputs.max_volume is not None:
        actionable = actionable.head(int(inputs.max_volume))

    actionable["cum_cost"] = actionable["cost"].cumsum()
    selected = actionable[actionable["cum_cost"] <= float(inputs.budget)].copy()

    if "customer_id" in df_users.columns:
        selected = selected.join(df_users["customer_id"], how="left")

    return selected


def recommend_from_payload(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    """
    Phase D: Compute best action + best uplift using ONLY payload contents.

    Payload structure:
    - preprocessor
    - treatment_models: {label: {"treated": sklearn_model, "control": sklearn_model}}

    Steps:
    1) transform X using preprocessor
    2) for each treatment: tau = p1 - p0
    3) choose max tau, but fall back to control if max tau <= 0
    """
    from src.config import FEATURE_COLS

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

    rec = pd.DataFrame(
        {"best_action": best_action, "best_uplift": best_uplift}, index=df.index
    )
    return rec


if __name__ == "__main__":
    from src.utils.io import load_joblib
    from src.config import ARTIFACTS_DIR

    val = pd.read_csv("data/processed/val.csv")

    payload = load_joblib(ARTIFACTS_DIR / "uplift_model.joblib")
    rec = recommend_from_payload(val, payload)

    inputs = PolicyInputs(
        budget=200.0,
        cost_mens=0.02,
        cost_womens=0.02,
        ltv_value=1.0,
        max_volume=5000,
        objective="retained_customers",
    )

    selected = select_targets(val, rec, inputs)
    print("Selected:", len(selected))
    print(selected["best_action"].value_counts())
    print("Total cost:", float(selected["cost"].sum()) if len(selected) else 0.0)
