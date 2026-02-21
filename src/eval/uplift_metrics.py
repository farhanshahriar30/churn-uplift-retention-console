"""
Phase A: Goal
Evaluate how good a predicted uplift score is.

We need evaluation that answers:
- If we target users with the highest predicted uplift first,
  do we actually get more conversions compared to random?

Two common outputs:
1) Uplift-by-decile table:
   - rank users by predicted uplift
   - within each decile, compare conversion in treated vs control
2) Qini / AUUC-style curve summary:
   - cumulative incremental conversions vs population targeted

Phase B: Important note (how we estimate incremental conversions)
For each prefix of the ranked list, we estimate:
  incremental = (#treated * rate_treated) - (#treated * rate_control)
That is:
  incremental = n_t * (p_t - p_c)
This is a standard way to visualize the benefit of targeting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def uplift_by_decile(
    df: pd.DataFrame, uplift_scores: np.ndarray, outcome_col: str = "conversion"
) -> pd.DataFrame:
    """
    Phase C: Uplift-by-decile table
    - df must contain columns: T (0/1), outcome_col (0/1)
    - uplift_scores aligns with df rows

    Output columns:
    - decile (1=highest uplift)
    - n, n_treated, n_control
    - treated_rate, control_rate
    - uplift = treated_rate - control_rate
    """
    tmp = df.copy()
    tmp = tmp.assign(uplift=uplift_scores)

    # Rank by uplift descending and split into 10 equal buckets
    tmp = tmp.sort_values("uplift", ascending=False).reset_index(drop=True)
    tmp["decile"] = (tmp.index * 10 // len(tmp)) + 1  # 1..10, 1 = highest uplift
    tmp["decile"] = tmp["decile"].clip(upper=10)

    rows = []
    for d in range(1, 11):
        g = tmp[tmp["decile"] == d]
        treated = g[g["T"] == 1]
        control = g[g["T"] == 0]

        treated_rate = treated[outcome_col].mean() if len(treated) else np.nan
        control_rate = control[outcome_col].mean() if len(control) else np.nan

        rows.append(
            {
                "decile": d,
                "n": int(len(g)),
                "n_treated": int(len(treated)),
                "n_control": int(len(control)),
                "treated_rate": float(treated_rate)
                if treated_rate == treated_rate
                else np.nan,
                "control_rate": float(control_rate)
                if control_rate == control_rate
                else np.nan,
                "uplift": float(treated_rate - control_rate)
                if (treated_rate == treated_rate and control_rate == control_rate)
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def qini_curve(
    df: pd.DataFrame, uplift_scores: np.ndarray, outcome_col: str = "conversion"
) -> pd.DataFrame:
    """
    Phase D: Qini-style cumulative incremental conversions curve.

    Steps:
    1) Sort by predicted uplift descending.
    2) Walk down the list; at each step compute:
       incremental_conversions = n_treated_so_far * (rate_treated_so_far - rate_control_so_far)
    3) Return a curve over population fraction.

    Output:
    - frac: fraction of population included (0..1)
    - incremental: estimated cumulative incremental conversions
    """
    tmp = df.copy().assign(uplift=uplift_scores)
    tmp = tmp.sort_values("uplift", ascending=False).reset_index(drop=True)

    t = tmp["T"].to_numpy(dtype=int)
    y = tmp[outcome_col].to_numpy(dtype=int)

    inc = []
    frac = []

    n = len(tmp)
    treated_count = 0
    control_count = 0
    treated_y = 0
    control_y = 0

    for i in range(n):
        if t[i] == 1:
            treated_count += 1
            treated_y += y[i]
        else:
            control_count += 1
            control_y += y[i]

        # rates so far (avoid div by 0)
        rt = treated_y / treated_count if treated_count > 0 else 0.0
        rc = control_y / control_count if control_count > 0 else 0.0

        incremental = treated_count * (rt - rc)
        inc.append(incremental)
        frac.append((i + 1) / n)

    return pd.DataFrame({"frac": frac, "incremental": inc})


def auuc(curve: pd.DataFrame) -> float:
    """
    Phase E: AUUC proxy via trapezoidal rule on incremental-vs-frac curve.
    Higher is better.
    """
    x = curve["frac"].to_numpy()
    y = curve["incremental"].to_numpy()
    return float(np.trapezoid(y, x))


if __name__ == "__main__":
    # Smoke test: evaluate Mens uplift on validation split.
    import pandas as pd

    from src.data.treatment import filter_binary_task
    from src.features.build import fit_feature_pipeline, transform_with_pipeline
    from src.models.uplift.t_learner import fit_t_learner

    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")

    train_task = filter_binary_task(train, "Womens E-Mail")
    val_task = filter_binary_task(val, "Womens E-Mail")

    bundle = fit_feature_pipeline(train_task)
    Xtr, ytr = transform_with_pipeline(bundle, train_task)
    Xva, yva = transform_with_pipeline(bundle, val_task)

    t_tr = train_task["T"].to_numpy(dtype=int)
    uplift_model = fit_t_learner(Xtr, ytr.to_numpy(), t_tr)

    tau_val = uplift_model.predict_uplift(Xva)

    dec = uplift_by_decile(val_task, tau_val, outcome_col="conversion")
    curve = qini_curve(val_task, tau_val, outcome_col="conversion")
    print("AUUC:", auuc(curve))
    print(dec)
