"""
Phase A: Goal
We need clean treatment labels to support uplift modeling.

Your dataset has 3 groups in `segment`:
- No E-Mail (control)
- Mens E-Mail (treatment A)
- Womens E-Mail (treatment B)

Uplift modeling often needs either:
1) Binary treatment setup (T=1 treated vs T=0 control), or
2) Multi-treatment setup handled via "one-vs-control" wrappers.

This module provides both in a consistent way.
"""

from __future__ import annotations

import pandas as pd

from src.config import TREATMENT_COL, CONTROL_LABEL, TREATMENT_LABELS


def normalize_segment(df: pd.DataFrame) -> pd.Series:
    """
    Phase B: Normalize treatment labels (defensive cleaning).
    Ensures values match the canonical labels in config.
    """
    seg = df[TREATMENT_COL].astype(str).str.strip()
    return seg


def make_binary_treatment(df: pd.DataFrame, treat_label: str) -> pd.Series:
    """
    Phase C: Create a binary treatment indicator for "treat_label vs control".

    Returns:
    - T = 1 for rows in treat_label
    - T = 0 for rows in CONTROL_LABEL
    - Rows from the other treatment are excluded by returning NaN
      (caller should filter them out)
    """
    if treat_label not in TREATMENT_LABELS:
        raise ValueError(
            f"Unknown treat_label={treat_label}. Expected one of {TREATMENT_LABELS}"
        )

    seg = normalize_segment(df)

    t = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    t[seg == CONTROL_LABEL] = 0
    t[seg == treat_label] = 1
    return t


def filter_binary_task(df: pd.DataFrame, treat_label: str) -> pd.DataFrame:
    """
    Phase D: Filter dataframe down to only {control, treat_label}.
    Adds a 'T' column (0/1).
    """
    out = df.copy()
    out["T"] = make_binary_treatment(out, treat_label)
    out = out.dropna(subset=["T"]).copy()
    out["T"] = out["T"].astype(int)
    return out


if __name__ == "__main__":
    # Quick sanity check
    import pandas as pd

    df = pd.read_csv("data/processed/train.csv")
    for label in TREATMENT_LABELS:
        task = filter_binary_task(df, label)
        print(label, "rows:", len(task), "T mean:", task["T"].mean())
