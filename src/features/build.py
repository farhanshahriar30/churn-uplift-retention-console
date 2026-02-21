"""
Phase A: Purpose
- Build a reproducible feature pipeline:
  - numeric passthrough
  - one-hot encode categorical features
- Provide helper functions to fit on train, transform val/test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLS, OUTCOME_CONVERSION


# Phase B: We keep this small and explicit so we can save/load it later.
@dataclass
class FeatureBundle:
    preprocessor: ColumnTransformer
    feature_names: list[str]


def _split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS].copy()
    y = df[OUTCOME_CONVERSION].astype(int)
    return X, y
