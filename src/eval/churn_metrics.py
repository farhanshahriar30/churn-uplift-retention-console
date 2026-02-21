"""
Phase A: Goal
For probability models, AUC is not the whole story.
We also want:
- calibration quality (are predicted probabilities meaningful?)
- lift (do top-scored users actually convert at higher rates?)

Phase B: What we compute
1) Brier score: mean squared error of probability forecasts.
   Lower is better; it captures calibration + sharpness.

2) Lift@k: conversion rate in top k% ranked by predicted probability,
   divided by overall conversion rate.

This tells us whether the model concentrates converters near the top.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.config import OUTCOME_CONVERSION


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float) -> dict:
    """
    Phase C: Lift@k
    - Sort by predicted probability (descending)
    - Take top k% rows
    - Compare their conversion rate to overall rate
    """
    n = len(y_true)
    cut = max(1, int(n * k))
    order = np.argsort(-y_prob)
    top_idx = order[:cut]

    base = float(np.mean(y_true))
    top_rate = float(np.mean(y_true[top_idx]))
    lift = float(top_rate / base) if base > 0 else float("nan")

    return {
        "k": k,
        "top_rate": top_rate,
        "base_rate": base,
        "lift": lift,
        "n_top": int(cut),
    }


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Phase D: Basic calibration metric
    Brier score is the most common simple calibration error measure.
    """
    return {"brier": float(brier_score_loss(y_true, y_prob))}


def evaluate_split(df: pd.DataFrame, y_prob: np.ndarray) -> dict:
    y_true = df[OUTCOME_CONVERSION].to_numpy(dtype=int)
    out = {}
    out.update(calibration_metrics(y_true, y_prob))
    out["lift_10"] = lift_at_k(y_true, y_prob, 0.10)
    out["lift_20"] = lift_at_k(y_true, y_prob, 0.20)
    return out


if __name__ == "__main__":
    # Minimal runner: load saved probs by recomputing from model artifacts
    import joblib

    preprocess_payload = joblib.load("artifacts/preprocess.joblib")
    model = joblib.load("artifacts/churn_model.joblib")
    preprocessor = preprocess_payload["preprocessor"]

    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")

    X_val = preprocessor.transform(val.drop(columns=[OUTCOME_CONVERSION]))
    X_test = preprocessor.transform(test.drop(columns=[OUTCOME_CONVERSION]))

    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        "val": evaluate_split(val, p_val),
        "test": evaluate_split(test, p_test),
    }

    print(json.dumps(metrics, indent=2))
