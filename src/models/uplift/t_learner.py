"""
Phase A: Goal
Estimate individual uplift for a binary treatment:
  tau(x) = P(Y=1 | T=1, X=x) - P(Y=1 | T=0, X=x)

We do this with a T-learner:
- Train one outcome model on treated users only  -> p1(x)
- Train one outcome model on control users only  -> p0(x)
- Uplift is the difference p1(x) - p0(x)

Phase B: Key design choices
- Outcome is conversion (0/1).
- Models are Logistic Regression for a strong, fast baseline.
- This file is binary-only (one treatment vs control). Multi-treatment comes next.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class TLearnerModel:
    """
    Phase C: What we store
    - model_treated: outcome model trained on T=1 rows
    - model_control: outcome model trained on T=0 rows
    """

    model_treated: LogisticRegression
    model_control: LogisticRegression

    def predict_uplift(self, X) -> np.ndarray:
        """
        Phase D: Produce tau(x) for each row in X.
        """
        p1 = self.model_treated.predict_proba(X)[:, 1]
        p0 = self.model_control.predict_proba(X)[:, 1]
        return p1 - p0

    def predict_p1_p0(self, X) -> tuple[np.ndarray, np.ndarray]:
        p1 = self.model_treated.predict_proba(X)[:, 1]
        p0 = self.model_control.predict_proba(X)[:, 1]
        return p1, p0


def fit_t_learner(X, y, t) -> TLearnerModel:
    """
    Phase E: Train the two outcome models.
    Inputs:
    - X: transformed feature matrix
    - y: outcome (0/1)
    - t: treatment indicator (0/1)

    Returns:
    - TLearnerModel capable of predicting uplift for new X
    """
    X_treat = X[t == 1]
    y_treat = y[t == 1]
    X_ctrl = X[t == 0]
    y_ctrl = y[t == 0]

    # Logistic Regression baseline
    m1 = LogisticRegression(max_iter=2000, class_weight="balanced")
    m0 = LogisticRegression(max_iter=2000, class_weight="balanced")

    m1.fit(X_treat, y_treat)
    m0.fit(X_ctrl, y_ctrl)

    return TLearnerModel(model_treated=m1, model_control=m0)


if __name__ == "__main__":
    # Smoke test: train Mens vs Control uplift on train split only.
    import pandas as pd

    from src.data.treatment import filter_binary_task
    from src.features.build import fit_feature_pipeline, transform_with_pipeline
    from src.config import OUTCOME_CONVERSION

    train = pd.read_csv("data/processed/train.csv")

    # Build binary task for Mens (change to "Womens E-Mail" if you want)
    task = filter_binary_task(train, "Mens E-Mail")

    bundle = fit_feature_pipeline(task)
    X, y = transform_with_pipeline(bundle, task)
    t = task["T"].to_numpy(dtype=int)

    uplift_model = fit_t_learner(X, y.to_numpy(), t)
    tau = uplift_model.predict_uplift(X)

    print(
        "tau summary:",
        "min",
        float(np.min(tau)),
        "mean",
        float(np.mean(tau)),
        "max",
        float(np.max(tau)),
    )
