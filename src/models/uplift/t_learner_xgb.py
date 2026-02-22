"""
Phase A: Goal
Same T-learner idea, but with XGBoost as the base learner:
- model treated: P(Y=1 | T=1, X)
- model control: P(Y=1 | T=0, X)
uplift = p1 - p0

Phase B: Why XGBoost
Even if the outcome model isn't improved by boosting,
uplift ranking can still improve because we're learning different response
functions for treated vs control.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from xgboost import XGBClassifier


@dataclass
class XGBTlearnerModel:
    model_treated: XGBClassifier
    model_control: XGBClassifier

    def predict_uplift(self, X) -> np.ndarray:
        p1 = self.model_treated.predict_proba(X)[:, 1]
        p0 = self.model_control.predict_proba(X)[:, 1]
        return p1 - p0


def fit_xgb_t_learner(X, y, t) -> XGBTlearnerModel:
    X_treat = X[t == 1]
    y_treat = y[t == 1]
    X_ctrl = X[t == 0]
    y_ctrl = y[t == 0]

    # scale_pos_weight helps with imbalance: neg/pos
    def _scale_pos_weight(y_bin):
        pos = max(1, int(y_bin.sum()))
        neg = max(1, int(len(y_bin) - pos))
        return neg / pos

    m1 = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=_scale_pos_weight(y_treat),
    )

    m0 = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=_scale_pos_weight(y_ctrl),
    )

    m1.fit(X_treat, y_treat)
    m0.fit(X_ctrl, y_ctrl)

    return XGBTlearnerModel(model_treated=m1, model_control=m0)


if __name__ == "__main__":
    # Smoke test on Mens vs control (train split)
    import pandas as pd
    from src.data.treatment import filter_binary_task
    from src.features.build import fit_feature_pipeline, transform_with_pipeline

    train = pd.read_csv("data/processed/train.csv")
    task = filter_binary_task(train, "Mens E-Mail")

    bundle = fit_feature_pipeline(task)
    X, y = transform_with_pipeline(bundle, task)
    t = task["T"].to_numpy(dtype=int)

    model = fit_xgb_t_learner(X, y.to_numpy(), t)
    tau = model.predict_uplift(X)
    print("tau summary:", float(np.min(tau)), float(np.mean(tau)), float(np.max(tau)))
