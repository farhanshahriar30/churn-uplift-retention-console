"""
Phase A: Goal
T-learner uplift with XGBoost:
- model treated: P(Y=1 | T=1, X)
- model control: P(Y=1 | T=0, X)
uplift = p1 - p0

Phase B: Realism strategy (compatible across XGBoost versions)
Some XGBoost sklearn wrappers (older versions) do not support early stopping
arguments like `early_stopping_rounds` or `callbacks`.

So we:
1) Use more conservative model settings to reduce overfitting/extreme probabilities.
2) Attempt early stopping only if the current XGBoost version supports it.
3) Fall back gracefully to a normal fit if not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
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
    """
    Phase C: Train treated/control models with conservative settings.

    Why this helps realism:
    - Lower tree depth + higher min_child_weight + stronger regularization
      reduce overly confident fits in rare-event settings.
    - Subsampling and column sampling also reduce overfitting.
    """
    X_treat = X[t == 1]
    y_treat = y[t == 1]
    X_ctrl = X[t == 0]
    y_ctrl = y[t == 0]

    # Phase C1: scale_pos_weight helps with imbalance: neg/pos
    def _scale_pos_weight(y_bin):
        pos = max(1, int(np.sum(y_bin)))
        neg = max(1, int(len(y_bin) - pos))
        return neg / pos

    # Phase C2: internal validation split (for early stopping if supported)
    rng = np.random.default_rng(42)

    def _train_val_split(Xg, yg, val_frac=0.15):
        n = len(yg)
        if n < 400:
            return Xg, yg, None, None  # too small -> skip validation

        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(n * val_frac)

        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        X_tr, y_tr = Xg[tr_idx], yg[tr_idx]
        X_va, y_va = Xg[val_idx], yg[val_idx]

        # Validation should contain at least a couple positives for stable early stopping
        if int(np.sum(y_va)) < 2:
            return Xg, yg, None, None

        return X_tr, y_tr, X_va, y_va

    def _build_model(spw: float) -> XGBClassifier:
        # Phase C3: conservative model to reduce extreme probabilities
        return XGBClassifier(
            n_estimators=600,  # lower ceiling than before
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=20,
            gamma=1.0,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            reg_alpha=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=spw,
            verbosity=0,
        )

    def _fit_with_optional_early_stopping(model: XGBClassifier, X_tr, y_tr, X_va, y_va):
        """
        Phase C4: Fit model.
        - If XGBoost supports early stopping args, use them.
        - Otherwise, fall back to normal fit.
        """
        fit_sig = inspect.signature(model.fit)
        supports_es = "early_stopping_rounds" in fit_sig.parameters
        supports_callbacks = "callbacks" in fit_sig.parameters

        if X_va is not None:
            if supports_es:
                # Newer sklearn API versions
                return model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                    early_stopping_rounds=50,
                )
            if supports_callbacks:
                # Some versions support callbacks, but yours does not
                # Keep here for forward-compatibility.
                from xgboost.callback import EarlyStopping  # only imported if supported

                es = EarlyStopping(rounds=50, save_best=True)
                return model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                    callbacks=[es],
                )

        # Fallback: normal fit
        return model.fit(X_tr, y_tr, verbose=False)

    def _fit_one(Xg, yg) -> XGBClassifier:
        spw = _scale_pos_weight(yg)
        model = _build_model(spw)

        X_tr, y_tr, X_va, y_va = _train_val_split(Xg, yg)

        # If no validation split, just train on full group
        if X_va is None:
            _fit_with_optional_early_stopping(model, Xg, yg, None, None)
        else:
            _fit_with_optional_early_stopping(model, X_tr, y_tr, X_va, y_va)

        return model

    m1 = _fit_one(X_treat, y_treat)
    m0 = _fit_one(X_ctrl, y_ctrl)

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
