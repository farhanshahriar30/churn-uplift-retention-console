"""
Phase A: Goal
Train uplift models for multiple treatments vs a shared control:
- Mens E-Mail vs No E-Mail
- Womens E-Mail vs No E-Mail

Then, for each user x:
- compute uplift for each treatment
- pick the best treatment if the best uplift > 0, otherwise pick control

Phase B: Realism improvement (probability calibration)
Uplift is computed as p1(x) - p0(x). If p1/p0 are poorly calibrated, uplift can
become unrealistically large. To reduce this, we calibrate probabilities using
CalibratedClassifierCV with sigmoid (Platt scaling) on each group separately.

Phase C: Artifact saving
We save a plain dict payload:
- preprocessor
- feature_names
- for each treatment: calibrated treated/control models
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from src.data.treatment import filter_binary_task
from src.features.build import (
    fit_feature_pipeline,
    transform_with_pipeline,
    FeatureBundle,
)
from src.models.uplift.t_learner_xgb import fit_xgb_t_learner, XGBTlearnerModel
from src.config import TREATMENT_LABELS, CONTROL_LABEL, RANDOM_SEED


@dataclass
class MultiTreatmentUpliftModel:
    bundle: FeatureBundle
    models: dict[str, XGBTlearnerModel]

    def predict_uplifts(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X, _ = transform_with_pipeline(self.bundle, df)
        return {label: m.predict_uplift(X) for label, m in self.models.items()}

    def recommend_action(self, df: pd.DataFrame) -> pd.DataFrame:
        uplifts = self.predict_uplifts(df)

        tau_stack = np.vstack([uplifts[label] for label in TREATMENT_LABELS]).T
        best_idx = np.argmax(tau_stack, axis=1)
        best_uplift = tau_stack[np.arange(len(df)), best_idx]

        best_action = np.array([TREATMENT_LABELS[i] for i in best_idx], dtype=object)
        best_action = np.where(best_uplift > 0, best_action, CONTROL_LABEL)

        out = pd.DataFrame(
            {"best_action": best_action, "best_uplift": best_uplift}, index=df.index
        )

        for label in TREATMENT_LABELS:
            col = "uplift_" + label.lower().replace(" ", "_").replace("-", "_")
            out[col] = uplifts[label]

        return out


def _fit_and_calibrate_group(Xg, yg, base_estimator) -> object:
    """
    Phase D: Calibrate probabilities for a single group.

    We use sigmoid calibration with CV. This avoids deprecated/removed 'prefit'
    behavior and works across modern scikit-learn versions.

    If a group is too small or has too few positives, calibration can be unstable,
    so we fall back to the base estimator.
    """
    n = len(yg)
    pos = int(np.sum(yg))
    if n < 300 or pos < 5:
        return base_estimator

    # CalibratedClassifierCV will clone and refit the estimator internally.
    # Using cv=3 keeps it reasonably fast.
    calib = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=3)
    calib.fit(Xg, yg)
    return calib


def fit_multitreatment_tlearner(df_train: pd.DataFrame) -> MultiTreatmentUpliftModel:
    """
    Phase E: Training logic (with calibration)

    1) Fit ONE shared feature pipeline on the full training set.
    2) For each treatment label:
       - filter to {control, treatment}
       - transform to X,y,t
       - split into fit/cal subsets (keeps balance across t and y)
       - fit base T-learner on fit subset
       - calibrate treated and control models on their respective fit-subset rows
         (calibration uses CV and refits internally)
    """
    bundle = fit_feature_pipeline(df_train)
    models: dict[str, XGBTlearnerModel] = {}

    for label in TREATMENT_LABELS:
        task = filter_binary_task(df_train, label)
        X_all, y_all = transform_with_pipeline(bundle, task)
        t_all = task["T"].to_numpy(dtype=int)
        y_np = y_all.to_numpy()

        # Stratify by treatment + outcome so both are represented in both splits
        strat = t_all.astype(str) + "_" + y_np.astype(str)
        idx = np.arange(len(task))

        idx_fit, _idx_unused = train_test_split(
            idx,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=strat,
        )

        X_fit, y_fit, t_fit = X_all[idx_fit], y_np[idx_fit], t_all[idx_fit]

        # Phase E1: fit base T-learner on fit subset
        base = fit_xgb_t_learner(X_fit, y_fit, t_fit)

        # Phase E2: calibrate treated and control models separately (on their own group rows)
        treat_mask = t_fit == 1
        ctrl_mask = t_fit == 0

        m1 = _fit_and_calibrate_group(
            X_fit[treat_mask], y_fit[treat_mask], base.model_treated
        )
        m0 = _fit_and_calibrate_group(
            X_fit[ctrl_mask], y_fit[ctrl_mask], base.model_control
        )

        models[label] = XGBTlearnerModel(model_treated=m1, model_control=m0)

    return MultiTreatmentUpliftModel(bundle=bundle, models=models)


def to_payload(mt: MultiTreatmentUpliftModel) -> dict:
    """
    Phase F: Artifact payload (plain dict)
    Stored models are already calibrated wrappers (or base estimators if fallback).
    Both expose predict_proba, so downstream code stays unchanged.
    """
    return {
        "preprocessor": mt.bundle.preprocessor,
        "feature_names": mt.bundle.feature_names,
        "treatment_models": {
            label: {
                "treated": mt.models[label].model_treated,
                "control": mt.models[label].model_control,
            }
            for label in mt.models.keys()
        },
    }


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")

    mt = fit_multitreatment_tlearner(train)
    rec = mt.recommend_action(val)

    print(rec["best_action"].value_counts())
    print(
        "best_uplift summary:",
        float(rec["best_uplift"].min()),
        float(rec["best_uplift"].mean()),
        float(rec["best_uplift"].max()),
    )

    from src.config import ARTIFACTS_DIR
    from src.utils.io import save_joblib

    payload = to_payload(mt)
    save_joblib(payload, ARTIFACTS_DIR / "uplift_model.joblib")
    print("Saved:", ARTIFACTS_DIR / "uplift_model.joblib")
