"""
src/models/uplift/multitreatment.py

Phase A: Goal
Train uplift models for multiple treatments vs a shared control:
- Mens E-Mail vs No E-Mail
- Womens E-Mail vs No E-Mail

Then, for each user x:
- compute uplift for each treatment
- pick the best treatment if the best uplift > 0, otherwise pick control

Phase B: Important implementation choice (artifact saving)
We do NOT save a MultiTreatmentUpliftModel object to disk because pickling custom
classes can break depending on import paths.
Instead, we save a plain dict (payload) containing:
- fitted preprocessor
- feature_names
- for each treatment: the treated and control sklearn models
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.data.treatment import filter_binary_task
from src.features.build import (
    fit_feature_pipeline,
    transform_with_pipeline,
    FeatureBundle,
)
from src.models.uplift.t_learner_xgb import fit_xgb_t_learner
from src.config import TREATMENT_LABELS, CONTROL_LABEL


@dataclass
class MultiTreatmentUpliftModel:
    bundle: FeatureBundle
    models: dict[str, object]

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


def fit_multitreatment_tlearner(df_train: pd.DataFrame) -> MultiTreatmentUpliftModel:
    """
    Phase C: Training logic
    - Fit ONE shared feature pipeline on the full training set
    - For each treatment label:
      - filter to {control, treatment}
      - transform using the shared pipeline
      - fit a T-learner (two outcome models: treated vs control)
    """
    bundle = fit_feature_pipeline(df_train)
    models: dict[str, object] = {}

    for label in TREATMENT_LABELS:
        task = filter_binary_task(df_train, label)
        X, y = transform_with_pipeline(bundle, task)
        t = task["T"].to_numpy(dtype=int)
        # We store treated/control sklearn models inside the payload, so we don't need a TLearner class here.
        models[label] = fit_xgb_t_learner(X, y.to_numpy(), t)

    return MultiTreatmentUpliftModel(bundle=bundle, models=models)


def to_payload(mt: MultiTreatmentUpliftModel) -> dict:
    """
    Phase D: Artifact payload (plain dict only)
    We store only joblib-safe objects (sklearn estimators + simple python types).
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
