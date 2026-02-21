"""
src/models/uplift/multitreatment.py

Phase A: Goal
Support multi-treatment uplift for this dataset:

Treatments:
- Mens E-Mail
- Womens E-Mail
Control:
- No E-Mail

Strategy:
- Train two separate binary uplift models (one-vs-control):
  1) Mens vs Control
  2) Womens vs Control
- For each user x, compute:
  tau_mens(x), tau_womens(x)
- Choose the best action:
  - if both <= 0 -> choose Control
  - else choose treatment with max uplift

This makes downstream policy optimization easy.
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
from src.models.uplift.t_learner import fit_t_learner, TLearnerModel
from src.config import TREATMENT_LABELS, CONTROL_LABEL, OUTCOME_CONVERSION


@dataclass
class MultiTreatmentUpliftModel:
    """
    Phase B: What we store
    - bundle: feature pipeline fitted on training data for consistency
    - models: dict mapping treatment label -> fitted binary uplift model
    """

    bundle: FeatureBundle
    models: dict[str, TLearnerModel]

    def predict_uplifts(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Phase C: Predict uplift for each treatment vs control.
        Returns: { "Mens E-Mail": tau_mens, "Womens E-Mail": tau_womens }
        """
        X, _ = transform_with_pipeline(self.bundle, df)
        return {label: m.predict_uplift(X) for label, m in self.models.items()}

    def recommend_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase D: Recommend best action per row based on predicted uplift.
        - If best uplift <= 0 -> CONTROL_LABEL
        - Else -> argmax uplift treatment
        Returns a small dataframe with:
          - best_action
          - best_uplift
          - uplift_mens, uplift_womens (columns for transparency)
        """
        uplifts = self.predict_uplifts(df)

        # Stack in a consistent label order
        tau_stack = np.vstack(
            [uplifts[label] for label in TREATMENT_LABELS]
        ).T  # shape (n, 2)
        best_idx = np.argmax(tau_stack, axis=1)
        best_uplift = tau_stack[np.arange(len(df)), best_idx]

        best_action = np.array([TREATMENT_LABELS[i] for i in best_idx], dtype=object)
        best_action = np.where(best_uplift > 0, best_action, CONTROL_LABEL)

        out = pd.DataFrame(
            {
                "best_action": best_action,
                "best_uplift": best_uplift,
            },
            index=df.index,
        )

        # Add individual treatment uplift columns for visibility
        for label in TREATMENT_LABELS:
            col = "uplift_" + label.lower().replace(" ", "_").replace("-", "_")
            out[col] = uplifts[label]

        return out


def fit_multitreatment_tlearner(df_train: pd.DataFrame) -> MultiTreatmentUpliftModel:
    """
    Phase E: Train one-vs-control T-learners for each treatment.

    Implementation detail:
    - Fit the feature pipeline ONCE using the full training data (all segments).
      This keeps feature space consistent across the two binary tasks.
    - For each treatment label:
      - filter to {control, treatment}
      - transform using the shared feature pipeline
      - train a binary T-learner uplift model
    """
    # Fit shared feature pipeline on full training data
    bundle = fit_feature_pipeline(df_train)

    models: dict[str, TLearnerModel] = {}

    for label in TREATMENT_LABELS:
        task = filter_binary_task(df_train, label)
        X, y = transform_with_pipeline(bundle, task)
        t = task["T"].to_numpy(dtype=int)
        models[label] = fit_t_learner(X, y.to_numpy(), t)

    return MultiTreatmentUpliftModel(bundle=bundle, models=models)


if __name__ == "__main__":
    # Smoke test: train multi-treatment model and recommend actions on validation set.
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

    save_joblib(mt, ARTIFACTS_DIR / "uplift_model.joblib")
    print("Saved:", ARTIFACTS_DIR / "uplift_model.joblib")
