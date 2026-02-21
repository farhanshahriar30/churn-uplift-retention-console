"""
src/models/outcome_lgbm.py

Phase A: Goal
Train a stronger outcome model to predict:
  P(conversion = 1 | X)

We keep the same feature pipeline and splits, but swap the classifier to LightGBM.

Phase B: Training flow
1) Load train/val/test splits
2) Fit feature pipeline on train, transform all
3) Train LightGBMClassifier
4) Report ROC-AUC + PR-AUC
5) Save artifacts:
   - artifacts/outcome_lgbm.joblib
   - artifacts/outcome_lgbm_metrics.json
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import ARTIFACTS_DIR
from src.features.build import fit_transform_all
from src.utils.io import save_joblib


def _metrics(y_true, y_prob) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")

    bundle, (Xtr, ytr), (Xva, yva), (Xte, yte) = fit_transform_all(train, val, test)

    # Phase C: LightGBM setup
    # Notes:
    # - class imbalance: use is_unbalance=True
    # - keep it modest for now; we can tune later
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        is_unbalance=True,
    )

    model.fit(Xtr, ytr)

    p_val = model.predict_proba(Xva)[:, 1]
    p_test = model.predict_proba(Xte)[:, 1]

    out = {
        "model": "lightgbm_outcome",
        "val": _metrics(yva, p_val),
        "test": _metrics(yte, p_test),
    }

    print(json.dumps(out, indent=2))

    save_joblib(model, ARTIFACTS_DIR / "outcome_lgbm.joblib")
    (ARTIFACTS_DIR / "outcome_lgbm_metrics.json").write_text(json.dumps(out, indent=2))
    print("Saved:", ARTIFACTS_DIR / "outcome_lgbm.joblib")
    print("Saved:", ARTIFACTS_DIR / "outcome_lgbm_metrics.json")


if __name__ == "__main__":
    main()
