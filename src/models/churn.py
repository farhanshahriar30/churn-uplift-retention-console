"""
src/models/churn.py
Baseline outcome model: predict conversion probability.

Phase A: Goal
Train a baseline model that ranks users by P(conversion=1 | X),
and also produce calibrated probabilities for UI display.

Phase B: Why calibration
Using class_weight="balanced" helps learning under class imbalance,
but raw predict_proba is often not calibrated to the true base rate.
So we fit a calibration layer (sigmoid) using CV on the training data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV

from src.config import ARTIFACTS_DIR
from src.features.build import fit_transform_all, FeatureBundle


def _ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_probs(y_true, y_prob) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def save_bundle(bundle: FeatureBundle, path: Path) -> None:
    import joblib

    payload = {
        "preprocessor": bundle.preprocessor,
        "feature_names": bundle.feature_names,
    }
    joblib.dump(payload, path)


def save_model(model, path: Path) -> None:
    import joblib

    joblib.dump(model, path)


def main() -> None:
    _ensure_artifacts_dir()

    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")

    bundle, (Xtr, ytr), (Xva, yva), (Xte, yte) = fit_transform_all(train, val, test)

    # Phase C: Base model for ranking (imbalance-aware)
    base = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )
    base.fit(Xtr, ytr)

    # Phase D: Calibrate probabilities using CV on training data
    # sigmoid = Platt scaling, usually stable for small positive rates
    calib = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    calib.fit(Xtr, ytr)

    # Evaluate both (ranking metrics should be similar; calibrated is for probability readout)
    p_val_base = base.predict_proba(Xva)[:, 1]
    p_test_base = base.predict_proba(Xte)[:, 1]

    p_val_cal = calib.predict_proba(Xva)[:, 1]
    p_test_cal = calib.predict_proba(Xte)[:, 1]

    metrics = {
        "model": "logreg_baseline",
        "val": {
            "base": evaluate_probs(yva, p_val_base),
            "calibrated": evaluate_probs(yva, p_val_cal),
        },
        "test": {
            "base": evaluate_probs(yte, p_test_base),
            "calibrated": evaluate_probs(yte, p_test_cal),
        },
    }

    print("Metrics:", json.dumps(metrics, indent=2))

    save_bundle(bundle, ARTIFACTS_DIR / "preprocess.joblib")
    save_model(base, ARTIFACTS_DIR / "churn_model.joblib")
    save_model(calib, ARTIFACTS_DIR / "churn_model_calibrated.joblib")
    (ARTIFACTS_DIR / "churn_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Saved:")
    print(" -", ARTIFACTS_DIR / "preprocess.joblib")
    print(" -", ARTIFACTS_DIR / "churn_model.joblib")
    print(" -", ARTIFACTS_DIR / "churn_model_calibrated.joblib")
    print(" -", ARTIFACTS_DIR / "churn_metrics.json")


if __name__ == "__main__":
    main()
