"""
Baseline outcome model: predict conversion probability.

Phase A: Goal
Train a simple, strong baseline model that outputs:
  P(conversion = 1 | X)

We'll start with Logistic Regression because:
- it's fast
- probabilities are usually reasonable
- it's a good baseline to compare against more complex models later

Phase B: Training flow
1) Load processed splits (train/val/test)
2) Build/fit the feature pipeline on train
3) Train logistic regression on transformed train features
4) Evaluate on val/test with ROC-AUC + PR-AUC
5) Save the trained artifacts (model + preprocessor + feature names)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import ARTIFACTS_DIR
from src.features.build import fit_transform_all, FeatureBundle


def _ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def train_logreg(X_train, y_train) -> LogisticRegression:
    """
    Phase C: Model training details
    - We use class_weight="balanced" because conversion can be imbalanced.
    - max_iter increased to ensure convergence.
    """
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_probs(y_true, y_prob) -> dict:
    """
    Phase D: Basic probability metrics
    - ROC-AUC: ranking quality overall
    - PR-AUC: more informative when positives are rare
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def save_bundle(bundle: FeatureBundle, path: Path) -> None:
    """
    Phase E: Persist the preprocessor + feature names.
    We store both because the model expects the *transformed* feature matrix.
    """
    import joblib

    payload = {
        "preprocessor": bundle.preprocessor,
        "feature_names": bundle.feature_names,
    }
    joblib.dump(payload, path)


def save_model(model: LogisticRegression, path: Path) -> None:
    import joblib

    joblib.dump(model, path)


def main() -> None:
    _ensure_artifacts_dir()

    # Phase F: Load processed splits
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")

    # Phase G: Features -> fit on train, transform all splits
    bundle, (Xtr, ytr), (Xva, yva), (Xte, yte) = fit_transform_all(train, val, test)

    # Phase H: Train baseline model
    model = train_logreg(Xtr, ytr)

    # Phase I: Evaluate on val/test
    p_val = model.predict_proba(Xva)[:, 1]
    p_test = model.predict_proba(Xte)[:, 1]

    metrics = {
        "val": evaluate_probs(yva, p_val),
        "test": evaluate_probs(yte, p_test),
        "model": "logreg_baseline",
    }

    print("Metrics:", json.dumps(metrics, indent=2))

    # Phase J: Save artifacts (local)
    save_bundle(bundle, ARTIFACTS_DIR / "preprocess.joblib")
    save_model(model, ARTIFACTS_DIR / "churn_model.joblib")
    (ARTIFACTS_DIR / "churn_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Saved:")
    print(" -", ARTIFACTS_DIR / "preprocess.joblib")
    print(" -", ARTIFACTS_DIR / "churn_model.joblib")
    print(" -", ARTIFACTS_DIR / "churn_metrics.json")


if __name__ == "__main__":
    main()
