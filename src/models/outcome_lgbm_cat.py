"""
src/models/outcome_lgbm_cat.py

Phase A: Goal
Train LightGBM on the original dataframe features and let it handle categoricals
natively (instead of one-hot). This often works better for tree boosting.

Phase B: Key idea
- Convert categorical columns to pandas 'category'
- Pass categorical_feature list to LightGBM fit
"""

from __future__ import annotations

import json
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import ARTIFACTS_DIR, FEATURE_COLS, OUTCOME_CONVERSION
from src.utils.io import save_joblib


CAT_COLS = ["history_segment", "zip_code", "channel"]


def _prep_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    return X


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

    Xtr = _prep_X(train)
    ytr = train[OUTCOME_CONVERSION].astype(int)

    Xva = _prep_X(val)
    yva = val[OUTCOME_CONVERSION].astype(int)

    Xte = _prep_X(test)
    yte = test[OUTCOME_CONVERSION].astype(int)

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        is_unbalance=True,
    )

    model.fit(Xtr, ytr, categorical_feature=CAT_COLS)

    p_val = model.predict_proba(Xva)[:, 1]
    p_test = model.predict_proba(Xte)[:, 1]

    out = {
        "model": "lightgbm_outcome_categorical",
        "val": _metrics(yva, p_val),
        "test": _metrics(yte, p_test),
    }

    print(json.dumps(out, indent=2))

    save_joblib(model, ARTIFACTS_DIR / "outcome_lgbm_cat.joblib")
    (ARTIFACTS_DIR / "outcome_lgbm_cat_metrics.json").write_text(
        json.dumps(out, indent=2)
    )
    print("Saved:", ARTIFACTS_DIR / "outcome_lgbm_cat.joblib")
    print("Saved:", ARTIFACTS_DIR / "outcome_lgbm_cat_metrics.json")


if __name__ == "__main__":
    main()
