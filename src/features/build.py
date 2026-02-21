"""
src/features/build.py

Phase A: What this module does
We want ONE consistent feature-processing pipeline that:
1) Fits only on the training split (to avoid leakage).
2) Applies the same transformations to val/test (and later to new data).
3) Produces a numeric feature matrix that ML models can learn from.

For this dataset:
- Numeric features: recency, history, mens, womens, newbie
- Categorical features: history_segment, zip_code, channel
We will one-hot encode categorical columns and pass numeric columns through.

Phase B: Why we do it this way (key ideas)
- OneHotEncoder(handle_unknown="ignore"):
  When val/test contains a category unseen during training, we don't crash.
  Instead, that category becomes all-zeros in the encoded vector.

- ColumnTransformer:
  Lets us apply different preprocessing to different column subsets cleanly.

- Feature names:
  We store the expanded feature names (after one-hot encoding) so later
  we can debug models and understand what each coefficient/importance refers to.

Note:
- We intentionally do NOT include the treatment column ('segment') here.
  Treatment is used in uplift modeling; features X should be pre-treatment attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLS, OUTCOME_CONVERSION


# Phase C: We keep this small and explicit so we can save/load it later.
@dataclass
class FeatureBundle:
    """
    What we store
    - preprocessor: the fitted ColumnTransformer (this is the "feature machine")
    - feature_names: list of final column names after encoding (for reproducibility/debug)
    """

    preprocessor: ColumnTransformer
    feature_names: list[str]


# PHASE D: Split raw dataframe into:
def _split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    - X: only the feature columns (pre-treatment attributes)
    - y: the label we predict (conversion; primary outcome)

    Important:
    - We cast conversion to int because it's 0/1 in the CSV.
    """
    X = df[FEATURE_COLS].copy()
    y = df[OUTCOME_CONVERSION].astype(int)
    return X, y


# Phase E: Fit the preprocessing pipeline on TRAIN only.
def fit_feature_pipeline(df_train: pd.DataFrame) -> FeatureBundle:
    """
    Steps:
    1) Decide which feature columns are categorical vs numeric.
    2) Build a ColumnTransformer:
       - 'cat' branch: one-hot encode categorical cols
       - 'num' branch: passthrough numeric cols
    3) Fit the transformer on training features.
    4) Extract final expanded feature names.
    """
    # Explicitly mark categorical columns.
    categorical_cols = ["history_segment", "zip_code", "channel"]
    # Everything else in FEATURE_COLS is treated as numeric.
    numeric_cols = [c for c in FEATURE_COLS if c not in categorical_cols]

    # - OneHotEncoder converts categories to sparse indicator columns.
    # - handle_unknown="ignore" avoids errors on unseen categories at inference.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )
    # Fit on training features only.
    preprocessor.fit(df_train[FEATURE_COLS])

    # COmpute final feature names.
    # COlumnTransformer stores fitted transformers in named_transformers_.
    cat_ohe: OneHotEncoder = preprocessor.named_transformers_["cat"]

    # get_feature_names_out expands categories into names like:
    # zip_code_Suburban, channel_Web, history_segment_2) $100 - $200, ...
    cat_feature_names = list(cat_ohe.get_feature_names_out(categorical_cols))

    # Final order matches ColumnTransformer order: cat features first, then numeric passthrough.
    feature_names = cat_feature_names + numeric_cols

    return FeatureBundle(preprocessor=preprocessor, feature_names=feature_names)


def transform_with_pipeline(budle: FeatureBundle, df: pd.DataFrame):
    """
    Phase G: Transform any split (train/val/test) with a fitted FeatureBundle.

    Returns:
    - X_t: transformed numeric matrix (often sparse)
    - y: label vector
    """
    X, y = _split_X_y(df)
    X_t = budle.preprocessor.transform(X)
    return X_t, y


def fit_transform_all(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
):
    """
    Phase H: Convenience function used by training scripts.

    - Fit the pipeline on df_train
    - Transform train/val/test consistently
    """
    bundle = fit_feature_pipeline(df_train)
    X_train, y_train = transform_with_pipeline(bundle, df_train)
    X_val, y_val = transform_with_pipeline(bundle, df_val)
    X_test, y_test = transform_with_pipeline(bundle, df_test)
    return bundle, (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    # Phase I: Smoke test for local development
    # This checks that:
    # - processed splits exist
    # - pipeline fits and transforms without errors
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")

    bundle, (Xtr, ytr), (Xva, yva), (Xte, yte) = fit_transform_all(train, val, test)

    print("X_train:", Xtr.shape, "y_train:", ytr.shape)
    print("X_val:", Xva.shape, "y_val:", yva.shape)
    print("X_test:", Xte.shape, "y_test:", yte.shape)
    print("n_features:", len(bundle.feature_names))
