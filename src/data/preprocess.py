"""
src/data/preprocess.py

Phase A: What this module does
- Loads the raw Hillstrom dataset (your email RCT dataset).
- Creates a lightweight dataset profile (base rates + missingness).
- Creates reproducible train/val/test splits (no leakage).
- Saves:
  - artifacts/dataset_profile.json
  - artifacts/splits.json
  - data/processed/train.csv, val.csv, test.csv
"""

# Phase B: Imports and configuration
# - json/pathlib: save metadata files in a reproducible way
# - pandas: data table operations
# - train_test_split: deterministic splitting with stratification
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# We keep "settings" in config.py so paths/columns are not duplicated all over the repo.
from src.config import (
    ARTIFACTS_DIR,
    DATA_PROCESSED_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
    TREATMENT_COL,
    CONTROL_LABEL,
    TREATMENT_LABELS,
    OUTCOME_CONVERSION,
    FEATURE_COLS,
)

# Loader that reads data/raw/hillstrom.csv
from src.data.load import load_raw


# Phase C: Helper utilities
def _ensure_dirs() -> None:
    """
    Phase C1: Ensure output directories exist.

    We write two kinds of outputs:
    1) artifacts/  -> metadata JSON files (profile + split info)
    2) data/processed/ -> the split datasets (train/val/test)

    mkdir(..., exist_ok=True) means "create if missing, don't error if already there".
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def dataset_profile(df: pd.DataFrame) -> dict:
    """
    Phase C2: Create a dataset profile (summary stats you can show in the UI/README).

    What we capture:
    - n_rows, n_cols, columns
    - missing values per column
    - overall conversion rate (primary outcome base rate)
    - treatment group proportions (Mens/Womens/No Email)
    - feature/treatment/outcome column names (for reproducibility)
    """
    profile = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_by_col": {c: int(df[c].isna().sum()) for c in df.columns},
        "conversion_rate": float(df[OUTCOME_CONVERSION].mean()),
        "treatment_rate_overall": {
            label: float((df[TREATMENT_COL] == label).mean())
            for label in [CONTROL_LABEL, *TREATMENT_LABELS]
        },
        "feature_cols": FEATURE_COLS,
        "treatment_col": TREATMENT_COL,
        "outcome_col": OUTCOME_CONVERSION,
    }
    return profile


def make_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Phase C3: Create reproducible train/val/test splits.

    Key design choices:
    1) We add a stable `customer_id` since Hillstrom doesn't include one.
       This makes it easier to reference users in the UI tables later.

    2) We stratify the split by BOTH:
       - treatment group (segment)
       - outcome (conversion)

       Why?
       - If we split randomly without stratification, the proportions of
         'Mens E-Mail' vs 'Womens E-Mail' vs 'No E-Mail' might drift slightly
         across train/val/test.
       - Also, conversion is sparse. Stratification keeps conversion rates stable.

    3) We do the split in two steps:
       - Step 1: trainval vs test
       - Step 2: train vs val (inside trainval)

       This is a common pattern and avoids leakage issues.
    """
    df = df.copy()

    # Phase C3.1: create a stable id (0..N-1)
    df["customer_id"] = range(len(df))

    # Phase C3.2: stratification key
    # Example: "Mens E-Mail_1" or "No E-Mail_0"
    strat = df[TREATMENT_COL].astype(str) + "_" + df[OUTCOME_CONVERSION].astype(str)

    # Phase C3.3: split off the test set
    df_trainval, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=strat,
    )

    # Phase C3.4: split trainval into train and validation
    strat_tv = (
        df_trainval[TREATMENT_COL].astype(str)
        + "_"
        + df_trainval[OUTCOME_CONVERSION].astype(str)
    )

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=VAL_SIZE,  # applied on trainval portion
        random_state=RANDOM_SEED,
        stratify=strat_tv,
    )

    # Phase C3.5: build split metadata for reproducibility and reporting
    def _rates(d: pd.DataFrame) -> dict:
        return {
            "n": int(len(d)),
            "conversion_rate": float(d[OUTCOME_CONVERSION].mean()),
            "treatment_rate": {
                label: float((d[TREATMENT_COL] == label).mean())
                for label in [CONTROL_LABEL, *TREATMENT_LABELS]
            },
        }

    splits_meta = {
        "seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "val_size_on_trainval": VAL_SIZE,
        "train": _rates(df_train),
        "val": _rates(df_val),
        "test": _rates(df_test),
    }

    return df_train, df_val, df_test, splits_meta


# Phase D: Main entrypoint
def save_profile_and_splits() -> None:
    """
    Phase D1: Orchestrator function that runs everything and saves outputs.

    Steps:
    1) Ensure output dirs exist
    2) Load raw data
    3) Compute profile -> save JSON
    4) Make splits -> save splits JSON + save split CSV files
    """
    _ensure_dirs()

    # Step 2: Load raw data
    df = load_raw()

    # Step 3: dataset profile -> JSON
    prof = dataset_profile(df)
    (ARTIFACTS_DIR / "dataset_profile.json").write_text(json.dumps(prof, indent=2))

    # Step 4: splits -> JSON + CSV files
    train, val, test, meta = make_splits(df)
    (ARTIFACTS_DIR / "splits.json").write_text(json.dumps(meta, indent=2))

    # Save split datasets (CSV for now; later we can switch to Parquet for speed)
    train.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(DATA_PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(DATA_PROCESSED_DIR / "test.csv", index=False)

    # Friendly terminal output so you know what was produced
    print("Saved:")
    print(" -", ARTIFACTS_DIR / "dataset_profile.json")
    print(" -", ARTIFACTS_DIR / "splits.json")
    print(" -", DATA_PROCESSED_DIR / "train.csv")
    print(" -", DATA_PROCESSED_DIR / "val.csv")
    print(" -", DATA_PROCESSED_DIR / "test.csv")


if __name__ == "__main__":
    # Phase D2: Running this file directly will generate the artifacts and splits.
    save_profile_and_splits()
