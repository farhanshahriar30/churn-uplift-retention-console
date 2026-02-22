"""
app/pages/1_Overview.py

Phase A: Goal
Show the project snapshot using saved artifacts:
- dataset_profile.json
- splits.json
- policy_simulation_val.json

This page should load instantly and confirm the pipeline ran successfully.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import streamlit as st


ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


st.header("Overview")

col1, col2, col3 = st.columns(3)

profile_path = ARTIFACTS_DIR / "dataset_profile.json"
splits_path = ARTIFACTS_DIR / "splits.json"
policy_path = ARTIFACTS_DIR / "policy_simulation_val.json"

with col1:
    st.subheader("Dataset")
    if profile_path.exists():
        prof = _load_json(profile_path)
        st.metric("Rows", f"{prof['n_rows']:,}")
        st.metric("Columns", f"{prof['n_cols']:,}")
        st.metric("Base conversion rate", f"{prof['conversion_rate']:.4f}")
        st.write("Treatment mix:")
        st.json(prof["treatment_rate_overall"])
    else:
        st.warning(
            "Missing artifacts/dataset_profile.json (run: python -m src.data.preprocess)"
        )

with col2:
    st.subheader("Splits")
    if splits_path.exists():
        splits = _load_json(splits_path)
        st.write("Train/Val/Test sizes:")
        st.json(
            {
                "train_n": splits["train"]["n"],
                "val_n": splits["val"]["n"],
                "test_n": splits["test"]["n"],
            }
        )
        st.write("Conversion rates:")
        st.json(
            {
                "train": splits["train"]["conversion_rate"],
                "val": splits["val"]["conversion_rate"],
                "test": splits["test"]["conversion_rate"],
            }
        )
    else:
        st.warning("Missing artifacts/splits.json (run: python -m src.data.preprocess)")

with col3:
    st.subheader("Latest Policy Simulation (Val)")
    if policy_path.exists():
        sim = _load_json(policy_path)
        st.metric("Budget ($)", f"{sim['inputs']['budget']:.2f}")
        st.metric("Emails selected", f"{sim['uplift']['n_selected']:,}")
        st.metric(
            "Expected incremental conversions (uplift)",
            f"{sim['uplift']['expected_incremental_conversions']:.1f}",
        )
        st.metric(
            "Cost per incremental conversion",
            f"{sim['uplift']['cost_per_incremental']:.4f}",
        )

        st.write("Strategy comparison (expected incremental conversions):")
        st.bar_chart(
            {
                "uplift": sim["uplift"]["expected_incremental_conversions"],
                "risk": sim["risk"]["expected_incremental_conversions"],
                "random": sim["random"]["expected_incremental_conversions"],
            }
        )
    else:
        st.warning(
            "Missing artifacts/policy_simulation_val.json (run: python -m src.eval.policy_metrics)"
        )

st.divider()
st.subheader("Artifacts present")

# Simple artifact checklist
artifact_files = [
    "dataset_profile.json",
    "splits.json",
    "churn_model.joblib",
    "preprocess.joblib",
    "uplift_model.joblib",
    "policy_simulation_val.json",
]

present = []
missing = []
for f in artifact_files:
    if (ARTIFACTS_DIR / f).exists():
        present.append(f)
    else:
        missing.append(f)

st.write("✅ Present:", present)
if missing:
    st.write("⚠️ Missing:", missing)
