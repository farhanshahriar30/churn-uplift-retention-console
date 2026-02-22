"""
Phase A: Goal
Tiny helper functions to save/load Python objects as artifacts.
We use joblib because it handles sklearn objects well.
"""

from __future__ import annotations

from pathlib import Path
import joblib


def save_joblib(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path):
    return joblib.load(path)
