import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "hillstrom.csv"


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {RAW_PATH}")
    return pd.read_csv(RAW_PATH)


if __name__ == "__main__":
    df = load_raw()
    print("Loaded:", RAW_PATH)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(3))
