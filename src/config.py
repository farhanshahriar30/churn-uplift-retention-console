from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw" / "hillstrom.csv"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# Columns
ID_COL = None  # Hillstrom has no customer_id; we'll create an index-based id later
TREATMENT_COL = "segment"  # {'Mens E-Mail','Womens E-Mail','No E-Mail'}
OUTCOME_CONVERSION = "conversion"  # primary outcome
OUTCOME_SPEND = "spend"  # secondary
AUX_VISIT = "visit"

FEATURE_COLS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]

# Treatment labels (canonical)
CONTROL_LABEL = "No E-Mail"
TREATMENT_LABELS = ["Mens E-Mail", "Womens E-Mail"]

# Repro
RANDOM_SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # applied on remaining after test split (i.e., val ~= 16% overall)
