


from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "dataset" / "processed" / "split" / "train.csv"
TEST_FILE = BASE_DIR / "dataset" / "processed" / "split" / "test.csv"

SAMPLE_SIZE = 100_000
TARGET = "Attack_label"

print("Loading samples...")

train = pd.read_csv(TRAIN_FILE, nrows=SAMPLE_SIZE)
test = pd.read_csv(TEST_FILE, nrows=SAMPLE_SIZE)

# Use features only to identify matching patterns
features = [c for c in train.columns if c != TARGET]

train_patterns = train[features].copy()
train_patterns[TARGET] = train[TARGET].values

test_patterns = test[features].copy()
test_patterns[TARGET] = test[TARGET].values

# Keep one label per feature pattern, or multiple labels if conflicting
label_counts = (
    train_patterns.groupby(features, dropna=False)[TARGET]
    .agg(["min", "max", "count"])
    .reset_index()
)

# A test pattern matches a training pattern if all feature values match
matched = test_patterns.merge(
    label_counts,
    on=features,
    how="inner"
)

print("\n=== MATCHED FEATURE PATTERNS ===")
print("Matching test rows:", len(matched))

if len(matched) > 0:
    print(
        "Matching patterns with conflicting training labels:",
        int((matched["min"] != matched["max"]).sum())
    )
    print(
        "Matching rows whose test label differs from the "
        "training pattern's single label:",
        int(
            (
                (matched["min"] == matched["max"])
                & (matched[TARGET] != matched["min"])
            ).sum()
        )
    )

print("\nSample investigation complete.")
print("No dataset files were modified.")

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "dataset" / "processed" / "split" / "train.csv"
TEST_FILE = BASE_DIR / "dataset" / "processed" / "split" / "test.csv"

TARGET = "Attack_label"
CHUNK_SIZE = 50_000

print("Starting full train/test feature-overlap check...")

# Store hashes of training feature rows, excluding the target.
train_hashes = set()
train_rows = 0

for chunk in pd.read_csv(TRAIN_FILE, chunksize=CHUNK_SIZE):
    features = chunk.drop(columns=[TARGET])
    hashes = pd.util.hash_pandas_object(features, index=False)

    train_hashes.update(int(value) for value in hashes)
    train_rows += len(chunk)

    if train_rows % 500_000 < CHUNK_SIZE:
        print(f"Training rows checked: {train_rows:,}")

print(f"\nTraining rows scanned: {train_rows:,}")
print(f"Unique training feature hashes: {len(train_hashes):,}")

# Compare every test row against the training feature hashes.
test_rows = 0
matching_rows = 0

for chunk in pd.read_csv(TEST_FILE, chunksize=CHUNK_SIZE):
    features = chunk.drop(columns=[TARGET])
    hashes = pd.util.hash_pandas_object(features, index=False)

    matching_rows += sum(
        int(value) in train_hashes for value in hashes
    )
    test_rows += len(chunk)

    if test_rows % 100_000 < CHUNK_SIZE:
        print(f"Testing rows checked: {test_rows:,}")

print("\n=== FULL OVERLAP RESULTS ===")
print(f"Training rows: {train_rows:,}")
print(f"Testing rows: {test_rows:,}")
print(f"Test rows matching training features: {matching_rows:,}")
print(
    "Matching test rows (%):",
    round(matching_rows / test_rows * 100, 3)
)

print("\nAudit complete. Dataset files were not modified.")
print("Note: this uses row hashes; hash collisions are theoretically possible.")
