
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "raw"
    / "DNN-EdgeIIoT-dataset.csv"
)

CHUNK_SIZE = 50_000

total_rows = 0
missing_counts = None
unique_samples = {}
numeric_min = {}
numeric_max = {}
nonfinite_counts = {}

print("Starting full-dataset feature audit...")

for chunk in pd.read_csv(
    DATASET_PATH,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    total_rows += len(chunk)

    if missing_counts is None:
        columns = chunk.columns.tolist()
        missing_counts = pd.Series(0, index=columns, dtype="int64")

        for col in columns:
            unique_samples[col] = set()
            nonfinite_counts[col] = 0

    missing_counts += chunk.isna().sum()

    for col in chunk.columns:
        # Retain at most two distinct non-missing values.
        # Two are enough to distinguish constant from nonconstant
        # columns without storing every unique value.
        if len(unique_samples[col]) < 2:
            for value in chunk[col].dropna().unique():
                unique_samples[col].add(str(value))
                if len(unique_samples[col]) >= 2:
                    break

        # Check whether values are numeric and finite.
        numeric = pd.to_numeric(chunk[col], errors="coerce")
        finite = numeric[np.isfinite(numeric)]

        nonfinite_counts[col] += int(
            (numeric.notna() & ~np.isfinite(numeric)).sum()
        )

        if not finite.empty:
            chunk_min = float(finite.min())
            chunk_max = float(finite.max())

            numeric_min[col] = min(
                numeric_min.get(col, chunk_min),
                chunk_min
            )
            numeric_max[col] = max(
                numeric_max.get(col, chunk_max),
                chunk_max
            )

print("\n" + "=" * 60)
print("FULL DATASET AUDIT RESULTS")
print("=" * 60)

print(f"Total rows scanned: {total_rows:,}")
print(f"Total columns: {len(missing_counts)}")

print("\nColumns with missing values:")
print(missing_counts[missing_counts > 0].to_string()
      or "None")

constant_columns = [
    col for col, values in unique_samples.items()
    if len(values) <= 1
]

print("\nPotential constant columns:")
print(constant_columns or "None detected")

print("\nColumns containing positive or negative infinity:")
infinite_columns = {
    col: count
    for col, count in nonfinite_counts.items()
    if count > 0
}
print(infinite_columns or "None detected")

print("\nNumeric ranges for selected target-independent columns:")

for col in [
    "tcp.len",
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
    "udp.time_delta",
    "Attack_label"
]:
    if col in numeric_min:
        print(
            f"{col}: min={numeric_min[col]}, "
            f"max={numeric_max[col]}"
        )

print("\nAudit complete. Original CSV was not modified.")
