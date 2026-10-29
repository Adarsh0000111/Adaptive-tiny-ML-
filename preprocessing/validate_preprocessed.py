from pathlib import Path

import numpy as np
import pandas as pd

from pathlib import Path



PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR
    / "dataset"
    / "processed"
    / "ids_numeric.csv"
)

print("Validating processed dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("\n=== DATASET SHAPE ===")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\n=== TARGET DISTRIBUTION ===")
print(df["Attack_label"].value_counts(dropna=False).sort_index())

print("\n=== MISSING VALUES ===")
missing = df.isna().sum()
print(missing[missing > 0] if missing.any() else "None")

print("\n=== INFINITE VALUES ===")
numeric_df = df.select_dtypes(include=[np.number])
infinite_counts = np.isinf(numeric_df).sum()
print(
    infinite_counts[infinite_counts > 0]
    if (infinite_counts > 0).any()
    else "None"
)

print("\n=== DUPLICATE ROWS ===")
print(f"Duplicate rows: {df.duplicated().sum():,}")
print("\n=== DUPLICATE TARGET CHECK ===")

duplicate_mask = df.duplicated(keep=False)

print(
    df.loc[duplicate_mask, "Attack_label"]
    .value_counts()
    .sort_index()
)

print("\n=== DUPLICATE EXAMPLES ===")
print(
    df.loc[duplicate_mask]
    .head(10)
    .to_string(index=False)
)

print("\n=== TARGET VALIDITY ===")
print("Only 0 and 1:", set(df["Attack_label"].dropna().unique()).issubset({0, 1}))

print("\nValidation inspection complete.")
