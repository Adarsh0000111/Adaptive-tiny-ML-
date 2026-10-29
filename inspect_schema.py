from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Locate the dataset
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "raw"
    / "DNN-EdgeIIoT-dataset.csv"
)

SAMPLE_SIZE = 20_000


# --------------------------------------------------
# 2. Verify that the dataset exists
# --------------------------------------------------

print("=" * 70)
print("EDGE-IIoTset FEATURE SCHEMA INSPECTION")
print("=" * 70)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}"
    )


# --------------------------------------------------
# 3. Load a limited sample
# --------------------------------------------------

print(f"\nReading the first {SAMPLE_SIZE:,} records...")

df = pd.read_csv(
    DATASET_PATH,
    nrows=SAMPLE_SIZE,
    low_memory=False
)

print("Sample loaded successfully.")

print(f"\nSample rows: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")


# --------------------------------------------------
# 4. Display column data types
# --------------------------------------------------

print("\n" + "=" * 70)
print("COLUMN DATA TYPES")
print("=" * 70)

for column in df.columns:
    print(f"{column:35} {df[column].dtype}")


# --------------------------------------------------
# 5. Show example values
# --------------------------------------------------

print("\n" + "=" * 70)
print("EXAMPLE VALUES")
print("=" * 70)

for column in df.columns:
    examples = (
        df[column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(5)
        .tolist()
    )

    print(f"\nColumn: {column}")
    print(f"Data type: {df[column].dtype}")
    print(f"Example values: {examples}")


# --------------------------------------------------
# 6. Identify columns with missing values in the sample
# --------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES IN THE SAMPLE")
print("=" * 70)

missing = df.isna().sum()

columns_with_missing = missing[missing > 0]

if columns_with_missing.empty:
    print("No missing values found in this sample.")
else:
    print(columns_with_missing.to_string())


# --------------------------------------------------
# 7. Identify possible feature columns
# --------------------------------------------------

print("\n" + "=" * 70)
print("POTENTIAL INPUT FEATURES")
print("=" * 70)

excluded_columns = {
    "Attack_label",
    "Attack_type"
}

candidate_features = [
    column
    for column in df.columns
    if column not in excluded_columns
]

print(f"Candidate feature count: {len(candidate_features)}")

for column in candidate_features:
    print(column)

print("\n" + "=" * 70)
print("SCHEMA INSPECTION COMPLETE")
print("=" * 70)
# --------------------------------------------------
# 8. Save schema inspection results
# --------------------------------------------------

REPORT_PATH = PROJECT_DIR / "schema_report.txt"

with open(REPORT_PATH, "w", encoding="utf-8") as report:
    report.write("EDGE-IIoTset SCHEMA REPORT\n")
    report.write("=" * 70 + "\n")

    report.write(f"\nSample rows: {len(df)}\n")
    report.write(f"Total columns: {len(df.columns)}\n")

    report.write("\nCOLUMN DATA TYPES\n")
    report.write("-" * 70 + "\n")
    report.write(df.dtypes.to_string())

    report.write("\n\nEXAMPLE VALUES\n")
    report.write("-" * 70 + "\n")

    for column in df.columns:
        examples = (
            df[column]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        report.write(f"\nColumn: {column}\n")
        report.write(f"Data type: {df[column].dtype}\n")
        report.write(f"Examples: {examples}\n")

    report.write("\n\nMISSING VALUES IN SAMPLE\n")
    report.write("-" * 70 + "\n")
    report.write(df.isna().sum().to_string())

print(f"\nSchema report saved to: {REPORT_PATH}")