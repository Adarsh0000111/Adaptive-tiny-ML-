from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "raw"
    / "DNN-EdgeIIoT-dataset.csv"
)

# Read only 20,000 rows
df = pd.read_csv(
    DATASET_PATH,
    nrows=20_000,
    low_memory=False
)

print("\n===== DATA TYPES =====")
print(df.dtypes.to_string())

print("\n===== TARGET COLUMNS =====")
for col in ["Attack_label", "Attack_type"]:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).to_string())

print("\n===== TEXT/CATEGORICAL COLUMNS =====")
for col in df.columns:
    if (
        df[col].dtype == "object"
        or isinstance(df[col].dtype, pd.CategoricalDtype)
    ):
        if col not in ["Attack_label", "Attack_type"]:
            print(f"\n{col}")
            print(df[col].dropna().astype(str).unique()[:5])

print("\n===== NUMERIC COLUMNS =====")
print(df.select_dtypes(include="number").columns.tolist())

print("\n===== MISSING VALUES IN SAMPLE =====")
missing = df.isna().sum()
print(missing[missing > 0].to_string())

print("\n===== SUMMARY COMPLETE =====")