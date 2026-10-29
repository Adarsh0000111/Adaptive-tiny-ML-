
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "raw"
    / "DNN-EdgeIIoT-dataset.csv"
)

columns_to_check = [
    "icmp.unused",
    "http.tls_port",
    "dns.qry.type",
    "mqtt.msg_decoded_as",
    "tcp.options",
    "tcp.payload",
]

print("Inspecting selected columns across the full dataset...")

unique_values = {col: set() for col in columns_to_check}
missing_counts = {col: 0 for col in columns_to_check}
infinite_counts = {col: 0 for col in columns_to_check}
rows_scanned = 0

for chunk in pd.read_csv(
    DATASET_PATH,
    usecols=columns_to_check,
    chunksize=50_000,
    low_memory=False,
):
    rows_scanned += len(chunk)

    for col in columns_to_check:
        series = chunk[col]

        missing_counts[col] += int(series.isna().sum())

        for value in series.dropna().unique():
            unique_values[col].add(str(value))

        numeric = pd.to_numeric(series, errors="coerce")
        values = numeric.to_numpy(dtype=float, na_value=np.nan)

        infinite_counts[col] += int(np.isinf(values).sum())

print(f"\nRows inspected: {rows_scanned:,}")

for col in columns_to_check:
    values = sorted(unique_values[col])[:10]

    print(f"\nColumn: {col}")
    print(f"Distinct non-missing values (up to 10): {values}")
    print(f"Missing values: {missing_counts[col]}")
    print(f"Infinite numeric values: {infinite_counts[col]}")

print("\nInspection complete. Original dataset unchanged.")
