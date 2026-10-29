
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = (
    PROJECT_DIR
    / "dataset"
    / "raw"
    / "DNN-EdgeIIoT-dataset.csv"
)

OUTPUT_DIR = PROJECT_DIR / "dataset" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "ids_numeric.csv"

CHUNK_SIZE = 50_000
TARGET = "Attack_label"

EXCLUDE = {
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "tcp.options",
    "tcp.payload",
    "Attack_type",
    "icmp.unused",
    "http.tls_port",
    "dns.qry.type",
    "mqtt.msg_decoded_as",
    TARGET,
}

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read only the header and a small sample to identify
    # candidate numeric input columns.
    sample = pd.read_csv(
        RAW_PATH,
        nrows=20_000,
        low_memory=False,
    )

    if TARGET not in sample.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    candidates = [
        col
        for col in sample.columns
        if col not in EXCLUDE
        and pd.api.types.is_numeric_dtype(sample[col])
    ]

    if not candidates:
        raise ValueError("No candidate numeric features found.")

    print(f"Candidate features: {len(candidates)}")
    print("Pass 1/2: validating the complete dataset...")

    invalid_counts = {col: 0 for col in candidates}
    rows_seen = 0

    # Pass 1: identify columns containing unexpected text.
    for chunk in pd.read_csv(
        RAW_PATH,
        usecols=candidates + [TARGET],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        rows_seen += len(chunk)

        for col in candidates:
            original = chunk[col]
            converted = pd.to_numeric(
                original,
                errors="coerce",
            )

            invalid = original.notna() & converted.isna()
            invalid_counts[col] += int(invalid.sum())

        if rows_seen % 500_000 < len(chunk):
            print(f"Validated {rows_seen:,} rows...")

    bad_features = [
        col
        for col, count in invalid_counts.items()
        if count > 0
    ]

    features = [
        col for col in candidates
        if col not in bad_features
    ]

    if not features:
        raise ValueError(
            "No valid numeric features remain after validation."
        )

    print("\nFeatures excluded for unexpected text:")
    print(bad_features or "None")

    print(f"\nFeatures retained: {len(features)}")
    print("Pass 2/2: writing processed data...")

    # Pass 2: write numeric features and the binary target.
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    rows_written = 0

    for chunk in pd.read_csv(
        RAW_PATH,
        usecols=features + [TARGET],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        for col in features:
            chunk[col] = pd.to_numeric(
                chunk[col],
                errors="raise",
            )

            chunk[col] = chunk[col].replace(
                [np.inf, -np.inf],
                np.nan,
            )

        # Verify target values before writing.
        labels = set(chunk[TARGET].dropna().unique())

        if not labels.issubset({0, 1}):
            raise ValueError(
                f"Unexpected target labels: {labels}"
            )

        if chunk[TARGET].isna().any():
            raise ValueError("Target contains missing values.")

        chunk.to_csv(
            OUTPUT_PATH,
            mode="a",
            header=(rows_written == 0),
            index=False,
        )

        rows_written += len(chunk)

        if rows_written % 500_000 < len(chunk):
            print(f"Processed {rows_written:,} rows...")

    print("\nPREPROCESSING COMPLETE")
    print(f"Rows written: {rows_written:,}")
    print(f"Features retained: {len(features)}")
    print(f"Output: {OUTPUT_PATH}")
    print(
        "Note: infinite values were converted to missing "
        "values; imputation comes later."
    )


if __name__ == "__main__":
    main()
