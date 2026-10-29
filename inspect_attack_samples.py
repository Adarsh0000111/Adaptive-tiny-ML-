
from pathlib import Path

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "raw"
    / "DNN-EdgeIIoT-dataset.csv"
)

CHUNK_SIZE = 50_000
SAMPLES_PER_CATEGORY = 2

print("Reading dataset in chunks...")

samples = {}
rows_scanned = 0

try:
    for chunk in pd.read_csv(
        DATASET_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        rows_scanned += len(chunk)

        # Collect examples from every Attack_type.
        for category, group in chunk.groupby(
            "Attack_type",
            dropna=False,
            sort=False
        ):
            if category not in samples:
                samples[category] = []

            remaining = SAMPLES_PER_CATEGORY - len(samples[category])

            if remaining > 0:
                samples[category].extend(
                    group.head(remaining).to_dict("records")
                )

        # Stop when every category seen so far has two examples
        # and the binary labels 0 and 1 have both been encountered.
        if (
            all(len(rows) >= SAMPLES_PER_CATEGORY
                for rows in samples.values())
            and set(chunk["Attack_label"].dropna().unique())
                >= {0, 1}
        ):
            # This only checks the current chunk for both labels.
            # Continue scanning to discover any categories not yet seen.
            pass

    print(f"\nRows scanned: {rows_scanned:,}")
    print(f"Attack categories found: {len(samples)}")

    print("\n=== REPRESENTATIVE ATTACK CATEGORIES ===")

    for category, rows in samples.items():
        print(f"{category}: {len(rows)} example(s)")

    print("\n=== STRING / OBJECT COLUMNS ===")

    # Inspect dtypes from the first 20,000 rows.
    schema_sample = pd.read_csv(
        DATASET_PATH,
        nrows=20_000,
        low_memory=False
    )

    text_columns = [
        column
        for column in schema_sample.columns
        if is_object_dtype(schema_sample[column].dtype)
        or is_string_dtype(schema_sample[column].dtype)
    ]

    print(text_columns)

    print("\n=== EXAMPLE VALUES FROM TEXT COLUMNS ===")

    for column in text_columns:
        examples = []

        for rows in samples.values():
            for row in rows:
                value = row.get(column)

                if pd.notna(value) and str(value) not in examples:
                    examples.append(str(value))

                if len(examples) >= 3:
                    break

            if len(examples) >= 3:
                break

        print(f"{column}: {examples}")

    print("\nInspection complete.")

except Exception as error:
    print(f"\nInspection failed: {error}")
    raise
