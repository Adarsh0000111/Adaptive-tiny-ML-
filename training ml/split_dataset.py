from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_DIR
    / "dataset"
    / "processed"
    / "ids_numeric.csv"
)

OUTPUT_DIR = PROJECT_DIR / "dataset" / "processed" / "split"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Attack_label"


def main():
    print("Loading processed dataset...")

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Unexpected target labels.")

    # Stratified split preserves the class proportions.
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSPLIT COMPLETE")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")

    print("\nTraining class distribution:")
    print(train_df[TARGET].value_counts(normalize=True).sort_index())

    print("\nTesting class distribution:")
    print(test_df[TARGET].value_counts(normalize=True).sort_index())

    print(f"\nTraining file: {train_path}")
    print(f"Testing file: {test_path}")


if __name__ == "__main__":
    main()
