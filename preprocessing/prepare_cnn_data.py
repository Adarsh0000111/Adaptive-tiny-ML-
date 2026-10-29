
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_DIR / "dataset" / "processed" / "split"
OUTPUT_DIR = PROJECT_DIR / "dataset" / "processed" / "cnn"
MODEL_DIR = PROJECT_DIR / "models"

TARGET = "Attack_label"
CHUNK_SIZE = 50_000


def main():
    train_path = SPLIT_DIR / "train.csv"
    test_path = SPLIT_DIR / "test.csv"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Read column names without loading the full dataset.
    columns = pd.read_csv(train_path, nrows=0).columns.tolist()

    if TARGET not in columns:
        raise ValueError(f"Missing target column: {TARGET}")

    features = [c for c in columns if c != TARGET]
    scaler = StandardScaler()

    # Pass 1: fit the scaler incrementally on training features only.
    print("Pass 1/2: fitting scaler on training chunks...", flush=True)

    for chunk in pd.read_csv(train_path, chunksize=CHUNK_SIZE):
        X = chunk[features].to_numpy(dtype=np.float32)
        scaler.partial_fit(X)
        print(f"Fitted through {min(scaler.n_samples_seen_, 1_775_360):,} rows",
              flush=True)

    # Save the fitted scaler.
    scaler_path = MODEL_DIR / "cnn_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # Pass 2: transform one CSV at a time, writing compressed arrays.
    for split_name, input_path in [
        ("train", train_path),
        ("test", test_path),
    ]:
        output_path = OUTPUT_DIR / f"{split_name}_scaled.npz"
        temp_path = OUTPUT_DIR / f"{split_name}_scaled.tmp"

        print(f"Transforming {split_name} data...", flush=True)

        # Stream scaled features and labels to temporary binary files.
        x_path = OUTPUT_DIR / f"{split_name}_X.tmp"
        y_path = OUTPUT_DIR / f"{split_name}_y.tmp"
        row_count = 0

        with open(x_path, "wb") as xf, open(y_path, "wb") as yf:
            for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):
                X = chunk[features].to_numpy(dtype=np.float32)
                X_scaled = scaler.transform(X).astype(np.float32)
                y = chunk[TARGET].to_numpy(dtype=np.int8)

                X_scaled.tofile(xf)
                y.tofile(yf)
                row_count += len(chunk)

                print(f"{split_name}: processed {row_count:,} rows",
                      flush=True)

        # Build memory-mapped arrays and save compressed output.
        X_map = np.memmap(
            x_path, dtype=np.float32, mode="r",
            shape=(row_count, len(features))
        )
        y_map = np.memmap(
            y_path, dtype=np.int8, mode="r", shape=(row_count,)
        )

        np.savez_compressed(
            output_path,
            X=X_map,
            y=y_map,
            feature_names=np.array(features),
        )

        del X_map, y_map
        x_path.unlink()
        y_path.unlink()

        print(f"Saved: {output_path}", flush=True)

    print("\nCNN preparation complete.")
    print(f"Features: {len(features)}")
    print(f"Scaler: {scaler_path}")


if __name__ == "__main__":
    main()
