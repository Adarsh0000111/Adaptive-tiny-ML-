"""Prepare one client's locally labeled packet CSV for network FedAvg."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.edge_packet_schema import NUMERIC_FEATURES, PROTOCOL_FEATURE, SCHEMA_VERSION, prepare_official_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="A node-local labeled CSV with the project's packet fields and Attack_type")
    parser.add_argument("--out", required=True, help="Local output NPZ; keep it on this client")
    args = parser.parse_args()
    pipeline_path = ROOT / "models/preprocessing_pipeline.joblib"
    pipeline = joblib.load(pipeline_path)
    frame = pd.read_csv(args.csv)
    rows, metadata = prepare_official_rows(frame)
    if rows.empty:
        parser.error("No rows with labels recognized by the project mapping were found")
    numeric = rows[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    categories = pd.DataFrame({PROTOCOL_FEATURE: rows[PROTOCOL_FEATURE].astype(str)})
    transformed_numeric = pipeline["scaler"].transform(pipeline["imputer"].transform(numeric.to_numpy(dtype=np.float64)))
    transformed_categories = pipeline["cat_encoder"].transform(categories)
    x = np.hstack((transformed_numeric, transformed_categories)).astype(np.float32)
    y = pipeline["label_encoder"].transform(rows["Attack_type"].to_numpy()).astype(np.int64)
    feature_names = [str(value) for value in pipeline["feature_names"]]
    class_names = [str(value) for value in pipeline["class_names"]]
    if x.shape[1] != 18 or len(class_names) != 6 or not np.isfinite(x).all():
        parser.error("Prepared client data does not match the project's finite 18-feature/six-class schema")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X_train=x, y_train_multi=y,
                        feature_names=np.asarray(feature_names), class_names=np.asarray(class_names),
                        schema_version=np.asarray(SCHEMA_VERSION),
                        preprocessing_sha256=np.asarray(hashlib.sha256(pipeline_path.read_bytes()).hexdigest()))
    print(f"Prepared {len(y)} locally labeled samples at {output}; raw CSV stays on this computer.")
    print(f"Class counts: {metadata['class_counts']}")
    print("Labels are only as trustworthy as the Attack_type ground truth provided by this node.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
