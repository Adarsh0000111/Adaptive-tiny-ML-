"""Create disjoint demo shards from the checked-in preprocessing output for FL nodes."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.federated.torch_fedavg import _partition_dirichlet
from src.data.edge_packet_schema import SCHEMA_VERSION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/processed_dataset.npz")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet label skew; lower means more non-IID")
    args = parser.parse_args()
    if not 2 <= args.nodes <= 32 or not 0.1 <= args.alpha <= 10:
        parser.error("--nodes must be 2-32 and --alpha 0.1-10")
    with np.load(args.input, allow_pickle=False) as archive:
        x = np.asarray(archive["X_train"], dtype=np.float32)
        y = np.asarray(archive["y_train_multi"], dtype=np.int64)
        feature_names = [str(value) for value in archive["feature_names"].tolist()]
    pipeline_path = ROOT / "models/preprocessing_pipeline.joblib"
    pipeline = joblib.load(pipeline_path)
    class_names = [str(value) for value in pipeline["class_names"]]
    preprocessing_sha256 = hashlib.sha256(pipeline_path.read_bytes()).hexdigest()
    if feature_names != [str(value) for value in pipeline["feature_names"]] or len(feature_names) != 18 or len(class_names) != 6:
        parser.error("Saved dataset, preprocessing pipeline, and six-class schema do not match")
    if x.ndim != 2 or x.shape[1] != 18 or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isin(y, np.arange(6)).all():
        parser.error("Input data does not match the finite 18-feature, six-class training schema")
    shards = _partition_dirichlet(y, args.nodes, args.alpha, seed=42)
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    for index, indices in enumerate(shards, start=1):
        if not len(indices):
            parser.error(f"Empty node shard {index}; use fewer nodes or a larger alpha")
        path = output / f"node-{index}.npz"
        np.savez_compressed(path, X_train=x[indices], y_train_multi=y[indices],
                            feature_names=np.asarray(feature_names), class_names=np.asarray(class_names),
                            schema_version=np.asarray(SCHEMA_VERSION), preprocessing_sha256=np.asarray(preprocessing_sha256))
        print(f"{path}: {len(indices)} rows")
    print("Copy each node's shard to its computer over your chosen trusted channel; do not put shards in a public repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
