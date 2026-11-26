"""Rebuild all model artifacts from the official Edge-IIoTset packet CSV.

Usage: python scripts/retrain_official.py [path/to/ML-EdgeIIoT-dataset.csv]
The path can also be supplied through ADAPTIVEML_DATASET. No synthetic fallback is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.pipeline import run_preprocessing
from src.evaluation.benchmarker import ModelBenchmarker
from src.tinyml.optimizer import TinyMLOptimizer
from research.evaluate_artifacts import evaluate


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else None
    data = run_preprocessing(dataset_path=dataset)
    print("Rows after label mapping and duplicate filtering:", data["X_train"].shape[0] + data["X_val"].shape[0] + data["X_test"].shape[0])
    print("Schema:", data["schema_version"], "features:", len(data["feature_names"]), "classes:", data["class_names"])
    print("Tier benchmarks:", ModelBenchmarker().train_and_evaluate_all(data))
    print("ESP32 fixed-point export:", TinyMLOptimizer().optimize_and_benchmark())
    result = evaluate()
    output = ROOT / "results/experiments/artifact_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    import json
    output.write_text(json.dumps(result, indent=2))
    print("Saved final evaluation:", output)


if __name__ == "__main__":
    main()
