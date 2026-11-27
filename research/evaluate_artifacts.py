"""Evaluate the checked-in model artifacts on the fixed saved test split.

This script does not retrain models or estimate physical energy.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = {"lightweight": ROOT / "models/lightweight/model.joblib", "medium": ROOT / "models/medium/model.joblib", "heavy": ROOT / "models/heavy/model.joblib"}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate() -> dict:
    data_path = ROOT / "data/processed/processed_dataset.npz"
    artifact = joblib.load(ROOT / "models/preprocessing_pipeline.joblib")
    dataset = np.load(data_path, allow_pickle=False)
    x_test = dataset["X_test"].astype(np.float32, copy=False)
    y_test = dataset["y_test_multi"].astype(np.int64, copy=False)
    class_names = [str(value) for value in artifact["class_names"]]
    feature_names = list(artifact["feature_names"])
    if dataset["feature_names"].tolist() != feature_names:
        raise ValueError("Processed split and saved preprocessing feature orders differ")
    if y_test.min(initial=0) < 0 or y_test.max(initial=0) >= len(class_names):
        raise ValueError("Test labels are outside the saved label mapping")

    results = {
        "evaluation": "fixed held-out test split; inference only; no retraining",
        "dataset": {"path": "data/processed/processed_dataset.npz", "sha256": checksum(data_path), "samples": int(len(y_test)), "features": int(x_test.shape[1]), "feature_names": feature_names, "class_names": class_names},
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "platform": platform.platform()},
        "models": {},
        "physical_energy": "not measured",
    }
    for tier, path in MODELS.items():
        model = joblib.load(path)
        if getattr(model, "n_features_in_", None) != len(feature_names):
            raise ValueError(f"{tier} feature width differs from saved schema")
        start = time.perf_counter()
        predictions = model.predict(x_test)
        batch_elapsed = time.perf_counter() - start
        timer_ms = []
        for row in x_test[: min(1000, len(x_test))]:
            start = time.perf_counter_ns()
            model.predict(row.reshape(1, -1))
            timer_ms.append((time.perf_counter_ns() - start) / 1_000_000)
        cm = confusion_matrix(y_test, predictions, labels=np.arange(len(class_names)))
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        results["models"][tier] = {
            "architecture": type(model).__name__,
            "artifact": str(path.relative_to(ROOT)),
            "artifact_sha256": checksum(path),
            "artifact_size_bytes": path.stat().st_size,
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_macro_precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            "test_macro_recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            "test_macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "per_class": classification_report(y_test, predictions, labels=np.arange(len(class_names)), target_names=class_names, output_dict=True, zero_division=0),
            "confusion_matrix": cm.tolist(),
            "macro_one_vs_rest_fpr": float(np.mean(fp / np.maximum(1, fp + tn))),
            "macro_one_vs_rest_fnr": float(np.mean(fn / np.maximum(1, fn + tp))),
            "batch_test_inference_seconds": batch_elapsed,
            "python_host_single_row_latency_ms": {"mean": float(np.mean(timer_ms)), "median": float(np.median(timer_ms)), "p95": float(np.percentile(timer_ms, 95)), "samples_timed": len(timer_ms), "hardware": "current API host; not ESP32 or edge-node latency"},
            "peak_ram": "not measured",
            "energy_per_inference": "not measured",
        }
    return results


if __name__ == "__main__":
    output = evaluate()
    target = ROOT / "results/experiments/artifact_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2))
    print(f"Wrote fixed-test artifact evaluation to {target.relative_to(ROOT)}")
    for tier, metrics in output["models"].items():
        print(f"{tier}: accuracy={metrics['test_accuracy']:.4f} macro_f1={metrics['test_macro_f1']:.4f} size={metrics['artifact_size_bytes']} bytes p95_host_ms={metrics['python_host_single_row_latency_ms']['p95']:.6f}")
