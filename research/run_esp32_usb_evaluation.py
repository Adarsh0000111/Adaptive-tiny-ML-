"""Run the complete saved test split through the ESP32 USB inference firmware."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.inference import InferenceService
from src.api.serial_device import USBReplayService


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(port: str | None) -> dict:
    inference = InferenceService()
    y_true = inference.y_test
    class_names = inference.class_names
    labels = np.arange(len(class_names))
    aggregate = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    total_started = time.perf_counter()

    result = USBReplayService().replay(port, len(y_true), 0, inference, include_latency_samples=True)
    aggregate = np.asarray(result["confusion_matrix"], dtype=np.int64)
    device_us = result["device_inference_samples_us"]
    round_trip_ms = result["usb_round_trip_samples_ms"]

    support = aggregate.sum(axis=1)
    true_positive = np.diag(aggregate)
    precision = np.divide(true_positive, aggregate.sum(axis=0), out=np.zeros(len(labels), dtype=float), where=aggregate.sum(axis=0) != 0)
    recall = np.divide(true_positive, support, out=np.zeros(len(labels), dtype=float), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(labels), dtype=float), where=(precision + recall) != 0)
    per_class = {
        name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1-score": float(f1[i]), "support": int(support[i])}
        for i, name in enumerate(class_names)
    }
    per_class["macro avg"] = {"precision": float(np.mean(precision)), "recall": float(np.mean(recall)), "f1-score": float(np.mean(f1)), "support": int(support.sum())}
    per_class["weighted avg"] = {
        "precision": float(np.average(precision, weights=support)),
        "recall": float(np.average(recall, weights=support)),
        "f1-score": float(np.average(f1, weights=support)),
        "support": int(support.sum()),
    }
    artifact = ROOT / "models/lightweight/model.joblib"
    dataset = ROOT / "data/processed/processed_dataset.npz"
    result = {
        "evaluation": "full held-out saved test split replayed through the ESP32 over USB; not live traffic capture",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "board": "ESP32-S3 identified by esptool; USB-only firmware handshake verified",
        "firmware_handshake": result["firmware"],
        "transport": "USB serial at 115200 baud; Wi-Fi and MQTT not used",
        "model": "lightweight fixed-point decision tree exported from the checked-in sklearn artifact",
        "feature_schema": "edge-iiotset-18-v1; laptop sends the saved, train-preprocessed test vectors",
        "dataset_sha256": sha256(dataset),
        "model_artifact_sha256": sha256(artifact),
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "serial_port": port or "auto-detected",
        "samples": int(support.sum()),
        "accuracy": float(true_positive.sum() / support.sum()),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": aggregate.tolist(),
        "device_inference_us": {"mean": float(np.mean(device_us)), "median": float(np.median(device_us)), "p95": float(np.percentile(device_us, 95))},
        "usb_per_sample_ms": {"mean": float(np.mean(round_trip_ms)), "median": float(np.median(round_trip_ms)), "p95": float(np.percentile(round_trip_ms, 95))},
        "usb_session": {"samples": len(y_true), "serial_startup_ms": result["usb_startup_ms"], "usb_transfer_total_ms": result["usb_round_trip_ms"]["total"]},
        "serial_protocol_retries": result["serial_protocol_retries"],
        "last_physical_sensor_reading": result["device_telemetry"],
        "total_wall_seconds": float(time.perf_counter() - total_started),
        "energy": "INA219 board integration from boot; measurement path and calibration were not independently validated",
        "scope_note": "This validates on-device inference on replayed saved features. The ESP32 is not extracting features from live network traffic.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Serial port, e.g. /dev/ttyACM0; default auto-detects the USB adapter")
    args = parser.parse_args()
    result = run(args.port)
    output = ROOT / "results/experiments/esp32_usb_replay.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved complete hardware replay results to {output}")
    print(f"Accuracy={result['accuracy']:.4f}; macro-F1={result['macro_f1']:.4f}; n={result['samples']}")


if __name__ == "__main__":
    main()
