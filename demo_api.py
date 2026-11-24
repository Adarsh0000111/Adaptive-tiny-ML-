"""Local JSON API used by the existing React dashboard demonstration.

Run from the repository root with ``python -m dashboard.demo_api``.
All sample feature rows come from the saved, preprocessed test split.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np

from src.adaptive.fsm import AdaptiveModelFSM, SwitchingPolicy

ROOT = Path(__file__).resolve().parents[1]
LOW_ENERGY_THRESHOLD = float(os.environ.get("FSM_ENOUGH_ENERGY_PCT", "35"))
fsm = AdaptiveModelFSM(SwitchingPolicy(sufficient_energy_pct=LOW_ENERGY_THRESHOLD))
lock = threading.RLock()

ARTIFACTS = {
    "decision_tree": ROOT / "models/lightweight/model.joblib",
    "random_forest": ROOT / "models/medium/model.joblib",
    # Check the CNN script's documented output locations. The available
    # models/heavy/model.joblib is an MLP and is deliberately not substituted.
    "cnn_1d": ROOT / "models/cnn_model.keras",
}
models: dict[str, object] = {}
model_errors: dict[str, str] = {}
sample_error: str | None = None
features: np.ndarray | None = None
feature_names: list[str] = []
class_names: list[str] = []
sample_labels: np.ndarray | None = None
sample_index = 0


def initialize() -> None:
    global features, feature_names, class_names, sample_labels, sample_error
    for name, path in ARTIFACTS.items():
        try:
            if not path.is_file():
                raise FileNotFoundError(f"Trained artifact not found: {path.relative_to(ROOT)}")
            if name == "cnn_1d":
                from tensorflow import keras
                models[name] = keras.models.load_model(path)
            else:
                models[name] = joblib.load(path)
        except Exception as exc:
            model_errors[name] = f"{type(exc).__name__}: {exc}"

    try:
        pipeline_path = ROOT / "models/preprocessing_pipeline.joblib"
        pipeline = joblib.load(pipeline_path)
        feature_names = list(pipeline["feature_names"])
        class_names = list(pipeline.get("class_names", pipeline["label_encoder"].classes_))
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Saved preprocessing feature order contains duplicate names")
        dataset = np.load(ROOT / "data/processed/processed_dataset.npz", allow_pickle=False)
        features = dataset["X_test"]
        sample_labels = dataset["y_test_bin"] if "y_test_bin" in dataset.files else None
        data_names = dataset["feature_names"].tolist() if "feature_names" in dataset.files else feature_names
        if data_names != feature_names:
            raise ValueError("Saved test data feature order does not match the preprocessing artifact")
        if features.ndim != 2 or features.shape[1] != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} ordered features; test data has shape {features.shape}")
        incompatible = []
        for key, model in models.items():
            expected = getattr(model, "n_features_in_", len(feature_names))
            if key != "cnn_1d" and expected != len(feature_names):
                model_errors[key] = f"Model expects {expected} features but preprocessing provides {len(feature_names)}"
                incompatible.append(key)
        for key in incompatible:
            models.pop(key, None)
    except Exception as exc:
        sample_error = f"{type(exc).__name__}: {exc}"
        features = None


initialize()


def status() -> dict:
    return {
        "policy": {"sufficient_energy_pct": LOW_ENERGY_THRESHOLD},
        "models": {
            name: {"available": name in models, "artifact": str(path.relative_to(ROOT)), "error": model_errors.get(name)}
            for name, path in ARTIFACTS.items()
        },
        "heavy_mlp": {"available": (ROOT / "models/heavy/model.joblib").is_file(), "used_as_cnn": False},
        "data": {
            "available": features is not None,
            "source": "saved preprocessed X_test sample",
            "samples": 0 if features is None else int(len(features)),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "matches_requested_38_feature_contract": len(feature_names) == 38,
            "error": sample_error,
        },
        "fsm": {"state": fsm.state, "model": fsm.model, "history": fsm.history},
    }


def _decode_class(value: object) -> str:
    try:
        index = int(value)
        if float(value) == index and 0 <= index < len(class_names):
            return str(class_names[index])
    except (TypeError, ValueError):
        pass
    return str(value)


def _binary_prediction(model_name: str, row: np.ndarray) -> tuple[str, object]:
    model = models[model_name]
    if model_name == "cnn_1d":
        batch = row.astype(np.float32).reshape((1, row.shape[0], 1))
        raw = np.asarray(model.predict(batch, verbose=0)).reshape(-1)[0]
        return ("Attack" if float(raw) >= 0.5 else "Normal"), float(raw)
    raw = model.predict(row.reshape(1, -1))[0]
    decoded = _decode_class(raw)
    label = "Normal" if decoded.strip().lower() in {"normal", "0", "false"} else "Attack"
    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row.reshape(1, -1))[0]
        decoded_classes = [_decode_class(value) for value in model.classes_]
        probability = float(sum(probabilities[i] for i, value in enumerate(decoded_classes) if value.strip().lower() not in {"normal", "0", "false"}))
    return label, probability


def predict(payload: dict) -> dict:
    global sample_index
    threat = str(payload.get("threat", "low"))
    energy = float(payload.get("energy_pct", 100))
    energy_source = str(payload.get("energy_source", "simulated"))
    if energy_source not in {"simulated", "hardware"}:
        raise ValueError("energy_source must be simulated or hardware")
    with lock:
        selection = fsm.select(threat, energy, energy_source)
        if features is None or len(features) == 0:
            raise RuntimeError(sample_error or "Saved test samples are unavailable")
        row_index = int(payload.get("sample_index", sample_index)) % len(features)
        row = np.asarray(features[row_index], dtype=np.float32)
        sample_index = (row_index + 1) % len(features)
        model_name = selection["model"]
        reason = selection["reason"]
        try:
            label, attack_probability = _binary_prediction(model_name, row)
        except Exception as exc:
            failed_model = model_name
            model_errors[failed_model] = f"Prediction failed: {type(exc).__name__}: {exc}"
            if "decision_tree" not in models:
                raise RuntimeError(f"{failed_model} prediction failed and Decision Tree fallback is unavailable") from exc
            model_name = "decision_tree"
            fsm.model = model_name
            reason = f"{failed_model} prediction failed; Decision Tree fallback used: {type(exc).__name__}: {exc}"
            fsm.history.insert(0, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "from_model": failed_model, "to_model": model_name,
                "state": fsm.state, "reason": reason, "energy_source": energy_source,
            })
            fsm.history = fsm.history[:100]
            label, attack_probability = _binary_prediction(model_name, row)
        return {
            "sample_index": row_index,
            "sample_count": int(len(features)),
            "prediction": label,
            "attack_probability": attack_probability,
            "model": model_name,
            "requested_model": selection["model"],
            "threat": threat.lower(),
            "energy_pct": energy,
            "energy_source": energy_source,
            "inference_ms": round(float(payload.get("_elapsed_ms", 0)), 4),
            "state": fsm.state,
            "reason": reason,
            "switching_history": fsm.history,
            "true_label": None if sample_labels is None else ("Attack" if int(sample_labels[row_index]) == 1 else "Normal"),
        }


def select(payload: dict) -> dict:
    with lock:
        chosen = fsm.select(
            str(payload.get("threat", "low")),
            float(payload.get("energy_pct", 100)),
            str(payload.get("energy_source", "simulated")),
        )
        return {**chosen, "history": fsm.history}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, value: dict) -> None:
        body = json.dumps(value, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send(204, {})

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._send(200, status())
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path not in {"/api/predict", "/api/select"}:
            self._send(404, {"error": "Not found"})
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if self.path == "/api/select":
                self._send(200, select(payload))
                return
            start = time.perf_counter()
            result = predict(payload)
            result["inference_ms"] = round((time.perf_counter() - start) * 1000, 4)
            self._send(200, result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            self._send(503, {"error": str(exc), "status": status()})

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[demo-api] {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("DASHBOARD_API_PORT", "8765"))), Handler)
    print(f"Adaptive IDS demo API: http://127.0.0.1:{server.server_port}")
    print(f"Loaded models: {', '.join(models) or 'none'}; CNN status: {model_errors.get('cnn_1d', 'available')}")
    server.serve_forever()
