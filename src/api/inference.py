from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.data.edge_packet_schema import NUMERIC_FEATURES, PROTOCOL_CATEGORIES, PROTOCOL_FEATURE, SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[2]
TIERS = {
    "lightweight": ROOT / "models/lightweight/model.joblib",
    "medium": ROOT / "models/medium/model.joblib",
    "heavy": ROOT / "models/heavy/model.joblib",
}


class InferenceService:
    """Loads the checked-in multi-class artifacts and enforces their saved schema."""

    def __init__(self):
        pipeline = joblib.load(ROOT / "models/preprocessing_pipeline.joblib")
        self._pipeline = pipeline
        self.feature_names = list(pipeline["feature_names"])
        self.schema_version = pipeline.get("schema_version", SCHEMA_VERSION)
        self.class_names = [str(x) for x in pipeline["class_names"]]
        data = np.load(ROOT / "data/processed/processed_dataset.npz", allow_pickle=False)
        self.X_val = data["X_val"].astype(np.float32, copy=False)
        self.y_val = data["y_val_multi"].astype(np.int64, copy=False)
        self.X_test = data["X_test"].astype(np.float32, copy=False)
        self.y_test = data["y_test_multi"].astype(np.int64, copy=False)
        schema_from_data = data["feature_names"].tolist()
        if schema_from_data != self.feature_names:
            raise ValueError("Saved model and processed-data feature schemas differ")
        if len(self.feature_names) != len(NUMERIC_FEATURES) + len(PROTOCOL_CATEGORIES):
            raise ValueError(f"Expected the saved packet schema width 18; found {len(self.feature_names)}")
        if self.y_val.max(initial=0) >= len(self.class_names):
            raise ValueError("Validation labels do not match the saved class mapping")
        self.models: dict[str, Any] = {}
        self.profiles: dict[str, dict[str, Any]] = {}
        self._load_and_profile()

    def _load_and_profile(self) -> None:
        for tier, path in TIERS.items():
            model = joblib.load(path)
            if int(getattr(model, "n_features_in_", -1)) != len(self.feature_names):
                raise ValueError(f"{tier} model input width does not match the saved feature schema")
            self.models[tier] = model
            predictions = model.predict(self.X_val)
            metrics = {
                "accuracy": float(accuracy_score(self.y_val, predictions)),
                "macro_precision": float(precision_score(self.y_val, predictions, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(self.y_val, predictions, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(self.y_val, predictions, average="macro", zero_division=0)),
            }
            test_predictions = model.predict(self.X_test)
            test_metrics = {
                "accuracy": float(accuracy_score(self.y_test, test_predictions)),
                "macro_precision": float(precision_score(self.y_test, test_predictions, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(self.y_test, test_predictions, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(self.y_test, test_predictions, average="macro", zero_division=0)),
            }
            per_class_recall = recall_score(self.y_val, predictions, labels=np.arange(len(self.class_names)), average=None, zero_division=0)
            timings: list[float] = []
            for row in self.X_val[: min(300, len(self.X_val))]:
                start = time.perf_counter_ns()
                model.predict(row.reshape(1, -1))
                timings.append((time.perf_counter_ns() - start) / 1_000_000)
            timing_array = np.asarray(timings)
            try:
                benchmark = json.loads((ROOT / "results/tables/model_benchmark.json").read_text())[tier]
            except (OSError, KeyError, ValueError):
                benchmark = {}
            self.profiles[tier] = {
                "name": type(model).__name__,
                "artifact": str(path.relative_to(ROOT)),
                "artifact_size_bytes": path.stat().st_size,
                "input_features": len(self.feature_names),
                "classes": self.class_names,
                "metrics": metrics,
                "test_metrics": test_metrics,
                "per_class_recall": {self.class_names[i]: float(per_class_recall[i]) for i in range(len(self.class_names))},
                "latency_ms": {
                    "mean": float(np.mean(timing_array)) if len(timing_array) else None,
                    "median": float(np.median(timing_array)) if len(timing_array) else None,
                    "p95": float(np.percentile(timing_array, 95)) if len(timing_array) else None,
                    "source": "measured on current API host during startup; validation sample; not device latency",
                },
                "energy": {"value_uj": benchmark.get("est_energy_uJ"), "source": "historical estimate; not measured by this API"},
                "evaluation": {"split": "validation", "samples": int(len(self.y_val)), "source": "saved processed_dataset.npz"},
            }

    def validate_vector(self, values: list[float], names: list[str]) -> np.ndarray:
        if names != self.feature_names:
            raise ValueError("feature_names must exactly match the saved packet feature order")
        row = np.asarray(values, dtype=np.float32)
        if row.ndim != 1 or row.shape[0] != len(self.feature_names):
            raise ValueError(f"Expected exactly {len(self.feature_names)} feature values")
        if not np.isfinite(row).all():
            raise ValueError("Feature values must be finite numbers")
        return row

    def transform_packet_rows(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Apply the train-fitted scaler and one-hot encoding to live packet fields."""
        numeric = pd.DataFrame([{name: row.get(name, 0.0) for name in NUMERIC_FEATURES} for row in rows])
        numeric = numeric.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        categories = pd.DataFrame({PROTOCOL_FEATURE: [row.get(PROTOCOL_FEATURE, "other") for row in rows]})
        transformed_num = self._pipeline["scaler"].transform(self._pipeline["imputer"].transform(numeric.to_numpy(dtype=np.float64)))
        transformed_cat = self._pipeline["cat_encoder"].transform(categories)
        output = np.hstack((transformed_num, transformed_cat)).astype(np.float32)
        if output.shape[1] != len(self.feature_names):
            raise ValueError("Live packet transform produced a feature vector with the wrong schema")
        return output

    def infer(self, tier: str, row: np.ndarray, *, source: str, sample_index: int | None = None) -> dict[str, Any]:
        if tier not in self.models:
            raise ValueError(f"Unknown model tier: {tier}")
        model = self.models[tier]
        start = time.perf_counter_ns()
        encoded = int(model.predict(row.reshape(1, -1))[0])
        probabilities = model.predict_proba(row.reshape(1, -1))[0] if hasattr(model, "predict_proba") else None
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000
        confidence = float(np.max(probabilities)) if probabilities is not None else None
        entropy = None
        if probabilities is not None:
            p = np.asarray(probabilities, dtype=np.float64)
            nonzero = p[p > 0]
            entropy = float(-np.sum(nonzero * np.log(nonzero)) / math.log(max(2, len(p))))
        return {
            "prediction": self.class_names[encoded],
            "class_index": encoded,
            "confidence": confidence,
            "normalized_entropy": entropy,
            "model": tier,
            "inference_ms": latency_ms,
            "input_source": source,
            "sample_index": sample_index,
            "feature_schema": {"version": self.schema_version, "features": self.feature_names, "scale": "preprocessed using saved train-fitted pipeline"},
        }

    def sample_row(self, index: int) -> tuple[np.ndarray, int, int]:
        if not 0 <= int(index) < len(self.X_test):
            raise ValueError(f"sample_index must be between 0 and {len(self.X_test) - 1}")
        return self.X_test[int(index)], int(self.y_test[int(index)]), len(self.X_test)
