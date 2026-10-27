"""Centralized configuration loader for the Adaptive TinyML IIoT IDS.

Provides a cached configuration object so every module does not re-read/parse
``config.yaml`` independently. Falls back to safe defaults when the file is
missing or malformed so the system keeps working in CI / development.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.yaml")
)

_FALLBACK_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "paths": {
        "raw_data": "data/raw/iiot_intrusion_dataset.csv",
        "processed_data": "data/processed/processed_dataset.npz",
        "data_summary": "data/processed/data_summary.json",
        "models_dir": "models",
        "results_dir": "results",
        "figures_dir": "results/figures",
        "tables_dir": "results/tables",
        "experiments_dir": "results/experiments",
    },
    "dataset": {
        "name": "Edge-IIoTset Cybersecurity Dataset",
        "target_col": "Attack_type",
        "binary_col": "Attack_label",
        "sample_size": 25000,
        "test_size": 0.2,
        "val_size": 0.1,
        "drop_features": [
            "frame.time", "ip.src", "ip.dst",
            "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
            "http.file_data", "tcp.payload",
        ],
    },
    "models": {
        "lightweight": {
            "type": "DecisionTreeClassifier",
            "params": {"max_depth": 4, "min_samples_leaf": 5, "criterion": "gini", "random_state": 42},
            "est_inference_latency_ms": 0.12,
            "est_ram_kb": 4.5,
            "est_energy_uJ_per_inf": 8.5,
        },
        "medium": {
            "type": "RandomForestClassifier",
            "params": {"n_estimators": 25, "max_depth": 10, "random_state": 42, "n_jobs": -1},
            "est_inference_latency_ms": 1.45,
            "est_ram_kb": 48.0,
            "est_energy_uJ_per_inf": 62.0,
        },
        "heavy": {
            "type": "MLPClassifier",
            "params": {"hidden_layer_sizes": [128, 64, 32], "max_iter": 100, "activation": "relu", "solver": "adam", "random_state": 42},
            "est_inference_latency_ms": 8.70,
            "est_ram_kb": 320.0,
            "est_energy_uJ_per_inf": 340.0,
        },
    },
    "federated": {
        "num_clients": 5,
        "num_rounds": 10,
        "local_epochs": 3,
        "batch_size": 64,
        "learning_rate": 0.01,
        "dirichlet_alpha": 0.5,
        "client_selection": {
            "strategy": "energy_aware",
            "min_battery_threshold": 20.0,
            "battery_weight": 0.6,
            "data_size_weight": 0.4,
        },
    },
    "threat_scoring": {
        "weights": {"confidence": 0.25, "severity": 0.45, "anomaly_frequency": 0.30},
        "thresholds": {"low": 30, "medium": 60, "high": 80},
        "severity_scores": {
            "Normal": 0, "Scanning": 35, "PortScan": 40,
            "SQL_Injection": 60, "Backdoor": 75, "DDoS": 90, "Ransomware": 95,
        },
    },
    "energy_simulation": {
        "initial_battery_pct": 100.0,
        "battery_capacity_mAh": 1200.0,
        "voltage_v": 3.3,
        "idle_power_mW": 15.0,
        "comm_energy_uJ_per_kb": 45.0,
        "fed_train_energy_uJ_per_epoch": 12500.0,
    },
    "adaptive_switching": {
        "utility_weights": {"threat_weight": 0.50, "energy_weight": 0.30, "latency_weight": 0.20},
        "battery_critical_pct": 15.0,
        "battery_low_pct": 35.0,
    },
    "simulation": {"realtime_speed_hz": 5, "default_events_count": 200},
}

_lock = threading.RLock()
_config_cache: Dict[str, Any] | None = None


def load_config(config_path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and cache the project configuration from ``config.yaml``.

    The config is read once and cached in memory. Subsequent calls return the
    cached dict unless ``config_path`` differs. A fallback config is returned if
    the file cannot be read or parsed, keeping the system runnable without the
    file during development / CI.
    """
    global _config_cache
    config_path = os.path.abspath(config_path)

    with _lock:
        if _config_cache is not None:
            return _config_cache

        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                if not isinstance(cfg, dict):
                    raise ValueError("config.yaml top-level must be a mapping")
            else:
                cfg = {}
        except Exception:  # pragma: no cover - defensive fallback
            cfg = {}

        _config_cache = _deep_merge(_FALLBACK_CONFIG, cfg)
        return _config_cache


def reload_config() -> Dict[str, Any]:
    """Clear the config cache and reload from disk."""
    global _config_cache
    with _lock:
        _config_cache = None
    return load_config()


def get(path: str, default: Any = None, config_path: str = _DEFAULT_CONFIG_PATH) -> Any:
    """Retrieve a dotted-path config value, e.g.
    ``get('models.lightweight.est_energy_uJ_per_inf')``."""
    cur: Any = load_config(config_path)
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    result = dict(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result

