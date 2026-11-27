"""Authenticated, synchronous network FedAvg coordinator for computer clients.

The coordinator receives model parameters and sample counts, never packet rows.
This is a basic deployment protocol, not secure aggregation or differential privacy.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.data.edge_packet_schema import SCHEMA_VERSION
from src.federated.torch_fedavg import FEDERATED_ARCHITECTURE_VERSION, FederatedMLP

ROOT = Path(__file__).resolve().parents[2]


def _tokens() -> dict[str, str]:
    try:
        value = json.loads(os.getenv("SARA_FL_NODE_TOKENS", "{}"))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


class NetworkedFedAvgCoordinator:
    def __init__(self, model_path: Path | None = None):
        self.lock = threading.RLock()
        self.run: dict | None = None
        self.history: list[dict] = []
        self.model_path = Path(model_path) if model_path else ROOT / "models/federated/sara_mlp_networked.pt"
        self.candidate_path = self.model_path.with_name(f"{self.model_path.stem}_candidate.pt")
        self.history_path = self.model_path.with_name(f"{self.model_path.stem}_history.json")
        self.last_completed = self._load_last_completed()
        self.node_status: dict[str, dict] = copy.deepcopy(self.last_completed.get("node_status", {})) if self.last_completed else {}
        if self.last_completed:
            self.history = copy.deepcopy(self.last_completed.get("all_history", self.last_completed.get("history", [])))
        self.validation_x = None
        self.validation_y = None
        self.feature_names: list[str] = []
        self.class_names: list[str] = []
        self.preprocessing_sha256 = ""
        self.class_weights: list[float] = []
        self.baseline_heavy_validation: dict[str, float] = {}
        try:
            pipeline = joblib.load(ROOT / "models/preprocessing_pipeline.joblib")
            data = np.load(ROOT / "data/processed/processed_dataset.npz", allow_pickle=False)
            self.feature_names = [str(value) for value in pipeline["feature_names"]]
            self.class_names = [str(value) for value in pipeline["class_names"]]
            self.preprocessing_sha256 = hashlib.sha256((ROOT / "models/preprocessing_pipeline.joblib").read_bytes()).hexdigest()
            if data["feature_names"].tolist() == self.feature_names and len(self.feature_names) == 18 and len(self.class_names) == 6:
                self.validation_x = data["X_val"].astype(np.float32, copy=False)
                self.validation_y = data["y_val_multi"].astype(np.int64, copy=False)
                train_labels = data["y_train_multi"].astype(np.int64, copy=False)
                counts = np.bincount(train_labels, minlength=len(self.class_names)).astype(np.float64)
                weights = np.sqrt(len(train_labels) / (len(self.class_names) * np.maximum(counts, 1)))
                self.class_weights = (weights / weights.mean()).astype(np.float32).tolist()
                baseline = joblib.load(ROOT / "models/heavy/model.joblib")
                self.baseline_heavy_validation = self._metrics(baseline.predict(self.validation_x), self.validation_y)
        except (OSError, KeyError, ValueError, TypeError):
            # Training remains possible when the coordinator has no validation set,
            # but that run cannot be promoted into live SARA inference.
            pass

    def _load_last_completed(self) -> dict | None:
        try:
            saved = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (not isinstance(saved, dict) or saved.get("status") != "complete"
                or not isinstance(saved.get("history"), list) or not saved["history"]
                or not isinstance(saved.get("nodes"), list)):
            return None
        return saved

    def _persist_completed_run(self) -> None:
        run = self.run
        if not run or run.get("status") != "complete":
            return
        summary = {
            "status": "complete", "round": run["round"], "rounds": run["rounds"],
            "local_epochs": run["local_epochs"], "min_clients": run["min_clients"],
            "timeout_seconds": run["timeout_seconds"], "nodes": list(run["nodes"]),
            "local_demo": run.get("local_demo", False),
            "base_model_source": run.get("base_model_source"),
            "history": copy.deepcopy(run["history"]),
            "all_history": copy.deepcopy(self.history),
            "node_status": {node: {"connected": False, "last_seen": None, "state": "complete",
                                    "updates_accepted": self.node_status.get(node, {}).get("updates_accepted", 0)}
                            for node in run["nodes"]},
            "updates_received": len(run["updates"]),
            "promotion_approved": run.get("promotion_approved", False),
            "promotion_reason": run.get("promotion_reason"),
            "validation_metrics": (run.get("best_candidate") or {}).get("validation_metrics"),
            "model_path": str(self.model_path), "candidate_model_path": str(self.candidate_path),
            "completed_at": run["history"][-1]["timestamp"],
        }
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.history_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        os.replace(temporary, self.history_path)
        self.last_completed = summary

    def _metrics(self, predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        }

    def _evaluate(self, model: FederatedMLP) -> dict[str, float] | None:
        if self.validation_x is None or self.validation_y is None:
            return None
        model.eval()
        with torch.no_grad():
            predictions = model(torch.from_numpy(self.validation_x)).argmax(dim=1).cpu().numpy()
        return self._metrics(predictions, self.validation_y)

    def _promotion_decision(self, metrics: dict[str, float] | None) -> tuple[bool, str]:
        if metrics is None or not self.baseline_heavy_validation:
            return False, "Coordinator validation data or heavy-tier baseline is unavailable."
        min_recall = max(0.90, self.baseline_heavy_validation["macro_recall"])
        min_accuracy = self.baseline_heavy_validation["accuracy"] - 0.02
        min_f1 = self.baseline_heavy_validation["macro_f1"] - 0.02
        approved = (metrics["macro_recall"] >= min_recall and metrics["accuracy"] >= min_accuracy and metrics["macro_f1"] >= min_f1)
        reason = ("Passed the held-out validation gate for the SARA high laptop tier." if approved else
                  f"Not promoted: validation must reach macro recall {min_recall:.3f}, accuracy {min_accuracy:.3f}, and macro-F1 {min_f1:.3f}.")
        return approved, reason

    def _node(self, node_id: str, token: str | None) -> bool:
        expected = _tokens().get(node_id, "")
        return bool(expected and token and hmac.compare_digest(expected, token))

    def start(self, nodes: list[str], rounds: int, local_epochs: int, min_clients: int, timeout: int, local_demo: bool = False) -> dict:
        with self.lock:
            if self.run and self.run["status"] == "running":
                raise RuntimeError("A networked FL run is already active")
            configured = _tokens()
            if not configured or any(node not in configured for node in nodes):
                raise ValueError("Every requested node must have a token in SARA_FL_NODE_TOKENS")
            if not 2 <= len(nodes) <= 32 or len(set(nodes)) != len(nodes):
                raise ValueError("Provide 2-32 unique configured node IDs")
            if not 2 <= min_clients <= len(nodes):
                raise ValueError("min_clients must be between 2 and the number of nodes")
            model = FederatedMLP()
            base_source = "validated_federated_checkpoint" if self.model_path.exists() else "existing_high_tier_mlp"
            checkpoint = self.model_path
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if (saved.get("promotion_approved") is True
                        and saved.get("input_features") == 18 and saved.get("class_count") == 6
                        and saved.get("schema_version") == SCHEMA_VERSION
                        and saved.get("architecture_version") == FEDERATED_ARCHITECTURE_VERSION):
                    model.load_state_dict(saved["state_dict"])
                else:
                    base_source = "existing_high_tier_mlp"
            if base_source == "existing_high_tier_mlp":
                baseline = joblib.load(ROOT / "models/heavy/model.joblib")
                linear_layers = [layer for layer in model.layers if isinstance(layer, torch.nn.Linear)]
                if (not hasattr(baseline, "coefs_") or len(baseline.coefs_) != len(linear_layers)
                        or [weight.shape for weight in baseline.coefs_] != [(18, 128), (128, 64), (64, 32), (32, 6)]
                        or not np.array_equal(np.asarray(baseline.classes_), np.arange(6))):
                    raise ValueError("The deployed high-tier model is not compatible with this federated MLP architecture/class mapping")
                with torch.no_grad():
                    for layer, weights, biases in zip(linear_layers, baseline.coefs_, baseline.intercepts_):
                        layer.weight.copy_(torch.as_tensor(weights.T, dtype=layer.weight.dtype))
                        layer.bias.copy_(torch.as_tensor(biases, dtype=layer.bias.dtype))
            state = {key: val.detach().cpu().numpy().tolist() for key, val in model.state_dict().items()}
            self.run = {"status": "running", "round": 1, "rounds": rounds, "local_epochs": local_epochs,
                        "min_clients": min_clients, "timeout_seconds": timeout, "deadline": time.time()+timeout,
                        "nodes": nodes, "global_state": state, "updates": {}, "history": [], "version": int(time.time()),
                        "local_demo": local_demo, "base_model_source": base_source, "best_candidate": None,
                        "promotion_approved": False, "promotion_reason": "Training run has not completed validation."}
            self.node_status = {node: {"connected": False, "last_seen": None, "state": "awaiting_client", "updates_accepted": 0}
                                for node in nodes}
            return self.status()

    def assignment(self, node_id: str, token: str | None) -> dict:
        if not self._node(node_id, token):
            raise PermissionError("Invalid federated node credentials")
        with self.lock:
            node = self.node_status.setdefault(node_id, {"connected": False, "last_seen": None, "state": "idle", "updates_accepted": 0})
            node.update({"connected": True, "last_seen": datetime.now(timezone.utc).isoformat()})
            run = self.run
            if not run or node_id not in run["nodes"]:
                node["state"] = "idle"
                return {"status": "idle"}
            if run["status"] != "running":
                node["state"] = run["status"]
                return {"status": run["status"]}
            self._expire_if_needed()
            if run["status"] != "running":
                return {"status": run["status"]}
            if node_id in run["updates"]:
                node["state"] = "update_accepted_waiting_for_peers"
                return {"status": "waiting", "round": run["round"]}
            node["state"] = "assigned"
            return {"status": "train", "round": run["round"], "local_epochs": run["local_epochs"],
                    "input_features": 18, "class_count": 6, "architecture_version": FEDERATED_ARCHITECTURE_VERSION,
                    "schema_version": SCHEMA_VERSION, "feature_names": self.feature_names,
                    "class_names": self.class_names, "preprocessing_sha256": self.preprocessing_sha256,
                    "class_weights": self.class_weights,
                    "global_state": run["global_state"]}

    def submit(self, node_id: str, token: str | None, round_id: int, sample_count: int, state_dict: dict, loss: float) -> dict:
        if not self._node(node_id, token):
            raise PermissionError("Invalid federated node credentials")
        if not 1 <= sample_count <= 10_000_000 or not np.isfinite(loss) or loss < 0:
            raise ValueError("Invalid sample count or loss")
        with self.lock:
            run = self.run
            if not run or run["status"] != "running" or node_id not in run["nodes"]:
                raise ValueError("No active assignment for this node")
            if round_id != run["round"] or node_id in run["updates"]:
                raise ValueError("Stale round or duplicate client update")
            expected_model = FederatedMLP()
            expected = expected_model.state_dict()
            if set(state_dict) != set(expected):
                raise ValueError("Model parameter names do not match")
            clean = {}
            for key, ref in expected.items():
                value = np.asarray(state_dict[key], dtype=np.float32)
                if tuple(value.shape) != tuple(ref.shape) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid model tensor: {key}")
                clean[key] = value
            run["updates"][node_id] = {"state": clean, "samples": sample_count, "loss": float(loss)}
            node = self.node_status.setdefault(node_id, {"updates_accepted": 0})
            node.update({"state": "update_accepted", "updates_accepted": node.get("updates_accepted", 0) + 1})
            self._aggregate_if_ready()
            return {"accepted": True, "status": run["status"], "round": run.get("round", round_id),
                    "updates_received": len(run.get("updates", {})),
                    "promotion_approved": run.get("promotion_approved"),
                    "promotion_reason": run.get("promotion_reason")}

    def _aggregate_if_ready(self):
        run = self.run
        if len(run["updates"]) < run["min_clients"]:
            return
        updates = list(run["updates"].values())
        total = sum(item["samples"] for item in updates)
        averaged = {}
        for key in updates[0]["state"]:
            averaged[key] = sum(item["state"][key] * (item["samples"] / total) for item in updates)
        run["global_state"] = {key: value.tolist() for key, value in averaged.items()}
        record = {"round": run["round"], "timestamp": datetime.now(timezone.utc).isoformat(),
                  "participating_nodes": list(run["updates"]), "postponed_nodes": [n for n in run["nodes"] if n not in run["updates"]],
                  "samples_per_node": {n: v["samples"] for n, v in run["updates"].items()},
                  "mean_local_loss": float(np.average([v["loss"] for v in updates], weights=[v["samples"] for v in updates])),
                  "raw_training_data_uploaded": False}
        run["history"].append(record)
        self.history.append(record)
        model = FederatedMLP()
        model.load_state_dict({k: torch.tensor(v) for k, v in averaged.items()})
        validation_metrics = self._evaluate(model)
        promotion_approved, promotion_reason = self._promotion_decision(validation_metrics)
        record.update({"validation_metrics": validation_metrics, "promotion_approved": promotion_approved,
                       "promotion_reason": promotion_reason})
        best = run["best_candidate"]
        if validation_metrics is not None and (best is None or
                (validation_metrics["macro_recall"], validation_metrics["macro_f1"]) >
                (best["validation_metrics"]["macro_recall"], best["validation_metrics"]["macro_f1"])):
            run["best_candidate"] = {"state_dict": {key: value.copy() for key, value in averaged.items()},
                                     "validation_metrics": validation_metrics}
        if run["round"] >= run["rounds"]:
            best = run["best_candidate"]
            final_metrics = best["validation_metrics"] if best else None
            approved, reason = self._promotion_decision(final_metrics)
            run["promotion_approved"] = approved
            run["promotion_reason"] = reason
            if best:
                final_model = FederatedMLP()
                final_model.load_state_dict({key: torch.as_tensor(value) for key, value in best["state_dict"].items()})
                checkpoint = {"state_dict": final_model.state_dict(), "input_features": 18, "class_count": 6,
                              "schema_version": SCHEMA_VERSION, "feature_names": self.feature_names,
                              "class_names": self.class_names, "preprocessing_sha256": self.preprocessing_sha256,
                              "architecture_version": FEDERATED_ARCHITECTURE_VERSION,
                              "version": run["version"] + run["round"], "federated": True,
                              "promotion_approved": approved, "promotion_reason": reason,
                              "validation_metrics": final_metrics, "validation_gate": {
                                  "min_macro_recall": max(0.90, self.baseline_heavy_validation.get("macro_recall", 0.90)),
                                  "min_accuracy": self.baseline_heavy_validation.get("accuracy", 1.0) - 0.02,
                                  "min_macro_f1": self.baseline_heavy_validation.get("macro_f1", 1.0) - 0.02,
                                  "baseline_heavy_validation": self.baseline_heavy_validation}}
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                candidate_temporary = self.candidate_path.with_suffix(".tmp")
                torch.save(checkpoint, candidate_temporary)
                os.replace(candidate_temporary, self.candidate_path)
                if approved:
                    active_temporary = self.model_path.with_suffix(".tmp")
                    torch.save(checkpoint, active_temporary)
                    os.replace(active_temporary, self.model_path)
            run["status"] = "complete"
            for node in run["nodes"]:
                self.node_status.setdefault(node, {}).update({"state": "complete"})
            self._persist_completed_run()
        else:
            run["round"] += 1
            run["updates"] = {}
            run["deadline"] = time.time() + run["timeout_seconds"]

    def _expire_if_needed(self):
        run = self.run
        if run and run["status"] == "running" and time.time() >= run["deadline"]:
            if len(run["updates"]) >= run["min_clients"]:
                self._aggregate_if_ready()
            else:
                run["status"] = "timed_out"

    def status(self) -> dict:
        with self.lock:
            nodes = copy.deepcopy(self.node_status)
            now = datetime.now(timezone.utc)
            for item in nodes.values():
                try:
                    seen = datetime.fromisoformat(item["last_seen"])
                    item["connected"] = 0 <= (now - seen).total_seconds() <= 10
                except (KeyError, TypeError, ValueError):
                    item["connected"] = False
                if not item["connected"] and item.get("state") not in {"complete", "timed_out"}:
                    item["state"] = "disconnected"
            if self.run:
                self._expire_if_needed()
                return {k: copy.deepcopy(v) for k, v in self.run.items() if k not in {"global_state", "updates", "deadline", "best_candidate"}} | {
                    "updates_received": len(self.run["updates"]), "model_path": str(self.model_path),
                    "candidate_model_path": str(self.candidate_path),
                    "node_status": {node: nodes.get(node, {}) for node in self.run["nodes"]}}
            if self.last_completed:
                return copy.deepcopy(self.last_completed) | {
                    "model_path": str(self.model_path), "candidate_model_path": str(self.candidate_path),
                    "node_status": nodes or copy.deepcopy(self.last_completed.get("node_status", {})),
                }
            return {"status": "idle", "history": copy.deepcopy(self.history), "model_path": str(self.model_path),
                    "candidate_model_path": str(self.candidate_path), "node_status": nodes}

