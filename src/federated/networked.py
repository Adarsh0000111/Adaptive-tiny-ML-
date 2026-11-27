"""Authenticated, synchronous network FedAvg coordinator for computer clients.

The coordinator receives model parameters and sample counts, never packet rows.
This is a basic deployment protocol, not secure aggregation or differential privacy.
"""
from __future__ import annotations

import copy
import hmac
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.federated.torch_fedavg import FederatedMLP

ROOT = Path(__file__).resolve().parents[2]


def _tokens() -> dict[str, str]:
    try:
        value = json.loads(os.getenv("SARA_FL_NODE_TOKENS", "{}"))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


class NetworkedFedAvgCoordinator:
    def __init__(self):
        self.lock = threading.RLock()
        self.run: dict | None = None
        self.history: list[dict] = []
        self.model_path = ROOT / "models/federated/sara_mlp_networked.pt"

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
            checkpoint = self.model_path
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if saved.get("input_features") != 18 or saved.get("class_count") != 6:
                    raise ValueError("Existing global checkpoint has an incompatible schema")
                model.load_state_dict(saved["state_dict"])
            state = {key: val.detach().cpu().numpy().tolist() for key, val in model.state_dict().items()}
            self.run = {"status": "running", "round": 1, "rounds": rounds, "local_epochs": local_epochs,
                        "min_clients": min_clients, "timeout_seconds": timeout, "deadline": time.time()+timeout,
                        "nodes": nodes, "global_state": state, "updates": {}, "history": [], "version": int(time.time()),
                        "local_demo": local_demo}
            return self.status()

    def assignment(self, node_id: str, token: str | None) -> dict:
        if not self._node(node_id, token):
            raise PermissionError("Invalid federated node credentials")
        with self.lock:
            run = self.run
            if not run or node_id not in run["nodes"]:
                return {"status": "idle"}
            if run["status"] != "running":
                return {"status": run["status"]}
            self._expire_if_needed()
            if run["status"] != "running":
                return {"status": run["status"]}
            if node_id in run["updates"]:
                return {"status": "waiting", "round": run["round"]}
            return {"status": "train", "round": run["round"], "local_epochs": run["local_epochs"],
                    "input_features": 18, "class_count": 6, "global_state": run["global_state"]}

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
            self._aggregate_if_ready()
            return {"accepted": True, "status": run["status"], "round": run.get("round", round_id),
                    "updates_received": len(run.get("updates", {}))}

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
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.model_path.with_suffix(".tmp")
        torch.save({"state_dict": model.state_dict(), "input_features": 18, "class_count": 6,
                    "version": run["version"] + run["round"], "federated": True}, tmp)
        os.replace(tmp, self.model_path)
        if run["round"] >= run["rounds"]:
            run["status"] = "complete"
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
            if self.run:
                self._expire_if_needed()
                return {k: copy.deepcopy(v) for k, v in self.run.items() if k not in {"global_state", "updates", "deadline"}} | {
                    "updates_received": len(self.run["updates"]), "model_path": str(self.model_path)}
            return {"status": "idle", "history": copy.deepcopy(self.history), "model_path": str(self.model_path)}

