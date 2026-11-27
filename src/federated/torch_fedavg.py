"""Reproducible CPU FedAvg simulation using one shared small MLP architecture.

Raw client arrays remain in process-local partitions. This is a local simulation,
not a networked or authenticated multi-machine FL deployment.
"""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import load_config

ROOT = Path(__file__).resolve().parents[2]
FEDERATED_ARCHITECTURE_VERSION = "mlp18-128-64-32-6-v3"


class FederatedMLP(nn.Module):
    def __init__(self, input_size: int = 18, classes: int = 6):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, classes))

    def forward(self, x):
        return self.layers(x)


def _partition_dirichlet(y: np.ndarray, clients: int, alpha: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[] for _ in range(clients)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(clients, alpha))
        cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
        for client, batch in enumerate(np.split(indices, cuts)):
            partitions[client].extend(batch.tolist())
    return [np.asarray(sorted(part), dtype=np.int64) for part in partitions]


class TorchFedAvgExperiment:
    def __init__(self):
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.status_data = {"running": False, "round": 0, "rounds": 0, "history": [], "version": None, "message": "No federated training has run."}
        self.history_path = ROOT / "results/experiments/federated_sara_history.json"
        self.model_path = ROOT / "models/federated/sara_mlp_global.pt"

    def status(self) -> dict:
        with self.lock:
            return {"available": True, "algorithm": "sample-weighted parameter FedAvg over a shared CPU MLP", "simulated": True, "networked_clients": False, "raw_data_exchange": False, **copy.deepcopy(self.status_data)}

    def rounds(self) -> list[dict]:
        with self.lock:
            return copy.deepcopy(self.status_data["history"])

    def start(self, rounds: int = 3, clients: int = 5, local_epochs: int = 1, alpha: float = 0.5) -> None:
        if not 1 <= rounds <= 20 or not 2 <= clients <= 8 or not 1 <= local_epochs <= 3 or not 0.1 <= alpha <= 10:
            raise ValueError("Use rounds 1-20, clients 2-8, local_epochs 1-3, and Dirichlet alpha 0.1-10")
        with self.lock:
            if self.status_data["running"]:
                raise RuntimeError("Federated training is already running")
            self.status_data = {"running": True, "round": 0, "rounds": rounds, "history": [], "version": self.status_data.get("version"), "message": "Federated simulation started."}
            self.thread = threading.Thread(target=self._run, args=(rounds, clients, local_epochs, alpha), daemon=True, name="sara-fedavg")
            self.thread.start()

    @staticmethod
    def _evaluate(model, x, y) -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(x).float()).argmax(dim=1).numpy()
        return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro", zero_division=0))

    def _run(self, rounds: int, clients: int, local_epochs: int, alpha: float) -> None:
        try:
            torch.set_num_threads(1)
            np.random.seed(42)
            torch.manual_seed(42)
            data = np.load(ROOT / "data/processed/processed_dataset.npz", allow_pickle=False)
            x_train = data["X_train"].astype(np.float32)
            y_train = data["y_train_multi"].astype(np.int64)
            x_val = data["X_val"].astype(np.float32)
            y_val = data["y_val_multi"].astype(np.int64)
            x_test = data["X_test"].astype(np.float32)
            y_test = data["y_test_multi"].astype(np.int64)
            indices = _partition_dirichlet(y_train, clients, alpha, 42)
            test_indices = _partition_dirichlet(y_test, clients, alpha, 43)
            model = FederatedMLP(x_train.shape[1], int(np.max(y_train)) + 1)
            best_acc, best_f1 = self._evaluate(model, x_val, y_val)
            best_state = copy.deepcopy(model.state_dict())
            base_version = int(time.time())
            global_version = base_version
            history = []
            cfg = load_config()
            energy_cfg = cfg["energy_simulation"]
            initial_battery = [95.0, 85.0, 18.0, 70.0, 12.0, 90.0, 65.0, 50.0][:clients]
            battery = (initial_battery + [80.0] * clients)[:clients]
            participation = [0] * clients
            data_size_scale = [max(0.0, len(part) / max(1, np.mean([len(p) for p in indices]))) for part in indices]
            battery_capacity_j = float(energy_cfg["battery_capacity_mAh"]) / 1000.0 * float(energy_cfg["voltage_v"]) * 3.6
            for round_id in range(1, rounds + 1):
                round_start_version = global_version
                # Explicitly simulated scheduler from estimated battery and local data availability.
                eligible = [i for i in range(clients) if battery[i] >= 20.0 and len(indices[i]) > 0]
                target_count = clients if clients <= 3 else max(3, (clients * 3) // 4)
                scores = {i: 0.6 * battery[i] + 40.0 * min(1.0, data_size_scale[i]) - 20.0 * participation[i] for i in eligible}
                selected_ids = sorted(sorted(scores, key=scores.get, reverse=True)[:target_count])
                selected_ids = [client_id for client_id in selected_ids if len(indices[client_id]) > 0]
                local_states = []
                sample_counts = []
                local_times = []
                local_losses = []
                for client_id in selected_ids:
                    local = FederatedMLP(x_train.shape[1], int(np.max(y_train)) + 1)
                    local.load_state_dict(model.state_dict())
                    client_x = torch.from_numpy(x_train[indices[client_id]])
                    client_y = torch.from_numpy(y_train[indices[client_id]])
                    loader = DataLoader(TensorDataset(client_x, client_y), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(42 + round_id + client_id))
                    optimizer = torch.optim.SGD(local.parameters(), lr=0.03, momentum=0.0)
                    start = time.perf_counter()
                    local.train()
                    weighted_loss = 0.0
                    trained_samples = 0
                    for _ in range(local_epochs):
                        for batch_x, batch_y in loader:
                            optimizer.zero_grad(set_to_none=True)
                            loss = nn.functional.cross_entropy(local(batch_x), batch_y)
                            weighted_loss += float(loss.detach()) * len(batch_y)
                            trained_samples += len(batch_y)
                            loss.backward()
                            optimizer.step()
                    local_times.append(time.perf_counter() - start)
                    local_losses.append(weighted_loss / max(1, trained_samples))
                    local_states.append(copy.deepcopy(local.state_dict()))
                    sample_counts.append(len(client_y))
                total = sum(sample_counts)
                if total == 0:
                    raise RuntimeError("Selected federated clients have no local samples")
                aggregated = {}
                for key in model.state_dict():
                    aggregated[key] = sum(local_states[i][key] * (sample_counts[i] / total) for i in range(len(local_states)))
                candidate = FederatedMLP(x_train.shape[1], int(np.max(y_train)) + 1)
                candidate.load_state_dict(aggregated)
                accuracy, macro_f1 = self._evaluate(candidate, x_val, y_val)
                accepted = accuracy + 1e-4 >= best_acc
                # Quantify model-update payload serialization; no network transmission occurs in this simulation.
                payload_buffer = io.BytesIO()
                torch.save(aggregated, payload_buffer)
                payload_bytes = len(payload_buffer.getbuffer())
                client_evaluations = []
                for client_id, local_test_indices in enumerate(test_indices):
                    if len(local_test_indices):
                        local_accuracy, local_f1 = self._evaluate(candidate, x_test[local_test_indices], y_test[local_test_indices])
                        client_evaluations.append({"client_id": client_id, "test_samples": int(len(local_test_indices)), "accuracy": local_accuracy, "macro_f1": local_f1})
                    else:
                        client_evaluations.append({"client_id": client_id, "test_samples": 0, "accuracy": None, "macro_f1": None})
                if accepted:
                    model = candidate
                    best_state = copy.deepcopy(aggregated)
                    best_acc, best_f1 = accuracy, macro_f1
                    global_version = base_version + round_id
                    self.model_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint = {"state_dict": best_state, "input_features": int(x_train.shape[1]), "class_count": int(np.max(y_train)) + 1, "version": base_version + round_id, "validation_accuracy": accuracy, "validation_macro_f1": macro_f1}
                    temporary = self.model_path.with_suffix(".tmp")
                    torch.save(checkpoint, temporary)
                    backup = self.model_path.with_suffix(".previous.pt")
                    if self.model_path.exists():
                        shutil.copy2(self.model_path, backup)
                    os.replace(temporary, self.model_path)
                item = {"round": round_id, "timestamp": datetime.now(timezone.utc).isoformat(), "selected_clients": selected_ids, "client_samples": sample_counts, "global_accuracy": accuracy, "global_macro_f1": macro_f1, "accepted": accepted, "validation_samples": int(len(y_val)), "mean_local_training_seconds": float(np.mean(local_times)), "mean_local_training_loss": float(np.average(local_losses, weights=sample_counts)), "client_local_losses": {str(client_id): float(loss) for client_id, loss in zip(selected_ids, local_losses)}, "energy_measurement": "not measured; no power sensor was attached to these simulation clients"}
                item["client_test_metrics"] = client_evaluations
                item["update_payload_bytes_serialized_locally"] = payload_bytes
                item["simulated_upload_bytes"] = int(payload_bytes * len(selected_ids))
                item["simulated_download_bytes"] = int(payload_bytes * len(selected_ids))
                item["simulated_total_communication_bytes"] = int(2 * payload_bytes * len(selected_ids))
                item["network_communication"] = "not measured; no network clients are connected"
                item["postponed_clients"] = [i for i in range(clients) if i not in selected_ids]
                item["client_availability"] = {str(client_id): {"eligible_by_simulated_battery": bool(battery[client_id] >= 20.0), "local_samples_available": int(len(indices[client_id])), "selected": client_id in selected_ids} for client_id in range(clients)}
                item["model_freshness"] = {"global_version_used": int(round_start_version), "client_model_age_rounds": {str(client_id): 0 for client_id in selected_ids}, "source": "simulated synchronous round; clients start from the current global model"}
                for slot, client_id in enumerate(selected_ids):
                    participation[client_id] += 1
                    size_ratio = len(indices[client_id]) / max(1, np.mean([len(indices[i]) for i in selected_ids]))
                    estimated_training_uj = float(energy_cfg["fed_train_energy_uJ_per_epoch"]) * local_epochs * size_ratio
                    estimated_comm_uj = payload_bytes / 1024.0 * float(energy_cfg["comm_energy_uJ_per_kb"])
                    battery_capacity_uj = battery_capacity_j * 1e6
                    if battery_capacity_uj > 0:
                        battery[client_id] = max(0.0, battery[client_id] - (estimated_training_uj + estimated_comm_uj) / battery_capacity_uj * 100.0)
                item["battery_pct_simulated_estimate"] = [round(value, 4) for value in battery]
                item["energy_cost_source"] = "config-based simulated estimate; not measured electrical energy"
                history.append(item)
                with self.lock:
                    self.status_data.update({"round": round_id, "history": copy.deepcopy(history), "version": base_version + round_id if accepted else self.status_data.get("version"), "message": "Candidate accepted by validation gate." if accepted else "Candidate rejected; previous global checkpoint retained."})
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps({"algorithm": "FedAvg", "simulated": True, "rounds": history}, indent=2))
            with self.lock:
                self.status_data["running"] = False
                self.status_data["message"] = "Federated simulation finished; accepted checkpoint retained for rollback-safe review."
        except Exception as exc:
            with self.lock:
                self.status_data["running"] = False
                self.status_data["message"] = f"Federated simulation failed: {type(exc).__name__}: {exc}"
