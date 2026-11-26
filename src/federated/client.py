"""Run one authenticated network FedAvg participant using its local NPZ data."""
from __future__ import annotations

import argparse
import os
import time
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated.torch_fedavg import FederatedMLP


def train_one_round(x: np.ndarray, y: np.ndarray, assignment: dict) -> tuple[dict, float]:
    model = FederatedMLP(assignment["input_features"], assignment["class_count"])
    model.load_state_dict({key: torch.as_tensor(value, dtype=value_dtype(model, key)) for key, value in assignment["global_state"].items()})
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=128, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    model.train()
    losses: list[float] = []
    for _ in range(assignment["local_epochs"]):
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(features), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return {key: tensor.detach().cpu().tolist() for key, tensor in model.state_dict().items()}, float(np.mean(losses))


def value_dtype(model: FederatedMLP, key: str):
    return model.state_dict()[key].dtype


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an AdaptiveML networked FedAvg node")
    parser.add_argument("--coordinator", required=True, help="API base URL, for example https://host:8765")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--data", required=True, help="Local preprocessed NPZ containing X_train and y_train_multi")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    args = parser.parse_args()
    token = os.getenv("SARA_FL_NODE_TOKEN", "")
    if not token:
        parser.error("Set SARA_FL_NODE_TOKEN in the environment")
    parsed = urlparse(args.coordinator)
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("Remote coordinators must use HTTPS")
    base = args.coordinator.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    with np.load(args.data, allow_pickle=False) as data:
        x = np.asarray(data["X_train"], dtype=np.float32)
        y = np.asarray(data["y_train_multi"], dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != 18 or y.shape != (len(x),) or len(x) == 0:
        parser.error("Local NPZ must have non-empty X_train (N,18) and y_train_multi (N,)")
    if not np.isfinite(x).all() or not np.isin(y, np.arange(6)).all():
        parser.error("Local data must be finite and labels must be encoded 0-5")
    torch.set_num_threads(1)
    print(f"Node {args.node_id}: loaded {len(y)} local rows; raw data will not be uploaded.")
    while True:
        response = requests.get(f"{base}/api/federated/networked/assignment/{args.node_id}", headers=headers, timeout=30)
        response.raise_for_status()
        assignment = response.json()
        if assignment["status"] == "idle":
            print("No active networked FL run; exiting.")
            return 0
        if assignment["status"] in {"complete", "timed_out"}:
            print(f"Networked FL {assignment['status']}.")
            return 0 if assignment["status"] == "complete" else 2
        if assignment["status"] != "train":
            time.sleep(args.poll_seconds)
            continue
        state, loss = train_one_round(x, y, assignment)
        result = requests.post(f"{base}/api/federated/networked/update/{args.node_id}", headers=headers,
                              json={"round": assignment["round"], "sample_count": len(y), "local_loss": loss, "state_dict": state}, timeout=120)
        if result.status_code == 409:
            print(f"Round {assignment['round']} closed before this node arrived; discarding its stale update.")
            continue
        result.raise_for_status()
        print(f"Round {assignment['round']} update accepted; local loss={loss:.5f}")


if __name__ == "__main__":
    raise SystemExit(main())
