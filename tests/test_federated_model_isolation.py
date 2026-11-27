import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.api import inference
from src.federated.networked import NetworkedFedAvgCoordinator
from src.federated.torch_fedavg import FederatedCNN


class TestFederatedModelIsolation(unittest.TestCase):
    def test_validation_pass_still_saves_candidate_without_overwriting_global_path(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "networked_global.pt"
            model_path.write_bytes(b"keep-existing-global-artifact")
            coordinator = NetworkedFedAvgCoordinator(model_path)
            architecture = FederatedCNN().state_dict()
            state = {key: np.zeros(tuple(value.shape), dtype=np.float32) for key, value in architecture.items()}
            coordinator.run = {
                "status": "running", "round": 1, "rounds": 1, "min_clients": 2,
                "timeout_seconds": 60, "local_epochs": 1, "nodes": ["node-a", "node-b"], "global_state": {},
                "updates": {node: {"state": state, "samples": 2, "loss": 0.1} for node in ("node-a", "node-b")},
                "history": [], "version": 10, "local_demo": False,
                "base_model_source": "existing_high_tier_cnn", "best_candidate": None,
            }
            coordinator.node_status = {node: {"updates_accepted": 1} for node in ("node-a", "node-b")}
            metrics = {"accuracy": 0.95, "macro_recall": 0.95, "macro_f1": 0.95}
            with patch.object(coordinator, "_evaluate", return_value=metrics), patch.object(
                coordinator, "_promotion_decision", return_value=(True, "validation passed")
            ):
                coordinator._aggregate_if_ready()
            self.assertEqual(model_path.read_bytes(), b"keep-existing-global-artifact")
            self.assertTrue(coordinator.candidate_path.exists())
            candidate = torch.load(coordinator.candidate_path, map_location="cpu", weights_only=True)
            self.assertFalse(candidate["promotion_approved"])
            self.assertTrue(candidate["validation_gate_passed"])
            self.assertFalse(coordinator.run["promotion_approved"])

    def test_inference_tiers_are_never_replaced_by_federated_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "models/federated/sara_cnn_networked_candidate.pt"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"placeholder-isolation-checkpoint")
            service = inference.InferenceService.__new__(inference.InferenceService)
            original_models = {tier: object() for tier in ("lightweight", "medium", "heavy")}
            service.models = original_models.copy()
            with patch.object(inference, "ROOT", root):
                service._load_federated_artifact_status()
            for tier, model in original_models.items():
                self.assertIs(service.models[tier], model)
            self.assertEqual(service.federated_model_status["status"], "isolated")


if __name__ == "__main__":
    unittest.main()
