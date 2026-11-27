import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.federated import networked


class TestNetworkedFederatedStatus(unittest.TestCase):
    def test_authenticated_nodes_have_independent_status(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            networked, "_tokens", return_value={"node-a": "secret-a", "node-b": "secret-b"}
        ):
            coordinator = networked.NetworkedFedAvgCoordinator(Path(directory) / "model.pt")
            coordinator.assignment("node-a", "secret-a")
            coordinator.assignment("node-b", "secret-b")
            nodes = coordinator.status()["node_status"]
            self.assertTrue(nodes["node-a"]["connected"])
            self.assertTrue(nodes["node-b"]["connected"])
            self.assertNotEqual(nodes["node-a"], nodes["node-b"])
            with self.assertRaises(PermissionError):
                coordinator.assignment("node-a", "secret-b")

    def test_completed_summary_survives_coordinator_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            coordinator = networked.NetworkedFedAvgCoordinator(model_path)
            coordinator.node_status = {"node-a": {"updates_accepted": 1}, "node-b": {"updates_accepted": 1}}
            coordinator.run = {
                "status": "complete", "round": 1, "rounds": 1, "local_epochs": 1,
                "min_clients": 2, "timeout_seconds": 60, "nodes": ["node-a", "node-b"],
                "local_demo": True, "base_model_source": "existing_high_tier_mlp",
                "history": [{"round": 1, "timestamp": "2026-01-01T00:00:00+00:00",
                              "participating_nodes": ["node-a", "node-b"],
                              "samples_per_node": {"node-a": 5, "node-b": 6}}],
                "updates": {"node-a": {}, "node-b": {}}, "promotion_approved": False,
                "promotion_reason": "validation unavailable", "best_candidate": None,
            }
            coordinator._persist_completed_run()
            restored = networked.NetworkedFedAvgCoordinator(model_path).status()
            self.assertEqual(restored["status"], "complete")
            self.assertEqual(restored["updates_received"], 2)
            self.assertEqual(restored["history"][0]["participating_nodes"], ["node-a", "node-b"])
            self.assertEqual(restored["node_status"]["node-b"]["updates_accepted"], 1)


if __name__ == "__main__":
    unittest.main()
