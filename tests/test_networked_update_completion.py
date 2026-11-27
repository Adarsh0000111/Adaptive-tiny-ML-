import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.api import app as api
from src.api.app import NetworkedFederatedUpdate


class TestNetworkedUpdateCompletion(unittest.TestCase):
    def test_completed_update_refreshes_isolated_artifact_status(self):
        coordinator = Mock()
        coordinator.status.return_value = {"local_demo": False}
        coordinator.submit.return_value = {
            "accepted": True,
            "status": "complete",
            "promotion_approved": False,
        }
        inference = SimpleNamespace(
            federated_model_status={"status": "isolated"},
            _load_federated_artifact_status=Mock(),
        )
        state = api.app.state._state
        old_coordinator = state.get("networked_federated")
        old_inference = state.get("inference")
        state["networked_federated"] = coordinator
        state["inference"] = inference
        try:
            response = api.networked_federated_update(
                "node-a",
                NetworkedFederatedUpdate(round=1, sample_count=5, local_loss=0.1, state_dict={}),
                SimpleNamespace(client=SimpleNamespace(host="127.0.0.1")),
                "Bearer test-token",
            )
        finally:
            if old_coordinator is None:
                state.pop("networked_federated", None)
            else:
                state["networked_federated"] = old_coordinator
            if old_inference is None:
                state.pop("inference", None)
            else:
                state["inference"] = old_inference

        self.assertTrue(response["accepted"])
        self.assertEqual(response["status"], "complete")
        inference._load_federated_artifact_status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
