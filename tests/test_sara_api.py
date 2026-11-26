import os
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.api.app import app
from src.sara import CrossLayerRiskEngine, SARAEngine


class TestSARAEngine(unittest.TestCase):
    def test_resource_pressure_alone_is_not_attack_evidence(self):
        engine = CrossLayerRiskEngine()
        calm = engine.evaluate(network_risk=0, resource_stress=0, uncertainty=0)
        stressed = engine.evaluate(network_risk=0, resource_stress=1, uncertainty=0)
        self.assertEqual(calm["score"], 0)
        self.assertEqual(stressed["score"], 0)

    def test_model_disagreement_is_included_in_security_risk(self):
        engine = CrossLayerRiskEngine()
        consensus = engine.evaluate(network_risk=0, resource_stress=0, uncertainty=0, model_disagreement=0)
        disagreement = engine.evaluate(network_risk=0, resource_stress=0, uncertainty=0, model_disagreement=1)
        self.assertEqual(consensus["score"], 0)
        self.assertGreater(disagreement["score"], consensus["score"])
        self.assertEqual(disagreement["model_disagreement"], 1)

    def test_high_risk_rejects_local_models_below_recall_floor(self):
        policy = SARAEngine({
            "lightweight": {"macro_recall": 0.80, "estimated_energy_uj": 8},
            "medium": {"macro_recall": 0.92, "estimated_energy_uj": 60},
            "heavy": {"macro_recall": 0.97, "estimated_energy_uj": 300},
        })
        decision = policy.select(0.9, 0.1)
        self.assertEqual(decision["action"], "heavy")
        self.assertEqual(len(decision["rejected_actions"]), 2)


class TestSARAApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        env = os.environ.copy()
        env.update({"ADAPTIVEML_DB": str(Path(cls.temp.name) / "runtime.sqlite3"), "ADAPTIVEML_HOST": "127.0.0.1", "ADAPTIVEML_PORT": str(cls.port), "SARA_TRUSTED_NODES": "[]"})
        cls.process = subprocess.Popen([sys.executable, "-m", "src.api.app"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        deadline = time.time() + 30
        while time.time() < deadline:
            if cls.process.poll() is not None:
                raise RuntimeError("API process exited before becoming ready")
            try:
                cls.request("GET", "/api/health")
                break
            except Exception:
                time.sleep(0.2)
        else:
            cls.process.terminate()
            raise RuntimeError("API process did not become ready")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(timeout=5)
        cls.temp.cleanup()

    @classmethod
    def request(cls, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(cls.base_url + path, data=data, headers={"Content-Type": "application/json"}, method=method)
        try:
            with urlopen(request, timeout=15) as response:
                return response.status, json.load(response)
        except HTTPError as response:
            return response.code, json.load(response)

    def test_health_and_saved_artifacts_match_18_feature_six_class_schema(self):
        status, data = self.request("GET", "/api/models")
        self.assertEqual(status, 200)
        self.assertEqual(len(data["feature_schema"]["feature_names"]), 18)
        self.assertEqual(len(data["feature_schema"]["class_names"]), 6)
        self.assertEqual(data["model_count"], 3)

    def test_saved_test_replay_returns_actual_multiclass_prediction(self):
        status, result = self.request("POST", "/api/inference", {"mode": "saved_sample", "sample_index": 0, "model": "auto"})
        self.assertEqual(status, 200, result)
        self.assertIn(result["prediction"], {"Normal", "Scanning", "DDoS", "Ransomware", "SQL_Injection", "Backdoor"})
        self.assertEqual(result["input_source"], "saved_test_replay")
        self.assertIn("decision", result["adaptive"])
        self.assertIn("model_disagreement", result["adaptive"]["risk"])
        self.assertEqual(set(result["model_diagnostics"]["model_predictions"]), {"lightweight", "medium", "heavy"})

    def test_mismatched_live_feature_schema_is_rejected(self):
        status, _ = self.request("POST", "/api/inference", {"mode": "preprocessed_features", "features": [0] * 18, "feature_names": [f"feature_{i}" for i in range(18)]})
        self.assertEqual(status, 422)

    def test_disagreement_is_normalized_and_exposed_to_sara(self):
        status, result = self.request("POST", "/api/inference", {"mode": "saved_sample", "sample_index": 29, "model": "auto"})
        self.assertEqual(status, 200, result)
        votes = list(result["model_diagnostics"]["model_predictions"].values())
        self.assertEqual(len(votes), 3)
        expected = (1 - max(votes.count(label) for label in set(votes)) / 3) / (2 / 3)
        self.assertAlmostEqual(result["model_diagnostics"]["disagreement_score"], expected)
        self.assertAlmostEqual(result["adaptive"]["risk"]["model_disagreement"], expected)

    def test_usb_inference_status_is_explicitly_offline_replay(self):
        status, data = self.request("GET", "/api/device/usb/status")
        self.assertEqual(status, 200)
        self.assertEqual(data["transport"], "USB serial")
        self.assertIn("saved preprocessed test split", data["input_source"])
        self.assertIn("exact READY handshake", data["firmware_contract"])

    def test_timestamped_power_is_integrated_and_source_is_retained(self):
        first = {"node_id": "test-node", "timestamp": "2026-01-01T00:00:00Z", "source": "physical", "voltage_v": 3.3, "current_a": 0.1}
        second = {"node_id": "test-node", "timestamp": "2026-01-01T00:00:02Z", "source": "physical", "voltage_v": 3.3, "current_a": 0.1}
        status, _ = self.request("POST", "/api/telemetry", first)
        self.assertEqual(status, 200)
        status, result = self.request("POST", "/api/telemetry", second)
        self.assertEqual(status, 200, result)
        self.assertAlmostEqual(result["cumulative_energy_j"], 0.66, places=4)
        self.assertEqual(result["source"], "physical")

    def test_network_capture_statistics_are_derived_and_missing_probe_values_stay_null(self):
        status, result = self.request("POST", "/api/network/stats", {
            "node_id": "test-network", "source": "capture", "window_seconds": 2,
            "packets": 10, "bytes": 900, "tcp_syn": 4, "tcp_connections": 4,
            "protocol_counts": {"tcp": 8, "udp": 2}, "link_capacity_bps": 12000,
        })
        self.assertEqual(status, 200, result)
        self.assertEqual(result["packets_per_second"], 5)
        self.assertEqual(result["average_packet_size_bytes"], 90)
        self.assertEqual(result["connection_frequency_per_second"], 2)
        self.assertEqual(result["bandwidth_bps"], 3600)
        self.assertEqual(result["bandwidth_utilization_pct"], 30)
        self.assertEqual(result["protocol_types"], ["tcp", "udp"])
        self.assertIsNone(result["round_trip_latency_ms"])
        self.assertIsNone(result["packet_loss_pct"])
        self.assertIsNone(result["jitter_ms"])

    def test_network_probe_accepts_only_literal_ip_and_capture_needs_acknowledgement(self):
        status, _ = self.request("POST", "/api/network/probe", {"target_ip": "example.com", "packets": 5})
        self.assertEqual(status, 422)
        status, _ = self.request("POST", "/api/network/capture", {"authorized": False})
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
