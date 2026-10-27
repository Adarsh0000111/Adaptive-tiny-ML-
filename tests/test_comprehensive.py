"""Comprehensive unit tests for the Adaptive TinyML IIoT IDS."""

import os
import unittest
import numpy as np

from src.config import load_config, get, reload_config
from src.adaptive.threat_scorer import ThreatScorer
from src.adaptive.utility_engine import AdaptiveModelSelector
from src.energy.simulated_provider import SimulationEnergyProvider
from src.energy.esp32_interface import ESP32INA219EnergyProvider
from src.federated.energy_selection import EnergyAwareClientSelector
from src.federated.non_iid import create_non_iid_partitions
from src.tinyml.optimizer import TinyMLOptimizer
from sklearn.tree import DecisionTreeClassifier


class TestConfig(unittest.TestCase):
    def test_load_config(self):
        cfg = load_config()
        self.assertIn("models", cfg)
        self.assertEqual(cfg["dataset"]["target_col"], "Attack_type")

    def test_get_dotted(self):
        self.assertEqual(get("models.lightweight.est_energy_uJ_per_inf"), 8.5)
        self.assertIsNone(get("nonexistent.path", default=None))

    def test_reload_config(self):
        reload_config()
        cfg = load_config()
        self.assertIsInstance(cfg, dict)


class TestThreatScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = ThreatScorer()

    def test_normal_low_severity(self):
        res = self.scorer.calculate_threat("Normal", 0.99)
        self.assertEqual(res["severity"], "LOW")
        self.assertLessEqual(res["threat_score"], 30)

    def test_ddos_high_severity(self):
        res = self.scorer.calculate_threat("DDoS", 0.95)
        self.assertIn(res["severity"], ["HIGH", "CRITICAL"])
        self.assertGreaterEqual(res["threat_score"], 60)

    def test_ransomware_critical(self):
        res = self.scorer.calculate_threat("Ransomware", 1.0)
        self.assertEqual(res["severity"], "CRITICAL")

    def test_output_fields(self):
        res = self.scorer.calculate_threat("Scanning", 0.5)
        for key in ["prediction", "confidence", "threat_score", "severity",
                    "category_severity", "anomaly_frequency"]:
            self.assertIn(key, res)
        self.assertTrue(0.0 <= res["threat_score"] <= 100.0)


class TestAdaptiveModelSelector(unittest.TestCase):
    def setUp(self):
        self.selector = AdaptiveModelSelector()

    def test_critical_battery_forces_lightweight(self):
        threat = {"threat_score": 40, "severity": "MEDIUM"}
        sel = self.selector.select_optimal_model(threat, battery_pct=10.0)
        self.assertEqual(sel["selected_model"], "lightweight")
        self.assertIn("Critical battery", sel["reason"])

    def test_critical_threat_uses_heavy(self):
        threat = {"threat_score": 90, "severity": "CRITICAL"}
        sel = self.selector.select_optimal_model(threat, battery_pct=80.0)
        self.assertEqual(sel["selected_model"], "heavy")

    def test_low_battery_medium_or_light(self):
        threat = {"threat_score": 50, "severity": "MEDIUM"}
        sel = self.selector.select_optimal_model(threat, battery_pct=25.0)
        self.assertIn(sel["selected_model"], ["lightweight", "medium"])
        self.assertIn("expected_energy_uJ", sel)
        self.assertIn("utility_scores", sel)


class TestSimulationEnergyProvider(unittest.TestCase):
    def test_battery_deduction(self):
        prov = SimulationEnergyProvider(initial_battery_pct=100.0)
        start = prov.get_battery_percentage()
        prov.consume_inference_energy("heavy")
        prov.consume_communication_energy(10.0)
        end = prov.get_battery_percentage()
        self.assertLess(end, start)
        self.assertGreaterEqual(prov.total_energy_consumed_uj, 340.0 + 450.0)

    def test_battery_floor_zero(self):
        prov = SimulationEnergyProvider(initial_battery_pct=100.0)
        prov.consume_inference_energy("heavy")
        prov.consume_inference_energy("heavy")
        self.assertGreaterEqual(prov.get_battery_percentage(), 0.0)


class TestESP32InterfaceStub(unittest.TestCase):
    def test_stub_battery_reading(self):
        prov = ESP32INA219EnergyProvider()
        pct = prov.get_battery_percentage()
        self.assertTrue(0.0 <= pct <= 100.0)
        self.assertIsInstance(prov.get_bus_voltage_v(), float)
        prov.close()

    def test_energy_bookkeeping(self):
        prov = ESP32INA219EnergyProvider()
        before = prov.total_energy_consumed_uj
        prov.consume_inference_energy("lightweight")
        self.assertGreater(prov.total_energy_consumed_uj, before)


class TestFederatedEnergySelection(unittest.TestCase):
    def test_excludes_dead_clients(self):
        class FakeClient:
            def __init__(self, cid, batt, n):
                self.client_id = cid
                self.batt = batt
                self.X_local = np.zeros((n, 5))

            def get_battery_level(self):
                return self.batt

        selector = EnergyAwareClientSelector.__new__(EnergyAwareClientSelector)
        selector.min_battery_threshold = 20.0
        selector.w_battery = 0.6
        selector.w_data = 0.4

        clients = [
            FakeClient(0, 95.0, 100),
            FakeClient(1, 10.0, 100),
            FakeClient(2, 70.0, 50),
        ]
        selected = selector.select_clients(clients, fraction=1.0, strategy="energy_aware")
        ids = [c.client_id for c in selected]
        self.assertNotIn(1, ids)


class TestNonIIDPartitions(unittest.TestCase):
    def test_partitions_cover_all_samples(self):
        X = np.random.rand(100, 5)
        y = np.array([0] * 40 + [1] * 60)
        parts = create_non_iid_partitions(X, y, num_clients=3, alpha=0.5)
        total = sum(len(p[0]) for p in parts)
        self.assertEqual(total, 100)
        self.assertEqual(len(parts), 3)


class TestTinyMLQuantizedTree(unittest.TestCase):
    def test_quantized_matches_sklearn(self):
        np.random.seed(0)
        X = np.random.rand(200, 4)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

        opt = TinyMLOptimizer()
        mean = X.mean(axis=0)
        scale = X.std(axis=0) + 1e-6
        X_s = (X - mean) / scale
        q = opt.quantize_tree_to_int8(tree, X_s)
        preds_q = opt.predict_quantized_tree(q, X_s)
        preds_sk = tree.predict(X_s)
        acc = np.mean(preds_q == preds_sk)
        self.assertGreaterEqual(acc, 0.9,
                                "Quantized tree should closely match sklearn")


if __name__ == "__main__":
    unittest.main()
