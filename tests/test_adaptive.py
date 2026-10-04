import unittest
from src.adaptive.threat_scorer import ThreatScorer
from src.adaptive.utility_engine import AdaptiveModelSelector

class TestAdaptiveEngine(unittest.TestCase):
    def setUp(self):
        self.scorer = ThreatScorer()
        self.selector = AdaptiveModelSelector()

    def test_threat_scoring(self):
        res = self.scorer.calculate_threat("DDoS", 0.95)
        self.assertGreaterEqual(res["threat_score"], 60)
        self.assertIn(res["severity"], ["HIGH", "CRITICAL"])

    def test_critical_battery_switching(self):
        threat_dict = {"threat_score": 40, "severity": "MEDIUM"}
        selection = self.selector.select_optimal_model(threat_dict, battery_pct=10.0)
        self.assertEqual(selection["selected_model"], "lightweight")
        self.assertIn("Critical battery", selection["reason"])

if __name__ == "__main__":
    unittest.main()
