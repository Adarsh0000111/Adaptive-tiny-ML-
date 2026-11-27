import unittest

from src.api.app import USBLiveCaptureRequest, live_threat_summary
from src.data.edge_packet_schema import packet_features_from_scapy


class TestLiveThreatSummary(unittest.TestCase):
    def test_counts_non_normal_model_predictions_by_class(self):
        summary = live_threat_summary([
            {"prediction": "Normal"},
            {"prediction": "DDoS"},
            {"prediction": "DDoS"},
            {"prediction": "Web_Attack"},
        ])
        self.assertEqual(summary["threat_prediction_count"], 3)
        self.assertEqual(summary["threat_prediction_counts_by_class"], {"DDoS": 2, "Web_Attack": 1})

    def test_endpoint_collection_requires_separate_opt_in(self):
        request = USBLiveCaptureRequest(authorized=True)
        self.assertFalse(request.include_endpoint_addresses)
        opted_in = USBLiveCaptureRequest(authorized=True, include_endpoint_addresses=True)
        self.assertTrue(opted_in.include_endpoint_addresses)

    def test_live_packet_endpoints_are_only_returned_when_requested(self):
        try:
            import scapy.all as scapy
        except ImportError:
            self.skipTest("Scapy is not installed")
        packet = scapy.IP(src="192.0.2.10", dst="198.51.100.20") / scapy.TCP(sport=43123, dport=443)
        private_row = packet_features_from_scapy(packet, scapy, {})
        visible_row = packet_features_from_scapy(packet, scapy, {}, include_endpoint_metadata=True)
        self.assertNotIn("endpoint_source_ip", private_row)
        self.assertEqual(visible_row["endpoint_source_ip"], "192.0.2.10")
        self.assertEqual(visible_row["endpoint_destination_ip"], "198.51.100.20")
        self.assertEqual(visible_row["endpoint_source_port"], 43123)
        self.assertEqual(visible_row["endpoint_destination_port"], 443)


if __name__ == "__main__":
    unittest.main()
