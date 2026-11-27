import subprocess
import unittest
from unittest.mock import patch

from src.api import app as api


class TestWindowsNetworkProbe(unittest.TestCase):
    @patch.object(api.subprocess, "run")
    @patch.object(api.sys, "platform", "win32")
    def test_uses_windows_ping_arguments_and_reads_loss(self, run):
        api.app.state.network_stats = {}
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="Packets: Sent = 2, Received = 2, Lost = 0 (0% loss)\n"
                   "Reply from 192.0.2.1: time<1ms\nReply from 192.0.2.1: time=2ms\n",
            stderr="",
        )
        result = api.probe_network(api.NetworkProbeRequest(target_ip="192.0.2.1", packets=2))
        self.assertEqual(run.call_args.args[0], ["ping", "-n", "2", "-w", "2000", "192.0.2.1"])
        self.assertEqual(result["packets_received"], 2)
        self.assertEqual(result["packet_loss_pct"], 0)


if __name__ == "__main__":
    unittest.main()
