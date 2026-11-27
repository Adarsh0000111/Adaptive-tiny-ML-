"""Optional, privacy-conscious packet counters for authorized capture.

These observations intentionally do not claim compatibility with the fitted
Edge-IIoTset model schema. Payloads and endpoint addresses are not retained.
"""

from __future__ import annotations

import logging
import time
from collections import Counter

import pandas as pd
from src.data.edge_packet_schema import packet_features_from_scapy

logger = logging.getLogger(__name__)


class RealTimeNetworkSniffer:
    def __init__(self, interface: str | None = None):
        self.interface = interface
        try:
            import scapy.all as scapy
        except ImportError:
            scapy = None
        self.scapy = scapy
        self.scapy_available = scapy is not None

    def parse_packet_to_features(self, packet) -> dict | None:
        """Return directly observable counters without invented feature values."""
        if not self.scapy_available or packet is None:
            return None
        scapy = self.scapy
        protocol = "other"
        if packet.haslayer(scapy.TCP):
            protocol = "tcp"
        elif packet.haslayer(scapy.UDP):
            protocol = "udp"
        elif packet.haslayer(scapy.IP):
            protocol = "ip"
        record = {
            "timestamp_epoch_s": float(getattr(packet, "time", time.time())),
            "packet_bytes": int(len(packet)),
            "protocol": protocol,
            "tcp_fin": 0,
            "tcp_syn": 0,
            "tcp_ack": 0,
            "tcp_rst": 0,
            "source": "authorized_live_capture",
            "model_compatible": False,
        }
        if packet.haslayer(scapy.TCP):
            flags = str(packet[scapy.TCP].flags)
            record.update({"tcp_fin": int("F" in flags), "tcp_syn": int("S" in flags), "tcp_ack": int("A" in flags), "tcp_rst": int("R" in flags)})
        return record

    def capture_live_stream(self, max_packets: int = 100, timeout: int = 10) -> pd.DataFrame:
        if not self.scapy_available:
            raise RuntimeError("Scapy is not installed; live capture is unavailable. No synthetic packet records were generated.")
        if max_packets < 1 or timeout < 1:
            raise ValueError("max_packets and timeout must be positive")
        records = []
        start = time.time()

        def process(packet):
            record = self.parse_packet_to_features(packet)
            if record is not None:
                records.append(record)

        logger.warning("Starting packet capture on %s. Capture only networks for which you have authorization.", self.interface or "default interface")
        self.scapy.sniff(iface=self.interface, prn=process, count=max_packets, timeout=timeout, store=False)
        elapsed = max(time.time() - start, 1e-9)
        protocols = Counter(row["protocol"] for row in records)
        totals = {
            "timestamp_epoch_s": time.time(),
            "window_seconds": elapsed,
            "packets": len(records),
            "bytes": sum(row["packet_bytes"] for row in records),
            "packets_per_second": len(records) / elapsed,
            "bytes_per_second": sum(row["packet_bytes"] for row in records) / elapsed,
            "tcp_syn": sum(row["tcp_syn"] for row in records),
            "tcp_fin": sum(row["tcp_fin"] for row in records),
            "tcp_rst": sum(row["tcp_rst"] for row in records),
            "protocol_counts": dict(protocols),
            "source": "authorized_live_capture",
            "model_compatible": False,
        }
        return pd.DataFrame([totals])

    def capture_model_packets(self, max_packets: int = 100, timeout: int = 10) -> list[dict]:
        """Capture bounded packet-derived model rows; retain no packet bytes or endpoints."""
        if not self.scapy_available:
            raise RuntimeError("Scapy is not installed; live capture is unavailable.")
        if max_packets < 1 or timeout < 1:
            raise ValueError("max_packets and timeout must be positive")
        rows: list[dict] = []
        state: dict = {}

        def process(packet):
            row = packet_features_from_scapy(packet, self.scapy, state)
            if row is not None:
                rows.append(row)

        logger.warning("Starting authorized packet feature capture on %s", self.interface or "default interface")
        self.scapy.sniff(iface=self.interface, prn=process, count=max_packets, timeout=timeout, store=False)
        return rows

    def capture_with_handler(self, handler, max_packets: int = 100, timeout: int = 5, *, include_endpoint_metadata: bool = False) -> dict:
        """Extract and handle each packet inside Scapy's live callback, then discard it."""
        if not self.scapy_available:
            raise RuntimeError("Scapy is not installed; live capture is unavailable.")
        if max_packets < 1 or timeout < 1:
            raise ValueError("max_packets and timeout must be positive")
        state: dict = {}
        captured = 0
        failures: list[Exception] = []
        started = time.perf_counter()

        def process(packet):
            nonlocal captured
            if failures:
                return
            try:
                row = packet_features_from_scapy(packet, self.scapy, state, include_endpoint_metadata=include_endpoint_metadata)
                if row is not None:
                    handler(row)
                    captured += 1
            except Exception as exc:  # Surface callback failures after sniff returns.
                if not failures:
                    failures.append(exc)

        logger.warning("Processing authorized packets live on %s", self.interface or "default interface")
        self.scapy.sniff(iface=self.interface, prn=process, count=max_packets, timeout=timeout, store=False)
        if failures:
            raise failures[0]
        return {"packets_processed": captured, "window_seconds": time.perf_counter() - started}
