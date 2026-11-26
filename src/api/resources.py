from __future__ import annotations

from datetime import datetime, timezone
import threading
import time


def sample_resources() -> dict:
    """Read host resources when available; distinguish absent sensors from zero."""
    result = {"timestamp": datetime.now(timezone.utc).isoformat(), "source": "physical_host", "cpu_percent": None, "memory_percent": None, "available_memory_mb": None, "cpu_temperature_c": None}
    try:
        import psutil
        result["cpu_percent"] = float(psutil.cpu_percent(interval=None))
        mem = psutil.virtual_memory()
        result["memory_percent"] = float(mem.percent)
        result["available_memory_mb"] = float(mem.available / (1024 * 1024))
        temps = psutil.sensors_temperatures()
        readings = [item.current for rows in temps.values() for item in rows if item.current is not None]
        if readings:
            result["cpu_temperature_c"] = float(max(readings))
    except (ImportError, OSError, AttributeError):
        result["source"] = "unavailable"
    try:
        import psutil
        counters = psutil.net_io_counters()
        result["network_counters"] = {"bytes_sent": int(counters.bytes_sent), "bytes_recv": int(counters.bytes_recv), "packets_sent": int(counters.packets_sent), "packets_recv": int(counters.packets_recv), "source": "host_cumulative_counters; rate requires timestamped deltas"}
    except (ImportError, OSError, AttributeError):
        result["network_counters"] = None
    result["resource_stress"] = round(max(result["cpu_percent"] or 0, result["memory_percent"] or 0) / 100, 4)
    return result


class HostMetricsSampler:
    """Measure host resources and derive rates from successive OS counters."""

    def __init__(self):
        self._previous = None
        self._lock = threading.Lock()

    def sample(self) -> dict:
        result = sample_resources()
        now = time.monotonic()
        counters = result.get("network_counters")
        with self._lock:
            previous = self._previous
            rates = {"bytes_per_second": None, "packets_per_second": None, "average_packet_size_bytes": None, "bandwidth_bps": None, "source": "host cumulative interface counters"}
            if counters and previous:
                elapsed = now - previous["time"]
                byte_delta = max(0, counters["bytes_sent"] + counters["bytes_recv"] - previous["bytes"])
                packet_delta = max(0, counters["packets_sent"] + counters["packets_recv"] - previous["packets"])
                if elapsed > 0:
                    byte_rate = byte_delta / elapsed
                    rates.update({"bytes_per_second": byte_rate, "packets_per_second": packet_delta / elapsed, "average_packet_size_bytes": byte_delta / packet_delta if packet_delta else None, "bandwidth_bps": byte_rate * 8})
            if counters:
                self._previous = {"time": now, "bytes": counters["bytes_sent"] + counters["bytes_recv"], "packets": counters["packets_sent"] + counters["packets_recv"]}
            result["network_rates"] = rates
        return result
