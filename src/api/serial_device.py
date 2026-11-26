from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from serial import Serial, SerialException
from serial.tools import list_ports
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
BAUD_RATE = 115200
USB_VID_PID = {(0x1A86, 0x55D3)}


def enumerate_ports() -> list[dict[str, Any]]:
    devices = []
    for port in list_ports.comports():
        if port.device.startswith(('/dev/ttyACM', '/dev/ttyUSB', 'COM')):
            vidpid = (port.vid, port.pid)
            devices.append({
                "port": port.device,
                "description": port.description,
                "vid": port.vid,
                "pid": port.pid,
                "serial_number": port.serial_number,
                "likely_usb_serial_adapter": vidpid in USB_VID_PID,
                "accessible": os.access(port.device, os.R_OK | os.W_OK),
            })
    return devices


class USBReplayService:
    """Send saved, preprocessed test rows to the on-device fixed-point tree."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_run: dict[str, Any] | None = None

    def status(self) -> dict[str, Any]:
        return {
            "available": True,
            "transport": "USB serial",
            "baud_rate": BAUD_RATE,
            "ports": enumerate_ports(),
            "firmware_contract": "SARA-ESP32-USB; exact READY handshake required",
            "input_source": "saved test split replay or explicitly authorized per-packet SARA capture with USB low-tier inference",
            "last_run": self._last_run,
        }

    def replay(self, port: str | None, count: int, sample_start: int, inference_service, *, include_latency_samples: bool = False) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A USB inference/replay session is already running")
        try:
            return self._replay_locked(port, count, sample_start, inference_service, include_latency_samples)
        finally:
            self._lock.release()

    def live_capture(self, port: str | None, rows: list[dict], inference_service) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A USB inference/replay session is already running")
        try:
            return self._replay_locked(port, len(rows), 0, inference_service, False, raw_rows=rows)
        finally:
            self._lock.release()

    def live_session(self, port: str | None, inference_service):
        return USBLiveSession(self, port, inference_service)

    def _replay_locked(self, port: str | None, count: int, sample_start: int, inference_service, include_latency_samples: bool, raw_rows: list[dict] | None = None) -> dict[str, Any]:
        ports = enumerate_ports()
        chosen = port or next((p["port"] for p in ports if p["likely_usb_serial_adapter"]), None)
        if chosen is None:
            raise ValueError("No USB serial device detected. Connect the ESP32-S3 and check port permissions.")
        if chosen not in {p["port"] for p in ports}:
            raise ValueError(f"Requested serial port {chosen!r} is not currently detected")
        if count < 1 or count > 10000:
            raise ValueError("count must be between 1 and 10000")
        if raw_rows is None and (sample_start < 0 or sample_start + count > len(inference_service.X_test)):
            raise ValueError(f"Requested rows must fit in saved test split (0..{len(inference_service.X_test)-1})")

        if raw_rows is None:
            data = np.load(ROOT / "data/processed/processed_dataset.npz", allow_pickle=False)
            X = data["X_test"][sample_start:sample_start + count].astype(np.float32, copy=False)
            y = data["y_test_multi"][sample_start:sample_start + count].astype(np.int64, copy=False)
        else:
            if not raw_rows:
                raise ValueError("No TCP/UDP packets were observed in the authorized capture window")
            X = inference_service.transform_packet_rows(raw_rows)
            y = None
        class_names = inference_service.class_names
        connection_started = time.perf_counter()
        device_us: list[int] = []
        round_trip_ms: list[float] = []
        latest_sensors: dict[str, Any] | None = None
        sensor_samples: list[dict[str, Any]] = []
        observed: list[int] = []
        protocol_retries = 0

        try:
            with Serial(chosen, BAUD_RATE, timeout=0.25, write_timeout=2.0, dsrdtr=False, rtscts=False) as serial_port:
                # Opening a USB serial monitor can reset ESP32 boards; wait for the new firmware handshake.
                serial_port.dtr = False
                serial_port.rts = False
                handshake_deadline = time.monotonic() + 12
                handshake = None
                while time.monotonic() < handshake_deadline:
                    line = serial_port.readline(128).decode("ascii", errors="replace").strip()
                    if line.startswith("READY,"):
                        handshake = line
                        break
                expected_prefix = "READY,SARA-ESP32-USB,features=18,classes=6"
                if handshake is None:
                    raise RuntimeError("No SARA USB firmware READY response. Flash hardware/esp32_usb_ids/esp32_usb_ids.ino first; current Wi-Fi/sensor sketch is not this firmware.")
                if handshake != expected_prefix and not handshake.startswith(expected_prefix + ","):
                    raise RuntimeError(f"Unexpected ESP32 firmware contract: {handshake!r}; expected prefix {expected_prefix!r}")
                sensor_capabilities = {
                    key: value == "1"
                    for key, value in (item.split("=", 1) for item in handshake.split(",")[4:] if "=" in item)
                }
                serial_ready_ms = (time.perf_counter() - connection_started) * 1000.0
                transfer_started = time.perf_counter()

                for offset, row in enumerate(X):
                    request_id = offset if raw_rows is not None else sample_start + offset
                    values = ",".join(format(float(value), ".9g") for value in row)
                    command = f"INFER,{request_id},{values}\n".encode("ascii")
                    response = None
                    sent_at = time.perf_counter()
                    for attempt in range(3):
                        serial_port.write(command)
                        serial_port.flush()
                        response_deadline = time.monotonic() + 3
                        retry_row = False
                        while time.monotonic() < response_deadline:
                            raw = serial_port.readline(192)
                            if not raw:
                                continue
                            decoded = raw.decode("ascii", errors="replace").strip()
                            if decoded.startswith("ERROR,"):
                                if decoded == "ERROR,BAD_COMMAND" and attempt < 2:
                                    protocol_retries += 1
                                    retry_row = True
                                    break
                                raise RuntimeError(f"ESP32 rejected request {request_id}: {decoded}")
                            parts = decoded.split(",")
                            if len(parts) == 18 and parts[0] == "RESULT":
                                response = parts
                                break
                        if response is not None or not retry_row:
                            break
                    if response is None:
                        raise RuntimeError(f"Timed out waiting for ESP32 response to sample {request_id}")
                    if int(response[1]) != request_id:
                        raise RuntimeError(f"Serial request ID mismatch: expected {request_id}, got {response[1]}")
                    class_index = int(response[2])
                    device_class = response[3]
                    inference_us = int(response[4])
                    if not 0 <= class_index < len(class_names) or class_names[class_index] != device_class:
                        raise RuntimeError(f"ESP32 returned an invalid class mapping for sample {request_id}")
                    observed.append(class_index)
                    device_us.append(inference_us)
                    round_trip_ms.append((time.perf_counter() - sent_at) * 1000.0)
                    sensor_values = [None if value == "NA" else float(value) for value in response[6:17]]
                    sensor_values = [value if value is None or np.isfinite(value) else None for value in sensor_values]
                    latest_sensors = {
                        "node_id": "esp32-s3-usb",
                        "timestamp": time.time(),
                        "source": "physical",
                        "uptime_ms": int(response[5]),
                        "voltage_v": sensor_values[0],
                        "current_a": sensor_values[1],
                        "power_w": sensor_values[2],
                        "energy_j": sensor_values[3],
                        "temperature_ambient_c": sensor_values[4],
                        "humidity_pct": sensor_values[5],
                        "vibration": bool(int(response[12])),
                        "movement": bool(int(response[13])),
                        "accel_x_g": sensor_values[8],
                        "accel_y_g": sensor_values[9],
                        "accel_z_g": sensor_values[10],
                        "alert": bool(int(response[17])),
                        "sensor_status": sensor_capabilities,
                        "quality": "reported", "sensor_presence_quality": "complete" if all(sensor_capabilities.get(key, False) for key in ("ina219", "dht22", "mpu6500", "oled")) else "partial",
                    }
                    sensor_samples.append(latest_sensors)

        except SerialException as exc:
            raise RuntimeError(f"Cannot use {chosen}: {exc}. Check serial permissions and close other serial monitors.") from exc

        elapsed = time.perf_counter() - transfer_started
        y_pred = np.asarray(observed, dtype=np.int64)
        labels = np.arange(len(class_names))
        report = classification_report(y, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0) if y is not None else None
        device_latency_ms = np.asarray(device_us, dtype=np.float64) / 1000.0
        result = {
            "transport": "USB serial",
            "port": chosen,
            "firmware": handshake,
            "model": "lightweight fixed-point decision tree",
            "model_artifact": "models/lightweight/model.joblib",
            "feature_schema": inference_service.schema_version,
            "mode": "authorized_live_packet_inference_on_esp32" if raw_rows is not None else "saved_test_replay_on_esp32",
            "wifi_used": False,
            "samples": int(len(X)),
            "sample_start": int(sample_start) if raw_rows is None else None,
            "accuracy": float(accuracy_score(y, y_pred)) if y is not None else None,
            "macro_f1": float(f1_score(y, y_pred, labels=labels, average="macro", zero_division=0)) if y is not None else None,
            "per_class": {name: {metric: float(report[name][metric]) for metric in ("precision", "recall", "f1-score", "support")} for name in class_names} if report else None,
            "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist() if y is not None else None,
            "predictions": [{"packet_index": i, "prediction": class_names[p], "class_index": int(p), "inference_us": int(device_us[i])} for i, p in enumerate(y_pred)] if raw_rows is not None else None,
            "device_inference_us": {"mean": float(np.mean(device_latency_ms) * 1000), "median": float(np.median(device_latency_ms) * 1000), "p95": float(np.percentile(device_latency_ms, 95) * 1000)},
            "usb_startup_ms": float(serial_ready_ms),
            "usb_round_trip_ms": {"mean": float(np.mean(round_trip_ms)), "median": float(np.median(round_trip_ms)), "p95": float(np.percentile(round_trip_ms, 95)), "total": float(elapsed * 1000)},
            "serial_protocol_retries": protocol_retries,
            "energy": "not measured; USB-powered board has no calibrated power measurement in this test",
            "input_note": "Laptop captured bounded TCP/UDP packet metadata, applied saved training preprocessing, and sent feature vectors by USB; payloads and endpoints were not retained." if raw_rows is not None else "Features are preprocessed saved test rows sent by laptop; this is hardware inference replay, not live traffic detection.",
            "device_telemetry": latest_sensors,
            "completed_at_unix": time.time(),
        }
        if include_latency_samples:
            result["device_inference_samples_us"] = device_us
            result["usb_round_trip_samples_ms"] = round_trip_ms
            result["device_telemetry_samples"] = sensor_samples
            result["hardware_prediction_indices"] = observed
        self._last_run = result
        return result


class USBLiveSession:
    """Keep one verified USB session open while packet callbacks make tier choices."""
    def __init__(self, service: USBReplayService, port: str | None, inference_service):
        self.service = service
        self.inference_service = inference_service
        self.port = port
        self.serial = None
        self.request_id = 0
        self.inference_us: list[int] = []
        self.round_trip_ms: list[float] = []
        self.latest_sensors = None
        self.firmware = None
        self.sensor_capabilities = {}

    def _sensor_snapshot(self, parts: list[str], *, uptime_index: int = 1, fields_index: int = 2):
        try:
            values = [None if value == "NA" else float(value) for value in parts[fields_index:fields_index + 12]]
            values = [value if value is None or np.isfinite(value) else None for value in values]
            self.latest_sensors = {
                "node_id": "esp32-s3-usb", "timestamp": time.time(), "source": "physical",
                "uptime_ms": int(parts[uptime_index]), "voltage_v": values[0], "current_a": values[1],
                "power_w": values[2], "energy_j": values[3], "temperature_ambient_c": values[4],
                "humidity_pct": values[5], "vibration": bool(int(parts[fields_index + 6])),
                "movement": bool(int(parts[fields_index + 7])), "accel_x_g": values[8],
                "accel_y_g": values[9], "accel_z_g": values[10], "alert": bool(int(parts[fields_index + 11])),
                "sensor_status": self.sensor_capabilities,
                "quality": "reported", "sensor_presence_quality": "complete" if all(self.sensor_capabilities.get(key, False) for key in ("ina219", "dht22", "mpu6500", "oled")) else "partial",
            }
        except (ValueError, IndexError):
            return

    def refresh_sensor_snapshot(self):
        """Drain periodic SENSOR frames without waiting; snapshots are emitted by ESP32 firmware."""
        if self.serial is None:
            return self.latest_sensors
        while self.serial.in_waiting:
            raw = self.serial.readline(192)
            if not raw:
                break
            parts = raw.decode("ascii", errors="replace").strip().split(",")
            if len(parts) == 14 and parts[0] == "SENSOR":
                self._sensor_snapshot(parts)
        return self.latest_sensors

    def __enter__(self):
        if not self.service._lock.acquire(blocking=False):
            raise RuntimeError("A USB inference/replay session is already running")
        try:
            ports = enumerate_ports()
            chosen = self.port or next((p["port"] for p in ports if p["likely_usb_serial_adapter"]), None)
            if chosen is None or chosen not in {p["port"] for p in ports}:
                raise ValueError("No USB serial device detected. Connect the ESP32-S3 and check port permissions.")
            self.port = chosen
            self.serial = Serial(chosen, BAUD_RATE, timeout=0.25, write_timeout=2.0, dsrdtr=False, rtscts=False)
            self.serial.dtr = False
            self.serial.rts = False
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                line = self.serial.readline(128).decode("ascii", errors="replace").strip()
                if line.startswith("READY,"):
                    self.firmware = line
                    break
            prefix = "READY,SARA-ESP32-USB,features=18,classes=6"
            if self.firmware is None:
                raise RuntimeError("No SARA USB firmware READY response; flash hardware/esp32_usb_ids first")
            if self.firmware != prefix and not self.firmware.startswith(prefix + ","):
                raise RuntimeError(f"Unexpected ESP32 firmware contract: {self.firmware!r}")
            self.sensor_capabilities = {key: value == "1" for key, value in
                (item.split("=", 1) for item in self.firmware.split(",")[4:] if "=" in item)}
            return self
        except Exception:
            if self.serial is not None:
                self.serial.close()
            self.service._lock.release()
            raise

    def infer(self, row: np.ndarray) -> dict[str, Any]:
        if self.serial is None:
            raise RuntimeError("USB live session is not open")
        request_id = self.request_id
        command = f"INFER,{request_id}," + ",".join(format(float(value), ".9g") for value in row) + "\n"
        started = time.perf_counter()
        self.serial.write(command.encode("ascii"))
        self.serial.flush()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            raw = self.serial.readline(192)
            if not raw:
                continue
            parts = raw.decode("ascii", errors="replace").strip().split(",")
            if parts[0] == "ERROR":
                raise RuntimeError(f"ESP32 rejected live packet features: {','.join(parts[1:])}")
            if len(parts) != 18 or parts[0] != "RESULT":
                continue
            if int(parts[1]) != request_id:
                continue
            index = int(parts[2])
            names = self.inference_service.class_names
            if not 0 <= index < len(names) or names[index] != parts[3]:
                raise RuntimeError("ESP32 returned a class outside the trained label mapping")
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            inference = int(parts[4])
            self.request_id += 1
            self.inference_us.append(inference)
            self.round_trip_ms.append(elapsed_ms)
            sensors = [None if value == "NA" else float(value) for value in parts[6:17]]
            sensors = [value if value is None or np.isfinite(value) else None for value in sensors]
            self.latest_sensors = {
                "node_id": "esp32-s3-usb", "timestamp": time.time(), "source": "physical",
                "uptime_ms": int(parts[5]), "voltage_v": sensors[0], "current_a": sensors[1],
                "power_w": sensors[2], "energy_j": sensors[3], "temperature_ambient_c": sensors[4],
                "humidity_pct": sensors[5], "vibration": bool(int(parts[12])),
                "movement": bool(int(parts[13])), "accel_x_g": sensors[8], "accel_y_g": sensors[9],
                "accel_z_g": sensors[10], "alert": bool(int(parts[17])),
                "sensor_status": self.sensor_capabilities,
                "quality": "good" if all(self.sensor_capabilities.get(key, False) for key in ("ina219", "dht22", "mpu6500", "oled")) else "partial",
            }
            self.refresh_sensor_snapshot()
            return {"class_index": index, "prediction": names[index], "inference_us": inference, "round_trip_ms": elapsed_ms}
        raise RuntimeError(f"Timed out waiting for ESP32 response to live packet {request_id}")

    def summary(self) -> dict[str, Any]:
        return {
            "port": self.port, "firmware": self.firmware,
            "device_inference_us": {"mean": float(np.mean(self.inference_us)), "median": float(np.median(self.inference_us)), "p95": float(np.percentile(self.inference_us, 95))} if self.inference_us else None,
            "usb_round_trip_ms": {"mean": float(np.mean(self.round_trip_ms)), "median": float(np.median(self.round_trip_ms)), "p95": float(np.percentile(self.round_trip_ms, 95)), "total": float(np.sum(self.round_trip_ms))} if self.round_trip_ms else None,
            "device_telemetry": self.latest_sensors,
        }

    def __exit__(self, exc_type, exc, tb):
        if self.serial is not None:
            self.serial.close()
            self.serial = None
        self.service._lock.release()
