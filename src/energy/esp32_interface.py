"""
ESP32 Hardware Interface for the Adaptive TinyML IDS.

Provides:
  * ESP32INA219EnergyProvider - reads battery voltage/current from an INA219
                                over I2C (smbus2), with a graceful stub.
  * ESP32MQTTClient           - optional WiFi/MQTT bridge to the dashboard.

Hardware deps (smbus2, paho-mqtt) are optional; module degrades to stub mode.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from src.config import load_config
from src.energy.provider_base import EnergyProvider

logger = logging.getLogger(__name__)

try:  # pragma: no cover - hardware dependent
    import smbus2 as smbus  # type: ignore
    _HAS_SMBUS = True
except Exception:
    smbus = None  # type: ignore
    _HAS_SMBUS = False

try:  # pragma: no cover - network dependent
    import paho.mqtt.client as mqtt  # type: ignore
    _HAS_MQTT = True
except Exception:
    mqtt = None  # type: ignore
    _HAS_MQTT = False

_INA219_DEFAULT_ADDR = 0x40
_INA219_REG_CONFIG = 0x00
_INA219_REG_BUS_VOLTAGE = 0x02
_INA219_REG_CURRENT = 0x04
_INA219_CONFIG_BRNG_32V = 0x2000
_INA219_CONFIG_PGA_40MV = 0x0000
_INA219_CONFIG_BADC_12BIT = 0x0100
_INA219_CONFIG_SADC_12BIT = 0x0080
_INA219_CONFIG_MODE_SANDBVOLT_CONTINUOUS = 0x0007
_SHUNT_OHMS = 0.1


def _read_u16(bus, addr: int, reg: int) -> int:
    if not _HAS_SMBUS:
        raise RuntimeError("smbus2 is required for hardware I2C access")
    val = bus.read_word_data(addr, reg)
    return ((val & 0xFF) << 8) | (val >> 8)


class ESP32INA219EnergyProvider(EnergyProvider):
    """Hardware-ready energy provider for ESP32 + INA219 I2C sensor."""

    def __init__(
        self,
        i2c_bus: int = 1,
        i2c_address: int = _INA219_DEFAULT_ADDR,
        shunt_ohms: float = _SHUNT_OHMS,
        voltage_min_v: float = 3.0,
        voltage_max_v: float = 4.2,
        capacity_mah: float = 1200.0,
        config_path: str = "config.yaml",
    ):
        config = load_config(config_path)
        e_cfg = config.get("energy_simulation", {})

        self.i2c_bus = i2c_bus
        self.i2c_address = i2c_address
        self.shunt_ohms = float(shunt_ohms)
        self.voltage_min_v = float(voltage_min_v)
        self.voltage_max_v = float(voltage_max_v)
        self.capacity_mah = float(e_cfg.get("battery_capacity_mAh", capacity_mah))
        self.voltage_v = float(e_cfg.get("voltage_v", 3.3))

        self._bus = None
        self.connected = False
        self._last_bus_voltage = 3.85

        self.comm_energy_uj_per_kb = float(e_cfg.get("comm_energy_uJ_per_kb", 45.0))
        self.fed_train_energy_uj_per_epoch = float(
            e_cfg.get("fed_train_energy_uJ_per_epoch", 12500.0)
        )
        self.model_energy_uj = {
            "lightweight": float(config["models"]["lightweight"]["est_energy_uJ_per_inf"]),
            "medium": float(config["models"]["medium"]["est_energy_uJ_per_inf"]),
            "heavy": float(config["models"]["heavy"]["est_energy_uJ_per_inf"]),
        }
        self.idle_power_mw = float(e_cfg.get("idle_power_mW", 15.0))
        self.total_energy_consumed_uj = 0.0
        self.battery_pct = 100.0

    # ------------------------------------------------------------------ #
    # Hardware connection
    # ------------------------------------------------------------------ #
    def connect_sensor(self) -> bool:
        """Initialise the I2C bus and configure the INA219."""
        if _HAS_SMBUS:
            try:
                self._bus = smbus.SMBus(self.i2c_bus)
                config_val = (
                    _INA219_CONFIG_BRNG_32V
                    | _INA219_CONFIG_PGA_40MV
                    | _INA219_CONFIG_BADC_12BIT
                    | _INA219_CONFIG_SADC_12BIT
                    | _INA219_CONFIG_MODE_SANDBVOLT_CONTINUOUS
                )
                self._write_u16(_INA219_REG_CONFIG, config_val)
                self.connected = True
                logger.info(
                    f"INA219 connected on bus {self.i2c_bus} addr 0x{self.i2c_address:02X}"
                )
                return True
            except Exception as exc:  # pragma: no cover - hardware dependent
                logger.warning(f"Failed to connect to INA219: {exc}. Falling back to stub.")
                self._bus = None
        self.connected = False
        return False

    def _write_u16(self, reg: int, value: int) -> None:
        if not _HAS_SMBUS or self._bus is None:
            return
        high = (value >> 8) & 0xFF
        low = value & 0xFF
        self._bus.write_i2c_block_data(self.i2c_address, reg, [high, low])

    def close(self) -> None:
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
        self.connected = False

    # ------------------------------------------------------------------ #
    # EnergyProvider abstract API
    # ------------------------------------------------------------------ #
    def get_battery_percentage(self) -> float:
        if self.connected and self._bus is not None:
            try:
                raw = _read_u16(self._bus, self.i2c_address, _INA219_REG_BUS_VOLTAGE)
                # INA219 bus voltage register is 13-bit, 4 mV per LSB
                self._last_bus_voltage = (raw >> 3) * 0.004
            except Exception as exc:  # pragma: no cover - hardware dependent
                logger.warning(f"INA219 read failed: {exc}")
        pct = max(
            0.0,
            min(
                100.0,
                ((self._last_bus_voltage - self.voltage_min_v)
                 / (self.voltage_max_v - self.voltage_min_v)) * 100.0,
            ),
        )
        self.battery_pct = pct
        return pct

    def get_bus_voltage_v(self) -> float:
        return self._last_bus_voltage

    def get_shunt_current_ma(self) -> float:
        if not (self.connected and self._bus is not None):
            return 0.0
        try:
            raw = _read_u16(self._bus, self.i2c_address, _INA219_REG_CURRENT)
            if raw & 0x8000:
                raw -= 0x10000
            shunt_voltage_mv = raw * 0.01
            return shunt_voltage_mv / (self.shunt_ohms * 1000.0) * 1000.0
        except Exception:
            return 0.0

    def get_battery_capacity_mah(self) -> float:
        return self.capacity_mah

    def consume_inference_energy(self, model_level: str) -> float:
        uj = self.model_energy_uj.get(model_level.lower(), 10.0)
        self.total_energy_consumed_uj += uj
        self._deduct_uj(uj)
        return uj

    def consume_communication_energy(self, payload_kb: float) -> float:
        uj = float(payload_kb) * self.comm_energy_uj_per_kb
        self.total_energy_consumed_uj += uj
        self._deduct_uj(uj)
        return uj

    def consume_federated_training_energy(self, num_epochs: int) -> float:
        uj = int(num_epochs) * self.fed_train_energy_uj_per_epoch
        self.total_energy_consumed_uj += uj
        self._deduct_uj(uj)
        return uj

    def update_idle_drain(self, duration_s: float) -> float:
        # Power (mW) * time (s) = mJ -> microjoules (* 1000)
        uj = self.idle_power_mw * float(duration_s) * 1000.0
        self.total_energy_consumed_uj += uj
        self._deduct_uj(uj)
        return uj

    def _deduct_uj(self, uj: float) -> None:
        capacity_j = (self.capacity_mah / 1000.0) * self.voltage_v * 3.6
        if capacity_j <= 0:
            return
        joules = uj / 1e6
        self.battery_pct = max(0.0, self.battery_pct - (joules / capacity_j) * 100.0)


# ---------------------------------------------------------------------------
# Optional MQTT bridge: publish energy/metrics to the dashboard
# ---------------------------------------------------------------------------
class ESP32MQTTClient:
    """Minimal MQTT client for the ESP32 side (JSON telemetry)."""

    def __init__(
        self,
        device_id: str = "esp32-ids-01",
        broker_host: str = "localhost",
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        keepalive: int = 60,
    ):
        self.device_id = device_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.keepalive = keepalive
        self.client = None
        self.connected = False
        self._command_callback = None

        if not _HAS_MQTT:
            logger.warning(
                "paho-mqtt not installed. ESP32MQTTClient will operate in stub mode."
            )

    def connect(self) -> bool:
        if not _HAS_MQTT:
            return False
        self.client = mqtt.Client(client_id=self.device_id)  # type: ignore
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        try:
            self.client.connect(self.broker_host, self.broker_port, self.keepalive)
            self.client.loop_start()
            for _ in range(50):
                if self.connected:
                    break
                time.sleep(0.1)
            return self.connected
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning(f"MQTT connect failed: {exc}")
            return False

    def _on_connect(self, client, userdata, flags, rc):  # type: ignore
        if rc == 0:
            self.connected = True
            client.subscribe(f"esp32/{self.device_id}/command")
            logger.info(f"MQTT connected, subscribed to esp32/{self.device_id}/command")
        else:
            logger.warning(f"MQTT connect returned code {rc}")

    def _on_message(self, client, userdata, msg):  # type: ignore
        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            payload = {"raw": msg.payload.decode(errors="ignore")}
        if self._command_callback:
            self._command_callback(payload)

    def set_command_callback(self, callback) -> None:
        self._command_callback = callback

    def publish_telemetry(self, data: dict) -> bool:
        return self._publish(f"esp32/{self.device_id}/telemetry", data)

    def publish_inference(self, data: dict) -> bool:
        return self._publish(f"esp32/{self.device_id}/inference", data)

    def _publish(self, topic: str, data: dict) -> bool:
        if not (_HAS_MQTT and self.client and self.connected):
            return False
        try:
            self.client.publish(topic, json.dumps(data), qos=1)
            return True
        except Exception as exc:
            logger.warning(f"MQTT publish failed: {exc}")
            return False

    def disconnect(self) -> None:
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        self.connected = False


__all__ = [
    "ESP32INA219EnergyProvider",
    "ESP32MQTTClient",
    "EnergyProvider",
]
