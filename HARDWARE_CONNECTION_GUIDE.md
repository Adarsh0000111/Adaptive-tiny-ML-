# 🔌 ESP32 & Sensor Connection Guide

This guide explains how to connect physical sensors and an ESP32 edge device to the
Adaptive Multi-Level TinyML IDS, and how to stream live telemetry to the Streamlit
dashboard.

---

## 1. System Overview (PC ↔ ESP32)

```
+---------------------+      WiFi / MQTT      +-------------------+
|   PC (Linux/Win)    | <===================> |   ESP32 DevKit    |
|  Streamlit Dashboard|      (port 1883)      |   + INA219        |
|  (Python backend)   |                       |   + (optional)    |
|                     |                       |   DHT22 / PIR /   |
|  subscribes to:     |                       |   MQ-2            |
|  esp32/<id>/telemetry|                      |                   |
|  esp32/<id>/inference|                      |  runs: esp32_ids.ino|
+---------------------+                       +-------------------+
        ^
        | runs TinyML quantized model (esp32_model.h)
        | loads config.yaml, simulates or reads real energy
```

The PC side is the **dashboard + inference backend**. The ESP32 side is the **edge sensor node** that measures battery energy, runs the TinyML model, and publishes results.

---

## 2. Required Hardware

| Item | Qty | Purpose |
|------|-----|---------|
| ESP32 DevKit V1 (or ESP32-S3) | 1 | Main MCU, WiFi, I2C, ADC |
| INA219 breakout board | 1 | Bus voltage + shunt current → battery energy |
| DHT22 (optional) | 1 | Ambient temperature / humidity |
| PIR motion sensor (optional) | 1 | Presence / tamper detection |
| MQ-2 gas sensor (optional) | 1 | Gas/smoke anomaly (analog) |
| Jumper wires, breadboard | — | Connections |
| LiPo battery 3.7V (optional) | 1 | Untethered power |

---

## 3. Wiring (ESP32 → INA219)

| ESP32 Pin | INA219 Pin | Note |
|-----------|------------|------|
| 3.3V | VCC | Power |
| GND | GND | Common ground |
| GPIO22 | SCL | I2C clock |
| GPIO21 | SDA | I2C data |

INA219 measures:
- Bus voltage (battery voltage)
- Shunt voltage → current
- From these, the firmware computes energy consumed per inference

---

## 4. Software on the PC (Dashboard)

### 4.1 Install dependencies

```bash
cd "/run/media/anjney-shah/New Volume/PROJECTS/Adaptive ML EL"
pip install -r requirements.txt
# optional hardware deps (only if using a Linux host I2C bus / MQTT broker):
# pip install smbus2 paho-mqtt
```

### 4.2 Generate the ESP32 model header

```bash
# Full pipeline (generates esp32_model.h):
python -m src.main

# Or just the TinyML phase:
python -m src.tinyml.optimizer
```

This produces `models/tinyml/esp32_model.h` — copy it into your Arduino sketch folder.

### 4.3 Run the dashboard

```bash
streamlit run dashboard/app.py
```

Then open `http://localhost:8501`.

### 4.4 Connect MQTT broker (for live ESP32 telemetry)

Install and run a local MQTT broker (e.g. Mosquitto):

```bash
# Ubuntu/Debian
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

In the dashboard sidebar:
- MQTT Broker Host: `localhost` (or your PC IP)
- MQTT Port: `1883`
- Device ID: `esp32-ids-01`
- Click **Connect MQTT**

Then on the ESP32 side, set `MQTT_BROKER` to your PC's IP and upload the firmware.

---

## 5. Software on the ESP32

### 5.1 Arduino IDE setup
1. Install ESP32 board package: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
2. Select board: **ESP32 Dev Module**
3. Install libraries (Library Manager):
   - `Adafruit INA219`
   - `PubSubClient`
   - `ArduinoJson`
   - `DHT sensor library` (optional)

### 5.2 Add the TinyML model
Copy `models/tinyml/esp32_model.h` into your sketch folder next to `esp32_ids.ino`.

### 5.3 Configure & upload
Edit `hardware/esp32_ids/esp32_ids.ino`:
- `WIFI_SSID`, `WIFI_PASSWORD`
- `MQTT_BROKER` → your PC's IP address
- `DEVICE_ID` → matches dashboard device id

Compile → Upload → Open Serial Monitor (115200 baud).

---

## 6. How the connection works (data flow)

1. **ESP32** reads INA219 every second → battery %, bus voltage, shunt current.
2. **ESP32** builds the feature vector (must match training features & scaling).
3. **ESP32** calls `predict_tinyml_int8(features)` → leaf index → attack class.
4. **ESP32** computes a threat score and publishes JSON to MQTT:
   - `esp32/esp32-ids-01/telemetry` → `{battery_pct, bus_voltage_v, shunt_current_ma, model_tier}`
   - `esp32/esp32-ids-01/inference` → `{prediction, threat_score, severity, latency_ms, energy_uj, model}`
5. **Dashboard** subscribes to these topics and shows live gauges, tables, and charts (Tab 1 "System Status" and Tab 7 "ESP32 & Hardware").
6. **Dashboard → ESP32**: commands on `esp32/esp32-ids-01/command` (e.g. change model tier, sleep mode).

---

## 7. Feature scaling (critical)

The quantized tree expects **scaled** features (same `StandardScaler` from training). Persist the scaler parameters:

```python
# On the PC, after training:
import joblib
art = joblib.load("models/preprocessing_pipeline.joblib")
scaler = art["scaler"]
print("mean:", scaler.mean_)
print("scale:", scaler.scale_)
```

Hard-code `mean_` and `scale_` into the ESP32 sketch and apply:
```
x_scaled[i] = (x_raw[i] - mean[i]) / scale[i]
```
before calling `predict_tinyml_int8`. Without scaling, predictions will be wrong.

---

## 8. Optional: direct I2C from a Linux host (no ESP32 yet)

If you want to test the INA219 interface from the PC (e.g. a Raspberry Pi with an INA219 on `/dev/i2c-1`):

```bash
sudo apt install i2c-tools
sudo i2cdetect -y 1   # confirm 0x40 present
pip install smbus2
```

Then in Python:
```python
from src.energy.esp32_interface import ESP32INA219EnergyProvider
p = ESP32INA219EnergyProvider(i2c_bus=1)
p.connect_sensor()
print(p.get_battery_percentage())       # %
print(p.get_bus_voltage_v())            # V
print(p.get_shunt_current_ma())         # mA
```

---

## 9. Troubleshooting

| Issue | Fix |
|-------|-----|
| INA219 not found | Check SDA/SCL wiring, pull-up resistors, I2C address (0x40 default) |
| WiFi won't connect | Verify SSID/password, ESP32 within range |
| MQTT connect fails | Broker not running, wrong IP, firewall blocking 1883 |
| Wrong predictions | Feature scaling mismatch with training (see §7) |
| Dashboard shows stale data | Refresh browser; confirm MQTT client connected in sidebar |

---

## 10. Files reference

| File | Purpose |
|------|---------|
| `src/energy/esp32_interface.py` | PC-side INA219 driver + MQTT client stub |
| `src/energy/provider_base.py` | Abstract energy-provider interface |
| `models/tinyml/esp32_model.h` | Exported quantized TinyML model (C header) |
| `hardware/esp32_ids/esp32_ids.ino` | ESP32 Arduino firmware |
| `dashboard/app.py` | Streamlit dashboard (ESP32 tab) |
| `config.yaml` | Energy, battery, model settings |

