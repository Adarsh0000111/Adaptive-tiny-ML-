# 🔌 ESP32 & Sensor Connection Guide

This guide explains how to connect an ESP32 and INA219 sensor to the Adaptive
Multi-Level TinyML IDS and stream live telemetry to its React frontend.

---

## 1. System Overview (PC ↔ ESP32)

```
+---------------------+      WiFi / MQTT      +-------------------+
|   PC (Linux/Win)    | <===================> |   ESP32 DevKit    |
|  React website      |      (MQTT)            |   + INA219        |
|  Mosquitto broker   |      1883 / 9001       |   + (optional)    |
|                     |                       |   DHT22 / PIR /   |
|  subscribes to:     |                       |   MQ-2            |
|  esp32/<id>/telemetry|                      |                   |
|  esp32/<id>/inference|                      |  runs: esp32_ids.ino|
+---------------------+                       +-------------------+
        ^
        | runs TinyML quantized model (esp32_model.h)
        | WebSockets to browser: port 9001
        | ESP32 MQTT: port 1883
```

The PC runs the **React dashboard and MQTT broker**. The ESP32 is the **edge sensor node** that reads INA219 measurements, runs the current TinyML firmware, and publishes results.

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
- Connect the source/battery positive to INA219 VIN+ and the load positive to VIN- to measure current; connect grounds together.
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

### 4.3 Start a local MQTT broker with WebSockets

```bash
sudo apt install mosquitto mosquitto-clients
mosquitto -c hardware/esp32_ids/mosquitto.conf -v
```

This prototype config exposes unauthenticated MQTT on port `1883` and WebSockets
on port `9001`. Use it only on a trusted private network; do not expose or
port-forward these ports to the internet. If Mosquitto is already running as a
system service, stop that service first or configure its existing listener.

### 4.4 Run the React website

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL, select **Live telemetry**, leave the broker URL as
`ws://localhost:9001/mqtt`, enter the firmware device ID, and click **Connect
device**. A browser uses WebSockets; it cannot connect directly to raw MQTT port
`1883`.

In another terminal, verify messages with:

```bash
mosquitto_sub -h localhost -t 'esp32/esp32-ids-01/#' -v
```

---

## 5. Software on the ESP32

### 5.1 Arduino IDE setup
1. Install ESP32 board package: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
2. Select board: **ESP32 Dev Module**
3. Install libraries (Library Manager):
   - `Adafruit INA219`
   - `PubSubClient`
   - `ArduinoJson`

### 5.2 Add the TinyML model
Copy `models/tinyml/esp32_model.h` into your sketch folder next to `esp32_ids.ino`.

### 5.3 Configure & upload
Edit `hardware/esp32_ids/esp32_ids.ino`:
- `WIFI_SSID`, `WIFI_PASSWORD`
- `MQTT_BROKER` → your PC's LAN IPv4 address (find it with `hostname -I`)
- `DEVICE_ID` → `esp32-ids-01`, matching the website's device ID

The ESP32 and PC must be on the same Wi-Fi/LAN. Do not use `localhost` for
`MQTT_BROKER`: on the ESP32 it means the ESP32 itself. Allow inbound TCP port
`1883` through the PC firewall on the private network. The website uses
`ws://localhost:9001/mqtt` because the browser runs on that PC. Compile and upload
in Arduino IDE, then open Serial Monitor at `115200` baud.

---

## 6. How the connection works (data flow)

1. **ESP32** reads INA219 every second → battery %, bus voltage, shunt current.
2. **ESP32** builds the feature vector (must match training features & scaling).
3. **ESP32** calls `predict_tinyml_int8(features)` → leaf index → attack class.
4. **ESP32** computes a threat score and publishes JSON to MQTT:
   - `esp32/esp32-ids-01/telemetry` → `{battery_pct, bus_voltage_v, shunt_current_ma, model_tier}`
   - `esp32/esp32-ids-01/inference` → `{prediction, threat_score, severity, latency_ms, energy_uj, model}`
5. **React website** subscribes over WebSockets to these topics and displays real device messages in **Live telemetry**.
6. The firmware subscribes to `esp32/esp32-ids-01/command`; sending device commands from the website is not implemented yet.

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
| `frontend/src/hooks/useMqttTelemetry.js` | Browser MQTT-over-WebSocket subscriber |
| `hardware/esp32_ids/mosquitto.conf` | Local development broker listeners |
| `frontend/src/main.jsx` | React dashboard live telemetry view |
| `config.yaml` | Energy, battery, model settings |

## 11. Sensor and model limitations

The current firmware reads and publishes INA219 battery percentage, bus voltage,
and shunt current. DHT22, PIR, and MQ-2 are not yet read by this sketch; its
temperature and humidity features are placeholders. Network-traffic features
are also placeholders, and the model input scaling is not applied. The site can
show real INA219 measurements, but do not treat the firmware's intrusion
predictions as validated sensor-based detections until those features and
training-time scaling are implemented.

