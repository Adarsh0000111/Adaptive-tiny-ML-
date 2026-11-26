# ESP32 sensors, MQTT, and AdaptiveML

This guide covers both the legacy MQTT sensor path and the no-Wi-Fi USB path. The ESP32 reads physical sensors and runs a fixed-point tree; the laptop can either replay held-out rows or explicitly capture bounded packet metadata, apply the matching transform, and send it over USB. Live capture is opt-in and is not yet validated on labeled arbitrary traffic.

## Architecture and operating modes

- **Local simulation/replay:** the FastAPI service uses the saved, preprocessed test set and the three compatible six-class scikit-learn artifacts. It labels predictions as saved-test replay.
- **Physical telemetry:** the legacy ESP32 sketch can publish INA219 and environmental readings over MQTT. The USB sketch returns physical sensor samples over serial and requires no network.
- **Live packet inference:** the dashboard can capture TCP/UDP packet headers only after explicit authorization. Laptop-side extraction uses the `edge-iiotset-live-packet-18-v2` packet schema and saved train-fitted transform. Sensor readings and aggregate counters remain separate model inputs.

## Offline ESP32 model replay over USB

The no-Wi-Fi path is implemented as `hardware/esp32_usb_ids/esp32_usb_ids.ino`. It runs the exported lightweight fixed-point tree on the ESP32-S3 and accepts one CSV line over USB serial: request ID plus exactly 18 already-preprocessed float features. Each result also carries physical INA219 voltage/current/power/energy, DHT22 temperature/humidity, SW-420 vibration, MPU6500 acceleration/movement, buzzer alert state, and device uptime. The OLED displays the same local sensor state and last IDS prediction. Missing I2C sensors are reported as unavailable; the sketch does not invent fallback readings. It does not start Wi-Fi or MQTT.

The saved held-out test split remains available for an accuracy-checked replay. A separate live mode accepts packet predictions without ground truth. This is packet-level classification, not flow analysis, and it still needs evaluation on labeled live or PCAP traffic before claiming detection performance.

1. With the board connected by USB, open `hardware/esp32_usb_ids/esp32_usb_ids.ino` in Arduino IDE. Select an ESP32-S3 board and the current serial port (usually `/dev/ttyACM0` on this host).
2. Compile and upload this sketch. It replaces the currently running sensor/Wi-Fi sketch; that sketch can be restored by uploading its `.ino` again.
3. Start the local API from the repository root: `python3 -m src.api.app`.
4. Start the React site: `cd frontend && npm run dev`.
5. Open **SARA Decisions → Offline ESP32 inference over USB** and run the 100-sample replay. The laptop reads the saved test vectors, sends them over serial, and the ESP32 returns predictions to the dashboard. Wi-Fi, MQTT, and the ACT captive portal are not used in this path.

Close Arduino Serial Monitor before using the dashboard because only one process can own the serial port. The API requires serial read/write permissions. The dashboard reports replay accuracy, macro F1, per-class metrics, device inference time, USB round-trip time, and the board's physical sensor readings. The INA219 energy counter is integrated on the ESP32 from power samples and resets when the board reboots; it is meaningful only for the load wired through the sensor. Sensor measurements do not enter the IDS feature vector.

Do not connect low-voltage measurement hardware to mains wiring. Measure only a properly rated low-voltage supply. Actual energy measurements apply to the device in the INA219 current path.

## Verified repository model contract

`models/preprocessing_pipeline.joblib` and `data/processed/processed_dataset.npz` define 18 ordered, scaled inputs and six labels: `Backdoor`, `DDoS`, `Normal`, `Ransomware`, `SQL_Injection`, and `Scanning`. The saved input schema is returned by `GET /api/models`.

The generated `hardware/esp32_usb_ids/esp32_model.h` is a fixed-point decision-tree C header with encoded class IDs. It is compiled into the USB sketch and has been exercised on the ESP32-S3 using saved, preprocessed held-out features. This validates hardware inference on replay only; the board still does not extract features from live network traffic.

## Install and start the website and API

From the repository root:

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env
python3 -m src.api.app
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The React website runs at the Vite URL. The local API is at `http://127.0.0.1:8765`; interactive API documentation is at `/docs`. Vite proxies `/api` and `/ws` to the API.

## Sensors and firmware

The legacy Wi-Fi sketch `hardware/esp32_ids/esp32_ids.ino` uses:

- INA219 over I2C on SDA GPIO 4 and SCL GPIO 5 (matching its `Wire.begin(4, 5)`).
- DHT22 data on GPIO 6.
- SW-420 vibration signal on GPIO 7.
- Buzzer on GPIO 15.
- MPU6050 and SSD1306 OLED on the same I2C bus.

The offline USB sketch `hardware/esp32_usb_ids/esp32_usb_ids.ino` uses INA219, MPU6500, and SSD1306 as its three I2C devices on SDA GPIO 4 / SCL GPIO 5; DHT22 is a separate one-wire-style data connection on GPIO 6; SW-420 is a digital input on GPIO 7; and the passive buzzer is a driven output on GPIO 15. The voltage/current module must be the compatible INA219 for the provided driver. Other voltage sensor modules need a matching, correctly scaled driver before their readings can be trusted. The firmware reports DHT22 available only when both temperature and humidity reads are finite. A software-driven buzzer is an output and is not detected as an I2C peripheral.

These pin selections are sketch-specific. Check your exact board and peripherals before wiring; do not assume the pins in older notes match the current sketch.

Install ESP32 board support plus Arduino libraries: Adafruit INA219, Adafruit GFX, Adafruit SSD1306, Adafruit MPU6050, Adafruit Unified Sensor, DHT sensor library, PubSubClient, and ArduinoJson.

Copy `hardware/esp32_ids/secrets.h.example` to the ignored local file `hardware/esp32_ids/secrets.h` and configure the Wi-Fi SSID/password and MQTT broker LAN address there. Never place credentials in the `.ino` file or commit `secrets.h`. Rotate any credential that was previously shared.

The sketch publishes topics `esp32/<device-id>/telemetry` and `esp32/<device-id>/inference`. Telemetry includes source, voltage, current, power, integrated energy, battery estimate, and ambient temperature/humidity. Inference currently reports `UNAVAILABLE` with an explanation; this is intentional to prevent a false prediction from incompatible inputs.

## MQTT broker

The sample Mosquitto configuration listens on raw MQTT port 1883 and browser WebSockets port 9001. It has no authentication or TLS. Use it only on a trusted isolated lab network, never port-forward it or expose it to the public internet. For a shared/production network, configure broker authentication, topic ACLs, and TLS in Mosquitto and in the ESP32 client before connecting devices. The browser uses the broker WebSocket URL; the ESP32 uses the broker's LAN address and raw MQTT port.

To inspect telemetry locally:

```bash
mosquitto_sub -h localhost -t 'esp32/esp32-ids-01/#' -v
```

## Power and energy calculations

The firmware obtains bus voltage and shunt current from INA219. It computes instantaneous power as `V × I` and integrates adjacent timestamped samples with the trapezoidal rule:

`energy_J += (previous_power_W + current_power_W) / 2 × elapsed_seconds`

The API independently integrates timestamped power records when no cumulative energy value is supplied. Both sources and their physical/simulated labels are retained. Calibration, sensor uncertainty, sampling resolution, and which device is in the measured current path must be documented for an experiment. CPU-host model timings are not ESP32 energy measurements.

## Troubleshooting

| Symptom | Check |
|---|---|
| Firmware cannot connect to Wi-Fi | Check the private `secrets.h`, signal, and network credentials. |
| MQTT does not connect | Confirm broker address, listener, LAN firewall, and topic ACLs. |
| Browser MQTT connection fails | Use `ws://<broker-host>:9001/mqtt` only on a trusted local network; confirm Mosquitto's WebSocket listener. |
| API reports feature-schema mismatch | Use exactly the ordered, preprocessed 18-feature contract returned by `/api/models`; do not send sensor fields as model features. |
| Firmware displays inference unavailable | Expected until a validated live network feature extractor and scaler are implemented. |
| Temperature or humidity is missing | Check DHT22 wiring and allow its measurement interval; missing sensor values must remain missing, not be presented as measured defaults. |
# Current packet-model USB path (2026-11-26)

The current board contract sends exactly 18 preprocessed packet features from the laptop over USB: 13 numeric values, then protocol indicators in the order `http, mqtt, other, tcp, udp`. The board runs only the low fixed-point tree and returns one of `Access_Compromise`, `DDoS`, `Normal`, `Ransomware`, `Scanning`, or `Web_Attack`. The laptop capture extracts headers/metadata from TCP and UDP packets and discards packet objects after feature extraction; no Wi-Fi, MQTT, endpoint addresses, or payload retention is involved.

Use the dashboard's **Start authorized live IDS window** control only on an interface you are authorized to monitor. It processes each packet in Scapy's capture callback and chooses a tier per packet: low goes over USB to the ESP32-S3, while medium/high run on the laptop. A window is bounded to at most 100 observed packets or five seconds. The regenerated firmware is compiled/flashed with PlatformIO; the 100-row saved-data replay is recorded in `results/experiments/esp32_usb_replay.json`. For later exports, run `pio run` here; upload with `pio run -t upload --upload-port <serial-port>`. Opening a serial monitor while the dashboard is connected will contend for the port. USB serial processing can limit capture throughput.

These inputs are packet-level fields, not reconstructed bidirectional flows. TLS-encrypted application content is not inspected, and the training dataset has no independent capture-session identifier. Live predictions are therefore model outputs without ground truth, not a claim of validated detection on arbitrary traffic. Physical electrical/environmental sensors continue to be displayed separately and do not feed these model inputs.

# Sensor-aware SARA policy

The IDS classifiers continue to consume only the saved 18 network packet features. ESP32 sensors are separate decision context and are not appended to the classifier input.

- **INA219:** SARA uses only fresh, physically reported supply measurements that pass plausibility checks (INA219 detected, voltage in 2.7–5.5 V, current within the configured 1 A range, and reported power consistent with voltage × current). Voltage below the configured floor or current at/above the configured limit blocks ESP32 execution: SARA uses a trusted peer when available, otherwise it runs the chosen tier on the laptop. Defaults are 3.0 V and 0.9 A. Set `ADAPTIVEML_ESP32_MIN_SUPPLY_V` and `ADAPTIVEML_ESP32_MAX_CURRENT_A` only after verifying which supply rail the INA219 measures. INA219 cumulative energy is recorded but is not treated as remaining battery or as a measured per-inference energy budget.
- **DHT22:** ambient temperature and humidity are logged as environmental context. DHT22 does not measure ESP32 die temperature, so these readings do not claim to predict processor throttling or attacks.
- **MPU6500 and SW-420:** movement and vibration are logged as physical events. They do not increase network threat severity or change the selected security tier.
- **Network load:** live capture records packet rate from packet timestamps and compares it with measured USB round-trip capacity. If the measured edge service capacity is reached, SARA keeps the selected model but runs it on the laptop to avoid queue growth. Packet features and model confidence/disagreement remain the security inputs.
- **Host resources and Federated Learning:** host CPU and memory pressure scale inference cost and gate new local FedAvg runs at 95%. FedAvg is currently a laptop simulation, so ESP32 sensor readings and captured-network rates do not gate simulated client rounds. The scheduler can evaluate `client_topology=physical_esp32` or `networked`: both require recent live packet risk/confidence before using live data; physical ESP32 clients additionally require qualified supply telemetry, while networked clients require fresh RTT/loss/link-utilization metrics. Missing or out-of-limit evidence defers scheduling; if preflight passes, the API still does not start training because those real client transports are not implemented. Thresholds are configurable in `.env.example`.

The Live IDS page has a paired sensor ablation action. It evaluates the same labeled packet rows with sensor context enabled and disabled, uses actual ESP32 replay telemetry, reports model metrics and execution placement, and saves full per-decision traces to `results/experiments/sensor_ablation_latest.json`. The current observed INA219 voltage is about 0.856 V, outside the qualification range; expect zero sensor-driven placement changes until the measured rail and INA219 wiring are verified.

