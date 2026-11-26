# AdaptiveML / SARA-IDS

AdaptiveML is a research prototype for intrusion detection in resource-constrained IoT/IIoT settings. SARA (Security-Aware Resource Adaptation) is a provisional policy name inside the project; no novelty claim is established.

## Current status

- React + Vite dashboard in `frontend/`; the old Streamlit UI has been removed.
- FastAPI backend in `src/api/` loads the Decision Tree, Random Forest, and MLP trained from the official packet-level Edge-IIoTset ML CSV. It exposes replay inference, model profiles, resources, telemetry, decision history, trusted-peer offloading, and a local FL simulation.
- The live IDS schema has 18 ordered packet fields and five protocol indicators and maps the original labels to six broader classes. Test-set accuracy / macro-F1: low 77.3% / 56.2%, medium 91.3% / 90.6%, high 91.0% / 89.3%. Packet rows are deduplicated; the source lacks capture-session IDs, so the random split does not establish session-independent generalization. See [`data_provenance.json`](results/experiments/data_provenance.json) and [`artifact_validation.json`](results/experiments/artifact_validation.json).
- ESP32-S3 USB firmware runs the lightweight fixed-point tree locally and returns physical sensor readings for the INA219, DHT22, SW-420, and MPU6500; the OLED and buzzer operate on the board.
- The ESP32-S3 runs the exported low-tier fixed-point tree. Its new firmware was compiled, flashed, and replayed against 100 held-out official-data rows over USB: 81% accuracy, macro-F1 56.7%, and device p95 inference 15 µs. During an explicitly authorized live capture window, SARA evaluates each arriving packet and automatically routes low to USB/ESP32, medium/high to laptop; no Wi-Fi is used. Capture is bounded to 100 packets or five seconds, stores no payloads or IP addresses, and reports predictions without ground-truth accuracy. See [the hardware replay](results/experiments/esp32_usb_replay.json).
- SARA compares the three saved model votes and uses disagreement with prediction entropy as a risk signal. Host CPU/RAM and host network-counter rates are reported separately from ESP32 sensor readings.
- The SARA page offers explicit user-authorized passive capture and a manual ICMP probe by IP for packet/throughput and RTT/loss/jitter measurements. The separate USB live path extracts packet-level fields compatible with the trained packet model; it is not flow-level IDS analysis.
- The local FedAvg simulation reports its training loss and simulated client/model freshness plus serialized update sizes. It does not send updates over a network.
- Federated training currently simulates client partitions within one process. It is not a networked deployment and provides no formal privacy guarantee.
- Device inference latency was measured on the ESP32 during replay. INA219-derived energy is recorded but its calibration and measured current path were not independently validated; board RAM and cross-layer attack performance remain unmeasured.

## Setup and run

Use Python 3.11+ and Node.js. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.api.app
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local Vite URL (usually `http://localhost:5173`). The API runs at `http://127.0.0.1:8765`; endpoint documentation is at `/docs`. The API's default mode is simulation. Saved-sample inference is test-split replay. Live USB capture only starts after checking the explicit authorization box and pressing its button. External feature submissions must match the exact ordered and preprocessed schema returned by `/api/models`.

## Dashboard and services

The website includes Overview and recorded-event replay, MQTT Live Telemetry, artifact-backed Model Tiers, browser-local device records, current-evidence Research Results, and SARA Decisions. Physical MQTT data appears only when a device publishes it. API host CPU/memory readings describe the API host, not automatically the ESP32. Live telemetry values retain their source and quality labels.

To retrain after obtaining the official CSV, run `python scripts/retrain_official.py /path/to/ML-EdgeIIoT-dataset.csv` from the repository root (or set `ADAPTIVEML_DATASET`). This rebuilds preprocessing, all three model artifacts, the fixed-point header, and final test evaluation. The old generated fixture is never selected implicitly.

Trusted offloading is disabled until peers are configured. Configure `SARA_TRUSTED_NODES` and `SARA_NODE_TOKEN` using the format in [.env.example](.env.example); non-local peers require HTTPS. The FL page launches a CPU simulation with local Dirichlet partitions; training runs outside the inference request path.

## Hardware

The electronics and firmware remain under `hardware/`. Follow [HARDWARE_CONNECTION_GUIDE.md](HARDWARE_CONNECTION_GUIDE.md) for pin assignments, wiring, MQTT setup, and limitations. Keep credentials in the ignored local `hardware/esp32_ids/secrets.h` based on `secrets.h.example`; do not commit secrets. The USB firmware consumes scaled packet features from the laptop; physical sensors are monitoring inputs and do not enter the IDS model.

## Evaluation and tests

The fixed saved test-split artifact evaluation is in [`results/experiments/artifact_validation.json`](results/experiments/artifact_validation.json). The separate [`ESP32-S3 USB replay results`](results/experiments/esp32_usb_replay.json) includes hardware predictions, per-class metrics, board inference time, USB transport, sensor snapshot, and artifact hashes. INA219 calibration/current path and peak RAM are unvalidated. The focused literature review is [research/LITERATURE_REVIEW.md](research/LITERATURE_REVIEW.md); it identifies overlap and does not establish novelty.

Run the tests and frontend production build:

```bash
python -m pytest -q
cd frontend && npm run build
```

The source CSV is distributed under CC BY-NC-SA 4.0: attribute the Edge-IIoTset authors and share adaptations under the same terms; it is not licensed for commercial use. Physical power calibration, independent-session generalization, multi-machine offloading/FL, and board RAM measurement remain future experiments. See [PROJECT_AUDIT.md](PROJECT_AUDIT.md) for the detailed status and [RESEARCH_REPORT.md](RESEARCH_REPORT.md) for the research plan and evidence limits.
