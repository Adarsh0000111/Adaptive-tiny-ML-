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
- The original `/api/federated/train` remains a one-process FedAvg simulation. A separate authenticated network FedAvg path is available for computer clients: each node trains on its own processed NPZ and sends model parameters and a sample count, not packet rows. Its global MLP checkpoint is separate from the three deployed low/medium/high inference tiers.
- Networked FedAvg exchanges model parameters over HTTPS and provides no secure aggregation, differential privacy, or formal privacy guarantee; model updates can reveal information. The coordinator does not centrally evaluate accuracy because client data remains local. ESP32 firmware, sensor reads, and USB inference are separate and unchanged; ESP32-S3 is not a training participant.
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

### Networked federated learning (computer nodes)

This is a separate backend path from the local FL simulation and the ESP32 USB path. The ESP32 sensor and USB inference flow are unchanged; the board is not an FL training client.

#### Two-client demo on one Windows computer

This verifies the coordinator/client protocol using two separate client processes and disjoint training shards. It still uses one computer, so it does not prove multi-computer connectivity.

1. From the repository root, create two shards:

   ```powershell
   python scripts/make_fl_shards.py --outdir .\fl-shards --nodes 2
   ```

   This reads `data/processed/processed_dataset.npz` and creates `node-1.npz` and `node-2.npz`. Keep the shard files private.

2. In the PowerShell window that will run the API, set three different long random tokens. You can generate each with `python -c "import secrets; print(secrets.token_hex(32))"`. Keep the values somewhere private, then set them like this, replacing the examples:

   ```powershell
   $env:SARA_FL_COORDINATOR_TOKEN = 'YOUR_COORDINATOR_TOKEN'
   $env:SARA_FL_NODE_TOKENS = '{"node-a":"YOUR_NODE_A_TOKEN","node-b":"YOUR_NODE_B_TOKEN"}'
   python -m src.api.app
   ```

   Leave this API window running. The environment variables must be set in this window before starting the API.

3. Open the website and run **Start authorized live IDS window** with the ESP32 connected. This gives SARA fresh live IDS risk and confidence data. Then open `http://127.0.0.1:8765/docs` and, within 30 seconds:

   - Call `POST /api/network/capture` with `authorized: true`, `max_packets: 50`, `timeout_seconds: 5`, and the network interface name. The interface must work with Scapy and allow the API to determine link capacity. On Windows, packet capture may require Npcap and running the API with the required capture privileges.
   - Call `POST /api/network/probe` with your own router's IP address (the **Default Gateway** shown by `ipconfig`) and `packets: 5`. Only probe a destination you are authorized to probe.

   SARA needs fresh link utilization as well as RTT/loss measurements. If the capture cannot measure link capacity, or the evidence is stale, SARA will defer the run with HTTP 409 and explain why in the response.

4. Start a two-round run from another PowerShell window. Replace the token with the coordinator token from step 2:

   ```powershell
   $body = @{
     node_ids = @("node-a", "node-b")
     rounds = 2
     local_epochs = 1
     min_clients = 2
     timeout_seconds = 300
   } | ConvertTo-Json

   Invoke-RestMethod -Method Post `
     -Uri "http://127.0.0.1:8765/api/federated/networked/start" `
     -Headers @{ Authorization = "Bearer YOUR_COORDINATOR_TOKEN" } `
     -ContentType "application/json" -Body $body
   ```

5. In two separate PowerShell windows, run one client per window. Set each window's node token to the matching value from step 2:

   ```powershell
   $env:SARA_FL_NODE_TOKEN = 'YOUR_NODE_A_TOKEN'
   python -m src.federated.client --coordinator http://127.0.0.1:8765 --node-id node-a --data .\fl-shards\node-1.npz
   ```

   ```powershell
   $env:SARA_FL_NODE_TOKEN = 'YOUR_NODE_B_TOKEN'
   python -m src.federated.client --coordinator http://127.0.0.1:8765 --node-id node-b --data .\fl-shards\node-2.npz
   ```

6. Watch the **Networked federated learning** panel on the website or query `GET /api/federated/networked/status`. A successful run reaches `complete`, records both participants for each round, and saves `models/federated/sara_mlp_networked.pt`. The checkpoint is separate; SARA does not automatically load it into its low/medium/high inference tiers.

#### One-computer network demo

For a basic networking demonstration, the coordinator and two client processes can run on one computer. This uses real HTTP requests and parameter updates over loopback, but does not demonstrate communication across a LAN/WAN or use live traffic/ESP32 evidence. It is enabled only by setting `SARA_FL_ALLOW_LOCAL_DEMO=true`, sending `local_demo: true`, and binding the API to `127.0.0.1`. The regular networked path retains SARA's fresh live IDS and link-quality gate.

Create two local training shards:

```bash
python scripts/make_fl_shards.py --outdir data/runtime/fl_client_shards --nodes 2
```

In one terminal, set three different secret values and start the API bound to loopback:

```bash
export SARA_FL_COORDINATOR_TOKEN='your-admin-secret'
export SARA_FL_NODE_TOKENS='{"node-a":"your-node-a-secret","node-b":"your-node-b-secret"}'
export SARA_FL_ALLOW_LOCAL_DEMO=true
uvicorn src.api.app:app --host 127.0.0.1 --port 8765
```

Start the two-round demo from another terminal, using the same coordinator secret:

```bash
curl -X POST http://127.0.0.1:8765/api/federated/networked/start \
  -H "Authorization: Bearer $SARA_FL_COORDINATOR_TOKEN" -H 'Content-Type: application/json' \
  -d '{"node_ids":["node-a","node-b"],"rounds":2,"local_epochs":1,"min_clients":2,"local_demo":true}'
```

Run each client in a separate terminal, setting its matching node token:

```bash
export SARA_FL_NODE_TOKEN='your-node-a-secret'
python -m src.federated.client --coordinator http://127.0.0.1:8765 --node-id node-a --data data/runtime/fl_client_shards/node-1.npz
```

```bash
export SARA_FL_NODE_TOKEN='your-node-b-secret'
python -m src.federated.client --coordinator http://127.0.0.1:8765 --node-id node-b --data data/runtime/fl_client_shards/node-2.npz
```

The API status is available at `http://127.0.0.1:8765/api/federated/networked/status`. Keep demo mode bound to loopback; do not use it as a substitute for the SARA gate on remote clients.

#### Using separate computers

Each client computer needs its own preprocessed NPZ containing `X_train` shaped `(N, 18)` and `y_train_multi` with labels encoded 0-5, using this project's same feature order, preprocessing, and class mapping. Real deployments should preprocess each node's own local data. For a controlled demo, the two shards above can be transferred to the respective client computers using a trusted channel.

Set `SARA_FL_COORDINATOR_TOKEN` and `SARA_FL_NODE_TOKENS` in the coordinator's environment. Bind the API to a reachable interface and put it behind a TLS reverse proxy; do not expose the development server directly to the public internet. Remote clients must use the HTTPS coordinator URL. Start the run with authenticated `POST /api/federated/networked/start`, then run this command on each client, setting that client's unique `SARA_FL_NODE_TOKEN` first:

```bash
python -m src.federated.client --coordinator https://COORDINATOR --node-id node-a --data /path/to/local_processed_dataset.npz
```

The coordinator receives model parameters and sample counts, not packet rows. This protocol has no secure aggregation or differential privacy, and model updates may still reveal information. It currently reports local loss and participation rather than global accuracy. The networked global model is not automatically used for tier selection.

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
