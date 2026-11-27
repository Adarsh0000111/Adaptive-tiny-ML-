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

#### Two-client run on one Windows computer

This uses the actual FastAPI coordinator over `http://127.0.0.1:8765`, with two separate Python processes, separate node identities/tokens, and disjoint local shards. Loopback proves HTTP update exchange on one computer; it does not prove multi-computer connectivity or independently collected site data.

Create two local shards once, from the project root:

```powershell
python scripts/make_fl_shards.py --outdir .\fl-shards --nodes 2
```

Keep these shards local. Create an untracked `fl_credentials.json` in the project root with three distinct private values (the file is gitignored):

```json
{"coordinator_token":"<private random token>","nodes":{"node-a":"<different private token>","node-b":"<another private token>"}}
```

**Coordinator terminal:**

```powershell
$c = Get-Content .\fl_credentials.json -Raw | ConvertFrom-Json
$env:SARA_FL_COORDINATOR_TOKEN = $c.coordinator_token
$env:SARA_FL_NODE_TOKENS = $c.nodes | ConvertTo-Json -Compress
$env:ADAPTIVEML_HOST = "127.0.0.1"
python -m src.api.app
```

Leave this terminal running. Start the dashboard in a separate terminal with `cd frontend; npm run dev`, then open its shown local URL.

**Node A terminal:**

```powershell
$c = Get-Content .\fl_credentials.json -Raw | ConvertFrom-Json
$env:SARA_FL_NODE_TOKEN = $c.nodes.'node-a'
python -m src.federated.client --coordinator http://127.0.0.1:8765 --node-id node-a --data .\fl-shards\node-1.npz --wait-for-run
```

**Node B terminal:**

```powershell
$c = Get-Content .\fl_credentials.json -Raw | ConvertFrom-Json
$env:SARA_FL_NODE_TOKEN = $c.nodes.'node-b'
python -m src.federated.client --coordinator http://127.0.0.1:8765 --node-id node-b --data .\fl-shards\node-2.npz --wait-for-run
```

These are independent long-running processes. They poll the coordinator and appear in the dashboard/API status as separate nodes. Check **GET `http://127.0.0.1:8765/api/federated/networked/status`** or the Networked federated learning panel; `node_status` reports each node's connection and accepted update count. A heartbeat is considered connected for 10 seconds.

To verify the clients are separate OS processes, run this from PowerShell while they are waiting/running; it should show two Python command lines with different `--node-id` and `--data` arguments:

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*src.federated.client*' } | Select-Object ProcessId, CommandLine
```

Before asking SARA to schedule training, use the dashboard's authorized live IDS action with the ESP32 connected to refresh sensor/security context, then collect measured link evidence from `/docs`: authorized passive capture on the actual adapter (Npcap/capture privileges may be required) and an ICMP probe to a destination you are authorized to probe, such as your own default gateway. These are measurements, not values to enter manually. The probe must be paired with fresh capture/link-utilization data. If SARA defers, the authenticated start endpoint returns HTTP 409 with the exact readiness reason; refresh the missing/expired evidence and retry. Do not start training until the decision says `schedule: true`.

**Start only after approval:**

```powershell
$c = Get-Content .\fl_credentials.json -Raw | ConvertFrom-Json
$body = @{ node_ids = @("node-a", "node-b"); rounds = 2; local_epochs = 1; min_clients = 2; timeout_seconds = 300 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8765/api/federated/networked/start" `
  -Headers @{ Authorization = "Bearer $($c.coordinator_token)" } -ContentType "application/json" -Body $body
```

The waiting clients receive assignments and train against their own shard. Confirm actual exchange by checking for `status: complete`, `updates_received`, both nodes in each round's `participating_nodes`, nonzero per-node `samples_per_node`, and each node's `updates_accepted` in coordinator status. These round records come from accepted HTTP update submissions and coordinator aggregation; check `GET http://127.0.0.1:8765/api/federated/networked/status` or the dashboard after the run. The coordinator persists a summary in the ignored `models/federated/sara_cnn_networked_history.json`; the API status endpoint returns the last completed run after restart. Networked clients use the high-tier 1D CNN architecture, while the federated candidate remains separate and never replaces any trained inference tier. A run that merely has two processes connected is not a successful training run.

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

Each client computer needs its own locally labeled CSV with the packet fields and a trustworthy `Attack_type` ground-truth column. Unlabeled live packets or the IDS's own predictions are not supervised training labels. Keep the CSV on that node and prepare its local NPZ with the project's exact scaler/encoder:

```bash
python scripts/prepare_fl_client_data.py --csv /secure/local/labeled_packets.csv --out data/runtime/client_train.npz
```

This NPZ includes the 18-feature order, six-class mapping, schema version, and preprocessing artifact hash; clients reject mismatches. For a controlled demo, the two generated shards above can instead be transferred to the computers over a trusted channel, but those shards are from one shared dataset and are not independent local site data.

Set `SARA_FL_COORDINATOR_TOKEN` and `SARA_FL_NODE_TOKENS` in the coordinator's environment. Bind the API to a reachable interface and put it behind a TLS reverse proxy; do not expose the development server directly to the public internet. Remote clients must use the HTTPS coordinator URL. Start the run with authenticated `POST /api/federated/networked/start`, then run this command on each client, setting that client's unique `SARA_FL_NODE_TOKEN` first:

```bash
python -m src.federated.client --coordinator https://COORDINATOR --node-id node-a --data /path/to/local_processed_dataset.npz
```

The coordinator receives model parameters and sample counts, not packet rows. This protocol has no secure aggregation or differential privacy, and model updates may still reveal information. The networked global 1D CNN starts from the compatible centrally trained high-tier weights, is validated for reporting, and is saved only as a separate federated candidate artifact. It is never loaded into, substituted for, or used to overwrite the three trained inference models. The recorded one-computer run used an earlier MLP implementation and is in [networked_federated_local_demo.json](results/experiments/networked_federated_local_demo.json), with its [validation results chart](results/experiments/networked_federated_results.png); the run uses shared dataset shards, not independent local site data.

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
