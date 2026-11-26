# AdaptiveML repository audit and implementation status

Audit date: 2026-11-26
Branch: `codex/sara-ids-upgrade`
Repository: `Adarsh0000111/Adaptive-tiny-ML-`

This audit records the checked-out code and artifacts, including the 2026-11-26 retraining and ESP32 update below. The physical replay is now validated; live traffic generalization, calibrated power, and distributed-node experiments are not.

## Verified starting point

- **Frontend:** React + Vite in `frontend/`. The former `dashboard/app.py` Streamlit UI has been removed. The React dashboard is retained and now calls the Python API for inference, model profiles, resource readings, SARA state, offloading status, and FL simulation.
- **Backend:** existing research code in `src/`; new FastAPI service in `src/api/`. The previous `demo_api.py` wrapper now starts that service.
- **Data:** the current processed dataset has 25,579 train, 3,654 validation, and 7,309 test rows, with 18 packet features and six grouped labels. Provenance and source license are recorded in `results/experiments/data_provenance.json`.
- **Artifacts:** current sklearn DecisionTreeClassifier (3,465 bytes), RandomForestClassifier (866,745 bytes), and MLPClassifier (217,608 bytes), each with 18 inputs and six classes. There is no CNN artifact.
- **Preprocessing:** `src/preprocessing/` produces the versioned packet schema. The shared extractor maps captured TCP/UDP packet headers into this schema; sensor and aggregate network counters are not passed as substitutes.
- **Hardware:** the regenerated fixed-point tree is compiled/flashed on the ESP32-S3 and validated with a 100-row USB replay. The live packet path is laptop capture → saved preprocessing → USB inference; local MQTT credentials remain separate.
- **Previous code issues:** the original demo API assumed an incompatible binary/CNN interface; live sniffer had synthetic fallback data; hardware fields did not match trained features; historic docs overstated energy, TinyML, DP, and FL results. These findings motivated explicit schemas, provenance, and unavailable states.

## Current implementation

| Component | Implemented behavior | Evidence and boundary |
|---|---|---|
| FastAPI | Loads the three saved artifacts, reports host CPU/RAM/network-counter rates, compares model votes, provides explicit ICMP probes and user-authorized passive packet capture, and exposes telemetry and device routes | `src/api/`; aggregate network counters remain separate from the trained 18-feature schema |
| React | Overview/replay, model profiles, physical MQTT telemetry, USB ESP32 replay and sensor readings, SARA decisions, and local simulated FL controls | Model output and decisions come from API. Saved recorded events remain visibly historical replay data. |
| SARA | Cross-layer risk includes model uncertainty and disagreement among the three artifact predictions; recall-floor-constrained model choice; resource stress alone is not attack evidence | Deterministic prototype. Model-label risk proxy is used unless a caller supplies risk; passive counters and ICMP probes are not a live model-compatible IDS detector. |
| Host monitor | CPU, memory, and available host telemetry through psutil | Describes the machine running the API, not necessarily the ESP32. |
| Offloading | Configurable trusted peer calls, bearer auth, HTTPS enforcement for non-local peers, bounded timeout, and local fallback | Requires externally configured node(s); no multi-node run has been performed. |
| Federated learning | CPU MLP, Dirichlet client partitions, sample-weighted parameter FedAvg, validation acceptance gate, checkpoint and rollback copy; exposes local loss, simulated client state/freshness, and serialized communication estimates | In-process simulation only. No independent clients/network exchange, formal DP, or measured energy. |
| ESP32 | USB-only firmware runs the exported fixed-point tree and returns readings from INA219, DHT22, SW-420, and MPU6500; updates OLED and buzzer | 100 held-out rows replayed on the flashed ESP32-S3; see current hardware artifact. INA219 calibration and RAM use are not independently validated. |
| Packet capture | Optional authorized Scapy packet feature extraction; no random fallback | The dedicated USB live path maps packet headers to model features, transfers over USB, and retains no payloads or endpoint addresses. It has not yet been evaluated on labeled arbitrary traffic. |
| Network quality | Explicit ICMP probe reports RTT, loss, and RTT variation; passive capture reports packet/throughput metrics | ICMP runs only after a user submits an IP. Bandwidth utilization is null unless link capacity is known. These signals do not become IDS model features. |
| TinyML export | Fixed-point tree threshold/input C header with class IDs | Compiled and flashed on ESP32-S3; `results/experiments/esp32_usb_replay.json` records on-board replay metrics and measured inference time. Not INT8 neural quantization. |
| Research evaluation | Fixed saved test split evaluation with per-class metrics, hashes, host latency, artifact sizes, plus full ESP32 USB replay | `results/experiments/artifact_validation.json` and `results/experiments/esp32_usb_replay.json`; board RAM and calibrated energy remain unmeasured. |
| Literature | Focused review identifies close adaptive IDS, edge/cloud, and resource-aware FL work | `research/LITERATURE_REVIEW.md` is not a systematic review and does not establish novelty. |

## Test and result snapshot

The prior API test run passed 31 tests before the packet-pipeline update. The current React production build succeeds (Vite reports a bundle-size advisory). PlatformIO compiled/flashed the packet-model firmware and the board returned its expected handshake. Current 100-row replay is recorded in `results/experiments/esp32_usb_replay.json`; INA219 calibration remains unverified.

On the current 7,309-row test split, artifacts score: tree 77.32% accuracy / 56.23% macro F1; forest 91.34% / 90.60%; MLP 90.98% / 89.26%. The tree remains weaker on macro recall and Ransomware; overall accuracy is not sufficient for security decisions. These metrics are from the current official-data artifacts. Host latency is machine-specific and not device latency.

## Remaining work and risks

1. Evaluate the extractor and model on independent, labeled capture sessions; report class-wise errors and session-level split lineage.
2. Validate domain transfer to labeled traffic from the intended laptop/ESP32 network, including encrypted applications and unsupported protocols.
3. Calibrate INA219 and device-side power measurement; characterize uncertainty and measure the device that executes inference.
4. MQTT and deployment broker authentication/TLS have not been exercised; the USB-only firmware path is the current offline run.
5. Test trusted inference between independent nodes, including unavailable peers, latency/loss, certificate and token rotation, and model/version compatibility.
6. Run FL on independent machines and test client dropout, poisoning, fairness, and actual communication cost. Implement formal privacy only with a defined mechanism and accountant.
7. Add reproducible baselines and ablations for fixed tiers, confidence-only switching, resource-only switching, offloading, hysteresis, and FL scheduling; report per-class security under benign and malicious resource stress.
8. Collect synchronized cross-layer experiment records and repeated trials with uncertainty intervals. No claimed energy saving, hardware deployment result, or novel contribution is justified until these are completed.

## Proposed research direction

Retain AdaptiveML and evaluate SARA as a security-constrained adaptive policy. Compare whether explicit per-class detection constraints reduce false negatives during resource stress, and whether trusted offloading improves that outcome under measured device/network constraints. Prior literature overlaps the broad areas; this is an experimental question, not a verified novelty claim.
# 2026-11-26 live packet pipeline update

The earlier project CSV at `data/raw/iiot_intrusion_dataset.csv` was generated by the repository's synthetic-data generator and is not official Edge-IIoTset. It is no longer accepted as a training fallback. The current three model artifacts were retrained from the official `ML-EdgeIIoT-dataset.csv` file using packet-level features; source hash, label grouping, feature schema, and row filtering are recorded in `results/experiments/data_provenance.json`.

The feature schema is `edge-iiotset-live-packet-18-v2`: 13 numeric packet fields followed by five protocol one-hot values. Original labels are grouped into Access_Compromise, DDoS, Normal, Ransomware, Scanning, and Web_Attack. Held-out test metrics are in `results/experiments/artifact_validation.json`; the tiny decision tree has lower macro recall than the medium and high models. Dataset deduplication and filtering leave 36,542 rows from 157,800 source rows. The official file has no capture/session identifier, so the stratified random split can share capture context; results do not demonstrate independent-session generalization.

The new fixed-point tree header was exported to `models/tinyml/esp32_model.h` and copied into both firmware folders. The fixed-point simulator matches the low model within 0.10 percentage points accuracy on the saved test split. PlatformIO compiled and flashed the regenerated firmware to the detected ESP32-S3 at `/dev/ttyACM0`; the board returned the expected 18-feature/six-class handshake and reported INA219, DHT22, MPU6500, and OLED available. A 100-row held-out replay reached 81.0% accuracy, macro-F1 0.5665, and 15 µs device p95 inference; see `results/experiments/esp32_usb_replay.json`. During authorized live capture, the API now evaluates and routes each packet from inside the capture callback; low inference uses USB/ESP32, while medium and high use laptop models. Capture remains explicit and predictions have no ground-truth metrics.

The source dataset license is CC BY-NC-SA 4.0. Attribute Edge-IIoTset and keep derived dataset adaptations under the same terms; commercial use is not permitted by that license. See the provenance manifest for DOI/attribution details.
