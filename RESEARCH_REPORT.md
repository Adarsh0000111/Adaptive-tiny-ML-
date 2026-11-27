# AdaptiveML / SARA-IDS: current research report

**Status:** software research prototype with a verified ESP32-S3 USB replay and an opt-in live packet path. The project has three models retrained on official packet-level Edge-IIoTset rows, a SARA policy prototype, React dashboard, FastAPI service, and simulated CPU federated learning. The current board replay (100 held-out rows) scored 81.0% accuracy and 0.5665 macro-F1, with 15 µs device p95 inference. Live capture features are implemented but have not yet been evaluated on labeled arbitrary traffic. INA219 calibration and separate-computer federated clients remain unvalidated; a two-client loopback CNN FedAvg demonstration has completed.

## Research question and scope

Can an IDS choose among classifiers or trusted remote inference under changing risk and resource conditions while respecting a measured security floor? This repository is a starting implementation, not evidence that the question has been answered. The name SARA is provisional and no novelty claim is made. A focused literature review and overlap assessment are in [research/LITERATURE_REVIEW.md](research/LITERATURE_REVIEW.md).

## Current data and model evidence

The processed dataset contains 25,579 training rows, 3,654 validation rows, and 7,309 held-out test rows; it has 18 ordered packet features and six grouped labels. Its source hash and filtering are recorded in [`results/experiments/data_provenance.json`](results/experiments/data_provenance.json). The current artifacts are a 3,465-byte DecisionTreeClassifier (lightweight), 866,745-byte RandomForestClassifier (medium), and a 378,666-byte PyTorch 1D CNN classifier (heavy). The CNN convolves across the ordered feature axis; this is not a time-series CNN over packet sequences.

The report in [`results/experiments/artifact_validation.json`](results/experiments/artifact_validation.json) evaluates the existing artifacts on the saved test split without retraining:

| Tier | Test accuracy | Macro recall | Macro F1 | Artifact size |
|---|---:|---:|---:|---:|
| Lightweight decision tree | 77.32% | 57.79% | 56.23% | 3.4 KB |
| Medium random forest | 91.34% | 89.65% | 90.60% | 846 KB |
| High 1D CNN classifier | 91.08% | 89.08% | 89.85% | 370 KB |

The lightweight model has lower macro recall than the medium and high models, and especially weak Ransomware recall; overall accuracy is not a sufficient security criterion. The stratified split is at packet level and the source has no session identifier, so context leakage is possible. Metrics and latency are host-specific software measurements. RAM and physical energy were not measured. Test results do not establish robustness, generalization, or causal benefit from adaptation.

## Implemented policy prototype

`src/sara.py` combines a network-risk signal, model uncertainty, and resource stress. Resource stress contributes to the score only in interaction with network risk, so high CPU or memory use by itself is not labeled an attack. The selection rule chooses the lowest estimated-energy local tier meeting validation macro-recall thresholds (0.95 for high risk, 0.90 for medium risk); under critical resource pressure it can request a configured trusted peer. It records a degraded-coverage condition when no local tier meets the applicable floor.

This is a deterministic rule, not a learned policy. Tier cost is an estimate derived from host-side profiling, not a measured energy model. The default risk evidence is a classifier-label proxy unless an external score is supplied; there is no independent live network IDS feeding SARA. The API's resource metrics describe the host running the service, not automatically the ESP32. A research evaluation must compare SARA against fixed tiers, confidence-only selection, resource-only selection, and an appropriate adaptive baseline, using per-class false negatives and synchronized device/network measurements.

## Federated learning and deployment status

The `/api/federated/train` route runs local CPU 1D CNN FedAvg over simulated Dirichlet partitions. The authenticated networked coordinator and clients use the same CNN architecture as the high tier. A two-client, two-round CNN FedAvg run completed over HTTP loopback using two local processes; the best candidate scored 88.23% validation accuracy, 82.93% macro recall, and 84.79% macro-F1, below the promotion gate, so it was saved separately. This demonstrates the client/coordinator exchange on one computer, not multi-computer connectivity or independent-site learning. Neither FL path provides secure aggregation, formal differential privacy, or measured communication energy. Config-based battery and energy values remain simulated estimates.

The React frontend replaces the former Streamlit dashboard. The USB-only ESP32-S3 sketch runs the exported fixed-point tree on laptop-supplied scaled packet vectors and returns physical sensor telemetry; this path needs no Wi-Fi or MQTT. Sensor readings remain separate from the model's 18 network features. A separate unchecked-by-default capture action extracts TCP/UDP packet metadata on the laptop, applies the training transform, and sends predictions to the board; this is packet-level inference, not flow reconstruction.

## Historical claims and evidence limits

Earlier project documents/results included energy-saving, INT8 deployment, federated accuracy, and privacy claims. Those figures are not reproduced by the current controlled evaluation and have been withdrawn from the current evidence. The TinyML export is fixed-point threshold/input quantization, not a demonstrated INT8 implementation. ESP32 compile, flash, and replay inference have been verified; peak RAM and calibrated energy remain unmeasured. See [the full USB replay report](results/experiments/esp32_usb_replay.json), [PROJECT_AUDIT.md](PROJECT_AUDIT.md), and [HARDWARE_CONNECTION_GUIDE.md](HARDWARE_CONNECTION_GUIDE.md).

## Minimum next experiment

1. Verify dataset release, licensing, exact source files, preprocessing, and split lineage; record immutable hashes.
2. Freeze train/validation/test splits and fit every preprocessing step on training data only. Report per-class precision, recall, F1, confusion matrices, macro FPR/FNR, and confidence intervals across seeds.
3. Calibrate security floors on validation data and test fixed lightweight/medium/heavy, confidence-only switching, resource-only switching, SARA, and literature-grounded adaptive baselines.
4. Separate attack traffic from benign resource stress. Sweep CPU/memory load, network latency/loss, peer availability, and battery state; synchronize decisions, predictions, resource readings, and energy measurements.
5. Measure energy at the inference device with a calibrated instrument, and report latency distributions, peak RAM, flash, energy per sample, and false-negative costs. Do not infer ESP32 performance from the development host.
6. For FL, use independent clients and report per-client class coverage, convergence, update bytes, wall time, measured energy, and adversarial tests. Add differential privacy only with a stated accountant, clipping/noise configuration, and privacy budget.
7. Validate laptop packet extraction against a labeled PCAP and test class-wise performance under different interface traffic, encryption, and capture privilege conditions.

## Conclusion

The current code supports artifact-level model comparison, a testable adaptive-policy prototype, and physical TinyML inference on saved-feature replay. It does not yet demonstrate energy savings, live network intrusion detection, multi-computer or independent-site FL, or a novel SARA contribution. The nearest literature overlaps adaptive model selection, security-energy adaptation, edge/cloud offloading, and resource-aware federated IDS; the defensible contribution, if any, must come from a carefully measured, security-constrained experiment.
# Data and live-inference update (2026-11-26)

The project's former 25,000-row CSV was generated by its own synthetic generator, despite its Edge-IIoTset naming. It was replaced for current model training by the official packet-level `ML-EdgeIIoT-dataset.csv`. Training uses 13 numeric packet fields and five protocol indicators and maps original attack labels into six broad groups. The exact mapping and source-file SHA-256 are in `results/experiments/data_provenance.json`; fixed held-out test results are in `results/experiments/artifact_validation.json`.

The official file has repeated packets and no capture/session identifier. Exact duplicate feature/label rows were removed and conflicting identical-feature rows discarded; the resulting stratified random split is useful for within-file evaluation but may share session context across splits. It cannot support claims of session-independent performance or generalization to arbitrary laptop traffic. The exported low-tier tree was compiled and flashed to the ESP32-S3; a 100-row held-out replay is recorded in `results/experiments/esp32_usb_replay.json`. During an explicitly authorized capture window, the laptop extracts each TCP/UDP packet in the capture callback, runs SARA selection, and immediately executes the selected tier on USB/ESP32 or laptop. Host capture does not retain packet payloads or IP addresses. Live arbitrary-traffic detection still needs evaluation on labeled traffic.

Dataset attribution/license: Mohamed Amine Ferrag et al., Edge-IIoTset, DOI [10.1109/ACCESS.2022.3165809](https://doi.org/10.1109/ACCESS.2022.3165809). The downloaded dataset is labeled CC BY-NC-SA 4.0; use requires attribution, share-alike for adaptations, and non-commercial use.
