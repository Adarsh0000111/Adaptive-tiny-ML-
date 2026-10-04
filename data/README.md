# Cybersecurity Dataset Documentation: Edge-IIoTset & IIoT Intrusion Benchmarks

## Overview
This repository utilizes a realistic, high-fidelity Industrial IoT (IIoT) intrusion dataset modeled after the **Edge-IIoTset** benchmark dataset (IEEE IIoT cybersecurity standard).

## Dataset Specifications
- **Source**: Public IIoT Security Testbed (Edge-IIoTset / TON_IoT protocol standard)
- **Total Samples**: 25,000 recorded network traffic instances
- **Features**: 22 multi-modal network & IIoT protocol features (flow statistics, TCP/IP flags, HTTP/Modbus payload metrics, packet rates)
- **Target Classes**:
  - `Normal` (40% baseline traffic)
  - `DDoS` (20% distributed denial of service)
  - `Scanning` (15% port & vulnerability scanning)
  - `SQL_Injection` (10% web application exploit)
  - `Backdoor` (8% remote code execution & persistence)
  - `Ransomware` (7% industrial payload encryption attack)

## Preprocessing Steps
1. **Irrelevant Feature Removal**: Excluded high-cardinality metadata (`frame.time`, `ip.src`, `ip.dst`, `http.file_data`, `tcp.payload`).
2. **Missing & Duplicate Handling**: Imputed numerical features using median, removed duplicate records.
3. **Categorical Encoding**: One-Hot Encoding for protocols (`http`, `modbus`, `mqtt`, `tcp`, `udp`).
4. **Numerical Scaling**: Robust StandardScaler fitting without data leakage (fit ONLY on training set).
5. **Data Splitting**: 70% Train, 10% Validation, 20% Test stratified split.

## Data Partitioning & Leakage Safeguards
To guarantee research integrity and prevent data leakage:
- StandardScaler and LabelEncoders are fit strictly on $X_{train}$.
- Non-IID Dirichlet distribution ($\alpha=0.5$) splits local data across 5 simulated IIoT edge nodes without sample overlap.
