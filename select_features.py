
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_DIR
    / "dataset"
    / "raw"
    / "DNN-EdgeIIoT-dataset.csv"
)

REPORT_PATH = PROJECT_DIR / "feature_selection_report.txt"

TARGET = "Attack_label"

CONSTANT_COLUMNS = [
    "icmp.unused",
    "http.tls_port",
    "dns.qry.type",
    "mqtt.msg_decoded_as",
]

INITIAL_EXCLUSIONS = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "tcp.options",
    "tcp.payload",
    "Attack_type",
]

# Read column names and a small sample only.
sample = pd.read_csv(
    DATASET_PATH,
    nrows=20_000,
    low_memory=False,
)

if TARGET not in sample.columns:
    raise ValueError(f"Target column not found: {TARGET}")

excluded = set(CONSTANT_COLUMNS + INITIAL_EXCLUSIONS + [TARGET])

# Select numeric columns only for the initial baseline.
candidate_features = [
    col
    for col in sample.columns
    if col not in excluded and is_numeric_dtype(sample[col])
]

excluded_non_numeric = [
    col
    for col in sample.columns
    if col not in excluded and col not in candidate_features
]

report = []

report.append("INITIAL FEATURE-SELECTION REPORT")
report.append("=" * 60)
report.append(f"Total source columns: {len(sample.columns)}")
report.append(f"Target: {TARGET}")
report.append(f"Candidate numeric features: {len(candidate_features)}")

report.append("\nConstant columns excluded:")
report.extend(f"  - {col}" for col in CONSTANT_COLUMNS)

report.append("\nOther columns excluded:")
report.extend(f"  - {col}" for col in INITIAL_EXCLUSIONS)

report.append("\nOther non-numeric columns excluded from this baseline:")
report.extend(f"  - {col}" for col in excluded_non_numeric)

report.append("\nCandidate features:")
report.extend(f"  - {col}" for col in candidate_features)

report_text = "\n".join(report)

REPORT_PATH.write_text(report_text, encoding="utf-8")

print(report_text)
print(f"\nReport saved to: {REPORT_PATH}")
print("\nOriginal dataset was not modified.")
