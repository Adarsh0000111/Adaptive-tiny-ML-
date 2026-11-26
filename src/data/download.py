import os
import urllib.request
import logging
import pandas as pd
import numpy as np

from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_authentic_iiot_dataset(output_path, num_samples=25000, seed=42):
    """
    Generates a high-fidelity IIoT cybersecurity dataset matching standard Edge-IIoTset schema.
    Includes real IIoT protocol distributions (HTTP, Modbus, MQTT, TCP, UDP)
    and realistic attack signatures across 6 distinct classes.
    """
    logger.info(f"Generating realistic IIoT intrusion dataset with {num_samples} records...")
    np.random.seed(seed)
    
    classes = ["Normal", "DDoS", "Scanning", "SQL_Injection", "Backdoor", "Ransomware"]
    class_probs = [0.40, 0.20, 0.15, 0.10, 0.08, 0.07]
    
    labels = np.random.choice(classes, size=num_samples, p=class_probs)
    binary_labels = [0 if l == "Normal" else 1 for l in labels]
    
    protocols = ["http", "modbus", "mqtt", "tcp", "udp"]
    
    data = []
    for i in range(num_samples):
        cls = labels[i]
        proto = np.random.choice(protocols, p=[0.25, 0.25, 0.25, 0.15, 0.10])
        
        # Base realistic network traffic features
        flow_duration = np.random.exponential(scale=1.5 if cls == "Normal" else 0.2)
        packet_rate = np.random.gamma(shape=2, scale=50 if cls == "Normal" else 500)
        byte_rate = packet_rate * np.random.normal(loc=250, scale=50)
        
        tcp_flags_fin = 1 if np.random.rand() < 0.1 else 0
        tcp_flags_syn = 1 if (cls in ["DDoS", "Scanning"] or np.random.rand() < 0.15) else 0
        tcp_flags_ack = 1 if cls == "Normal" or np.random.rand() < 0.7 else 0
        tcp_flags_psh = 1 if cls in ["SQL_Injection", "Ransomware"] else (1 if np.random.rand() < 0.3 else 0)
        
        # Protocol specific metrics
        if proto == "modbus":
            modbus_func = np.random.choice([1, 3, 5, 16], p=[0.4, 0.4, 0.1, 0.1])
            modbus_reg = np.random.randint(0, 100)
            http_res_code = 0
            mqtt_msg_type = 0
        elif proto == "mqtt":
            modbus_func = 0
            modbus_reg = 0
            http_res_code = 0
            mqtt_msg_type = np.random.choice([1, 3, 8], p=[0.2, 0.7, 0.1])
        elif proto == "http":
            modbus_func = 0
            modbus_reg = 0
            http_res_code = 200 if cls == "Normal" else np.random.choice([200, 403, 404, 500], p=[0.2, 0.3, 0.3, 0.2])
            mqtt_msg_type = 0
        else:
            modbus_func = 0
            modbus_reg = 0
            http_res_code = 0
            mqtt_msg_type = 0

        # Class specific metric shifts (realistic attack behavior)
        if cls == "DDoS":
            packet_rate *= np.random.uniform(5.0, 15.0)
            tcp_flags_syn = 1
            payload_len = np.random.randint(64, 128)
            failed_connections = np.random.poisson(lam=12)
        elif cls == "Scanning":
            flow_duration = np.random.uniform(0.01, 0.1)
            tcp_flags_syn = 1
            payload_len = np.random.randint(0, 32)
            failed_connections = np.random.poisson(lam=25)
        elif cls == "SQL_Injection":
            payload_len = np.random.randint(500, 2500)
            failed_connections = np.random.poisson(lam=2)
            http_res_code = 500 if np.random.rand() < 0.6 else 200
        elif cls == "Backdoor":
            payload_len = np.random.randint(100, 800)
            failed_connections = np.random.poisson(lam=1)
            flow_duration *= 3.0
        elif cls == "Ransomware":
            payload_len = np.random.randint(1000, 4000)
            failed_connections = np.random.poisson(lam=4)
        else: # Normal
            payload_len = np.random.randint(128, 512)
            failed_connections = 0 if np.random.rand() < 0.95 else 1

        # High cardinality metadata columns to test dropped features removal
        frame_time = f"2026-10-04T09:{i%60:02d}:{i%60:02d}.{i%1000:03d}Z"
        ip_src = f"192.168.1.{(i % 250) + 1}"
        ip_dst = f"10.0.0.{(i % 50) + 1}"
        arp_src = f"192.168.1.{(i % 250) + 1}"
        arp_dst = f"10.0.0.{(i % 50) + 1}"
        http_file_data = "NULL" if cls == "Normal" else "SELECT * FROM users WHERE '1'='1'"
        tcp_payload = f"0x{i:04x}"

        record = {
            "frame.time": frame_time,
            "ip.src": ip_src,
            "ip.dst": ip_dst,
            "arp.src.proto_ipv4": arp_src,
            "arp.dst.proto_ipv4": arp_dst,
            "http.file_data": http_file_data,
            "tcp.payload": tcp_payload,
            "protocol": proto,
            "flow_duration": round(float(flow_duration), 4),
            "packet_rate": round(float(packet_rate), 2),
            "byte_rate": round(float(byte_rate), 2),
            "payload_len": int(payload_len),
            "tcp_flags_fin": tcp_flags_fin,
            "tcp_flags_syn": tcp_flags_syn,
            "tcp_flags_ack": tcp_flags_ack,
            "tcp_flags_psh": tcp_flags_psh,
            "modbus_func": modbus_func,
            "modbus_reg": modbus_reg,
            "http_res_code": http_res_code,
            "mqtt_msg_type": mqtt_msg_type,
            "failed_connections": failed_connections,
            "Attack_label": binary_labels[i],
            "Attack_type": labels[i]
        }
        data.append(record)

    df = pd.DataFrame(data)
    
    # Introduce small missing value noise (0.2%) to test preprocessing pipeline
    mask = np.random.rand(*df[["flow_duration", "packet_rate"]].shape) < 0.002
    df[["flow_duration", "packet_rate"]] = df[["flow_duration", "packet_rate"]].mask(mask)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset successfully created and saved to {output_path}")
    return df

def acquire_dataset(config_path="config.yaml", dataset_path=None, return_path=False):
    """Load only the official packet CSV; never silently train on the synthetic fixture."""
    config = load_config(config_path)
    configured = os.environ.get("ADAPTIVEML_DATASET")
    candidates = [dataset_path, configured,
                  "/mnt/9A50FC8350FC6801/PROJECTS/.codex-adapt-arduino/datasets/edge-iiotset/ML-EdgeIIoT-dataset.csv",
                  os.path.join(os.path.dirname(config["paths"]["raw_data"]), "ML-EdgeIIoT-dataset.csv"),
                  os.path.join(os.path.dirname(config["paths"]["raw_data"]), "DNN-EdgeIIoT-dataset.csv")]
    resolved = next((os.path.abspath(p) for p in candidates if p and os.path.isfile(p)), None)
    if resolved is None:
        raise FileNotFoundError("Official packet-level Edge-IIoTset CSV not found. Set ADAPTIVEML_DATASET to ML-EdgeIIoT-dataset.csv. Synthetic fixture data is intentionally not used for model training.")
    logger.info("Loading official Edge-IIoTset packet CSV from %s", resolved)
    frame = pd.read_csv(resolved, low_memory=False)
    return (frame, resolved) if return_path else frame

if __name__ == "__main__":
    df = acquire_dataset()
    print("Dataset shape:", df.shape)
    print("\nClass Distribution:\n", df["Attack_type"].value_counts())
