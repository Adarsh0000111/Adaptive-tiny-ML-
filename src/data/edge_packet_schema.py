"""Shared packet feature mapping for Edge-IIoTset training and live capture.

The selected Edge-IIoTset ML CSV is packet-level. These fields are extracted
from each observed packet; payloads are never retained. Endpoint metadata is
excluded by default and available only through the explicit live-report opt-in.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "edge-iiotset-live-packet-18-v2"
NUMERIC_FEATURES = [
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.flags.ack",
    "tcp.flags",
    "tcp.len",
    "tcp.dstport",
    "udp.time_delta",
    "dns.qry.name.len",
    "http.response",
    "mqtt.msgtype",
    "mqtt.len",
]
PROTOCOL_FEATURE = "protocol"
PROTOCOL_CATEGORIES = ["http", "mqtt", "other", "tcp", "udp"]
RAW_COLUMNS = NUMERIC_FEATURES + ["Attack_type"]

ATTACK_GROUPS = {
    "Backdoor": "Access_Compromise",
    "MITM": "Access_Compromise",
    "Password": "Access_Compromise",
    "DDoS_HTTP": "DDoS",
    "DDoS_ICMP": "DDoS",
    "DDoS_TCP": "DDoS",
    "DDoS_UDP": "DDoS",
    "Normal": "Normal",
    "Ransomware": "Ransomware",
    "Fingerprinting": "Scanning",
    "OS_Fingerprinting": "Scanning",
    "Port_Scanning": "Scanning",
    "Vulnerability_scanner": "Scanning",
    "SQL_injection": "Web_Attack",
    "Uploading": "Web_Attack",
    "XSS": "Web_Attack",
}


def map_attack_label(value: Any) -> str | None:
    return ATTACK_GROUPS.get(str(value).strip())


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def protocol_from_dataset_row(row: pd.Series) -> str:
    source_port = _number(row.get("tcp.srcport"))
    destination_port = _number(row.get("tcp.dstport"))
    udp_port = _number(row.get("udp.port"))
    if _number(row.get("mqtt.msgtype")) > 0 or _number(row.get("mqtt.len")) > 0 or 1883 in (source_port, destination_port):
        return "mqtt"
    if _number(row.get("http.response")) > 0 or source_port in (80, 8080) or destination_port in (80, 8080):
        return "http"
    if source_port > 0 or destination_port > 0:
        return "tcp"
    if udp_port > 0:
        return "udp"
    return "other"


def prepare_official_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project official packet rows into the schema shared with live capture."""
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Official Edge-IIoTset ML file is missing required columns: {missing}")

    rows = frame[RAW_COLUMNS].copy()
    for name in NUMERIC_FEATURES:
        rows[name] = pd.to_numeric(rows[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rows[PROTOCOL_FEATURE] = frame.loc[rows.index].apply(protocol_from_dataset_row, axis=1).astype(str)
    rows["Attack_type"] = rows["Attack_type"].map(map_attack_label)
    unknown_label_count = int(rows["Attack_type"].isna().sum())
    rows = rows.dropna(subset=["Attack_type"]).reset_index(drop=True)

    feature_columns = NUMERIC_FEATURES + [PROTOCOL_FEATURE]
    duplicate_mask = rows.duplicated(feature_columns, keep=False)
    conflicts = rows.loc[duplicate_mask].groupby(feature_columns, dropna=False)["Attack_type"].nunique()
    conflicting_keys = conflicts[conflicts > 1].reset_index()[feature_columns]
    conflicting_rows_dropped = 0
    if len(conflicting_keys):
        matched = rows.merge(conflicting_keys.assign(_conflict=True), on=feature_columns, how="inner")
        conflicting_rows_dropped = int(len(matched))
        rows = rows.merge(conflicting_keys.assign(_conflict=True), on=feature_columns, how="left")
        rows = rows[rows["_conflict"].isna()].drop(columns="_conflict")
    before_dedup = len(rows)
    rows = rows.drop_duplicates(feature_columns + ["Attack_type"]).reset_index(drop=True)
    metadata = {
        "source_rows": int(len(frame)),
        "unknown_label_rows_dropped": unknown_label_count,
        "conflicting_feature_rows_dropped": conflicting_rows_dropped,
        "duplicate_rows_dropped": int(before_dedup - len(rows)),
        "label_mapping": ATTACK_GROUPS,
        "class_counts": {str(key): int(value) for key, value in rows["Attack_type"].value_counts().sort_index().items()},
        "protocol_counts": {str(key): int(value) for key, value in rows[PROTOCOL_FEATURE].value_counts().sort_index().items()},
    }
    return rows, metadata


def packet_features_from_scapy(packet, scapy, protocol_state: dict, *, include_endpoint_metadata: bool = False) -> dict[str, Any] | None:
    """Extract model fields and optionally endpoint metadata for the live report."""
    if packet is None or (not packet.haslayer(scapy.TCP) and not packet.haslayer(scapy.UDP)):
        return None

    from scapy.layers.dns import DNSQR

    raw_bytes = b""
    if packet.haslayer(scapy.Raw):
        raw_bytes = bytes(packet[scapy.Raw].load)
    tcp = packet[scapy.TCP] if packet.haslayer(scapy.TCP) else None
    udp = packet[scapy.UDP] if packet.haslayer(scapy.UDP) else None
    source_port = int(tcp.sport if tcp is not None else udp.sport)
    destination_port = int(tcp.dport if tcp is not None else udp.dport)

    flags = int(tcp.flags) if tcp is not None else 0
    fin = bool(flags & 0x01)
    syn = bool(flags & 0x02)
    rst = bool(flags & 0x04)
    psh = bool(flags & 0x08)
    ack = bool(flags & 0x10)

    packet_time = _number(getattr(packet, "time", 0.0))
    udp_delta = 0.0
    if udp is not None:
        # The training column is typically zero; approximate its packet timing
        # without keeping source/destination IPs or port tuples in memory.
        last_time = protocol_state.get("last_udp_time")
        if last_time is not None and packet_time >= last_time:
            udp_delta = packet_time - last_time
        protocol_state["last_udp_time"] = packet_time

    http_response = int(raw_bytes.startswith((b"HTTP/1.0 ", b"HTTP/1.1 ")))
    if not http_response:
        try:
            from scapy.layers.http import HTTPResponse
            http_response = int(packet.haslayer(HTTPResponse))
        except ImportError:
            pass

    dns_query_name_length = 0
    if packet.haslayer(DNSQR):
        query_name = getattr(packet[DNSQR], "qname", b"")
        if isinstance(query_name, str):
            query_name = query_name.encode("utf-8", errors="ignore")
        dns_query_name_length = len(bytes(query_name).rstrip(b".\x00"))
    mqtt_message_type = 0
    mqtt_length = 0
    is_mqtt = source_port == 1883 or destination_port == 1883
    if is_mqtt and raw_bytes:
        mqtt_message_type = raw_bytes[0] >> 4
        remaining = 0
        multiplier = 1
        for byte in raw_bytes[1:5]:
            remaining += (byte & 0x7F) * multiplier
            if byte < 0x80:
                mqtt_length = remaining
                break
            multiplier *= 128

    is_http = http_response or source_port in (80, 8080) or destination_port in (80, 8080)
    protocol = "mqtt" if is_mqtt else "http" if is_http else "tcp" if tcp is not None else "udp"
    row = {
        "observed_at_epoch_s": packet_time,
        "packet_bytes": int(len(packet)),
        "tcp.connection.fin": int(fin),
        "tcp.connection.rst": int(rst),
        "tcp.connection.syn": int(syn and not ack),
        "tcp.connection.synack": int(syn and ack),
        "tcp.flags.ack": int(ack),
        "tcp.flags": flags,
        "tcp.len": len(bytes(tcp.payload)) if tcp is not None else 0,
        "tcp.dstport": int(tcp.dport) if tcp is not None else 0,
        "udp.time_delta": udp_delta,
        "dns.qry.name.len": dns_query_name_length,
        "http.response": http_response,
        "mqtt.msgtype": mqtt_message_type,
        "mqtt.len": mqtt_length,
        "protocol": protocol,
    }
    if include_endpoint_metadata:
        ip_layer = next((layer for layer in (scapy.IP, scapy.IPv6) if packet.haslayer(layer)), None)
        if ip_layer is not None:
            ip_packet = packet[ip_layer]
            row["endpoint_source_ip"] = str(ip_packet.src)
            row["endpoint_destination_ip"] = str(ip_packet.dst)
            row["endpoint_source_port"] = source_port
            row["endpoint_destination_port"] = destination_port
    return row
