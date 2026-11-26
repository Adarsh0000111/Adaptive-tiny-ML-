from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import os
import re
import subprocess
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.inference import InferenceService, ROOT
from src.api.offloading import TrustedNodeRouter
from src.api.resources import HostMetricsSampler
from src.api.serial_device import USBReplayService
from src.api.store import EventStore
from src.data.live_sniffer import RealTimeNetworkSniffer
from src.data.edge_packet_schema import SCHEMA_VERSION
from src.sara import CrossLayerRiskEngine, SARAEngine
from src.federated.torch_fedavg import TorchFedAvgExperiment
from src.federated.networked import NetworkedFedAvgCoordinator

app = FastAPI(title="AdaptiveML SARA-IDS API", version="1.0.0", description="Inference, telemetry, resource monitoring, and security-constrained model selection for the 18-feature packet schema.")
origins = [part.strip() for part in os.getenv("ADAPTIVEML_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if part.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["saved_sample", "preprocessed_features"] = "saved_sample"
    model: Literal["auto", "lightweight", "medium", "heavy"] = "auto"
    sample_index: int = Field(default=0, ge=0)
    features: list[float] | None = None
    feature_names: list[str] | None = None
    network_risk: float | None = Field(default=None, ge=0, le=1)
    energy_pct: float | None = Field(default=None, ge=0, le=100)

    @field_validator("features")
    @classmethod
    def finite_features(cls, value):
        if value is not None and not all(math.isfinite(float(x)) for x in value):
            raise ValueError("features must contain only finite numbers")
        return value


class TelemetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    timestamp: datetime
    source: Literal["physical", "simulated"]
    voltage_v: float | None = Field(default=None, ge=0, le=50)
    current_a: float | None = Field(default=None, ge=-10, le=10)
    power_w: float | None = Field(default=None, ge=0, le=500)
    energy_j: float | None = Field(default=None, ge=0)
    temperature_ambient_c: float | None = Field(default=None, ge=-50, le=150)
    temperature_device_c: float | None = Field(default=None, ge=-50, le=200)
    humidity_pct: float | None = Field(default=None, ge=0, le=100)
    battery_pct: float | None = Field(default=None, ge=0, le=100)
    cpu_percent: float | None = Field(default=None, ge=0, le=100)
    memory_percent: float | None = Field(default=None, ge=0, le=100)
    free_heap_bytes: int | None = Field(default=None, ge=0)
    inference_queue_length: int = Field(default=0, ge=0, le=100000)
    hardware_status: Literal["online", "degraded", "offline"] = "online"
    quality: Literal["good", "partial", "invalid"] = "good"


class NetworkStatsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Literal["capture", "replay", "simulated"]
    window_seconds: float = Field(gt=0, le=3600)
    packets: int = Field(ge=0)
    bytes: int = Field(ge=0)
    tcp_syn: int = Field(default=0, ge=0)
    tcp_fin: int = Field(default=0, ge=0)
    tcp_rst: int = Field(default=0, ge=0)
    unique_destinations: int | None = Field(default=None, ge=0)
    protocol_counts: dict[str, int] = Field(default_factory=dict)
    tcp_connections: int = Field(default=0, ge=0)
    link_capacity_bps: int | None = Field(default=None, ge=1)
    round_trip_latency_ms: float | None = Field(default=None, ge=0)
    packet_loss_pct: float | None = Field(default=None, ge=0, le=100)
    jitter_ms: float | None = Field(default=None, ge=0)
    note: str | None = Field(default=None, max_length=300)

    @field_validator("protocol_counts")
    @classmethod
    def valid_protocol_counts(cls, value):
        if any(not key or count < 0 for key, count in value.items()):
            raise ValueError("protocol_counts must have non-empty names and non-negative counts")
        return value


class NetworkProbeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_ip: str = Field(min_length=3, max_length=45)
    packets: int = Field(default=5, ge=2, le=20)


class NetworkCaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    authorized: bool = False
    interface: str | None = Field(default=None, max_length=128)
    max_packets: int = Field(default=500, ge=1, le=5000)
    timeout_seconds: int = Field(default=10, ge=1, le=60)


class InternalInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: list[float]
    feature_names: list[str]
    model: Literal["lightweight", "medium", "heavy"]


class FederatedTrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rounds: int = Field(default=3, ge=1, le=20)
    clients: int = Field(default=5, ge=2, le=8)
    local_epochs: int = Field(default=1, ge=1, le=3)
    dirichlet_alpha: float = Field(default=0.5, ge=0.1, le=10)
    client_topology: Literal["simulated_host", "physical_esp32", "networked"] = "simulated_host"


class NetworkedFederatedStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_ids: list[str] = Field(min_length=2, max_length=32)
    rounds: int = Field(default=3, ge=1, le=100)
    local_epochs: int = Field(default=1, ge=1, le=5)
    min_clients: int = Field(default=2, ge=2, le=32)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)


class NetworkedFederatedUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round: int = Field(ge=1)
    sample_count: int = Field(ge=1, le=10_000_000)
    local_loss: float = Field(ge=0, le=1_000_000)
    state_dict: dict[str, Any]


class USBReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    port: str | None = Field(default=None, min_length=1, max_length=128)
    count: int = Field(default=100, ge=1, le=1000)
    sample_start: int = Field(default=0, ge=0)


class USBLiveCaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    authorized: bool = False
    port: str | None = Field(default=None, min_length=1, max_length=128)
    interface: str | None = Field(default=None, max_length=128)
    max_packets: int = Field(default=100, ge=1, le=500)
    timeout_seconds: int = Field(default=5, ge=1, le=30)


@app.on_event("startup")
def startup() -> None:
    app.state.inference = InferenceService()
    app.state.risk_engine = CrossLayerRiskEngine()
    app.state.sara = SARAEngine({key: {"macro_recall": value["metrics"]["macro_recall"], "estimated_energy_uj": value["energy"]["value_uj"] or 1e9} for key, value in app.state.inference.profiles.items()},
                                min_esp32_supply_v=float(os.getenv("ADAPTIVEML_ESP32_MIN_SUPPLY_V", "3.0")),
                                max_esp32_current_a=float(os.getenv("ADAPTIVEML_ESP32_MAX_CURRENT_A", "0.9")))
    app.state.store = EventStore(Path(os.getenv("ADAPTIVEML_DB", str(ROOT / "data/runtime/adaptiveml.sqlite3"))))
    app.state.offloader = TrustedNodeRouter()
    app.state.federated = TorchFedAvgExperiment()
    app.state.networked_federated = NetworkedFedAvgCoordinator()
    app.state.usb_replay = USBReplayService()
    app.state.host_metrics = HostMetricsSampler()
    app.state.energy_by_node = {}
    app.state.latest_telemetry = {}
    app.state.usb_sensor_snapshot = None
    app.state.federated_schedule_log = []
    app.state.last_live_security_context = None
    app.state.network_stats = {}
    app.state.ws_clients = set()
    app.state.lock = threading.RLock()


async def broadcast(event: dict[str, Any]) -> None:
    stale = []
    for socket in list(app.state.ws_clients):
        try:
            await socket.send_json(event)
        except Exception:
            stale.append(socket)
    for socket in stale:
        app.state.ws_clients.discard(socket)


def _predict(tier: str, row: np.ndarray, source: str, sample_index: int | None = None) -> dict[str, Any]:
    return app.state.inference.infer(tier, row, source=source, sample_index=sample_index)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AdaptiveML SARA-IDS", "mode": os.getenv("ADAPTIVEML_MODE", "simulation"), "mode_source": "environment default; physical mode requires verified device telemetry", "model_count": len(app.state.inference.models), "feature_schema": SCHEMA_VERSION}


@app.get("/api/models")
def models():
    return {"feature_schema": {"version": SCHEMA_VERSION, "feature_names": app.state.inference.feature_names, "class_names": app.state.inference.class_names}, "models": app.state.inference.profiles, "model_count": len(app.state.inference.models)}


@app.post("/api/inference")
async def inference(request: InferenceRequest):
    service = app.state.inference
    sample_index: int | None = None
    true_label: str | None = None
    if request.mode == "saved_sample":
        if request.features is not None or request.feature_names is not None:
            raise HTTPException(422, "Do not mix saved_sample mode with caller-supplied features")
        try:
            row, encoded_label, sample_count = service.sample_row(request.sample_index)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        sample_index = request.sample_index
        true_label = service.class_names[encoded_label]
        source = "saved_test_replay"
    else:
        if request.features is None or request.feature_names is None:
            raise HTTPException(422, "preprocessed_features mode requires features and feature_names")
        try:
            row = service.validate_vector(request.features, request.feature_names)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        sample_count = None
        source = "caller_preprocessed_features"

    initial = _predict("lightweight", row, source, sample_index)
    diagnostics_started = time.perf_counter()
    model_predictions = {"lightweight": initial["prediction"]}
    for tier, model in service.models.items():
        if tier != "lightweight":
            model_predictions[tier] = service.class_names[int(model.predict(row.reshape(1, -1))[0])]
    vote_counts = {name: list(model_predictions.values()).count(name) for name in set(model_predictions.values())}
    vote_share = max(vote_counts.values()) / len(model_predictions)
    disagreement = (1.0 - vote_share) / (1.0 - 1.0 / len(model_predictions))
    diagnostics_latency_ms = (time.perf_counter() - diagnostics_started) * 1000.0
    class_severity = {"Normal": 0.0, "Scanning": 0.35, "DDoS": 0.90, "Web_Attack": 0.60, "Access_Compromise": 0.75, "Ransomware": 0.95}
    network_score = request.network_risk if request.network_risk is not None else class_severity.get(initial["prediction"], 0.5)
    network_evidence_source = "caller_provided_score" if request.network_risk is not None else "lightweight_model_label_proxy; no independent live network detector is connected"
    resources = app.state.host_metrics.sample()
    resource_stress = resources["resource_stress"]
    uncertainty = initial["normalized_entropy"] or 0.0
    risk = app.state.risk_engine.evaluate(network_score, resource_stress, uncertainty, disagreement)
    risk["network_evidence_source"] = network_evidence_source
    decision = app.state.sara.select(risk["score"], resource_stress, remote_available=app.state.offloader.available, energy_pct=request.energy_pct,
                                     sensor_snapshot=app.state.usb_sensor_snapshot, network_context={"source": "packet feature model", "packet_features_used": len(service.feature_names)},
                                     confidence=initial.get("confidence"))
    selected_tier = request.model if request.model != "auto" else decision["action"]
    if selected_tier != "remote":
        decision["execution_target"] = "local_host"
    elif request.model != "auto":
        decision["execution_target"] = "local_host"
    remote_result = None
    if selected_tier == "remote":
        fallback_tier = decision.get("fallback_action") or "heavy"
        try:
            remote_result, remote_status = await asyncio.to_thread(app.state.offloader.infer, {"features": row.tolist(), "feature_names": service.feature_names, "model": fallback_tier})
            selected_tier = remote_result.get("model", fallback_tier)
        except RuntimeError as exc:
            decision["reason"] += f" Remote inference failed; local fallback used ({type(exc).__name__})."
            decision["offloading_failure"] = str(exc)
            selected_tier = fallback_tier
            decision["security_coverage_degraded"] = service.profiles[selected_tier]["metrics"]["macro_recall"] < decision["required_macro_recall"]
            remote_status = {"success": False, "fallback_model": fallback_tier}
    else:
        remote_status = {"success": False, "reason": "Local inference selected."}
    if remote_result is not None:
        result = remote_result
        result["inference_ms"] = remote_status["latency_ms"]
        result["input_source"] = "trusted_remote_inference"
    else:
        result = initial if selected_tier == "lightweight" else _predict(selected_tier, row, source, sample_index)
    result.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "true_label": true_label,
        "sample_count": sample_count,
        "mode": request.mode,
        "adaptive": {"risk": risk, "decision": decision, "remote_inference": remote_status},
        "resources": resources,
        "requested_model": request.model,
        "model_selection_source": "user_override" if request.model != "auto" else "SARA",
        "model_diagnostics": {"prediction_confidence": initial["confidence"], "normalized_entropy": uncertainty, "model_predictions": model_predictions, "disagreement_score": round(disagreement, 4), "measurement_latency_ms": diagnostics_latency_ms},
    })
    app.state.store.add_inference(result)
    await broadcast({"type": "inference", "data": result})
    return result


@app.post("/api/internal/inference")
def internal_inference(request: InternalInferenceRequest, authorization: str | None = Header(default=None)):
    token = os.getenv("SARA_NODE_TOKEN")
    if not token or authorization != f"Bearer {token}":
        raise HTTPException(401, "Trusted-node authentication failed")
    try:
        row = app.state.inference.validate_vector(request.features, request.feature_names)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _predict(request.model, row, "trusted_remote_request")


@app.get("/api/adaptive/status")
def adaptive_status():
    return {"current_model": app.state.sara.current_model, "policy": "SARA cross-layer, security-constrained selection", "history": app.state.sara.history[:100], "risk_weights": app.state.risk_engine.weights.__dict__, "resource_stress_is_attack_evidence": False, "sensor_role": "qualified INA219 supply/current can constrain ESP32 execution; ambient/motion are context only", "esp32_min_supply_v": app.state.sara.min_esp32_supply_v, "esp32_max_current_a": app.state.sara.max_esp32_current_a}


@app.get("/api/adaptive/history")
def adaptive_history():
    return {"items": app.state.sara.history[:100]}


@app.get("/api/resources")
def resources():
    return app.state.host_metrics.sample()


@app.get("/api/device/usb/status")
def usb_device_status():
    return app.state.usb_replay.status()


@app.post("/api/device/usb/replay")
async def usb_device_replay(request: USBReplayRequest):
    try:
        result = await asyncio.to_thread(
            app.state.usb_replay.replay,
            request.port,
            request.count,
            request.sample_start,
            app.state.inference,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    if result.get("device_telemetry") is not None:
        app.state.usb_sensor_snapshot = result["device_telemetry"]
    await broadcast({"type": "usb_inference_replay", "data": result})
    return result


@app.post("/api/device/usb/live-capture")
async def usb_device_live_capture(request: USBLiveCaptureRequest):
    if not request.authorized:
        raise HTTPException(403, "Explicit authorization is required before capturing packets on the selected interface")
    if not any(item.get("accessible") for item in app.state.usb_replay.status().get("ports", [])):
        raise HTTPException(409, "The ESP32 USB serial port is not accessible; no packets were captured")
    sniffer = RealTimeNetworkSniffer(request.interface)
    service = app.state.inference
    resources = app.state.host_metrics.sample()
    class_severity = {"Normal": 0.0, "Scanning": 0.35, "DDoS": 0.90, "Web_Attack": 0.60, "Access_Compromise": 0.75, "Ransomware": 0.95}

    def run_live_capture():
        selected_packets: list[dict[str, Any]] = []
        changes: list[dict[str, Any]] = []
        traffic_window: deque[tuple[float, int]] = deque()
        tier_counts: Counter[str] = Counter()
        previous_tier = app.state.sara.current_model
        with app.state.usb_replay.live_session(request.port, service) as device:
            def process_packet(raw_row):
                nonlocal previous_tier
                device.refresh_sensor_snapshot()
                sensor_snapshot = device.latest_sensors
                if sensor_snapshot is not None:
                    app.state.usb_sensor_snapshot = sensor_snapshot
                packet_resources = app.state.host_metrics.sample()
                feature_row = service.transform_packet_rows([raw_row])[0]
                per_tier = {tier: service.infer(tier, feature_row, source="authorized_live_packet", sample_index=len(selected_packets))
                            for tier in ("lightweight", "medium", "heavy")}
                model_predictions = {tier: item["prediction"] for tier, item in per_tier.items()}
                vote_counts = Counter(model_predictions.values())
                vote_share = max(vote_counts.values()) / len(model_predictions)
                disagreement = (1.0 - vote_share) / (1.0 - 1.0 / len(model_predictions))
                low_prediction = per_tier["lightweight"]
                uncertainty = float(low_prediction.get("normalized_entropy") or 0.0)
                network_score = class_severity.get(low_prediction["prediction"], 0.5)
                risk = app.state.risk_engine.evaluate(network_score, packet_resources["resource_stress"], uncertainty, disagreement)
                packet_time = float(raw_row.get("observed_at_epoch_s") or time.time())
                packet_bytes = int(raw_row.get("packet_bytes") or 0)
                traffic_window.append((packet_time, packet_bytes))
                while traffic_window and packet_time - traffic_window[0][0] > 5.0:
                    traffic_window.popleft()
                span = packet_time - traffic_window[0][0] if len(traffic_window) > 1 else 0.0
                packet_rate = len(traffic_window) / span if span >= 0.5 else None
                bytes_per_second = sum(item[1] for item in traffic_window) / span if span >= 0.5 else None
                edge_capacity = None
                if len(device.round_trip_ms) >= 3:
                    median_rtt = float(np.median(device.round_trip_ms[-100:]))
                    edge_capacity = 1000.0 / median_rtt if median_rtt > 0 else None
                network_context = {
                    "source": "current captured packet features and model predictions",
                    "packet_index": len(selected_packets), "protocol": raw_row.get("protocol"),
                    "packet_features_used": len(service.feature_names), "payload_retained": False,
                    "packet_bytes": packet_bytes, "window_packets": len(traffic_window),
                    "window_seconds": round(span, 4), "packet_rate_pps": packet_rate,
                    "bytes_per_second": bytes_per_second,
                    "edge_capacity_pps": edge_capacity,
                    "capacity_source": "median measured USB round-trip from preceding ESP32 inferences" if edge_capacity else "not yet measured",
                }
                decision = app.state.sara.select(risk["score"], packet_resources["resource_stress"], remote_available=app.state.offloader.available,
                                                 sensor_snapshot=sensor_snapshot, network_context=network_context,
                                                 confidence=low_prediction.get("confidence"))
                app.state.last_live_security_context = {
                    "source": "authorized_live_packet_capture", "timestamp": time.time(),
                    "risk": risk["score"], "confidence": low_prediction.get("confidence"),
                    "model_uncertainty": uncertainty, "model_disagreement": disagreement,
                    "network_context": network_context,
                }
                # Escalate only when all three tiers disagree and the high tier
                # is more confident than medium on this specific packet.
                if (disagreement >= 0.99 and per_tier["heavy"].get("confidence") is not None
                        and per_tier["medium"].get("confidence") is not None
                        and per_tier["heavy"]["confidence"] > per_tier["medium"]["confidence"]):
                    decision["previous_action"] = app.state.sara.current_model
                    decision["action"] = "heavy"
                    decision["changed"] = decision["previous_action"] != "heavy"
                    decision["reason"] += " Complete model disagreement; high tier was more confident than medium."
                    heavy_recall = float(app.state.sara.profiles["heavy"].get("macro_recall", 0.0))
                    decision["validation_macro_recall"] = round(heavy_recall, 4)
                    decision["security_coverage_degraded"] = heavy_recall < decision["required_macro_recall"]
                    app.state.sara.current_model = "heavy"
                remote_result = None
                if decision["action"] == "remote":
                    try:
                        remote_result, remote_status = app.state.offloader.infer({"features": feature_row.tolist(), "feature_names": service.feature_names,
                                                                                   "model": decision.get("fallback_action") or "lightweight"})
                    except RuntimeError as exc:
                        decision["action"] = decision.get("fallback_action") or "lightweight"
                        decision["reason"] += f" Trusted offload failed; local laptop fallback used ({type(exc).__name__})."
                        decision["offloading_failure"] = str(exc)
                tier = decision["action"] if decision["action"] in ("lightweight", "medium", "heavy") else (decision.get("fallback_action") or "heavy")
                if tier != previous_tier:
                    changes.append({"packet_index": len(selected_packets), "from": previous_tier, "to": tier, "risk": risk["score"], "reason": decision["reason"]})
                previous_tier = tier
                tier_counts[tier] += 1
                prediction = per_tier[tier]
                electrical_context = SARAEngine.sensor_context(sensor_snapshot, min_supply_v=app.state.sara.min_esp32_supply_v,
                                                               max_current_a=app.state.sara.max_esp32_current_a)
                edge_blocked = (electrical_context.get("edge_execution_allowed") is False
                                or (tier == "lightweight" and decision.get("execution_target") == "local_host"))
                device_result = device.infer(feature_row) if tier == "lightweight" and not edge_blocked else None
                decision["execution_target"] = "trusted_remote" if remote_result else "esp32_usb" if device_result else "local_host"
                if edge_blocked:
                    decision["execution_reason"] = decision.get("execution_reason") or "Qualified INA219 supply voltage is below the configured 3.0 V ESP32 execution floor; the selected tier ran on the laptop."
                elif tier == "lightweight":
                    decision["execution_reason"] = "ESP32 low-tier inference used; no qualified undervoltage was reported."
                else:
                    decision["execution_reason"] = "Selected medium/high tier is executed on the laptop; no trusted remote node was available."
                selected_packets.append({
                    "packet_index": len(selected_packets), "selected_model": tier,
                    "prediction": remote_result["prediction"] if remote_result else device_result["prediction"] if device_result else prediction["prediction"],
                    "confidence": remote_result.get("confidence", prediction.get("confidence")) if remote_result else prediction.get("confidence"),
                    "inference_ms": remote_status.get("latency_ms") if remote_result else prediction.get("inference_ms") if device_result is None else None,
                    "device_inference_us": device_result.get("inference_us") if device_result else None,
                    "inference_location": "esp32_usb" if device_result else "laptop",
                    "model_predictions": model_predictions, "risk": risk, "decision": decision,
                    "host_resources": packet_resources,
                })
                if app.state.sara.history:
                    app.state.sara.history[0] = decision.copy()

            capture = sniffer.capture_with_handler(process_packet, request.max_packets, request.timeout_seconds)
            if not selected_packets:
                raise ValueError("No TCP/UDP packets were observed in the authorized capture window")
            device_summary = device.summary()
        if device_summary["device_telemetry"] is not None:
            app.state.usb_sensor_snapshot = device_summary["device_telemetry"]
        return {
            "transport": "USB serial plus laptop inference", "port": device_summary["port"],
            "firmware": device_summary["firmware"], "feature_schema": service.schema_version,
            "mode": "authorized_live_adaptive_packet_inference", "wifi_used": False,
            "samples": len(selected_packets), "capture_window_seconds": capture["window_seconds"],
            "accuracy": None, "macro_f1": None, "selected_tier_counts": dict(tier_counts),
            "model_switches": changes, "predictions": selected_packets,
            "device_inference_us": device_summary["device_inference_us"],
            "usb_round_trip_ms": device_summary["usb_round_trip_ms"],
            "device_telemetry": device_summary["device_telemetry"], "host_resources": resources,
            "decision_log": [item["decision"] | {"packet_index": item["packet_index"], "selected_model": item["selected_model"], "execution_location": item["inference_location"], "risk": item["risk"]} for item in selected_packets],
            "input_note": "For each arriving packet, SARA evaluated risk and model votes, selected a tier, and immediately ran low-tier predictions on ESP32 over USB or medium/high on the laptop. Live packets have no labels, so accuracy is not reported.",
            "completed_at_unix": time.time(),
        }

    try:
        result = await asyncio.to_thread(run_live_capture)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, "Packet capture needs OS privileges for the selected interface") from exc
    except OSError as exc:
        raise HTTPException(422, f"Could not capture on requested interface: {exc}") from exc
    app.state.usb_replay._last_run = result
    await broadcast({"type": "usb_live_capture", "data": result})
    return result


@app.post("/api/telemetry")
async def telemetry(request: TelemetryRequest):
    payload = request.model_dump(mode="json")
    if request.power_w is None and request.voltage_v is not None and request.current_a is not None:
        payload["power_w"] = request.voltage_v * request.current_a
        payload["power_source"] = "derived_voltage_times_current"
    elif request.power_w is not None:
        payload["power_source"] = "reported_sensor_value"
    payload["timestamp"] = request.timestamp.astimezone(timezone.utc).isoformat()
    prior = app.state.latest_telemetry.get(request.node_id)
    cumulative = request.energy_j
    if prior and request.timestamp.timestamp() > prior["time"] and payload.get("power_w") is not None and prior.get("power_w") is not None:
        elapsed = request.timestamp.timestamp() - prior["time"]
        if elapsed <= 3600:
            integrated = (max(0.0, payload["power_w"]) + max(0.0, prior["power_w"])) * 0.5 * elapsed
            cumulative = (prior.get("energy_j") or 0.0) + integrated
            payload["energy_integration"] = {"method": "trapezoidal", "elapsed_seconds": elapsed, "source": "timestamped_power_samples"}
    app.state.latest_telemetry[request.node_id] = {"time": request.timestamp.timestamp(), "power_w": payload.get("power_w"), "energy_j": cumulative}
    payload["cumulative_energy_j"] = cumulative
    stored = app.state.store.add_telemetry(payload, cumulative)
    await broadcast({"type": "telemetry", "data": stored})
    return stored


@app.get("/api/telemetry")
def get_telemetry(limit: int = 100):
    if not 1 <= limit <= 1000:
        raise HTTPException(422, "limit must be between 1 and 1000")
    return {"items": app.state.store.recent_telemetry(limit)}


@app.post("/api/network/stats")
async def receive_network_stats(request: NetworkStatsRequest):
    data = request.model_dump(mode="json")
    data["timestamp"] = request.timestamp.astimezone(timezone.utc).isoformat()
    data["packets_per_second"] = request.packets / request.window_seconds
    data["bytes_per_second"] = request.bytes / request.window_seconds
    data["average_packet_size_bytes"] = request.bytes / request.packets if request.packets else None
    data["connection_frequency_per_second"] = request.tcp_connections / request.window_seconds
    data["bandwidth_bps"] = request.bytes * 8 / request.window_seconds
    data["bandwidth_utilization_pct"] = min(100.0, data["bandwidth_bps"] / request.link_capacity_bps * 100.0) if request.link_capacity_bps else None
    data["protocol_types"] = sorted(request.protocol_counts)
    data["network_measurement_note"] = "Rates and packet size are derived from the declared capture window. Latency, loss, jitter, and utilization require measured probe inputs and known link capacity; missing values stay null."
    data["source_label"] = {"capture": "physical_or_live_capture; authorization required", "replay": "recorded traffic replay", "simulated": "simulated"}[request.source]
    app.state.network_stats[request.node_id] = data
    await broadcast({"type": "network_stats", "data": data})
    return data


@app.post("/api/network/capture")
async def capture_network(request: NetworkCaptureRequest):
    if not request.authorized:
        raise HTTPException(403, "Set authorized=true after confirming you may monitor traffic on this interface")
    sniffer = RealTimeNetworkSniffer(request.interface)
    try:
        frame = await asyncio.to_thread(sniffer.capture_live_stream, request.max_packets, request.timeout_seconds)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, "Packet capture needs OS privileges for the selected interface") from exc
    except OSError as exc:
        raise HTTPException(422, f"Could not capture on the requested interface: {exc}") from exc
    summary = frame.iloc[0].to_dict()
    link_capacity_bps = None
    if request.interface:
        try:
            import psutil
            link_mbps = psutil.net_if_stats().get(request.interface).speed
            if link_mbps and link_mbps > 0:
                link_capacity_bps = int(link_mbps * 1_000_000)
        except (ImportError, AttributeError, OSError):
            pass
    stats = NetworkStatsRequest(
        node_id="laptop-capture",
        timestamp=datetime.fromtimestamp(float(summary["timestamp_epoch_s"]), timezone.utc),
        source="capture",
        window_seconds=max(0.001, float(summary["window_seconds"])),
        packets=int(summary["packets"]),
        bytes=int(summary["bytes"]),
        tcp_syn=int(summary["tcp_syn"]),
        tcp_fin=int(summary["tcp_fin"]),
        tcp_rst=int(summary["tcp_rst"]),
        protocol_counts={str(key): int(value) for key, value in summary["protocol_counts"].items()},
        tcp_connections=int(summary["tcp_syn"]),
        link_capacity_bps=link_capacity_bps,
        note="Authorized passive packet capture; payload and endpoint addresses are not stored.",
    )
    result = await receive_network_stats(stats)
    result["capture_limits"] = "Passive capture does not measure RTT, packet loss, or jitter. Aggregate counters are not model-compatible IDS features."
    return result


@app.get("/api/network/stats")
def get_network_stats():
    return {"items": list(app.state.network_stats.values()), "feature_compatibility": "These aggregate counters do not reproduce the training feature schema and are not passed to models."}


@app.post("/api/network/probe")
def probe_network(request: NetworkProbeRequest):
    try:
        target = str(ipaddress.ip_address(request.target_ip))
    except ValueError as exc:
        raise HTTPException(422, "Use a literal IPv4 or IPv6 address; hostnames are not accepted") from exc
    command = ["ping", "-n", "-c", str(request.packets), "-W", "2", target]
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=request.packets * 2 + 3, check=False, env={**os.environ, "LC_ALL": "C"})
    except FileNotFoundError as exc:
        raise HTTPException(503, "The system ping utility is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "Network probe timed out") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    output = completed.stdout + "\n" + completed.stderr
    times = [float(value) for value in re.findall(r"time[=<]([0-9.]+)\s*ms", output)]
    loss_match = re.search(r"([0-9.]+)%\s*packet loss", output)
    loss_pct = float(loss_match.group(1)) if loss_match else None
    deltas = [abs(times[index] - times[index - 1]) for index in range(1, len(times))]
    result = {
        "target_ip": target,
        "source": "explicit_active_icmp_probe",
        "packets_sent": request.packets,
        "packets_received": len(times),
        "packet_loss_pct": loss_pct,
        "rtt_ms": {"mean": sum(times) / len(times) if times else None, "min": min(times) if times else None, "max": max(times) if times else None, "samples": times},
        "jitter_ms": sum(deltas) / len(deltas) if deltas else None,
        "probe_wall_time_ms": elapsed_ms,
        "note": "RTT is round-trip time; jitter is the mean absolute difference between successive successful RTT samples. This is not a one-way latency measurement and does not produce IDS features.",
    }
    # Pair an explicit ICMP measurement with a very recent passive capture so SARA
    # can use measured latency/loss together with its measured link utilization.
    latest_capture = app.state.network_stats.get("laptop-capture")
    if latest_capture and latest_capture.get("source") == "capture" and latest_capture.get("bandwidth_utilization_pct") is not None:
        try:
            capture_age = time.time() - datetime.fromisoformat(latest_capture["timestamp"]).timestamp()
        except (KeyError, TypeError, ValueError):
            capture_age = float("inf")
        if 0 <= capture_age <= 30:
            latest_capture["round_trip_latency_ms"] = result["rtt_ms"]["mean"]
            latest_capture["packet_loss_pct"] = loss_pct
            latest_capture["jitter_ms"] = result["jitter_ms"]
            latest_capture["timestamp"] = datetime.now(timezone.utc).isoformat()
            latest_capture["network_measurement_note"] = "Fresh passive capture counters/link utilization paired with explicit ICMP RTT/loss/jitter probe."
            result["paired_with_fresh_capture"] = True
        else:
            result["paired_with_fresh_capture"] = False
    else:
        result["paired_with_fresh_capture"] = False
    if completed.returncode not in (0, 1) and loss_pct is None:
        result["error"] = output.strip()[-500:]
    return result


@app.get("/api/nodes")
def nodes():
    now = datetime.now(timezone.utc).timestamp()
    return {"items": [{"node_id": node, "source": data["source"], "timestamp": data["timestamp"], "stale": now - datetime.fromisoformat(data["timestamp"]).timestamp() > 30, "hardware_status": data["hardware_status"]} for node, data in app.state.latest_telemetry.items()]}


@app.get("/api/alerts")
def alerts():
    rows = app.state.store.recent_inferences(100)
    return {"items": [{"timestamp": item["timestamp"], "prediction": item["prediction"], "risk": item["adaptive"]["risk"]["score"], "coverage_degraded": item["adaptive"]["decision"]["security_coverage_degraded"]} for item in rows if item["prediction"] != "Normal" or item["adaptive"]["decision"]["security_coverage_degraded"]]}


@app.get("/api/offloading/status")
def offloading_status():
    return app.state.offloader.status()


@app.get("/api/federated/status")
def federated_status():
    return app.state.federated.status()


@app.get("/api/federated/rounds")
def federated_rounds():
    return {"items": app.state.federated.rounds(), "source": "current API process; simulated independent client partitions"}


@app.get("/api/federated/schedule")
def federated_schedule():
    return {"items": app.state.federated_schedule_log[:100], "latest": app.state.federated_schedule_log[0] if app.state.federated_schedule_log else None}


@app.get("/api/federated/networked/status")
def networked_federated_status():
    return app.state.networked_federated.status()


@app.post("/api/federated/networked/start", status_code=202)
def start_networked_federated(request: NetworkedFederatedStartRequest, authorization: str | None = Header(default=None)):
    expected = os.getenv("SARA_FL_COORDINATOR_TOKEN", "")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    import hmac
    if not expected or not hmac.compare_digest(expected, supplied):
        raise HTTPException(401, "Invalid coordinator credentials")
    resources = app.state.host_metrics.sample()
    schedule = app.state.sara.schedule_federated(resources, sensor_snapshot=app.state.usb_sensor_snapshot,
                                                 network_stats=list(app.state.network_stats.values()),
                                                 security_context=app.state.last_live_security_context,
                                                 client_topology="networked")
    app.state.federated_schedule_log.insert(0, {"timestamp": datetime.now(timezone.utc).isoformat(), **schedule})
    app.state.federated_schedule_log[:] = app.state.federated_schedule_log[:100]
    if not schedule["schedule"]:
        raise HTTPException(409, {"message": "SARA deferred networked federated training", "schedule_decision": schedule})
    try:
        status = app.state.networked_federated.start(request.node_ids, request.rounds, request.local_epochs, request.min_clients, request.timeout_seconds)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(409 if isinstance(exc, RuntimeError) else 422, str(exc)) from exc
    return {"accepted": True, "status": status, "schedule_decision": schedule,
            "message": "Network FedAvg started; clients train locally and upload parameter updates only."}


@app.get("/api/federated/networked/assignment/{node_id}")
def networked_federated_assignment(node_id: str, authorization: str | None = Header(default=None)):
    from fastapi import HTTPException
    try:
        return app.state.networked_federated.assignment(node_id, authorization.removeprefix("Bearer ") if authorization else None)
    except PermissionError as exc:
        raise HTTPException(401, str(exc)) from exc


@app.post("/api/federated/networked/update/{node_id}")
def networked_federated_update(node_id: str, request: NetworkedFederatedUpdate, authorization: str | None = Header(default=None)):
    from fastapi import HTTPException
    try:
        return app.state.networked_federated.submit(node_id, authorization.removeprefix("Bearer ") if authorization else None,
                                                   request.round, request.sample_count, request.state_dict, request.local_loss)
    except PermissionError as exc:
        raise HTTPException(401, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/federated/train", status_code=202)
def start_federated_training(request: FederatedTrainRequest):
    resources = app.state.host_metrics.sample()
    schedule = app.state.sara.schedule_federated(resources, sensor_snapshot=app.state.usb_sensor_snapshot,
                                                 network_stats=list(app.state.network_stats.values()),
                                                 security_context=app.state.last_live_security_context,
                                                 client_topology=request.client_topology)
    app.state.federated_schedule_log.insert(0, {"timestamp": datetime.now(timezone.utc).isoformat(), **schedule})
    app.state.federated_schedule_log[:] = app.state.federated_schedule_log[:100]
    if not schedule["schedule"]:
        raise HTTPException(409, {"message": "SARA deferred the federated training request based on the selected client topology's resource or link constraints", "schedule_decision": schedule})
    if request.client_topology != "simulated_host":
        raise HTTPException(501, {"message": "SARA approved the client preflight, but this build only implements local simulated FedAvg; no ESP32 or networked training was started", "schedule_decision": schedule})
    try:
        app.state.federated.start(request.rounds, request.clients, request.local_epochs, request.dirichlet_alpha)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(409 if isinstance(exc, RuntimeError) else 422, str(exc)) from exc
    return {"accepted": True, "status": app.state.federated.status(), "schedule_decision": schedule,
            "message": "Training runs in a background thread; SARA recorded host and sensor context before scheduling."}


@app.get("/api/experiments/results")
def experiment_results():
    path = ROOT / "results/tables/model_benchmark.json"
    return {"historical_results": json.loads(path.read_text()), "source": "committed benchmark artifact; mixed measured and estimated values; not a current hardware run"}


@app.post("/api/experiments/run")
def run_experiment():
    raise HTTPException(501, "Long-running reproducible experiment runner is not implemented yet; no result was fabricated.")


@app.post("/api/experiments/sensor-ablation")
async def sensor_ablation(request: USBReplayRequest):
    """Paired policy ablation over labeled test packets and real ESP32 readings."""
    service = app.state.inference
    try:
        replay = await asyncio.to_thread(app.state.usb_replay.replay, request.port, request.count,
                                         request.sample_start, service, include_latency_samples=True)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    if replay.get("device_telemetry") is not None:
        app.state.usb_sensor_snapshot = replay["device_telemetry"]
    data = np.load(ROOT / "data/processed/processed_dataset.npz", allow_pickle=False)
    X = data["X_test"][request.sample_start:request.sample_start + request.count].astype(np.float32, copy=False)
    y = data["y_test_multi"][request.sample_start:request.sample_start + request.count].astype(np.int64, copy=False)
    tiers = ("lightweight", "medium", "heavy")
    matrix = {tier: app.state.inference.models[tier].predict(X).astype(np.int64) for tier in tiers}
    severity = {"Normal": 0.0, "Scanning": 0.35, "DDoS": 0.90, "Web_Attack": 0.60, "Access_Compromise": 0.75, "Ransomware": 0.95}
    physical_rows = replay.get("device_telemetry_samples", [])
    hardware_predictions = replay.get("hardware_prediction_indices", [])
    resources = app.state.host_metrics.sample()
    def evaluate(use_sensors: bool):
        policy = SARAEngine(app.state.sara.profiles, min_esp32_supply_v=app.state.sara.min_esp32_supply_v,
                            max_esp32_current_a=app.state.sara.max_esp32_current_a)
        selected, locations, decisions = [], Counter(), []
        for i, row in enumerate(X):
            low = service.infer("lightweight", row, source="sensor_ablation", sample_index=request.sample_start + i)
            preds = {tier: service.class_names[int(matrix[tier][i])] for tier in tiers}
            votes = Counter(preds.values())
            share = max(votes.values()) / len(tiers)
            disagree = (1.0 - share) / (1.0 - 1.0 / len(tiers))
            risk = app.state.risk_engine.evaluate(severity.get(preds["lightweight"], 0.5), resources["resource_stress"],
                                                  float(low.get("normalized_entropy") or 0.0), disagree)
            snapshot = physical_rows[i] if use_sensors and i < len(physical_rows) else None
            decision = policy.select(risk["score"], resources["resource_stress"], remote_available=False,
                                     sensor_snapshot=snapshot,
                                     sensor_reference_time=float(snapshot["timestamp"]) if snapshot and snapshot.get("timestamp") else None,
                                     network_context={"source": "saved labeled packet features", "packet_index": i},
                                     confidence=low.get("confidence"))
            tier = decision["action"] if decision["action"] in tiers else (decision.get("fallback_action") or "heavy")
            location = decision["execution_target"] if use_sensors else ("esp32_usb" if tier == "lightweight" else "local_host")
            selected.append(int(hardware_predictions[i]) if location == "esp32_usb" and i < len(hardware_predictions) else matrix[tier][i])
            locations[location] += 1
            decisions.append(decision)
        labels = np.arange(len(service.class_names))
        from sklearn.metrics import accuracy_score, f1_score
        return {"accuracy": float(accuracy_score(y, selected)),
                "macro_f1": float(f1_score(y, selected, labels=labels, average="macro", zero_division=0)),
                "selected_tiers": dict(Counter(item["action"] for item in decisions)),
                "execution_locations": dict(locations),
                "qualified_sensor_samples": sum(item["sensor_context"]["electrical_quality"] == "qualified" for item in decisions),
                "sensor_affected_decisions": sum(item["sensor_context"]["edge_execution_allowed"] is False for item in decisions),
                "physical_event_samples": sum(bool(item["sensor_context"]["physical_event"]) for item in decisions),
                "ambient_context_samples": sum(item["sensor_context"]["ambient_context"] is not None for item in decisions),
                "decisions": decisions}
    without = evaluate(False)
    with_sensors = evaluate(True)
    result = {
        "experiment": "paired physical-sensor ablation on labeled Edge-IIoTset test rows",
        "samples": int(len(X)), "sample_start": request.sample_start,
        "sensor_source": "actual ESP32-S3 USB telemetry sampled during the paired board replay",
        "trusted_remote_available": app.state.offloader.available,
        "ablation_execution_scope": "ESP32 versus laptop placement; trusted peers are not contacted by this replay study",
        "network_input": "the same saved 18 packet features in both arms; sensors never enter the IDS classifier",
        "comparisons": {"without_physical_sensor_inputs": without, "with_physical_sensor_context": with_sensors},
        "deltas_sensor_minus_baseline": {
            "accuracy": with_sensors["accuracy"] - without["accuracy"],
            "macro_f1": with_sensors["macro_f1"] - without["macro_f1"],
            "sensor_affected_decisions": with_sensors["sensor_affected_decisions"] - without["sensor_affected_decisions"],
            "execution_locations": {key: with_sensors["execution_locations"].get(key, 0) - without["execution_locations"].get(key, 0)
                                     for key in set(with_sensors["execution_locations"]) | set(without["execution_locations"])},
        },
        "interpretation": "Only qualified INA219 supply voltage/current can alter ESP32 execution/offload. Ambient temperature, humidity, vibration, and movement remain physical context and do not change network threat severity. If no supply readings qualify, this run cannot claim a sensor benefit.",
        "limitations": ["Replay supplies labeled packet rows, not a live attack capture.", "Sensor snapshots are paired in time with board inference; this is a policy ablation, not a randomized causal hardware stress test.", "Energy deltas are not reported because the INA219 supply path must be qualified before interpreting them."],
        "device_replay": {key: replay.get(key) for key in ("port", "firmware", "samples", "device_telemetry")},
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact = ROOT / "results/experiments/sensor_ablation_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    result["comparison_summary"] = {
        arm: {key: value for key, value in payload.items() if key != "decisions"}
        for arm, payload in result["comparisons"].items()
    }
    artifact.write_text(json.dumps(result, indent=2))
    result.pop("comparisons", None)
    result["artifact"] = str(artifact)
    return result


@app.websocket("/ws")
async def websocket_updates(socket: WebSocket):
    await socket.accept()
    app.state.ws_clients.add(socket)
    try:
        await socket.send_json({"type": "connected", "data": {"timestamp": datetime.now(timezone.utc).isoformat()}})
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        app.state.ws_clients.discard(socket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host=os.getenv("ADAPTIVEML_HOST", "127.0.0.1"), port=int(os.getenv("ADAPTIVEML_PORT", "8765")), reload=False)
