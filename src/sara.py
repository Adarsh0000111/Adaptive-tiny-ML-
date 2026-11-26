"""Interpretable security-constrained adaptive inference policy.

Resource pressure influences feasibility and action cost. It is never, by itself,
treated as evidence that an attack is occurring.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from datetime import datetime, timezone
import os
import time
from typing import Any


@dataclass(frozen=True)
class RiskWeights:
    network: float = 0.55
    uncertainty: float = 0.20
    model_disagreement: float = 0.15
    network_resource_interaction: float = 0.10


class CrossLayerRiskEngine:
    def __init__(self, weights: RiskWeights | None = None):
        self.weights = weights or RiskWeights()

    @staticmethod
    def _unit(value: float, name: str) -> float:
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return number

    def evaluate(self, network_risk: float, resource_stress: float, uncertainty: float, model_disagreement: float = 0.0) -> dict[str, Any]:
        network = self._unit(network_risk, "network_risk")
        stress = self._unit(resource_stress, "resource_stress")
        uncertain = self._unit(uncertainty, "uncertainty")
        disagreement = self._unit(model_disagreement, "model_disagreement")
        # Resource pressure contributes only when network evidence is also present.
        interaction = network * stress
        w = self.weights
        risk = w.network * network + w.uncertainty * uncertain + w.model_disagreement * disagreement + w.network_resource_interaction * interaction
        return {
            "score": round(min(1.0, max(0.0, risk)), 4),
            "network_evidence": round(network, 4),
            "resource_stress": round(stress, 4),
            "model_uncertainty": round(uncertain, 4),
            "model_disagreement": round(disagreement, 4),
            "network_resource_interaction": round(interaction, 4),
            "weights": {"network": w.network, "uncertainty": w.uncertainty, "model_disagreement": w.model_disagreement, "interaction": w.network_resource_interaction},
            "explanation": "Resource stress affects feasibility; it raises security risk only in interaction with network evidence.",
        }


class SARAEngine:
    """Choose the lowest estimated-cost local model meeting a risk-tier recall floor."""

    ORDER = ("lightweight", "medium", "heavy")

    def __init__(self, profiles: dict[str, dict[str, Any]], *, hysteresis_margin: float = 0.04, min_esp32_supply_v: float = 3.0, max_esp32_current_a: float = 0.9):
        self.profiles = profiles
        self.hysteresis_margin = float(hysteresis_margin)
        self.min_esp32_supply_v = float(min_esp32_supply_v)
        self.max_esp32_current_a = float(max_esp32_current_a)
        if not 2.7 <= self.min_esp32_supply_v <= 5.5:
            raise ValueError("min_esp32_supply_v must be between 2.7 and 5.5 V")
        if not 0.05 <= self.max_esp32_current_a <= 1.0:
            raise ValueError("max_esp32_current_a must be between 0.05 and the configured INA219 1 A range")
        self.current_model = "lightweight"
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def sensor_context(snapshot: dict[str, Any] | None, *, now: float | None = None, min_supply_v: float = 3.0, max_current_a: float = 0.9) -> dict[str, Any]:
        """Separate physical context from IDS evidence; qualify INA219 before using it."""
        result: dict[str, Any] = {
            "source": "unavailable", "age_seconds": None, "electrical_quality": "unavailable",
            "electrical_state": "unknown", "edge_execution_allowed": None,
            "physical_event": False, "ambient_context": None,
            "used_for_security_risk": False,
        }
        if not isinstance(snapshot, dict) or snapshot.get("source") != "physical":
            return result
        result["source"] = snapshot.get("node_id", "physical")
        try:
            stamp = float(snapshot.get("timestamp"))
            age = max(0.0, (time.time() if now is None else now) - stamp)
            result["age_seconds"] = round(age, 3)
        except (TypeError, ValueError):
            return result
        result["physical_event"] = bool(snapshot.get("vibration") or snapshot.get("movement"))
        result["ambient_context"] = {
            "temperature_ambient_c": snapshot.get("temperature_ambient_c"),
            "humidity_pct": snapshot.get("humidity_pct"),
            "role": "environmental context only; not attack evidence or ESP32 die temperature",
        }
        status = snapshot.get("sensor_status") or {}
        voltage, current, power = (snapshot.get(key) for key in ("voltage_v", "current_a", "power_w"))
        try:
            voltage, current, power = float(voltage), float(current), float(power)
            finite = all(math.isfinite(v) for v in (voltage, current, power))
        except (TypeError, ValueError):
            finite = False
        result["electrical_quality"] = "unqualified"
        if finite:
            result["reported_voltage_v"] = round(voltage, 4)
            result["reported_current_a"] = round(current, 6)
            result["reported_power_w"] = round(power, 6)
        if not status.get("ina219"):
            result["electrical_quality_reason"] = "INA219 not detected"
        elif age > 5:
            result["electrical_quality"] = "stale"
            result["electrical_quality_reason"] = "sensor frame is older than five seconds"
        elif not finite:
            result["electrical_quality_reason"] = "voltage, current, or power is missing/non-finite"
        elif not 2.7 <= voltage <= 5.5:
            result["electrical_quality_reason"] = "measured voltage is outside the 2.7–5.5 V plausibility range"
        elif abs(current) > 1.0:
            result["electrical_quality_reason"] = "current exceeds the INA219 1 A calibration range"
        elif abs(power - voltage * current) > max(0.03, abs(power) * 0.15):
            result["electrical_quality_reason"] = "reported power is inconsistent with voltage × current"
        # INA219 presence alone does not prove that its shunt is in the ESP32 supply path.
        # Plausibility and P=V*I checks avoid driving policy from disconnected/incorrect wiring.
        if age <= 5 and status.get("ina219") and finite and 2.7 <= voltage <= 5.5 and abs(current) <= 1.0:
            if abs(power - voltage * current) <= max(0.03, abs(power) * 0.15):
                result["electrical_quality"] = "qualified"
                result["electrical_state"] = "low_voltage" if voltage < min_supply_v else "high_current" if abs(current) >= max_current_a else "nominal"
                result["edge_execution_allowed"] = voltage >= min_supply_v and abs(current) < max_current_a
                result["minimum_supply_v"] = min_supply_v
                result["maximum_current_a"] = max_current_a
                result["voltage_v"] = round(voltage, 4)
                result["current_a"] = round(current, 6)
                result["power_w"] = round(power, 6)
                result["electrical_quality_reason"] = None
        result["sensor_roles"] = {
            "ina219": "qualified supply voltage/current constrain ESP32 execution availability; power must agree with voltage × current; cumulative energy is logged, not treated as remaining battery",
            "dht22": "ambient conditions only; not processor temperature",
            "mpu6500_sw420": "physical motion/vibration event only; not cyberattack evidence",
        }
        return result

    def select(self, risk: float, resource_stress: float, *, remote_available: bool = False, energy_pct: float | None = None, sensor_snapshot: dict[str, Any] | None = None, sensor_reference_time: float | None = None, network_context: dict[str, Any] | None = None, confidence: float | None = None) -> dict[str, Any]:
        risk = CrossLayerRiskEngine._unit(risk, "risk")
        stress = CrossLayerRiskEngine._unit(resource_stress, "resource_stress")
        recall_floor = 0.95 if risk >= 0.70 else 0.90 if risk >= 0.35 else 0.0
        eligible = []
        rejected = []
        network = network_context or {}
        packet_rate = network.get("packet_rate_pps")
        edge_capacity = network.get("edge_capacity_pps")
        network_load = 0.0
        if packet_rate is not None and edge_capacity is not None and float(edge_capacity) > 0:
            network_load = min(1.0, max(0.0, float(packet_rate) / float(edge_capacity)))
        for name in self.ORDER:
            profile = self.profiles.get(name, {})
            recall = float(profile.get("macro_recall", 0.0))
            cost = float(profile.get("estimated_energy_uj", float("inf")))
            if recall + 1e-12 < recall_floor:
                rejected.append({"action": name, "reason": f"Validation macro recall {recall:.3f} is below security floor {recall_floor:.3f}."})
            else:
                # Host pressure raises estimated cost, never lowers the security floor.
                eligible.append((cost * (1.0 + 0.5 * stress), name, recall))

        best_local = max(self.ORDER, key=lambda name: float(self.profiles.get(name, {}).get("macro_recall", 0.0)))
        if eligible:
            eligible.sort()
            _, chosen, recall = eligible[0]
            reason = f"Selected the lowest estimated-cost local model meeting validation macro-recall floor {recall_floor:.3f}."
            fallback_action = chosen
        elif remote_available:
            chosen, recall = "remote", 0.0
            reason = f"No local model meets validation macro-recall floor {recall_floor:.3f}; request trusted remote inference."
            fallback_action = best_local
        else:
            chosen = best_local
            recall = float(self.profiles.get(best_local, {}).get("macro_recall", 0.0))
            reason = "No local tier meets the security floor and no trusted node is available; use the strongest local tier and raise a coverage-degraded alert."
            fallback_action = chosen

        constrained = stress >= 0.90 or (energy_pct is not None and float(energy_pct) < 15.0)
        if chosen in self.ORDER and constrained and remote_available:
            fallback_action = chosen
            chosen = "remote"
            reason = "High local resource stress or critically low reported battery triggered trusted remote inference; local fallback is retained if the peer is unavailable."

        # Prevent noisy switches when the old model remains eligible and has nearly equal cost.
        if chosen in self.ORDER and self.current_model in self.ORDER and chosen != self.current_model:
            current = self.profiles.get(self.current_model, {})
            if float(current.get("macro_recall", 0.0)) >= recall_floor:
                chosen_cost = float(self.profiles[chosen].get("estimated_energy_uj", float("inf")))
                current_cost = float(current.get("estimated_energy_uj", float("inf")))
                if chosen_cost >= current_cost * (1.0 - self.hysteresis_margin):
                    reason += f" Kept {self.current_model} because the estimated cost gain is within the hysteresis margin."
                    chosen = self.current_model
                    recall = float(current.get("macro_recall", 0.0))

        previous = self.current_model
        if chosen in self.ORDER:
            self.current_model = chosen
        physical = self.sensor_context(sensor_snapshot, now=sensor_reference_time, min_supply_v=self.min_esp32_supply_v, max_current_a=self.max_esp32_current_a)
        execution_target = "trusted_remote" if chosen == "remote" else ("esp32_usb" if chosen == "lightweight" else "local_host")
        execution_reason = None
        if chosen == "lightweight" and physical["edge_execution_allowed"] is False:
            if remote_available:
                fallback_action = chosen
                chosen = "remote"
                execution_target = "trusted_remote"
                reason += " Qualified INA219 electrical limits block ESP32 execution; inference was offloaded to a trusted node."
            else:
                execution_target = "local_host"
                reason += " Qualified INA219 electrical limits block ESP32 execution; execute the selected lightweight tier on the laptop."
            execution_reason = f"Qualified INA219 reading violates the configured supply floor ({self.min_esp32_supply_v:.2f} V) or current limit ({self.max_esp32_current_a:.2f} A)."
        elif chosen == "lightweight" and network_load >= 1.0:
            execution_target = "local_host"
            execution_reason = "Observed packet rate reached or exceeded measured ESP32 USB service capacity; use the laptop to avoid a growing edge queue."
        decision = {
            "action": chosen,
            "previous_action": previous,
            "changed": chosen != previous,
            "risk": round(risk, 4),
            "resource_stress": round(stress, 4),
            "required_macro_recall": recall_floor,
            "validation_macro_recall": round(recall, 4),
            "security_coverage_degraded": chosen != "remote" and recall + 1e-12 < recall_floor,
            "fallback_action": fallback_action if chosen == "remote" else None,
            "energy_pct": energy_pct,
            "rejected_actions": rejected,
            "reason": reason,
            "execution_target": execution_target,
            "execution_reason": execution_reason,
            "sensor_context": physical,
            "network_context": network_context or {"source": "not_supplied"},
            "network_load_vs_edge_capacity": round(network_load, 4),
            "prediction_confidence": confidence,
            "influence_summary": {
                "network_and_model_evidence_select_tier": True,
                "host_resources_scale_cost": round(stress, 4),
                "observed_packet_load_scales_cost": round(network_load, 4),
                "qualified_electrical_limit_blocks_esp32_execution": physical["edge_execution_allowed"] is False,
                "measured_edge_capacity_exceeded": network_load >= 1.0,
                "ambient_and_motion_are_context_only": True,
            },
        }
        self.history.insert(0, decision.copy())
        self.history[:] = self.history[:500]
        return decision

    def schedule_federated(self, resources: dict[str, Any], *, sensor_snapshot: dict[str, Any] | None = None, network_stats: list[dict[str, Any]] | None = None, security_context: dict[str, Any] | None = None, client_topology: str = "simulated_host") -> dict[str, Any]:
        """Preflight FL against the actual training host; use INA only for physical clients."""
        physical = self.sensor_context(sensor_snapshot, min_supply_v=self.min_esp32_supply_v, max_current_a=self.max_esp32_current_a)
        cpu = resources.get("cpu_percent")
        memory = resources.get("memory_percent")
        stress = max(float(cpu or 0), float(memory or 0)) / 100.0
        reasons = []
        eligible = True
        if stress >= 0.95:
            eligible = False
            reasons.append("host CPU or memory pressure is at least 95%")
        security = {"quality": "not_applicable" if client_topology == "simulated_host" else "unavailable"}
        if client_topology != "simulated_host":
            security = dict(security_context or {})
            try:
                security_age = time.time() - float(security["timestamp"])
                security_fresh = 0 <= security_age <= 30
            except (KeyError, TypeError, ValueError):
                security_age, security_fresh = None, False
            if not security_fresh or security.get("source") != "authorized_live_packet_capture":
                eligible = False
                reasons.append("a fresh live packet risk and confidence assessment is required before scheduling a live-data FL client")
                security = {**security, "quality": "unavailable_or_stale", "age_seconds": security_age}
            else:
                max_risk = float(os.getenv("ADAPTIVEML_FL_MAX_SECURITY_RISK", "0.70"))
                min_confidence = float(os.getenv("ADAPTIVEML_FL_MIN_PREDICTION_CONFIDENCE", "0.50"))
                security.update({"quality": "fresh", "age_seconds": round(security_age, 2),
                                 "limits": {"max_risk": max_risk, "min_confidence": min_confidence}})
                if float(security.get("risk", 1.0)) >= max_risk or float(security.get("confidence", 0.0)) < min_confidence:
                    eligible = False
                    reasons.append("live threat severity is high or packet-model confidence is too low to trust a local FL update")
        if client_topology == "physical_esp32":
            if physical["electrical_quality"] != "qualified":
                eligible = False
                reasons.append("a fresh, qualified INA219 supply reading is required before scheduling a physical ESP32 client")
            elif physical["edge_execution_allowed"] is False:
                eligible = False
                reasons.append("qualified INA219 supply/current is outside the configured ESP32 execution limits")
        link_context: dict[str, Any] = {"quality": "not_applicable" if client_topology == "simulated_host" else "unavailable"}
        if client_topology == "networked":
            latest = max(network_stats or [], key=lambda item: str(item.get("timestamp", "")), default=None)
            if latest:
                try:
                    stamp = datetime.fromisoformat(str(latest["timestamp"])).timestamp()
                    fresh = 0 <= time.time() - stamp <= 30
                except (KeyError, TypeError, ValueError):
                    fresh = False
                link_context = {key: latest.get(key) for key in ("source", "round_trip_latency_ms", "packet_loss_pct", "bandwidth_utilization_pct", "timestamp")}
                link_context["age_seconds"] = round(max(0.0, time.time() - stamp), 2) if fresh else None
                required = ("round_trip_latency_ms", "packet_loss_pct", "bandwidth_utilization_pct")
                if not fresh or latest.get("source") not in ("capture", "replay") or any(latest.get(key) is None for key in required):
                    eligible = False
                    reasons.append("fresh measured RTT, packet loss, and link utilization are required for networked clients")
                else:
                    max_rtt = float(os.getenv("ADAPTIVEML_FL_MAX_RTT_MS", "250"))
                    max_loss = float(os.getenv("ADAPTIVEML_FL_MAX_PACKET_LOSS_PCT", "5"))
                    max_util = float(os.getenv("ADAPTIVEML_FL_MAX_BANDWIDTH_UTILIZATION_PCT", "90"))
                    if float(latest["round_trip_latency_ms"]) > max_rtt or float(latest["packet_loss_pct"]) > max_loss or float(latest["bandwidth_utilization_pct"]) > max_util:
                        eligible = False
                        reasons.append("measured link quality exceeds a configured FL scheduling limit")
                    link_context["limits"] = {"round_trip_latency_ms": max_rtt, "packet_loss_pct": max_loss, "bandwidth_utilization_pct": max_util}
                    link_context["quality"] = "eligible" if eligible else "degraded"
            else:
                eligible = False
                reasons.append("fresh measured RTT, packet loss, and link utilization are required for networked clients")
        if not reasons:
            reasons.append("host resources permit another training run")
        return {
            "schedule": eligible, "client_topology": client_topology,
            "host_resource_stress": round(stress, 4), "network": link_context,
            "sensor_context": physical,
            "security_context": security,
            "reason": "; ".join(reasons),
            "note": "Security risk/confidence govern per-packet inference. FL scheduling uses host resources; qualified INA219 applies to physical ESP32 clients, and fresh link quality applies to networked clients. Simulated-host FL has no client network to gate.",
        }
