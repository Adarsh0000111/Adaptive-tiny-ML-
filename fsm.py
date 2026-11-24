"""Configurable FSM for demonstration model selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class SwitchingPolicy:
    sufficient_energy_pct: float = 35.0


class AdaptiveModelFSM:
    """Select a model from threat and energy inputs and retain transition history."""

    def __init__(self, policy: SwitchingPolicy | None = None):
        self.policy = policy or SwitchingPolicy()
        self.state = "LOW_THREAT"
        self.model = "decision_tree"
        self.history: list[dict] = []

    def select(self, threat: str, energy_pct: float, energy_source: str = "simulated") -> dict:
        threat = threat.strip().lower()
        if threat not in {"low", "medium", "high"}:
            raise ValueError("Threat must be low, medium, or high")
        energy_pct = float(energy_pct)
        if not 0 <= energy_pct <= 100:
            raise ValueError("Energy must be between 0 and 100 percent")
        sufficient = energy_pct >= self.policy.sufficient_energy_pct
        if threat == "low":
            state, model, reason = "LOW_THREAT", "decision_tree", "Low threat uses the lightweight Decision Tree."
        elif threat == "medium" and sufficient:
            state, model, reason = "MEDIUM_THREAT_ENOUGH_ENERGY", "random_forest", "Medium threat and sufficient energy select the Random Forest."
        elif threat == "high" and sufficient:
            state, model, reason = "HIGH_THREAT_ENOUGH_ENERGY", "cnn_1d", "High threat and sufficient energy select the 1D CNN."
        else:
            state, model, reason = f"{threat.upper()}_THREAT_LOW_ENERGY", "decision_tree", f"{threat.title()} threat with energy below {self.policy.sufficient_energy_pct:g}% uses the Decision Tree fallback."

        previous = self.model
        self.state = state
        self.model = model
        if previous != model:
            self.history.insert(0, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "from_model": previous,
                "to_model": model,
                "state": state,
                "reason": reason,
                "energy_source": energy_source,
            })
            self.history = self.history[:100]
        return {"state": state, "model": model, "reason": reason, "switched": previous != model}
