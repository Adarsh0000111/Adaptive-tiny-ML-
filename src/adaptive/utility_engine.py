import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class AdaptiveModelSelector:
    """
    Adaptive Model Selection Engine implementing multi-attribute utility theory (MAUT).
    Selects optimal ML model (Lightweight, Medium, Heavy) per event dynamically based on:
    - Threat Score & Severity
    - Device Battery Percentage
    - Model Detection F1-Score
    - Energy Consumption Cost
    - Inference Latency
    """
    def __init__(self, config_path="config.yaml"):
        config = load_config(config_path)
        a_cfg = config["adaptive_switching"]
        
        self.w_threat = float(a_cfg["utility_weights"]["threat_weight"])
        self.w_energy = float(a_cfg["utility_weights"]["energy_weight"])
        self.w_latency = float(a_cfg["utility_weights"]["latency_weight"])
        
        self.batt_critical = float(a_cfg.get("battery_critical_pct", 15.0))
        self.batt_low = float(a_cfg.get("battery_low_pct", 35.0))
        
        # Static baseline specs of models for utility calculation
        self.model_specs = {
            "lightweight": {"f1": 0.7717, "energy_uJ": 8.5, "latency_ms": 0.12},
            "medium":      {"f1": 0.9780, "energy_uJ": 62.0, "latency_ms": 1.45},
            "heavy":       {"f1": 0.9749, "energy_uJ": 340.0, "latency_ms": 8.70}
        }

    def compute_model_utility(self, model_level: str, threat_score: int, battery_pct: float) -> float:
        specs = self.model_specs[model_level]
        
        # Detection Capability scaled by current threat urgency
        threat_scale = threat_score / 100.0
        detection_cap = specs["f1"] * (0.5 + 0.5 * threat_scale)
        
        # Energy penalty weighted heavily when battery is low
        battery_urgency = max(0.1, (100.0 - battery_pct) / 100.0)
        norm_energy = (specs["energy_uJ"] / 340.0) * (1.0 + 2.0 * battery_urgency)
        
        # Latency penalty
        norm_latency = specs["latency_ms"] / 10.0
        
        # Utility Equation
        utility = (self.w_threat * detection_cap) - (self.w_energy * norm_energy) - (self.w_latency * norm_latency)
        return utility

    def select_optimal_model(self, threat_dict: dict, battery_pct: float) -> dict:
        threat_score = threat_dict["threat_score"]
        severity = threat_dict["severity"]
        
        # HARD EDGE CONSTRAINTS to prevent power starvation or security breach
        if battery_pct < self.batt_critical and severity != "CRITICAL":
            selected_model = "lightweight"
            reason = f"Critical battery ({battery_pct:.1f}% < {self.batt_critical}%). Enforcing LIGHTWEIGHT model to preserve device life."
        elif battery_pct < self.batt_low and severity in ["LOW", "MEDIUM"]:
            # Evaluate lightweight vs medium
            u_light = self.compute_model_utility("lightweight", threat_score, battery_pct)
            u_med = self.compute_model_utility("medium", threat_score, battery_pct)
            selected_model = "medium" if u_med > u_light else "lightweight"
            reason = f"Low battery ({battery_pct:.1f}%). Selected {selected_model.upper()} based on energy-security tradeoff."
        elif severity == "CRITICAL" and battery_pct > self.batt_critical:
            selected_model = "heavy" if battery_pct > 30.0 else "medium"
            reason = f"CRITICAL threat level detected ({threat_score}/100). Prioritizing maximal detection security."
        else:
            # Full Utility Maximization across all 3 candidate models
            utilities = {
                level: self.compute_model_utility(level, threat_score, battery_pct)
                for level in ["lightweight", "medium", "heavy"]
            }
            selected_model = max(utilities, key=utilities.get)
            reason = f"Utility-maximized model selected ({selected_model.upper()}) based on Threat: {severity} and Battery: {battery_pct:.1f}%."
            
        specs = self.model_specs[selected_model]
        return {
            "selected_model": selected_model,
            "reason": reason,
            "expected_energy_uJ": specs["energy_uJ"],
            "expected_latency_ms": specs["latency_ms"],
            "utility_scores": {l: round(self.compute_model_utility(l, threat_score, battery_pct), 4) for l in ["lightweight", "medium", "heavy"]}
        }
