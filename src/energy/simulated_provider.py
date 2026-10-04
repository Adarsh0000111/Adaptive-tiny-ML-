import yaml
from src.energy.provider_base import EnergyProvider

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class SimulationEnergyProvider(EnergyProvider):
    """
    Simulated Energy Provider tracking battery state, inference costs,
    communication costs, and FL training energy.
    Note: Explicitly marks readings as ESTIMATED SIMULATION data.
    """
    def __init__(self, initial_battery_pct=100.0, config_path="config.yaml"):
        config = load_config(config_path)
        e_cfg = config["energy_simulation"]
        
        self.battery_pct = float(initial_battery_pct)
        self.capacity_mah = float(e_cfg.get("battery_capacity_mAh", 1200.0))
        self.voltage_v = float(e_cfg.get("voltage_v", 3.3))
        self.idle_power_mw = float(e_cfg.get("idle_power_mW", 15.0))
        self.comm_energy_uj_per_kb = float(e_cfg.get("comm_energy_uJ_per_kb", 45.0))
        self.fed_train_energy_uj_per_epoch = float(e_cfg.get("fed_train_energy_uJ_per_epoch", 12500.0))
        
        # Energy per model inference (microjoules)
        self.model_energy_uj = {
            "lightweight": float(config["models"]["lightweight"]["est_energy_uJ_per_inf"]),
            "medium": float(config["models"]["medium"]["est_energy_uJ_per_inf"]),
            "heavy": float(config["models"]["heavy"]["est_energy_uJ_per_inf"])
        }
        
        # Total energy capacity in Joules (1 mAh * 3.3V * 3.6 = Joules)
        self.total_joules = (self.capacity_mah / 1000.0) * self.voltage_v * 3600.0
        self.current_joules = (self.battery_pct / 100.0) * self.total_joules
        self.total_energy_consumed_uj = 0.0

    def get_battery_percentage(self) -> float:
        return max(0.0, min(100.0, self.battery_pct))

    def get_battery_capacity_mah(self) -> float:
        return self.capacity_mah

    def _subtract_energy_uj(self, uj):
        joules = uj / 1e6
        self.current_joules = max(0.0, self.current_joules - joules)
        self.battery_pct = (self.current_joules / self.total_joules) * 100.0
        self.total_energy_consumed_uj += uj
        return uj

    def consume_inference_energy(self, model_level: str) -> float:
        uj = self.model_energy_uj.get(model_level.lower(), 10.0)
        return self._subtract_energy_uj(uj)

    def consume_communication_energy(self, payload_kb: float) -> float:
        uj = payload_kb * self.comm_energy_uj_per_kb
        return self._subtract_energy_uj(uj)

    def consume_federated_training_energy(self, num_epochs: int) -> float:
        uj = num_epochs * self.fed_train_energy_uj_per_epoch
        return self._subtract_energy_uj(uj)

    def update_idle_drain(self, duration_s: float) -> float:
        # Power (mW) * time (s) = mJ -> uJ (* 1000)
        uj = self.idle_power_mw * duration_s * 1000.0
        return self._subtract_energy_uj(uj)
