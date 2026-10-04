import numpy as np
import joblib
from sklearn.base import clone
from src.energy.simulated_provider import SimulationEnergyProvider

class IIoTEdgeClient:
    """
    Simulated Industrial IoT Edge Client Node.
    Maintains local non-IID dataset, local energy provider, and performs local training.
    Raw data remains 100% local to this client node.
    """
    def __init__(self, client_id, X_local, y_local, initial_battery_pct=100.0, config_path="config.yaml"):
        self.client_id = client_id
        self.X_local = X_local
        self.y_local = y_local
        self.energy_provider = SimulationEnergyProvider(initial_battery_pct=initial_battery_pct, config_path=config_path)
        self.local_model = None

    def get_battery_level(self) -> float:
        return self.energy_provider.get_battery_percentage()

    def train_local_model(self, base_model, num_epochs=3):
        """
        Trains model locally on X_local, y_local.
        Consumes local FL training energy.
        Returns trained local model parameters and local sample count.
        """
        # Drain training energy
        self.energy_provider.consume_federated_training_energy(num_epochs)
        
        # Clone base architecture
        self.local_model = clone(base_model)
        self.local_model.fit(self.X_local, self.y_local)
        
        # Drain communication energy for transmitting model update
        payload_kb = 48.0 # Model update payload size
        self.energy_provider.consume_communication_energy(payload_kb)
        
        return {
            "client_id": self.client_id,
            "model": self.local_model,
            "data_size": len(self.X_local),
            "battery_pct": self.get_battery_level()
        }
