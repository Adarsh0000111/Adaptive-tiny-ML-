from abc import ABC, abstractmethod

from src.config import load_config

class EnergyProvider(ABC):
    """
    Abstract Interface for Energy Measurement & Battery Monitoring.
    Decouples simulation from physical INA219 sensor on ESP32.
    """
    @abstractmethod
    def get_battery_percentage(self) -> float:
        pass

    @abstractmethod
    def get_battery_capacity_mah(self) -> float:
        pass

    @abstractmethod
    def consume_inference_energy(self, model_level: str) -> float:
        """Consumes energy for model inference. Returns energy consumed in microjoules (uJ)."""
        pass

    @abstractmethod
    def consume_communication_energy(self, payload_kb: float) -> float:
        """Consumes energy for wireless payload transmission. Returns uJ."""
        pass

    @abstractmethod
    def consume_federated_training_energy(self, num_epochs: int) -> float:
        """Consumes energy for local FL model training. Returns uJ."""
        pass

    @abstractmethod
    def update_idle_drain(self, duration_s: float) -> float:
        """Consumes baseline idle device power over time."""
        pass
