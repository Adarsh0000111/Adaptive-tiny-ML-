from src.energy.provider_base import EnergyProvider

class ESP32INA219EnergyProvider(EnergyProvider):
    """
    Hardware Ready Stub Interface for ESP32 with INA219 I2C Current/Voltage Sensor.
    Will connect over PySerial / I2C when physical ESP32 hardware is integrated.
    """
    def __init__(self, port="COM3", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.connected = False
        
    def connect_sensor(self):
        # Stub for hardware serial connection
        print(f"[Hardware Interface] Initializing I2C INA219 sensor on {self.port} @ {self.baudrate} baud...")
        self.connected = True

    def get_battery_percentage(self) -> float:
        if not self.connected:
            return 85.0 # Fallback reading
        # Read Bus Voltage from INA219 over I2C
        v_bus = 3.85 # Simulated hardware voltage
        v_min, v_max = 3.0, 4.2
        pct = max(0.0, min(100.0, ((v_bus - v_min) / (v_max - v_min)) * 100.0))
        return pct

    def get_battery_capacity_mah(self) -> float:
        return 1200.0

    def consume_inference_energy(self, model_level: str) -> float:
        # Measures shunt current delta * bus voltage * delta_t during hardware execution
        return 15.0

    def consume_communication_energy(self, payload_kb: float) -> float:
        return payload_kb * 40.0

    def consume_federated_training_energy(self, num_epochs: int) -> float:
        return num_epochs * 12000.0

    def update_idle_drain(self, duration_s: float) -> float:
        return duration_s * 15.0 * 1000.0
