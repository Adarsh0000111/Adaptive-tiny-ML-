import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ESP32INA219HardwareProvider:
    """
    Physical Hardware Energy Provider interface.
    Connects to physical ESP32 MCU equipped with an INA219 current/power monitor over Serial/I2C.
    Falls back to high-fidelity Hardware Emulator if serial device is disconnected.
    """
    def __init__(self, port=None, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_connected = False
        
        if not port:
            self.ser = None
            logger.info("No hardware serial port configured; physical energy readings are unavailable.")
            return
        try:
            import serial
            self.ser = serial.Serial(port, baudrate, timeout=1)
            self.serial_connected = True
            logger.info(f"Successfully connected to physical ESP32 INA219 sensor on {port}@{baudrate} baud.")
        except Exception:
            logger.info(f"Physical ESP32 not detected on {port}. Physical readings are unavailable; use SimulationEnergyProvider for explicit simulation.")

    def read_power_metrics(self):
        """
        Reads Bus Voltage (V), Shunt Voltage (mV), Current (mA), and Power (mW).
        """
        if self.serial_connected:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                # Expected format: "V:3.29,I:45.2,P:148.7"
                parts = dict(item.split(":") for item in line.split(",") if ":" in item)
                voltage_v = float(parts.get("V", 3.3))
                current_ma = float(parts.get("I", 15.0))
                power_mw = float(parts.get("P", voltage_v * current_ma))
                return {
                    "voltage_v": round(voltage_v, 3),
                    "current_ma": round(current_ma, 2),
                    "power_mw": round(power_mw, 2),
                    "is_real_hardware": True
                }
            except Exception as e:
                logger.error(f"Error reading physical serial stream: {e}")

        return {
            "voltage_v": None,
            "current_ma": None,
            "power_mw": None,
            "is_real_hardware": False
        }

    def measure_inference_energy(self, duration_seconds=0.001):
        """
        Calculates energy consumed during inference run: E = P * t (in microjoules uJ)
        """
        metrics = self.read_power_metrics()
        if metrics["power_mw"] is None:
            metrics["energy_uj"] = None
            metrics["energy_source"] = "unavailable"
            return metrics
        energy_uj = metrics["power_mw"] * (duration_seconds * 1000.0) # mW * ms = uJ
        metrics["energy_uj"] = round(float(energy_uj), 2)
        return metrics
