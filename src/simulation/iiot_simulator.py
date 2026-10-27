import os
import json
import time
import logging
import joblib
import numpy as np

from src.config import load_config
from src.simulation.traffic_provider import DatasetTrafficProvider
from src.energy.simulated_provider import SimulationEnergyProvider
from src.adaptive.threat_scorer import ThreatScorer
from src.adaptive.utility_engine import AdaptiveModelSelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RealtimeIIoTSimulator:
    """
    Real-Time IIoT Intrusion Simulator with Adaptive Energy-Aware Model Switching.
    """
    def __init__(self, config_path="config.yaml", initial_battery_pct=100.0):
        self.config = load_config(config_path)
        self.traffic_provider = DatasetTrafficProvider(config_path)
        self.energy_provider = SimulationEnergyProvider(initial_battery_pct=initial_battery_pct, config_path=config_path)
        self.threat_scorer = ThreatScorer(config_path=config_path)
        self.selector = AdaptiveModelSelector(config_path=config_path)
        
        # Load trained models
        models_dir = self.config["paths"]["models_dir"]
        self.models = {
            "lightweight": joblib.load(os.path.join(models_dir, "lightweight", "model.joblib")),
            "medium": joblib.load(os.path.join(models_dir, "medium", "model.joblib")),
            "heavy": joblib.load(os.path.join(models_dir, "heavy", "model.joblib"))
        }
        
        # Load class names
        processed_data = np.load(self.config["paths"]["processed_data"], allow_pickle=True)
        self.class_names = processed_data["class_names"].tolist()
        
        self.previous_model = "lightweight"
        self.logs = []

    def process_next_event(self) -> dict:
        # 1. Fetch incoming event
        event = self.traffic_provider.get_next_event()
        event_id = event["event_id"]
        x_features = event["features"].reshape(1, -1)
        true_class = event["true_class"]
        
        # 2. Check current battery state
        current_battery = self.energy_provider.get_battery_percentage()
        
        # 3. Fast preliminary threat estimate using lightweight model
        start_t = time.perf_counter()
        probe_pred_idx = self.models["lightweight"].predict(x_features)[0]
        probe_class = self.class_names[probe_pred_idx]
        probe_proba = self.models["lightweight"].predict_proba(x_features)[0]
        probe_conf = float(np.max(probe_proba))
        
        # Calculate threat score
        threat_dict = self.threat_scorer.calculate_threat(probe_class, probe_conf)
        
        # 4. Select optimal model via Adaptive Engine
        selection_dict = self.selector.select_optimal_model(threat_dict, current_battery)
        selected_model_name = selection_dict["selected_model"]
        reason = selection_dict["reason"]
        
        # 5. Execute inference on selected model
        start_inf = time.perf_counter()
        final_pred_idx = self.models[selected_model_name].predict(x_features)[0]
        final_class = self.class_names[final_pred_idx]
        final_proba = self.models[selected_model_name].predict_proba(x_features)[0]
        final_conf = float(np.max(final_proba))
        inf_latency_ms = (time.perf_counter() - start_inf) * 1000.0
        
        # Recalculate threat with final model prediction
        final_threat_dict = self.threat_scorer.calculate_threat(final_class, final_conf)
        
        # 6. Update energy consumption
        energy_used_uj = self.energy_provider.consume_inference_energy(selected_model_name)
        self.energy_provider.update_idle_drain(duration_s=0.2)
        new_battery = self.energy_provider.get_battery_percentage()
        
        # Detect model switch event
        switch_occurred = (selected_model_name != self.previous_model)
        switch_str = f"{self.previous_model.upper()} -> {selected_model_name.upper()}" if switch_occurred else "UNCHANGED"
        self.previous_model = selected_model_name
        
        log_entry = {
            "event_id": event_id,
            "timestamp": time.strftime("%H:%M:%S"),
            "true_class": true_class,
            "predicted_class": final_class,
            "confidence": round(final_conf, 3),
            "threat_score": final_threat_dict["threat_score"],
            "severity": final_threat_dict["severity"],
            "battery_pct": round(new_battery, 2),
            "selected_model": selected_model_name.upper(),
            "model_switch": switch_str,
            "reason": reason,
            "inference_latency_ms": round(inf_latency_ms, 3),
            "energy_consumed_uj": round(energy_used_uj, 2)
        }
        
        self.logs.append(log_entry)
        return log_entry

    def run_simulation_batch(self, count=100):
        logger.info(f"Running Real-Time IIoT Simulation Batch ({count} events)...")
        results = []
        for _ in range(count):
            entry = self.process_next_event()
            results.append(entry)
            
        out_path = os.path.join(self.config["paths"]["experiments_dir"], "simulation_results.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Simulation completed. Logs saved to {out_path}")
        return results

if __name__ == "__main__":
    sim = RealtimeIIoTSimulator()
    batch = sim.run_simulation_batch(count=20)
    print("\nSAMPLE REAL-TIME EVENT LOG:")
    print(json.dumps(batch[0], indent=2))
