import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class EnergyAwareClientSelector:
    """
    Energy-Aware Federated Client Selector.
    Filters low-power edge nodes (< threshold battery) to prevent power exhaustion,
    and selects top optimal clients per FL communication round.
    """
    def __init__(self, config_path="config.yaml"):
        config = load_config(config_path)
        cs_cfg = config["federated"]["client_selection"]
        
        self.min_battery_threshold = float(cs_cfg.get("min_battery_threshold", 20.0))
        self.w_battery = float(cs_cfg.get("battery_weight", 0.6))
        self.w_data = float(cs_cfg.get("data_size_weight", 0.4))

    def select_clients(self, clients, fraction=0.8, strategy="energy_aware"):
        num_target = max(1, int(round(len(clients) * fraction)))
        
        if strategy == "standard":
            # Standard FedAvg: random or all participating clients regardless of battery
            indices = list(range(len(clients)))
            np_random_selected = np.random.choice(indices, size=min(num_target, len(clients)), replace=False)
            return [clients[i] for i in np_random_selected]
            
        # Energy-Aware Strategy
        eligible_clients = []
        for client in clients:
            batt = client.get_battery_level()
            if batt >= self.min_battery_threshold:
                eligible_clients.append(client)
                
        if len(eligible_clients) == 0:
            # Fallback to highest battery client if all are below threshold
            eligible_clients = sorted(clients, key=lambda c: c.get_battery_level(), reverse=True)[:1]
            
        max_data = max([len(c.X_local) for c in eligible_clients]) if eligible_clients else 1
        
        scored_clients = []
        for c in eligible_clients:
            norm_batt = c.get_battery_level() / 100.0
            norm_data = len(c.X_local) / float(max_data)
            score = (self.w_battery * norm_batt) + (self.w_data * norm_data)
            scored_clients.append((score, c))
            
        scored_clients.sort(key=lambda x: x[0], reverse=True)
        selected = [item[1] for item in scored_clients[:num_target]]
        return selected

import numpy as np
