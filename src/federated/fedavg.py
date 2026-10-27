import os
import json
import logging
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from src.config import load_config
from src.federated.non_iid import create_non_iid_partitions
from src.federated.client import IIoTEdgeClient
from src.federated.energy_selection import EnergyAwareClientSelector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FederatedServer:
    """
    Central Federated Learning Orchestrator.
    Manages global model rounds, client communication, FedAvg aggregation,
    and direct comparison between Standard FL vs Energy-Aware FL.
    """
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.fl_cfg = self.config["federated"]
        self.num_clients = self.fl_cfg["num_clients"]
        self.num_rounds = self.fl_cfg["num_rounds"]
        self.local_epochs = self.fl_cfg["local_epochs"]
        self.alpha = self.fl_cfg["dirichlet_alpha"]
        
        self.results_dir = self.config["paths"]["results_dir"]
        self.tables_dir = self.config["paths"]["tables_dir"]
        self.figures_dir = self.config["paths"]["figures_dir"]
        
        self.selector = EnergyAwareClientSelector(config_path)

    def run_federated_simulation(self, strategy="energy_aware"):
        logger.info(f"--- Starting Federated Learning Simulation (Strategy: {strategy.upper()}) ---")
        
        # Load processed data
        data = np.load(self.config["paths"]["processed_data"], allow_pickle=True)
        X_train = data["X_train"]
        y_train = data["y_train_multi"]
        X_test = data["X_test"]
        y_test = data["y_test_multi"]
        
        # Split train data into non-IID partitions across simulated clients
        partitions = create_non_iid_partitions(X_train, y_train, num_clients=self.num_clients, alpha=self.alpha)
        
        # Initialize IIoT Edge Clients with varying initial battery levels (95%, 85%, 18%, 70%, 12%)
        initial_batteries = [95.0, 85.0, 18.0, 70.0, 12.0]
        clients = [
            IIoTEdgeClient(i, partitions[i][0], partitions[i][1], initial_battery_pct=initial_batteries[i])
            for i in range(self.num_clients)
        ]
        
        global_model = RandomForestClassifier(n_estimators=15, max_depth=8, random_state=42, n_jobs=-1)
        # Initial fit on a small seed sample to initialize internal arrays
        global_model.fit(X_train[:100], y_train[:100])
        
        round_history = []
        total_comm_overhead_kb = 0.0
        
        for r in range(1, self.num_rounds + 1):
            # Select participating clients
            selected_clients = self.selector.select_clients(clients, fraction=0.8, strategy=strategy)
            selected_ids = [c.client_id for c in selected_clients]
            
            # Local client training
            client_updates = []
            for client in selected_clients:
                update = client.train_local_model(global_model, num_epochs=self.local_epochs)
                client_updates.append(update)
                total_comm_overhead_kb += 48.0 # Model update payload in KB
                
            # FedAvg Aggregation (Weighted by client dataset size)
            total_samples = sum([u["data_size"] for u in client_updates])
            
            # Aggregate decision trees from selected clients
            all_trees = []
            for u in client_updates:
                c_model = u["model"]
                weight = u["data_size"] / float(total_samples)
                n_trees_to_take = max(2, int(round(weight * 15)))
                all_trees.extend(c_model.estimators_[:n_trees_to_take])
                
            # Update global model estimators
            global_model.estimators_ = all_trees[:15]
            global_model.n_estimators = len(global_model.estimators_)
            
            # Evaluate global model performance on test set
            y_pred = global_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            
            avg_battery = np.mean([c.get_battery_level() for c in clients])
            min_battery = min([c.get_battery_level() for c in clients])
            
            logger.info(f"Round {r:02d}/{self.num_rounds} | Participating Clients: {selected_ids} | Acc: {acc:.4f} | F1: {f1:.4f} | Avg Battery: {avg_battery:.1f}%")
            
            round_history.append({
                "round": r,
                "participating_clients": selected_ids,
                "global_accuracy": round(float(acc), 4),
                "global_f1_macro": round(float(f1), 4),
                "avg_battery_pct": round(float(avg_battery), 2),
                "min_battery_pct": round(float(min_battery), 2),
                "cumulative_comm_kb": round(float(total_comm_overhead_kb), 1)
            })
            
        summary = {
            "strategy": strategy,
            "final_global_accuracy": round(float(round_history[-1]["global_accuracy"]), 4),
            "final_global_f1_macro": round(float(round_history[-1]["global_f1_macro"]), 4),
            "total_comm_overhead_kb": round(float(total_comm_overhead_kb), 1),
            "final_avg_battery_pct": round(float(round_history[-1]["avg_battery_pct"]), 2),
            "final_min_battery_pct": round(float(round_history[-1]["min_battery_pct"]), 2),
            "history": round_history
        }
        
        # Save global FL model
        save_path = os.path.join(self.config["paths"]["models_dir"], "federated", f"global_model_{strategy}.joblib")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(global_model, save_path)
        
        return summary

    def run_comparison_experiment(self):
        logger.info("Executing Centralized vs Standard FL vs Energy-Aware FL Comparison Experiment...")
        
        res_std = self.run_federated_simulation(strategy="standard")
        res_energy = self.run_federated_simulation(strategy="energy_aware")
        
        # Save comparison results
        comp_summary = {
            "standard_fl": res_std,
            "energy_aware_fl": res_energy,
            "energy_savings_pct": round(float((res_energy["final_avg_battery_pct"] - res_std["final_avg_battery_pct"]) / res_std["final_avg_battery_pct"] * 100.0), 2)
        }
        
        table_path = os.path.join(self.tables_dir, "federated_comparison.json")
        with open(table_path, "w") as f:
            json.dump(comp_summary, f, indent=2)
            
        logger.info(f"Federated Learning comparison summary saved to {table_path}")
        return comp_summary

if __name__ == "__main__":
    server = FederatedServer()
    comp = server.run_comparison_experiment()
    print("\nFEDERATED LEARNING EXPERIMENT SUMMARY:")
    print(json.dumps(comp, indent=2))
