import sys
import os
import json
import logging
import time

from src.data.download import acquire_dataset
from src.data.explorer import explore_dataset
from src.preprocessing.pipeline import run_preprocessing
from src.evaluation.benchmarker import run_benchmarking
from src.tinyml.optimizer import TinyMLOptimizer
from src.federated.fedavg import FederatedServer
from src.evaluation.baselines import BaselineExperimentSuite

# New Advanced Modules
from src.tinyml.distillation import KnowledgeDistiller
from src.federated.dp_fedavg import DifferentiallyPrivateFedAvg
from src.federated.robust_aggregation import RobustAggregator
from src.evaluation.drift import ConceptDriftMonitor
from src.data.live_sniffer import RealTimeNetworkSniffer
from src.energy.hardware_interface import ESP32INA219HardwareProvider
from src.utils.notifications import AlertDispatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    logger.info("=========================================================================")
    logger.info("STARTING ENHANCED ADVANCED RESEARCH PIPELINE FOR ADAPTIVE TINYML IIOT IDS")
    logger.info("=========================================================================")
    
    start_time = time.time()
    
    # PHASE 1: Dataset Acquisition & Exploration
    logger.info("[PHASE 1] Dataset Acquisition & Exploration...")
    df_raw = acquire_dataset()
    explore_summary = explore_dataset(df_raw)
    
    # PHASE 2: Data Preprocessing
    logger.info("[PHASE 2] Data Preprocessing...")
    data_dict = run_preprocessing()
    
    # PHASE 3 & 4: Model Training & Benchmarking
    logger.info("[PHASE 3 & 4] Training & Benchmarking ML Models...")
    bench_results = run_benchmarking()
    
    # ADVANCED FEATURE A: Knowledge Distillation (MLP -> Tiny Tree)
    logger.info("[ADVANCED FEATURE A] Knowledge Distillation (Heavy MLP -> Tiny Tree)...")
    from src.models.factory import get_model
    teacher_mlp = get_model("heavy")
    teacher_mlp.fit(data_dict["X_train"], data_dict["y_train_multi"])
    distiller = KnowledgeDistiller(temperature=2.5, student_max_depth=5)
    distill_results = distiller.distill(
        teacher_mlp.model, 
        data_dict["X_train"], data_dict["y_train_multi"], 
        data_dict["X_val"], data_dict["y_val_multi"]
    )
    distiller.save_student("models/tinyml/distilled_student_tree.joblib")
    
    # PHASE 5: TinyML Optimization
    logger.info("[PHASE 5] TinyML INT8 Optimization & ESP32 Code Export...")
    opt = TinyMLOptimizer()
    tinyml_results = opt.optimize_and_benchmark()
    
    # PHASE 6, 8, 10: Federated Learning & Client Selection
    logger.info("[PHASE 6, 8, 10] Federated Learning & Energy-Aware Client Selection...")
    fed_server = FederatedServer()
    fl_results = fed_server.run_comparison_experiment()
    
    # ADVANCED FEATURE B: Differential Privacy & Byzantine Robust FL Simulation
    logger.info("[ADVANCED FEATURE B] Differential Privacy & Byzantine-Robust Aggregation Testing...")
    dp_fed = DifferentiallyPrivateFedAvg(clipping_threshold=1.0, noise_multiplier=0.1)
    eps, delta = dp_fed.estimate_privacy_budget(total_rounds=10, sample_rate=0.8)
    logger.info(f"Estimated Privacy Budget: epsilon = {eps}, delta = {delta}")
    
    robust_agg = RobustAggregator()

    # ADVANCED FEATURE C: Live Network Sniffer & INA219 Power Reading Test
    logger.info("[ADVANCED FEATURE C] Testing Live Network Sniffer & INA219 Sensor Integration...")
    sniffer = RealTimeNetworkSniffer()
    live_df = sniffer.capture_live_stream(max_packets=50, timeout=2)
    
    hardware_power = ESP32INA219HardwareProvider()
    power_reading = hardware_power.read_power_metrics()
    logger.info(f"Hardware/Emulated INA219 Power Reading: {power_reading}")

    # ADVANCED FEATURE D: Concept Drift Monitor Test
    logger.info("[ADVANCED FEATURE D] Testing Concept Drift Monitor...")
    drift_mon = ConceptDriftMonitor(num_features=data_dict["X_train"].shape[1])
    sample_feat = data_dict["X_test"][0]
    drift_status = drift_mon.monitor_event(sample_feat, prediction_error=0)

    # ADVANCED FEATURE E: Alert Notifications System
    logger.info("[ADVANCED FEATURE E] Security Alert Notification Dispatcher Ready.")
    dispatcher = AlertDispatcher()
    dispatcher.dispatch_threat_alert("DDoS_Flood", 90, "Heavy MLP", 0.985, 85.0)

    # PHASE 7, 9, 11, 13, 16: Real-time Simulation & Baseline Experiments
    logger.info("[PHASE 7, 9, 11, 13, 16] Real-Time IIoT Simulation & Baseline Experiments...")
    suite = BaselineExperimentSuite()
    baseline_results = suite.run_all_baselines(num_events=300)
    
    elapsed_time = time.time() - start_time
    logger.info("=========================================================================")
    logger.info(f"RESEARCH PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_time:.2f} SECONDS!")
    logger.info("=========================================================================")
    
    return {
        "execution_time_seconds": round(elapsed_time, 2),
        "exploration": explore_summary,
        "benchmarking": bench_results,
        "distillation": distill_results,
        "dp_privacy": {"epsilon": eps, "delta": delta},
        "tinyml": tinyml_results,
        "federated": fl_results,
        "baselines": baseline_results
    }

if __name__ == "__main__":
    run_complete_pipeline()
