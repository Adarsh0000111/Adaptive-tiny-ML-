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
from src.evaluation.drift import ConceptDriftMonitor
from src.data.live_sniffer import RealTimeNetworkSniffer
from src.energy.hardware_interface import ESP32INA219HardwareProvider

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
    logger.info("[PHASE 5] Fixed-point tree export & ESP32 code generation...")
    opt = TinyMLOptimizer()
    tinyml_results = opt.optimize_and_benchmark()
    
    # PHASE 6, 8, 10: Federated Learning & Client Selection
    logger.info("[PHASE 6, 8, 10] Federated Learning & Energy-Aware Client Selection...")
    fed_server = FederatedServer()
    fl_results = fed_server.run_comparison_experiment()
    
    # Live capture is opt-in and requires explicit authorization for the chosen network.
    live_df = None
    if os.getenv("ADAPTIVEML_ENABLE_LIVE_CAPTURE", "false").lower() == "true":
        interface = os.getenv("ADAPTIVEML_CAPTURE_INTERFACE") or None
        logger.warning("Live packet capture explicitly enabled on interface %s; capture only authorized networks.", interface or "default")
        try:
            live_df = RealTimeNetworkSniffer(interface=interface).capture_live_stream(max_packets=50, timeout=2)
        except (RuntimeError, OSError) as exc:
            logger.warning("Live capture unavailable: %s", exc)
    else:
        logger.info("Live packet capture skipped; set ADAPTIVEML_ENABLE_LIVE_CAPTURE=true only for an authorized interface.")

    hardware_port = os.getenv("ADAPTIVEML_HARDWARE_PORT")
    if hardware_port:
        hardware_power = ESP32INA219HardwareProvider(port=hardware_port)
        logger.info("Physical power measurement status: %s", hardware_power.read_power_metrics())
    else:
        logger.info("Physical power measurement skipped; set ADAPTIVEML_HARDWARE_PORT after connecting an ESP32 sensor.")

    # ADVANCED FEATURE D: Concept Drift Monitor Test
    logger.info("[ADVANCED FEATURE D] Testing Concept Drift Monitor...")
    drift_mon = ConceptDriftMonitor(num_features=data_dict["X_train"].shape[1])
    sample_feat = data_dict["X_test"][0]
    drift_status = drift_mon.monitor_event(sample_feat, prediction_error=0)

    logger.info("External alert dispatch was not triggered with fabricated sample values.")

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
        "dp_privacy": {"implemented_and_validated": False, "reason": "The current clipping/noise prototype is not connected to a compatible federated aggregation path and has no validated privacy accountant."},
        "tinyml": tinyml_results,
        "federated": fl_results,
        "baselines": baseline_results
    }

if __name__ == "__main__":
    run_complete_pipeline()
