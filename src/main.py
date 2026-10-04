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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    logger.info("=========================================================================")
    logger.info("STARTING COMPLETE RESEARCH PIPELINE FOR ADAPTIVE TINYML IIOT IDS")
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
    
    # PHASE 5: TinyML Optimization
    logger.info("[PHASE 5] TinyML INT8 Optimization & ESP32 Code Export...")
    opt = TinyMLOptimizer()
    tinyml_results = opt.optimize_and_benchmark()
    
    # PHASE 6, 8, 10: Federated Learning & Client Selection
    logger.info("[PHASE 6, 8, 10] Federated Learning & Energy-Aware Client Selection...")
    fed_server = FederatedServer()
    fl_results = fed_server.run_comparison_experiment()
    
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
        "tinyml": tinyml_results,
        "federated": fl_results,
        "baselines": baseline_results
    }

if __name__ == "__main__":
    run_complete_pipeline()
