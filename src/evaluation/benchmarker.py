import os
import json
import logging
import time
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.models.factory import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class ModelBenchmarker:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.results_dir = self.config["paths"]["results_dir"]
        self.figures_dir = self.config["paths"]["figures_dir"]
        self.tables_dir = self.config["paths"]["tables_dir"]
        self.models_dir = self.config["paths"]["models_dir"]
        
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        
    def train_and_evaluate_all(self, data_dict):
        X_train = data_dict["X_train"]
        y_train = data_dict["y_train_multi"]
        X_val = data_dict["X_val"]
        y_val = data_dict["y_val_multi"]
        X_test = data_dict["X_test"]
        y_test = data_dict["y_test_multi"]
        class_names = data_dict["class_names"]
        
        levels = ["lightweight", "medium", "heavy"]
        benchmark_results = {}
        
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, level in enumerate(levels):
            logger.info(f"--- Training & Benchmarking Model Level: {level.upper()} ---")
            model = get_model(level)
            
            # Train model
            start_train = time.time()
            model.fit(X_train, y_train)
            train_duration_s = time.time() - start_train
            
            # Save model
            save_path = os.path.join(self.models_dir, level, "model.joblib")
            model.save(save_path)
            model_size_kb = model.get_file_size_kb(save_path)
            
            # Evaluate on Test Set
            y_pred = model.predict(X_test)
            
            # Measure actual latency
            measured_latency_ms = model.measure_inference_speed(X_test[:500])
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate FPR & FNR across classes
            fp = cm.sum(axis=0) - np.diag(cm)
            fn = cm.sum(axis=1) - np.diag(cm)
            tp = np.diag(cm)
            tn = cm.sum() - (fp + fn + tp)
            
            fpr = np.mean(fp / (fp + tn + 1e-8))
            fnr = np.mean(fn / (fn + tp + 1e-8))
            
            benchmark_results[level] = {
                "model_name": model.name,
                "model_level": level,
                "accuracy": round(float(acc), 4),
                "precision_macro": round(float(prec_macro), 4),
                "recall_macro": round(float(rec_macro), 4),
                "f1_macro": round(float(f1_macro), 4),
                "f1_weighted": round(float(f1_weighted), 4),
                "false_positive_rate": round(float(fpr), 4),
                "false_negative_rate": round(float(fnr), 4),
                "measured_latency_ms": round(float(measured_latency_ms), 4),
                "estimated_latency_ms": model.est_latency_ms,
                "model_size_kb": round(float(model_size_kb), 2),
                "est_ram_kb": model.est_ram_kb,
                "est_energy_uJ": model.est_energy_uJ,
                "training_time_s": round(float(train_duration_s), 2)
            }
            
            logger.info(f"Model {level.upper()} Results -> Accuracy: {acc:.4f} | F1-Macro: {f1_macro:.4f} | Latency: {measured_latency_ms:.4f}ms | Size: {model_size_kb:.2f}KB")
            
            # Confusion matrix subplot
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes_cm[idx],
                        xticklabels=class_names, yticklabels=class_names)
            axes_cm[idx].set_title(f"{model.name}\n(Acc: {acc*100:.1f}%)", fontsize=11, fontweight="bold")
            axes_cm[idx].set_xlabel("Predicted")
            axes_cm[idx].set_ylabel("True")
            
        plt.tight_layout()
        cm_path = os.path.join(self.figures_dir, "baseline_confusion_matrices.png")
        fig_cm.savefig(cm_path, dpi=300)
        plt.close(fig_cm)
        logger.info(f"Confusion matrices plot saved to {cm_path}")
        
        # Save benchmark summary table
        table_path = os.path.join(self.tables_dir, "model_benchmark.json")
        with open(table_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)
            
        df_bench = pd.DataFrame(benchmark_results).T
        csv_table_path = os.path.join(self.tables_dir, "model_benchmark.csv")
        df_bench.to_csv(csv_table_path)
        logger.info(f"Benchmark table saved to {table_path} and {csv_table_path}")
        
        # Generate model comparison plot
        self._generate_comparison_plots(benchmark_results)
        
        return benchmark_results

    def _generate_comparison_plots(self, results):
        levels = list(results.keys())
        accs = [results[l]["accuracy"] * 100 for l in levels]
        f1s = [results[l]["f1_macro"] * 100 for l in levels]
        latencies = [results[l]["measured_latency_ms"] for l in levels]
        sizes = [results[l]["model_size_kb"] for l in levels]
        energies = [results[l]["est_energy_uJ"] for l in levels]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Accuracy & F1-Score
        x = np.arange(len(levels))
        width = 0.35
        axes[0, 0].bar(x - width/2, accs, width, label="Accuracy (%)", color="#1f77b4")
        axes[0, 0].bar(x + width/2, f1s, width, label="F1-Score (%)", color="#ff7f0e")
        axes[0, 0].set_ylabel("Percentage (%)")
        axes[0, 0].set_title("Model Detection Performance", fontsize=12, fontweight="bold")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([l.capitalize() for l in levels])
        axes[0, 0].set_ylim(50, 100)
        axes[0, 0].legend()
        
        # Plot 2: Latency per Inference
        axes[0, 1].bar([l.capitalize() for l in levels], latencies, color="#2ca02c")
        axes[0, 1].set_ylabel("Latency (ms per sample)")
        axes[0, 1].set_title("Inference Latency (Lower is Better)", fontsize=12, fontweight="bold")
        
        # Plot 3: Model Storage Size
        axes[1, 0].bar([l.capitalize() for l in levels], sizes, color="#d62728")
        axes[1, 0].set_ylabel("Model Size (KB)")
        axes[1, 0].set_title("Storage Size Footprint", fontsize=12, fontweight="bold")
        
        # Plot 4: Energy per Inference
        axes[1, 1].bar([l.capitalize() for l in levels], energies, color="#9467bd")
        axes[1, 1].set_ylabel("Estimated Energy (uJ / inference)")
        axes[1, 1].set_title("Energy Consumption", fontsize=12, fontweight="bold")
        
        plt.tight_layout()
        plot_path = os.path.join(self.figures_dir, "model_metrics_comparison.png")
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        logger.info(f"Model comparison plot saved to {plot_path}")

def run_benchmarking(config_path="config.yaml"):
    processed_path = load_config(config_path)["paths"]["processed_data"]
    if not os.path.exists(processed_path):
        from src.preprocessing.pipeline import run_preprocessing
        data_dict = run_preprocessing(config_path)
    else:
        data = np.load(processed_path, allow_pickle=True)
        data_dict = {
            "X_train": data["X_train"], "X_val": data["X_val"], "X_test": data["X_test"],
            "y_train_multi": data["y_train_multi"], "y_val_multi": data["y_val_multi"], "y_test_multi": data["y_test_multi"],
            "y_train_bin": data["y_train_bin"], "y_val_bin": data["y_val_bin"], "y_test_bin": data["y_test_bin"],
            "feature_names": data["feature_names"].tolist(), "class_names": data["class_names"].tolist()
        }
        
    benchmarker = ModelBenchmarker(config_path)
    return benchmarker.train_and_evaluate_all(data_dict)

if __name__ == "__main__":
    results = run_benchmarking()
    print("\nBENCHMARK RESULTS SUMMARY:")
    print(json.dumps(results, indent=2))
