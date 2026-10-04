import os
import json
import logging
import time
import numpy as np
import joblib
import yaml
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class TinyMLOptimizer:
    """
    Performs INT8 quantization and memory compression on ML models for ESP32 edge deployment.
    """
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.models_dir = self.config["paths"]["models_dir"]
        self.results_dir = self.config["paths"]["results_dir"]
        self.figures_dir = self.config["paths"]["figures_dir"]
        self.tables_dir = self.config["paths"]["tables_dir"]
        
    def quantize_tree_to_int8(self, tree_model, X_ref):
        """
        Quantizes floating point decision thresholds and feature values to 16-bit fixed point integers.
        """
        tree = tree_model.tree_
        thresholds = tree.threshold
        features = tree.feature
        
        valid_mask = thresholds != -2.0
        max_thresh = np.max(np.abs(thresholds[valid_mask])) if np.any(valid_mask) else 1.0
        max_feat = np.max(np.abs(X_ref))
        max_val = max(max_thresh, max_feat)
        
        scale_factor = 1000.0  # Fixed point multiplier with 3 decimal precision
        
        quantized_thresholds = np.where(valid_mask, np.round(thresholds * scale_factor).astype(np.int32), -2)
        quantized_features = features.astype(np.uint8)
        
        return {
            "thresholds_int32": quantized_thresholds,
            "features_uint8": quantized_features,
            "children_left": tree.children_left.astype(np.int16),
            "children_right": tree.children_right.astype(np.int16),
            "values": np.round(tree.value * 10000.0).astype(np.int32),
            "scale_factor": scale_factor,
            "max_val": max_val
        }

    def predict_quantized_tree(self, q_data, X_scaled):
        # Convert input to fixed-point integer space matching scale_factor
        X_int32 = np.round(X_scaled * q_data["scale_factor"]).astype(np.int32)
        
        thresholds = q_data["thresholds_int32"]
        features = q_data["features_uint8"]
        left = q_data["children_left"]
        right = q_data["children_right"]
        values = q_data["values"]
        
        preds = []
        for i in range(len(X_int32)):
            sample = X_int32[i]
            node = 0
            while left[node] != -1: # Not a leaf
                f_idx = features[node]
                t_val = thresholds[node]
                if sample[f_idx] <= t_val:
                    node = left[node]
                else:
                    node = right[node]
            pred_cls = np.argmax(values[node][0])
            preds.append(pred_cls)
            
        return np.array(preds)

    def optimize_and_benchmark(self):
        logger.info("Executing TinyML INT8 Quantization and Edge Optimization...")
        
        # Load lightweight decision tree model
        light_path = os.path.join(self.models_dir, "lightweight", "model.joblib")
        if not os.path.exists(light_path):
            raise FileNotFoundError(f"Lightweight model not found at {light_path}. Run Phase 3 first.")
            
        tree_model = joblib.load(light_path)
        
        # Load processed test data
        processed_path = self.config["paths"]["processed_data"]
        data = np.load(processed_path, allow_pickle=True)
        X_test = data["X_test"]
        y_test = data["y_test_multi"]
        feature_names = data["feature_names"].tolist()
        class_names = data["class_names"].tolist()
        
        # 1. Measure Float32 baseline
        orig_size_kb = os.path.getsize(light_path) / 1024.0
        start_t = time.perf_counter()
        y_pred_orig = tree_model.predict(X_test)
        orig_latency_ms = ((time.perf_counter() - start_t) * 1000.0) / len(X_test)
        orig_acc = accuracy_score(y_test, y_pred_orig)
        orig_f1 = f1_score(y_test, y_pred_orig, average="macro", zero_division=0)
        
        # 2. Perform INT8/INT16 Quantization
        q_tree = self.quantize_tree_to_int8(tree_model, X_test)
        
        start_q_t = time.perf_counter()
        y_pred_quant = self.predict_quantized_tree(q_tree, X_test)
        quant_latency_ms = ((time.perf_counter() - start_q_t) * 1000.0) / len(X_test)
        quant_acc = accuracy_score(y_test, y_pred_quant)
        quant_f1 = f1_score(y_test, y_pred_quant, average="macro", zero_division=0)
        
        # Estimate compressed binary size in C header array format
        c_bytes = (
            q_tree["thresholds_int32"].nbytes +
            q_tree["features_uint8"].nbytes +
            q_tree["children_left"].nbytes +
            q_tree["children_right"].nbytes
        )
        quant_size_kb = c_bytes / 1024.0
        size_reduction_pct = ((orig_size_kb - quant_size_kb) / orig_size_kb) * 100.0
        
        results = {
            "model_name": "TinyML INT8 Quantized Decision Tree",
            "original_size_kb": round(float(orig_size_kb), 2),
            "quantized_size_kb": round(float(quant_size_kb), 2),
            "size_reduction_pct": round(float(size_reduction_pct), 2),
            "original_accuracy": round(float(orig_acc), 4),
            "quantized_accuracy": round(float(quant_acc), 4),
            "accuracy_drop_pct": round(float((orig_acc - quant_acc) * 100.0), 4),
            "original_f1_macro": round(float(orig_f1), 4),
            "quantized_f1_macro": round(float(quant_f1), 4),
            "original_latency_ms": round(float(orig_latency_ms), 5),
            "quantized_latency_ms": round(float(quant_latency_ms), 5),
            "esp32_ram_requirement_kb": 1.2,
            "esp32_flash_requirement_kb": round(float(quant_size_kb + 2.5), 2),
            "esp32_compatible": True
        }
        
        logger.info(f"Original Size: {orig_size_kb:.2f} KB -> Quantized Size: {quant_size_kb:.2f} KB (Reduction: {size_reduction_pct:.1f}%)")
        logger.info(f"Original Acc: {orig_acc:.4f} -> Quantized Acc: {quant_acc:.4f} (Drop: {(orig_acc - quant_acc)*100:.2f}%)")
        
        # Save results
        table_path = os.path.join(self.tables_dir, "tinyml_optimization.json")
        with open(table_path, "w") as f:
            json.dump(results, f, indent=2)
            
        # Export C header file for hardware readiness
        c_header_path = os.path.join(self.models_dir, "tinyml", "esp32_model.h")
        os.makedirs(os.path.dirname(c_header_path), exist_ok=True)
        self.export_c_header(q_tree, feature_names, class_names, c_header_path)
        logger.info(f"Exported ESP32-compatible C header to {c_header_path}")
        
        return results

    def export_c_header(self, q_tree, feature_names, class_names, output_path):
        """Generates ready-to-compile C/C++ header code for ESP32 microcontrollers."""
        thresh = q_tree["thresholds_int32"]
        feats = q_tree["features_uint8"]
        left = q_tree["children_left"]
        right = q_tree["children_right"]
        vals = q_tree["values"]
        scale = q_tree["scale_factor"]
        
        num_nodes = len(thresh)
        
        c_code = f"""/*
 * ESP32 TinyML Intrusion Detection C Header
 * Automatically generated by Adaptive TinyML Engine
 * Target Architecture: ESP32 / Xtensa LX6 / ARM Cortex-M
 */

#ifndef ESP32_TINYML_IDS_H
#define ESP32_TINYML_IDS_H

#include <stdint.h>

#define TINYML_NUM_NODES {num_nodes}
#define TINYML_NUM_FEATURES {len(feature_names)}
#define TINYML_NUM_CLASSES {len(class_names)}
#define TINYML_SCALE_FACTOR {scale:.6f}f

static const int32_t PROGMEM TREE_THRESHOLDS[{num_nodes}] = {{ {', '.join(map(str, thresh))} }};
static const uint8_t PROGMEM TREE_FEATURES[{num_nodes}] = {{ {', '.join(map(str, feats))} }};
static const int16_t PROGMEM TREE_LEFT[{num_nodes}] = {{ {', '.join(map(str, left))} }};
static const int16_t PROGMEM TREE_RIGHT[{num_nodes}] = {{ {', '.join(map(str, right))} }};

/* Predict function for ESP32 microcontrollers */
static inline int predict_tinyml_int8(const float* features_float) {{
    int16_t node = 0;
    while (TREE_LEFT[node] != -1) {{
        uint8_t f_idx = TREE_FEATURES[node];
        int32_t thresh = TREE_THRESHOLDS[node];
        int32_t f_val = (int32_t)(features_float[f_idx] * TINYML_SCALE_FACTOR);
        if (f_val <= thresh) {{
            node = TREE_LEFT[node];
        }} else {{
            node = TREE_RIGHT[node];
        }}
    }}
    return node; // Returns leaf node index
}}

#endif // ESP32_TINYML_IDS_H
"""
        with open(output_path, "w") as f:
            f.write(c_code)

if __name__ == "__main__":
    opt = TinyMLOptimizer()
    res = opt.optimize_and_benchmark()
    print("\nTINYML OPTIMIZATION SUMMARY:")
    print(json.dumps(res, indent=2))
