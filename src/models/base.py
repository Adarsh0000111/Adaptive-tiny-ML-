import time
import os
import sys
import joblib
import numpy as np

class BaseIDSModel:
    def __init__(self, name, model_level, est_latency_ms, est_ram_kb, est_energy_uJ):
        self.name = name
        self.model_level = model_level  # "lightweight", "medium", "heavy"
        self.est_latency_ms = est_latency_ms
        self.est_ram_kb = est_ram_kb
        self.est_energy_uJ = est_energy_uJ
        self.model = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without direct probability support
            preds = self.predict(X)
            num_classes = len(np.unique(preds))
            probs = np.zeros((len(preds), num_classes))
            for i, p in enumerate(preds):
                probs[i, p] = 1.0
            return probs

    def measure_inference_speed(self, X_sample, n_runs=100):
        start_time = time.perf_counter()
        for _ in range(n_runs):
            _ = self.model.predict(X_sample)
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        avg_latency_ms = total_time_ms / (n_runs * len(X_sample))
        return avg_latency_ms

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)

    def get_file_size_kb(self, filepath):
        if os.path.exists(filepath):
            return os.path.getsize(filepath) / 1024.0
        return 0.0
