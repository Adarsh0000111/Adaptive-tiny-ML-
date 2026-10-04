import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PageHinkleyDriftDetector:
    """
    Page-Hinkley Sequential Test for Concept Drift Detection in real-time IIoT network telemetry.
    Detects abrupt or gradual changes in prediction error or feature means.
    """
    def __init__(self, delta=0.005, threshold=3.5, alpha=0.999):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        
        self.mean = 0.0
        self.sum_cumulative = 0.0
        self.min_cumulative = float("inf")
        self.element_count = 0
        self.drift_detected = False

    def update(self, val):
        """
        Update Page-Hinkley statistics with a new sample value (e.g. prediction error 0/1 or feature value).
        Returns True if concept drift is detected.
        """
        self.element_count += 1
        self.mean = self.mean + (val - self.mean) / self.element_count
        
        self.sum_cumulative = self.sum_cumulative + (val - self.mean - self.delta)
        if self.sum_cumulative < self.min_cumulative:
            self.min_cumulative = self.sum_cumulative

        ph_stat = self.sum_cumulative - self.min_cumulative
        
        if ph_stat > self.threshold:
            self.drift_detected = True
            logger.warning(f"🚨 CONCEPT DRIFT DETECTED at sample #{self.element_count}! (Page-Hinkley Stat: {ph_stat:.4f} > Threshold: {self.threshold})")
            self.reset()
            return True
            
        return False

    def reset(self):
        self.sum_cumulative = 0.0
        self.min_cumulative = float("inf")
        self.drift_detected = False

class ConceptDriftMonitor:
    """
    Monitors overall pipeline feature streams and accuracy metrics for concept drift.
    """
    def __init__(self, num_features=18):
        self.feature_detectors = [PageHinkleyDriftDetector() for _ in range(num_features)]
        self.error_detector = PageHinkleyDriftDetector(delta=0.01, threshold=2.5)

    def monitor_event(self, feature_vector, prediction_error):
        """
        Check feature vector and prediction error for drift.
        """
        drift_in_error = self.error_detector.update(prediction_error)
        
        feature_drifts = []
        for i, val in enumerate(feature_vector):
            if self.feature_detectors[i].update(val):
                feature_drifts.append(i)

        return {
            "error_drift": drift_in_error,
            "feature_drifts": feature_drifts,
            "drift_occurred": drift_in_error or len(feature_drifts) > 0
        }
