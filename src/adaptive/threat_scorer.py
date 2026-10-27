from collections import deque

from src.config import load_config

class ThreatScorer:
    """
    Configurable Threat Scoring System evaluating prediction, confidence, attack category,
    and rolling anomaly frequency.
    Outputs Threat Score (0-100) and Severity Level (LOW, MEDIUM, HIGH, CRITICAL).
    """
    def __init__(self, buffer_size=20, config_path="config.yaml"):
        config = load_config(config_path)
        ts_cfg = config["threat_scoring"]
        
        self.weights = ts_cfg.get("weights", {"confidence": 0.25, "severity": 0.45, "anomaly_frequency": 0.30})
        self.thresholds = ts_cfg.get("thresholds", {"low": 30, "medium": 60, "high": 80})
        self.severity_scores = ts_cfg.get("severity_scores", {
            "Normal": 0, "Scanning": 35, "PortScan": 40,
            "SQL_Injection": 60, "Backdoor": 75, "DDoS": 90, "Ransomware": 95
        })
        
        self.recent_events = deque(maxlen=buffer_size)

    def calculate_threat(self, prediction_class: str, confidence: float) -> dict:
        """
        Calculates threat score and severity level.
        """
        # Determine base severity score of predicted attack
        cat_severity = float(self.severity_scores.get(prediction_class, 50))
        
        # Track anomaly occurrence (Normal = 0, Attack = 1)
        is_anomaly = 0.0 if prediction_class == "Normal" else 1.0
        self.recent_events.append(is_anomaly)
        
        # Calculate rolling anomaly frequency (0.0 to 1.0)
        anomaly_freq = sum(self.recent_events) / float(len(self.recent_events))
        
        # Composite Threat Score Formula (0 to 100)
        w_conf = self.weights["confidence"]
        w_sev = self.weights["severity"]
        w_freq = self.weights["anomaly_frequency"]
        
        raw_score = (w_sev * cat_severity) + (w_conf * confidence * cat_severity) + (w_freq * anomaly_freq * 100.0)
        threat_score = int(round(max(0.0, min(100.0, raw_score))))
        
        # Classify severity level based on thresholds
        if threat_score <= self.thresholds["low"]:
            severity_level = "LOW"
        elif threat_score <= self.thresholds["medium"]:
            severity_level = "MEDIUM"
        elif threat_score <= self.thresholds["high"]:
            severity_level = "HIGH"
        else:
            severity_level = "CRITICAL"
            
        return {
            "prediction": prediction_class,
            "confidence": round(float(confidence), 3),
            "threat_score": threat_score,
            "severity": severity_level,
            "category_severity": cat_severity,
            "anomaly_frequency": round(float(anomaly_freq), 2)
        }
