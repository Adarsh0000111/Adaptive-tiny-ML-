import numpy as np

from src.config import load_config

class DatasetTrafficProvider:
    """
    Streams network traffic samples from processed test dataset for real-time simulation.
    """
    def __init__(self, config_path="config.yaml"):
        config = load_config(config_path)
        processed_path = config["paths"]["processed_data"]
        data = np.load(processed_path, allow_pickle=True)
        
        self.X_test = data["X_test"]
        self.y_test_multi = data["y_test_multi"]
        self.class_names = data["class_names"].tolist()
        self.num_samples = len(self.X_test)
        self.index = 0

    def get_next_event(self):
        if self.index >= self.num_samples:
            self.index = 0 # Loop around
            
        sample_x = self.X_test[self.index]
        true_label_idx = self.y_test_multi[self.index]
        true_class = self.class_names[true_label_idx]
        
        event_id = self.index + 1000
        self.index += 1
        
        return {
            "event_id": event_id,
            "features": sample_x,
            "true_class": true_class
        }
