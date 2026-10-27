import numpy as np
import logging
import copy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DifferentiallyPrivateFedAvg:
    """
    DP-FedAvg implementation with clipping threshold C and Gaussian Noise Mechanism.
    Guarantees (epsilon, delta)-differential privacy across local client updates.
    """
    def __init__(self, clipping_threshold=1.0, noise_multiplier=0.1, delta=1e-5):
        self.clipping_threshold = clipping_threshold
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.accumulated_noise_scale = 0.0

    def clip_update(self, global_weights, client_weights):
        """
        Clip local update vector to L2 norm bound C.
        """
        clipped_weights = {}
        total_l2_norm = 0.0

        for key in global_weights.keys():
            diff = client_weights[key] - global_weights[key]
            total_l2_norm += np.sum(diff ** 2)
        total_l2_norm = np.sqrt(total_l2_norm)

        clip_coef = min(1.0, self.clipping_threshold / (total_l2_norm + 1e-8))

        for key in global_weights.keys():
            clipped_weights[key] = global_weights[key] + (client_weights[key] - global_weights[key]) * clip_coef

        return clipped_weights, total_l2_norm

    def aggregate_with_dp(self, global_weights, client_weights_list, num_samples_list):
        """
        Aggregates clipped client updates and adds calibrated Gaussian noise.
        """
        num_clients = len(client_weights_list)
        if num_clients == 0:
            return global_weights

        clipped_updates = []
        l2_norms = []

        for c_weights in client_weights_list:
            clipped, norm = self.clip_update(global_weights, c_weights)
            clipped_updates.append(clipped)
            l2_norms.append(norm)

        # Average clipped weights
        new_global_weights = {}
        for key in global_weights.keys():
            avg_layer = np.mean([u[key] for u in clipped_updates], axis=0)
            
            # Calibrate and add Gaussian noise: N(0, (sigma * C / num_clients)^2)
            noise_std = (self.noise_multiplier * self.clipping_threshold) / num_clients
            noise = np.random.normal(0, noise_std, size=avg_layer.shape)
            
            new_global_weights[key] = avg_layer + noise

        self.accumulated_noise_scale += noise_std
        logger.info(f"DP-FedAvg Aggregation complete. Avg Client Update L2 Norm: {np.mean(l2_norms):.4f} | Added Noise Std: {noise_std:.6f}")
        return new_global_weights

    def estimate_privacy_budget(self, total_rounds, sample_rate):
        """
        Rényi / Moments Accountant approximate privacy budget estimation.
        """
        if self.noise_multiplier == 0:
            return float("inf"), self.delta
        
        # Simplified moments accountant bound approximation
        epsilon = (2.0 * sample_rate * np.sqrt(total_rounds * np.log(1.0 / self.delta))) / max(self.noise_multiplier, 1e-4)
        return round(float(epsilon), 4), self.delta
