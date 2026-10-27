import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RobustAggregator:
    """
    Byzantine-Robust Federated Learning Aggregators to defend against adversarial client poisoning attacks.
    Supports Krum, Trimmed Mean, and Coordinate-wise Median.
    """

    @staticmethod
    def _flatten_weights(weights_dict):
        keys = sorted(weights_dict.keys())
        flat = np.concatenate([weights_dict[k].flatten() for k in keys])
        return flat, keys

    @staticmethod
    def _unflatten_weights(flat_vec, keys, template_weights):
        reconstructed = {}
        offset = 0
        for k in keys:
            shape = template_weights[k].shape
            size = template_weights[k].size
            reconstructed[k] = flat_vec[offset:offset + size].reshape(shape)
            offset += size
        return reconstructed

    def aggregate_krum(self, client_weights_list, num_byzantine=1):
        """
        Multi-Krum / Single Krum Aggregation:
        Selects the client update that minimizes sum of Euclidean distances to its n - k - 2 closest neighbors.
        """
        n = len(client_weights_list)
        if n <= 2 * num_byzantine + 2:
            logger.warning(f"Krum requirement n > 2f + 2 not strictly met (n={n}, f={num_byzantine}). Falling back to mean.")
            num_byzantine = max(0, (n - 3) // 2)

        flat_updates = []
        for w in client_weights_list:
            flat, keys = self._flatten_weights(w)
            flat_updates.append(flat)
        flat_updates = np.array(flat_updates)

        scores = []
        m = n - num_byzantine - 2
        for i in range(n):
            distances = np.sum((flat_updates - flat_updates[i]) ** 2, axis=1)
            sorted_dist = np.sort(distances)
            # Sum of distances to m closest neighbors (excluding itself at index 0)
            score = np.sum(sorted_dist[1:m + 1])
            scores.append(score)

        best_idx = int(np.argmin(scores))
        logger.info(f"Krum Selected Client Index {best_idx} out of {n} clients (Byzantine f={num_byzantine})")
        return self._unflatten_weights(flat_updates[best_idx], keys, client_weights_list[0])

    def aggregate_trimmed_mean(self, client_weights_list, trim_ratio=0.2):
        """
        Trimmed Mean Aggregation:
        Trims highest and lowest beta fraction of weights per coordinate before averaging.
        """
        n = len(client_weights_list)
        flat_updates = [self._flatten_weights(w)[0] for w in client_weights_list]
        keys = self._flatten_weights(client_weights_list[0])[1]
        flat_updates = np.array(flat_updates)

        k = int(n * trim_ratio)
        if k == 0 and n > 2:
            k = 1

        sorted_updates = np.sort(flat_updates, axis=0)
        trimmed = sorted_updates[k:n - k, :]
        mean_update = np.mean(trimmed, axis=0)

        logger.info(f"Trimmed Mean Aggregation completed (Trimmed {k} min/max clients per parameter)")
        return self._unflatten_weights(mean_update, keys, client_weights_list[0])

    def aggregate_median(self, client_weights_list):
        """
        Coordinate-wise Median Aggregation.
        """
        flat_updates = [self._flatten_weights(w)[0] for w in client_weights_list]
        keys = self._flatten_weights(client_weights_list[0])[1]
        flat_updates = np.array(flat_updates)

        median_update = np.median(flat_updates, axis=0)
        logger.info("Coordinate-wise Median Aggregation completed.")
        return self._unflatten_weights(median_update, keys, client_weights_list[0])
