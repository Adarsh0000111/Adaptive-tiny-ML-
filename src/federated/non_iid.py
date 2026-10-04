import numpy as np

def create_non_iid_partitions(X, y, num_clients=5, alpha=0.5, seed=42):
    """
    Partitions dataset X, y across num_clients using a Dirichlet distribution (alpha).
    Lower alpha -> higher Non-IID label skew.
    """
    np.random.seed(seed)
    num_classes = len(np.unique(y))
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        np.random.shuffle(idx_c)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(idx_c)).astype(int)
        
        # Ensure all samples are assigned
        diff = len(idx_c) - proportions.sum()
        for i in range(diff):
            proportions[i % num_clients] += 1
            
        start = 0
        for client_id in range(num_clients):
            end = start + proportions[client_id]
            client_indices[client_id].extend(idx_c[start:end])
            start = end
            
    client_datasets = []
    for client_id in range(num_clients):
        indices = np.array(client_indices[client_id])
        np.random.shuffle(indices)
        client_datasets.append((X[indices], y[indices]))
        
    return client_datasets
