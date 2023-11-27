import numpy as np
from scipy.signal import correlate

class LSH:
    def __init__(self, n_dimensions, n_hashes):
        self.n_dimensions = n_dimensions
        self.n_hashes = n_hashes
        self.hash_tables = [dict() for _ in range(n_hashes)]
        self.random_vectors = [np.random.randn(n_dimensions) for _ in range(n_hashes)]

    def hash(self, vector, hash_index):
        """Hash a vector into a binary hash using a random projection method."""
        return int(np.dot(vector, self.random_vectors[hash_index]) > 0)

    def add_to_hash_table(self, vector, label, hash_index):
        """Add a vector to a specific hash table."""
        hash_value = self.hash(vector, hash_index)
        if hash_value not in self.hash_tables[hash_index]:
            self.hash_tables[hash_index][hash_value] = []
        self.hash_tables[hash_index][hash_value].append(label)

    def fit(self, data, labels):
        """Fit LSH to data."""
        for i in range(self.n_hashes):
            for label, vector in zip(labels, data):
                self.add_to_hash_table(vector, label, i)

    def query(self, vector):
        """Query LSH with a vector to find nearest neighbors."""
        candidates = set()
        for i in range(self.n_hashes):
            hash_value = self.hash(vector, i)
            if hash_value in self.hash_tables[i]:
                candidates.update(self.hash_tables[i][hash_value])
        return list(candidates)

def generate_spike_trains(n, T):
    """Generate n spike trains with T time points."""
    return np.random.randint(0, 2, (n, T))

def compute_cross_correlation(spike_train1, spike_train2):
    """Compute cross-correlation between two spike trains."""
    correlation = correlate(spike_train1, spike_train2, mode='full', method='auto')
    normalized_correlation = correlation / (len(spike_train1) - np.abs(np.arange(-len(spike_train1) + 1, len(spike_train1))))
    return normalized_correlation

def main():
    n = 10  # Number of spike trains
    T = 1000  # Total time points
    n_hashes = 5  # Number of hash functions

    # Generate synthetic spike trains
    spike_trains = generate_spike_trains(n, T)

    # Initialize and fit LSH
    lsh = LSH(T, n_hashes)
    lsh.fit(spike_trains, labels=np.arange(n))

    # Query LSH with a target spike train
    target_spike_train = spike_trains[0]
    nearest_neighbors = lsh.query(target_spike_train)
    
    for neighbor_index in nearest_neighbors:
        if neighbor_index != 0:  # Avoid self-comparison
            cross_corr = compute_cross_correlation(target_spike_train, spike_trains[neighbor_index])
            print(f"Cross-correlation with spike train {neighbor_index}: {cross_corr}")


if __name__ == "__main__":
    main()