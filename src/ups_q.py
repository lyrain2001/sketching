import numpy as np
import scipy.sparse as sparse
import math
import time

#
# Unweighted Priority Sampling SimHash Sketch
#

class UnweightedSimRandomHasher:
    def __init__(self, n):
        self.hashed_values = [0] * (n + 1)  # Initialize list with n+1 zeros
        
        for i in range(n + 1):
            # generate a random float in N(0,1)
            # rand_value = np.random.randn()
            # generate a random float in [-1, 1]
            rand_value = np.random.uniform(-1, 1)
            self.hashed_values[i] = rand_value
    
    def get_hashed_values(self):
        return self.hashed_values

class UnweightedSimNormalMatrixGenerator:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
    
    def generate(self):
        return np.random.randn(self.rows, self.cols)
    
class UPSSHSketch():
    def __init__(self, sk_indices: np.ndarray, sk_values: np.ndarray, norm: float, tau: float) -> None:
        self.sk_indices: np.ndarray = sk_indices
        self.sk_values: np.ndarray = sk_values
        self.norm: float = norm
        self.tau: float = tau

    def inner_product(self, other: 'UPSSHSketch') -> float:
        i = 0
        j = 0
        intersection = []
        while i < len(self.sk_indices) and j < len(other.sk_indices):
            ka, va = self.sk_indices[i], self.sk_values[i]
            kb, vb = other.sk_indices[j], other.sk_values[j]
            if ka == kb:
                intersection.append(va-vb)
                i += 1
                j += 1
            elif ka < kb:
                i += 1
            else:
                j += 1

        difference = np.count_nonzero(intersection)
        # print(f"difference: {difference}")
        # print(f"2_cosine_sketch: {math.cos(difference / len(self.sk_values))}")
        # # print("2_sketch: {}".format(math.cos(math.pi * difference / len(self.sk_values)) * self.norm * other.norm))
        # print(f"with denomimator: {math.cos(math.pi * difference / len(self.sk_values)) * self.norm * other.norm / min(1, self.tau, other.tau)}")
        # print(f"without denomimator: {math.cos(math.pi * difference / len(self.sk_values)) * self.norm * other.norm}")
        return math.cos(math.pi * difference / len(self.sk_values)) * self.norm * other.norm / min(1, self.tau, other.tau)

class UnweightedPrioritySamplingSimHash():
    def __init__(self, sketch_size, vector_size) -> None:
        self.sketch_size: int = sketch_size  # Number of rows in the sketch matrix
        self.vector_size: int = vector_size
        self.hashed_values: np.ndarray = UnweightedSimRandomHasher(vector_size).get_hashed_values()
        self.phi = UnweightedSimNormalMatrixGenerator(sketch_size, sketch_size).generate()

    def sketch(self, vector: np.ndarray) -> UPSSHSketch:
        rank = [self.hashed_values[i] for i in range(len(vector))]
        abs_rank = sorted(abs(rank[i]) for i in range(len(vector)))
        tau = float('inf')  
        
        if len(abs_rank) > self.sketch_size:
            tau = abs_rank[self.sketch_size]  
        
        sk_indices = []
        sk_values = []
        
        for i in range(len(rank)):
            if abs(rank[i]) < tau:
                sk_indices.append(i)
                sk_values.append(vector[i])  
        
        norm = np.linalg.norm(sk_values)
        sk_values = np.sign(self.phi.dot(sk_values))
        
        return UPSSHSketch(sk_indices, sk_values, norm, tau)

if __name__ == "__main__":
    vector_length = 2000
    vector_a = np.random.rand(vector_length)
    vector_b = np.random.rand(vector_length)
    
    print(f"vector_a: {vector_a}")
    print(f"vector_b: {vector_b}")

    sh = UnweightedPrioritySamplingSimHash(sketch_size=1000, seed=1)

    sketch_a = sh.sketch(vector_a)
    sketch_b = sh.sketch(vector_b)

    inner_product = vector_a.dot(vector_b)
    inner_product_sketch = sketch_a.inner_product(sketch_b)

    # Print the results
    print("Inner product of the vector with itself: {}".format(inner_product))
    print("Inner product of the vector with itself using the SimHash: {}".format(inner_product_sketch))
    print("Relative error: {}".format(np.abs(inner_product - inner_product_sketch) / inner_product))
    