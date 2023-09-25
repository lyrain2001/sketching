import numpy as np
import scipy.sparse as sparse
import random

#
# Priority Sampling Sketch
#

class RandomHasher:
    def __init__(self, n):
        self.hashed_values = [0] * (n + 1)  # Initialize list with n+1 zeros
        
        for i in range(n + 1):
            rand_value = random.random()  # Generates a random float in [0, 1)
            self.hashed_values[i] = rand_value
    
    def get_hashed_values(self):
        return self.hashed_values
    
    
class PSSketch():
    def __init__(self, sk_indices: np.ndarray, sk_values: np.ndarray, tau: float) -> None:
        self.sk_values: np.ndarray = sk_values
        self.sk_indices: np.ndarray = sk_indices
        self.tau: float = tau

    def inner_product(self, other: 'PSSketch') -> float:
        sum_value = 0
        i = 0
        j = 0
        while i < len(self.sk_indices) and j < len(other.sk_indices):
            print(f"i: {i}\nj: {j}")
            ka, va = self.sk_indices[i], self.sk_values[i]
            kb, vb = other.sk_indices[j], other.sk_values[j]
            if ka == kb:
                denominator = min(1, va ** 2 * self.tau, vb ** 2 * other.tau)
                sum_value += va * vb / denominator
                i += 1
                j += 1
            elif ka < kb:
                i += 1
            else:
                j += 1
        return sum_value


class PrioritySampling():
    def __init__(self, sketch_size: int) -> None:
        self.sketch_size: int = sketch_size  # Number of rows in the sketch matrix

    def sketch(self, vector: np.ndarray) -> PSSketch:
        hasher = RandomHasher(len(vector))
        hashed_values = hasher.get_hashed_values()
        
        rank = [None] * len(vector)

        # Fill in the rank array with H[i]/A[i]^2 for A[i] != 0
        for i in range(len(vector)):
            if vector[i] != 0:
                rank[i] = hashed_values[i] / (vector[i] ** 2)
            else:
                rank[i] = None 
        
        tau = float('inf')  # Initialize tau_a to infinity
        
        # Filtering out values where corresponding A[i] is zero and then sorting the remaining values in rank.
        non_zero_rank = sorted(rank[i] for i in range(len(vector)) if vector[i] != 0)
        
        # If there are at least m + 1 non-zero values in A, find the (m + 1)-th smallest value in rank.
        if len(non_zero_rank) > self.sketch_size:
            tau = non_zero_rank[self.sketch_size]  # m is 1 less than (m + 1) due to 0-based indexing.
        
        print(tau)  # Output tau_a
        sk_indices = []
        sk_values = []
        
        # Iterate through the rank array and append to K and V as necessary
        for i in range(len(rank)):
            if vector[i] != 0 and rank[i] < tau:
                sk_indices.append(i)
                sk_values.append(vector[i])    
        
        return PSSketch(sk_indices, sk_values, tau)


if __name__ == "__main__":
    vector_length = 2000
    vector_a = np.random.rand(vector_length)
    vector_b = np.random.rand(vector_length)
    
    print(f"vector_a: {vector_a}")
    print(f"vector_b: {vector_b}")
    
    ps = PrioritySampling(sketch_size=1000)

    sketch_a = ps.sketch(vector_a)
    sketch_b = ps.sketch(vector_b)
    
    # print(sketch_a.sk_values)
    # print(sketch_b.sk_values)

    inner_product = vector_a.dot(vector_b)
    inner_product_sketch = sketch_a.inner_product(sketch_b)

    # Print the results
    print("Inner product of the vector with itself: {}".format(inner_product))
    print("Inner product of the vector with itself using Priority Sampling: {}".format(inner_product_sketch))
    print("Relative error: {}".format(np.abs(inner_product - inner_product_sketch) / inner_product))
    