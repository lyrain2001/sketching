import numpy as np
import scipy.sparse as sparse
import math
import time

#
# Normalized SimHash Sketch
#

class NormalMatrixGenerator:
    def __init__(self, rows, cols, seed):
        self.rows = rows
        self.cols = cols
        self.seed = seed
    
    def generate(self):
        np.random.seed(self.seed)
        return np.random.randn(self.rows, self.cols)


class NSHSketch():
    def __init__(self, sk_values: np.ndarray, norm: float) -> None:
        self.sk_values: np.ndarray = sk_values
        self.norm: float = norm

    def inner_product(self, other: 'NSHSketch') -> float:
        
        difference = np.count_nonzero(self.sk_values - other.sk_values)
        
        return math.cos(math.pi * difference / len(self.sk_values)) * self.norm * other.norm

class NormalizedSimHash():
    def __init__(self, sketch_size, vector_size, seed, p: float=1e-7) -> None:
        self.sketch_size: int = sketch_size  # Number of rows in the sketch matrix
        self.vector_size: int = vector_size
        self.L = 1/p
        # really time consuming
        self.phi = NormalMatrixGenerator(sketch_size, vector_size, seed).generate()

    def sketch(self, vector: np.ndarray) -> NSHSketch:
        vector_l1 = np.linalg.norm(vector, ord=1)
        tilte_a = self.vector_rounding(vector/vector_l1, self.L)
        # tilte_a = vector/vector_l1
        
        # sk_values = np.sign(self.phi.dot(vector))
        sk_values = np.sign(self.phi.dot(tilte_a))
        return NSHSketch(sk_values, np.linalg.norm(vector))
    
    @staticmethod
    def vector_rounding(z, L):
        # Step 1: Compute the rounded values for all elements in bv_z
        tilde_z = np.sign(z) * np.floor(np.abs(z) * L) / L
        # Step 2: Find the index i* with maximum absolute value in bv_z
        i_star = np.argmax(np.abs(z))
        # Step 3: Adjust bv_tilde_z[i*] to ensure it's a unit vector (norm = 1)
        delta = 1 - np.linalg.norm(tilde_z, ord=1)
        tilde_z[i_star] = np.sign(z[i_star]) * (tilde_z[i_star] + delta)
        return tilde_z
    
    @staticmethod
    def log_scaling(z, epsilon=1e-10):
        scaled_z = np.log(z + epsilon)
        return scaled_z
