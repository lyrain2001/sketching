import numpy as np
import scipy.sparse as sparse
import math
import time

#
# SimHashRound Sketch
#

class NormalMatrixGenerator:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
    
    def generate(self):
        # round to 2 decimal places
        return np.around(np.random.randn(self.rows, self.cols), 2)


class SHRSketch():
    def __init__(self, sk_values: np.ndarray, norm: float) -> None:
        self.sk_values: np.ndarray = sk_values
        self.norm: float = norm

    def inner_product(self, other: 'SHRSketch') -> float:
        
        difference = np.count_nonzero(self.sk_values - other.sk_values)
        return math.cos(math.pi * difference / len(self.sk_values)) * self.norm * other.norm

class SimHashRound():
    def __init__(self, sketch_size, vector_size) -> None:
        self.sketch_size: int = sketch_size  # Number of rows in the sketch matrix
        self.vector_size: int = vector_size
        # really time consuming
        self.phi = NormalMatrixGenerator(sketch_size, vector_size).generate()

    def sketch(self, vector: np.ndarray) -> SHRSketch:
        sk_values = np.sign(self.phi.dot(vector))
        return SHRSketch(sk_values, np.linalg.norm(vector))

if __name__ == "__main__":
    vector_length = 2000
    vector_a = np.random.rand(vector_length)
    vector_b = np.random.rand(vector_length)
    
    print(f"vector_a: {vector_a}")
    print(f"vector_b: {vector_b}")

    sh = SimHashRound(sketch_size=1000, seed=1)

    sketch_a = sh.sketch(vector_a)
    sketch_b = sh.sketch(vector_b)

    inner_product = vector_a.dot(vector_b)
    inner_product_sketch = sketch_a.inner_product(sketch_b)

    # Print the results
    print("Inner product of the vector with itself: {}".format(inner_product))
    print("Inner product of the vector with itself using the SimHashRound: {}".format(inner_product_sketch))
    print("Relative error: {}".format(np.abs(inner_product - inner_product_sketch) / inner_product))
    