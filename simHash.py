import numpy as np
import scipy.sparse as sparse
import math
import time

#
# SimHash Sketch
#

class NormalMatrixGenerator:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
    
    def generate(self):
        return np.random.randn(self.rows, self.cols)


class SHSketch():
    def __init__(self, sk_values: np.ndarray, norm: float) -> None:
        self.sk_values: np.ndarray = sk_values
        self.norm: float = norm

    def inner_product(self, other: 'SHSketch') -> float:
        
        # # norm_Sa = np.sqrt(len(self.sk_values))
        # # norm_Sb = np.sqrt(len(other.sk_values))
        # # print(f"cosine: {dot_product / (norm_Sa * norm_Sb)}")
        
        dot_product = np.dot(self.sk_values, other.sk_values)
        norm_Sa = np.linalg.norm(self.sk_values)
        norm_Sb = np.linalg.norm(other.sk_values)
        # print("cosine_sketch: {}".format(dot_product / (norm_Sa * norm_Sb)))
        return self.norm * other.norm * dot_product / (norm_Sa * norm_Sb)
        
        # print(f"self.sk_values: {self.sk_values}")
        # print(f"number of zeroes: {len(self.sk_values) - np.count_nonzero(self.sk_values)}")
        
        # difference = np.count_nonzero(self.sk_values - other.sk_values)
        # print(f"difference: {difference}")
        # print(f"cossim: {math.cos(difference / len(self.sk_values))}")
        # return math.cos(difference / len(self.sk_values)) * self.norm * other.norm

class SimHash():
    def __init__(self, sketch_size, vector_size) -> None:
        self.sketch_size: int = sketch_size  # Number of rows in the sketch matrix
        self.vector_size: int = vector_size
        # really time consuming
        self.phi = NormalMatrixGenerator(sketch_size, vector_size).generate()

    def sketch(self, vector: np.ndarray) -> SHSketch:

        sk_values = np.sign(self.phi.dot(vector))
        
        # print(f"np.linalg.norm(vector) = {np.linalg.norm(vector)}")
        return SHSketch(sk_values, np.linalg.norm(vector))

if __name__ == "__main__":
    vector_length = 2000
    vector_a = np.random.rand(vector_length)
    vector_b = np.random.rand(vector_length)
    
    print(f"vector_a: {vector_a}")
    print(f"vector_b: {vector_b}")

    sh = SimHash(sketch_size=1000, seed=1)

    sketch_a = sh.sketch(vector_a)
    sketch_b = sh.sketch(vector_b)

    inner_product = vector_a.dot(vector_b)
    inner_product_sketch = sketch_a.inner_product(sketch_b)

    # Print the results
    print("Inner product of the vector with itself: {}".format(inner_product))
    print("Inner product of the vector with itself using the SimHash: {}".format(inner_product_sketch))
    print("Relative error: {}".format(np.abs(inner_product - inner_product_sketch) / inner_product))
    