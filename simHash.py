import numpy as np
import scipy.sparse as sparse

#
# SimHash Sketch
#
class SHSketch():
    def __init__(self, sk_values: np.ndarray, norm: float) -> None:
        self.sk_values: np.ndarray = sk_values
        self.norm: float = norm

    def inner_product(self, other: 'SHSketch') -> float:
        dot_product = np.dot(self.sk_values, other.sk_values)
        norm_Sa = np.linalg.norm(self.sk_values)
        norm_Sb = np.linalg.norm(other.sk_values)
        print(f"cosine: {dot_product / (norm_Sa * norm_Sb)}")
        return dot_product / (norm_Sa * norm_Sb) * self.norm * other.norm

class SimHash():
    def __init__(self, sketch_size: int, seed: int) -> None:
        self.sketch_size: int = sketch_size  # Number of rows in the sketch matrix
        self.phi_rows: int = sketch_size  # Number of rows in the sketch matrix
        self.seed: int = seed  # Random seed for reproducibility

    def sketch(self, vector: np.ndarray) -> SHSketch:
        np.random.seed(self.seed)
        matrix_phi = np.random.randn(self.phi_rows, len(vector))
        # Compute the sketch values by taking dot product with the vector
        sk_values = matrix_phi.dot(vector)
        # Return the SimHash sketch as a SHSketchh object
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
    