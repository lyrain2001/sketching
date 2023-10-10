import numpy as np
import scipy.sparse as sparse

#
# Johnson-Lindenstrauss Sketch
#
class JLSketch():
    def __init__(self, sk_values: np.ndarray) -> None:
        self.sk_values: np.ndarray = sk_values

    def inner_product(self, other: 'JLSketch') -> float:
        return self.sk_values.dot(other.sk_values)

class JL():
    def __init__(self, sketch_size: int, seed: int) -> None:
        self.sketch_size: int = sketch_size  # Number of rows in the sketch matrix
        self.pi_rows: int = sketch_size  # Number of rows in the sketch matrix
        self.seed: int = seed  # Random seed for reproducibility

    def sketch(self, vector: np.ndarray) -> JLSketch:
        np.random.seed(self.seed)

        # Find the indices of nonzero elements in the vector
        # nonzero_index = sparse.find(vector != 0)[1]
        
        matrix_pi = np.random.choice([1, -1], size=(self.pi_rows, len(vector)))
        matrix_pi = matrix_pi * (1 / np.sqrt(self.pi_rows))

        # Compute the sketch values by taking dot product with the vector
        sk_values = matrix_pi.dot(vector)

        # Return the JL sketch as a JLSketch object
        return JLSketch(sk_values)

if __name__ == "__main__":
    vector_length = 2000
    vector_a = np.random.rand(vector_length)
    vector_b = np.random.rand(vector_length)
    
    print(f"vector_a: {vector_a}")
    print(f"vector_b: {vector_b}")
    
    jl = JL(sketch_size=1000, seed=1)

    sketch_a = jl.sketch(vector_a)
    sketch_b = jl.sketch(vector_b)

    inner_product = vector_a.dot(vector_b)
    inner_product_sketch = sketch_a.inner_product(sketch_b)

    # Print the results
    print("Inner product of the vector with itself: {}".format(inner_product))
    print("Inner product of the vector with itself using the JL sketch: {}".format(inner_product_sketch))
    print("Relative error: {}".format(np.abs(inner_product - inner_product_sketch) / inner_product))
    