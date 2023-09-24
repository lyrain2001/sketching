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
        nonzero_index = sparse.find(vector != 0)[1]
        
        matrix_pi = np.random.choice([1, -1], size=(self.pi_rows, len(nonzero_index)))
        matrix_pi = matrix_pi * (1 / np.sqrt(self.pi_rows))
        print(matrix_pi.shape)

        # Compute the sketch values by taking dot product with the vector
        sk_values = matrix_pi.dot(vector[nonzero_index])

        # Return the JL sketch as a JLSketch object
        return JLSketch(sk_values)

if __name__ == "__main__":
    vector = np.random.randint(0, 2, 1000)

    jl = JL(sketch_size=50, seed=1)

    sketch = jl.sketch(vector)
    sketch2 = jl.sketch(vector)

    inner_product = vector.dot(vector)
    inner_product_sketch = sketch.inner_product(sketch2)

    # Print the results
    print("Inner product of the vector with itself: {}".format(inner_product))
    print("Inner product of the vector with itself using the JL sketch: {}".format(inner_product_sketch))
    print("Relative error: {}".format(np.abs(inner_product - inner_product_sketch) / inner_product))
    