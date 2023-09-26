import numpy as np
import scipy.sparse as sparse

#
# SimHash Sketch with same length
#
class SHMSketch():
    def __init__(self, sk_values: np.ndarray, norm: float) -> None:
        self.sk_values: np.ndarray = sk_values
        self.norm: float = norm

    def inner_product(self, other: 'SHMSketch') -> float:
        dot_product = np.dot(self.sk_values, other.sk_values)
        norm_Sa = np.linalg.norm(self.sk_values)
        norm_Sb = np.linalg.norm(other.sk_values)
        # norm_Sa = np.sqrt(len(self.sk_values))
        # norm_Sb = np.sqrt(len(other.sk_values))
        # print(f"cosine: {dot_product / (norm_Sa * norm_Sb)}")
        return dot_product / (norm_Sa * norm_Sb) * self.norm * other.norm

class SimHashM():
    def __init__(self) -> None:
        pass

    def sketch(self, vector: np.ndarray) -> SHMSketch:        
        sk_values = np.sign(vector)
        return SHMSketch(sk_values, np.linalg.norm(vector))

if __name__ == "__main__":
    vector_length = 2000
    vector_a = np.random.rand(vector_length)
    vector_b = np.random.rand(vector_length)
    
    print(f"vector_a: {vector_a}")
    print(f"vector_b: {vector_b}")

    sh = SimHashM(sketch_size=1000, seed=1)

    sketch_a = sh.sketch(vector_a)
    sketch_b = sh.sketch(vector_b)

    inner_product = vector_a.dot(vector_b)
    inner_product_sketch = sketch_a.inner_product(sketch_b)

    # Print the results
    print("Inner product of the vector with itself: {}".format(inner_product))
    print("Inner product of the vector with itself using the SimHash: {}".format(inner_product_sketch))
    print("Relative error: {}".format(np.abs(inner_product - inner_product_sketch) / inner_product))
    