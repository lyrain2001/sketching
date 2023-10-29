import numpy as np
import scipy.sparse as sparse
import math
import time

#
# SimHashFunction Sketch
#

class HashFuncGenerator:
    def __init__(self, non_zero_index, sketch_size, vector_size):
        self.non_zero_index = non_zero_index
        self.sketch_size = sketch_size
        self.vector_size = vector_size
    
    def generate(self):
        hash_func = np.zeros((self.vector_size, self.sketch_size))
        for index in self.non_zero_index:
            hash_func[index] = np.around(np.random.randn(self.sketch_size), 2)
        # print(f"hash_func: {hash_func}")
        return hash_func.T


class SimHashFunction():
    def __init__(self, sketch_size, vector_size, vector_a, vector_b) -> None:
        self.sketch_size: int = sketch_size  
        self.vector_size: int = vector_size
        self.vector_a = vector_a
        self.vector_b = vector_b        

    def sketch(self) -> float:
        non_zero_index = np.union1d(np.nonzero(self.vector_a), np.nonzero(self.vector_b))
        hash_func = HashFuncGenerator(non_zero_index, self.sketch_size, self.vector_size).generate()
    
        sketch_a = np.sign(hash_func.dot(self.vector_a))
        sketch_b = np.sign(hash_func.dot(self.vector_b))
    
        diff = np.count_nonzero(sketch_a - sketch_b)
        return math.cos(math.pi * diff / len(sketch_a)) * np.linalg.norm(self.vector_a) * np.linalg.norm(self.vector_b)
    