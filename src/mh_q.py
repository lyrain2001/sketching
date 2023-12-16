import numpy as np
import random
from numba import njit

#
# Quantized MH Sketch
#
class QMHSketch():
    def __init__(self, sk_hashes: np.ndarray, vector_l1: int) -> None:
        self.sk_hashes: np.ndarray = sk_hashes
        self.vector_l1: int = vector_l1
    
    def inner_product_numba(self, sk_hashesA: np.ndarray, sk_hashesB: np.ndarray, vector_l1A: int, vector_l1B: int):
        m = len(sk_hashesA)
        sum_min = 0.0
        for i in range(m):
            if sk_hashesA[i] == sk_hashesB[i]:
                sum_min += 1
        # print("sum_min: {}".format(sum_min))
        sum_l1 = vector_l1A + vector_l1B
        # print("sum_l1: {}".format(sum_l1))
        union_size = m * sum_l1 / (2 * sum_min)
        # print("union_size: {}".format(union_size))
        ip_est = sum_l1 - union_size
        return ip_est

    def inner_product(self, other: 'QMHSketch') -> float:
        return self.inner_product_numba(self.sk_hashes, other.sk_hashes, self.vector_l1, other.vector_l1)


class QMH():
    def __init__(self, sketch_size: int, seed: int) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed

    def sketch(self, vector: np.ndarray) -> QMHSketch:
        vector_l1 = np.linalg.norm(vector, ord=1)
        sk_hashes = hash_kwise(vector, self.seed, dimension_num=self.sketch_size)
        return QMHSketch(sk_hashes, vector_l1)
    

def hash_kwise(vector, seed, dimension_num=1, k_wise=4, n_bits=1, PRIME=2147483587):
        # if input is vector, treat index of values as keys
        keys = [i for i,v in enumerate(vector) if v != 0]
        keys = np.array(keys, dtype=np.int32)
        return hash_kwise_kv(keys, seed, dimension_num, k_wise, n_bits, PRIME)
    

def hash_kwise_kv(keys, seed, dimension_num, k_wise, n_bits, PRIME):
    np.random.seed(seed)
    hash_parameters = np.random.randint(1, PRIME, (dimension_num, k_wise))
    hash_kwise = 0
    for exp in range(k_wise):
        hash_kwise += np.dot(np.transpose(np.array([keys])**exp), np.array([np.transpose(hash_parameters[:, exp])]))
    hash_kwise = np.mod(hash_kwise, PRIME)

    if dimension_num == 1:
        # Reshape the hash values as a 1D array
        hashes = hash_kwise.reshape(hash_kwise.shape[0],)
    else:
        # Find the minimum hash value for each column
        hashes = np.min(hash_kwise, axis=0)
    max_val = 2**n_bits
    hashes = np.mod(hashes, max_val)

    return hashes