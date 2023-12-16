import numpy as np
import random
from numba import njit

#
# Quantized MH Sketch
#
class QMHSketch():
    def __init__(self, sk_hashes: np.ndarray, sk_values: np.ndarray, sketch_size: int) -> None:
        self.sk_hashes: np.ndarray = sk_hashes
        self.sk_values: np.ndarray = sk_values
        self.sketch_size: int = sketch_size

    @staticmethod
    @njit(parallel=False)
    def inner_product_numba(sk_hashesA, sk_valuesA, sk_hashesB, sk_valuesB):
        m = len(sk_hashesA)
        sum_min = 0.0
        for hA,hB in zip(sk_hashesA, sk_hashesB):
            sum_min += min(hA,hB)
        mean_min = sum_min/m
        union_size_est = (1 / mean_min - 1)
        sum_k = 0.0
        for ha,hb,va,vb in zip(sk_hashesA, sk_hashesB, sk_valuesA, sk_valuesB):
            if ha==hb:
                sum_k += (va*vb)
        ip_est = union_size_est * (sum_k/m)
        return ip_est

    def inner_product(self, other: 'QMHSketch') -> float:
        return self.inner_product_numba(self.sk_hashes, self.sk_values, other.sk_hashes, other.sk_values)


class MH():
    def __init__(self, sketch_size: int, seed: int) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed

    def sketch(self, vector: np.ndarray) -> QMHSketch:
        sk_hashes, sk_values = hash_kwise(vector, self.seed, dimension_num=self.sketch_size)
        return QMHSketch(sk_hashes, sk_values, self.sketch_size)
    
    

def hash_kwise(vector, seed, dimension_num=1, k_wise=4, PRIME=2147483587):
        # if input is vector, treat index of values as keys
        keys = [i for i,v in enumerate(vector) if v != 0]
        values = [v for v in vector if v != 0]
        keys = np.array(keys, dtype=np.int32)
        values = np.array(values)
        return hash_kwise_kv(keys, values, seed, dimension_num, k_wise, PRIME)
    

def hash_kwise_kv(keys, values, seed, dimension_num=1, k_wise=4, PRIME=2147483587):
    np.random.seed(seed)
    hash_parameters = np.random.randint(1, PRIME, (dimension_num, k_wise))
    hash_kwise = 0
    for exp in range(k_wise):
        hash_kwise += np.dot(np.transpose(np.array([keys])**exp), np.array([np.transpose(hash_parameters[:, exp])]))
    hash_kwise = np.mod(hash_kwise, PRIME)/PRIME
    if dimension_num == 1:
        # Reshape the hash values as a 1D array
        hashes = hash_kwise.reshape(hash_kwise.shape[0],)
    else:
        # Find the minimum hash value for each column
        hashes = np.min(hash_kwise, axis=0)
        positions = np.argmin(hash_kwise, axis=0)
        values = values[positions]
    return hashes, values