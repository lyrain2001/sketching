import numpy as np
import random
from numba import njit
from numba.typed import List

# 
# Weighted MinHash sketch
# 
class WMHSketch():
    def __init__(self, sk_hashes: np.ndarray, sk_values: np.ndarray, vecotr_l2: float, sketch_size: int, p: float) -> None:
        self.sk_hashes: np.ndarray = sk_hashes
        self.sk_values: np.ndarray = sk_values
        self.vector_l2: float = vecotr_l2
        self.sketch_size: int = sketch_size
        self.p = p

    def inner_product(self, other: 'WMHSketch') -> float:
        mean_min = np.mean([min(hA, hB) for hA, hB in zip(self.sk_hashes, other.sk_hashes)])
        union_size_est = self.p * (1 / mean_min - 1) # p = 1/L
        sum_m = sum([(va * vb) / min(va ** 2, vb ** 2) for ha, hb, va, vb in
                    zip(self.sk_hashes, other.sk_hashes, self.sk_values, other.sk_values) if ha == hb])
        ip_est = self.vector_l2 * other.vector_l2 * union_size_est * (sum_m/self.sketch_size)
        return ip_est

class WMH():
    def __init__(self, sketch_size: int, seed: int, p: float=1e-7) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed
        self.p = p
        self.L = 1/p #  integer discretization parameter L for WMH block size: 1/p

    def sketch(self, vector: np.ndarray) -> WMHSketch:
        vector_l2 = np.linalg.norm(vector, ord=2)
        tilte_a = self.vector_rounding(vector/vector_l2, self.L)
        tilte_a_nonzeroIndex = np.nonzero(tilte_a)[0]
        tilte_a_repeat = [(v**2)*self.L for v in tilte_a]
        
        tilte_a_repeat_numba = List(tilte_a_repeat)
        all_hashes = self.sketch_geometric_numba(tilte_a_nonzeroIndex, tilte_a_repeat_numba, self.sketch_size, self.seed)
        
        all_min_indices = np.argmin(all_hashes, axis=0)
        all_min_nonzeroIndex = tilte_a_nonzeroIndex[all_min_indices]
        sk_values = tilte_a[all_min_nonzeroIndex]
        sk_hashes = np.min(all_hashes, axis=0)
        return WMHSketch(sk_hashes, sk_values, vector_l2, self.sketch_size, self.p)
    
    
    @staticmethod
    def vector_rounding(z, L):
        # Step 1: Compute the rounded values for all elements in bv_z
        tilde_z = np.sign(z) * np.sqrt(np.floor(z**2 * L) / L)
        print(f"tilde_z: {tilde_z[np.nonzero(tilde_z)]}")
        # Step 2: Find the index i* with maximum absolute value in bv_z
        i_star = np.argmax(np.abs(z))
        # Step 3: Adjust bv_tilde_z[i*] to ensure it's a unit vector (norm = 1)
        delta = 1 - np.linalg.norm(tilde_z)**2
        tilde_z[i_star] = np.sign(z[i_star]) * np.sqrt(tilde_z[i_star]**2 + delta)
        return tilde_z
    
    @staticmethod
    @njit(parallel=True)
    def sketch_geometric_numba(vec_norm_nonzeroIndex, vec_norm2_repeat, sample_size, seed):
        vec_hash = []
        for ind_ind in range(vec_norm_nonzeroIndex.shape[0]):
            ind = vec_norm_nonzeroIndex[ind_ind]
            k = ind + 1
            m_range = vec_norm2_repeat[ind]
            sub_hash = []
            for j in range(sample_size):
                random.seed(k * (1000000) + seed * sample_size + j)
                m = 1.0
                h_ = 1.0
                while m <= m_range:
                    u = random.uniform(0.0, 1.0)
                    h_ *= u
                    v = random.uniform(0.0, 1.0)
                    g = np.floor(np.log(v) / np.log(1.0 - h_))
                    m = m + 1.0 + g
                sub_hash.append(h_)
            vec_hash.append(sub_hash)
        vec_hash = np.array(vec_hash)
        return vec_hash