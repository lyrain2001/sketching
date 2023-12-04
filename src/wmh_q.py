import numpy as np
import random
from numba import njit, prange
from numba.typed import List

# 
# Quantized Weighted MinHash sketch
# 
class QWMHSketch():
    def __init__(self, sk_hashes: np.ndarray, sk_values: np.ndarray, sk_signs: np.ndarray, vecotr_l: float, sketch_size: int, p: float) -> None:
        self.sk_hashes: np.ndarray = sk_hashes
        self.sk_values: np.ndarray = sk_values
        self.sk_signs: np.ndarray = sk_signs
        self.vector_l: float = vecotr_l
        self.sketch_size: int = sketch_size
        self.p = p

    def inner_product(self, other: 'QWMHSketch') -> float:
        r = [max(sA, sB) * signA * signB for (sA, signA), (sB, signB) in zip(zip(self.sk_values, self.sk_signs), zip(other.sk_values, other.sk_signs))]

        mean = np.mean([1 if ha == hb else 0 for ha, hb in zip(self.sk_hashes, other.sk_hashes)])
        print(f"mean: {mean}")
        union_size_est = self.p * (1 / mean)  # p = 1/L
        print(f"union_size_est: {union_size_est}")
        ip_est = self.vector_l * other.vector_l * union_size_est / self.sketch_size * (np.sum(r * self.sk_signs * other.sk_signs) / self.sketch_size)
        
        # mean_union_size_est = self.p * (1 / np.sum([1 if ha == hb else 0 for ha, hb in zip(self.sk_hashes, other.sk_hashes)]))
        # ip_est = self.vector_l * other.vector_l * mean_union_size_est * (np.sum(r * self.sk_signs * other.sk_signs) / self.sketch_size)
        
        return ip_est


class QWMH_L1():
    def __init__(self, sketch_size: int, seed: int, p: float=1e-7) -> None:
        self.sketch_size: int = sketch_size
        self.seed: int = seed
        self.p = p
        self.L = 1/p #  integer discretization parameter L for WMH block size: 1/p

    def sketch(self, vector: np.ndarray) -> QWMHSketch:        
        vector_l1 = np.linalg.norm(vector, ord=1)
        tilte_a = self.vector_rounding(vector/vector_l1, self.L)
        tilte_a_nonzeroIndex = np.nonzero(tilte_a)[0]
        # bar_a without zeroes
        tilte_a_repeat = [v*self.L for v in tilte_a]
        
        # ind = tilte_a_nonzeroIndex[0]
        # print(f"tilte_a_repeat[ind]: {tilte_a_repeat[ind]}")
        
        tilte_a_repeat_numba = List(tilte_a_repeat)
        all_hashes = self.sketch_geometric_numba(tilte_a_nonzeroIndex, tilte_a_repeat_numba, self.sketch_size, self.seed)
        # print(f"all_hashes: {all_hashes}")
        all_ps = np.random.uniform(0, 1, self.sketch_size)
        
        all_min_indices = np.argmin(all_hashes, axis=0)
        all_min_nonzeroIndex = tilte_a_nonzeroIndex[all_min_indices]
        a_values = tilte_a[all_min_nonzeroIndex]
        # print(f"a_values: {a_values}")
        
        # sk_hashes = all_signs[all_min_nonzeroIndex]
        sk_hashes = np.random.choice([-1, 1], size=self.sketch_size)
        sk_values = np.where(all_ps < a_values, 1, 0)
        sk_signs = np.sign(a_values)
        
        return QWMHSketch(sk_hashes, sk_values, sk_signs, vector_l1, self.sketch_size, self.p)
    
    
    @staticmethod
    def vector_rounding(z, L):
        # Step 1: Compute the rounded values for all elements in bv_z
        tilde_z = np.sign(z) * np.floor(np.abs(z) * L) / L
        # Step 2: Find the index i* with maximum absolute value in bv_z
        i_star = np.argmax(np.abs(z))
        # Step 3: Adjust bv_tilde_z[i*] to ensure it's a unit vector (norm = 1)
        delta = 1 - np.linalg.norm(tilde_z, ord=1)
        tilde_z[i_star] = np.sign(z[i_star]) * (tilde_z[i_star] + delta)
        return tilde_z

    @staticmethod
    @njit(parallel=True)
    def sketch_geometric_numba(vec_norm_nonzeroIndex, vec_norm_repeat, sample_size, seed):
        vec_hash = []
        for ind_ind in range(vec_norm_nonzeroIndex.shape[0]):
            # non-zero index
            ind = vec_norm_nonzeroIndex[ind_ind]
            k = ind + 1
            # tilda_a value at ind
            m_range = vec_norm_repeat[ind]
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