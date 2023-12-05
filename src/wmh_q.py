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
        # print(f"r: {np.sum(r)}")
        
        mean = np.mean([1 if ha == hb else 0 for ha, hb in zip(self.sk_hashes, other.sk_hashes)])
        # print(f"mean: {mean}") 
        
        # TODO
        # union_size_est = self.p * (1 / mean)  # p = 1/L
        union_size_est = 1 / mean
        
        # print(f"union_size_est: {union_size_est}")
        
        r_sum = np.sum(r * self.sk_hashes * other.sk_hashes)
        # print(f"r_sum: {r_sum}")
        
        ip_est = self.vector_l * other.vector_l * union_size_est / self.sketch_size * r_sum
    
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
        # TODO  
        vector_l1 = np.linalg.norm(vector, ord=1)
        tilte_a = self.vector_rounding(vector/vector_l1, self.L)
        # vector_l2 = np.linalg.norm(vector, ord=2)
        # tilte_a = self.vector_rounding(vector/vector_l2, self.L)
        
        # print(f"tilte_a: {tilte_a}")
        tilte_a_nonzeroIndex = np.nonzero(tilte_a)[0]
        
        # number of non-zero elements in each row in bar_a
        # TODO
        tilte_a_repeat = [np.abs(v)*self.L for v in tilte_a]
        # tilte_a_repeat = [(v**2)*self.L for v in tilte_a]
        
        # ind = tilte_a_nonzeroIndex[0]
        # print(f"tilte_a_repeat[ind]: {tilte_a_repeat[ind]}")
        
        tilte_a_repeat_numba = List(tilte_a_repeat)
        all_hashes, all_signs = self.sketch_geometric_numba(tilte_a_nonzeroIndex, tilte_a_repeat_numba, self.sketch_size, self.seed)
        # print(f"all_hashes: {all_hashes}") 
        # print(f"all_hashes.shape: {all_hashes.shape}")
        # print(f"all_signs: {all_signs}")
        np.random.seed(self.seed)
        all_ps = np.random.uniform(0, 1, self.sketch_size)
        
        all_min_indices = np.argmin(all_hashes, axis=0)
        all_min_nonzeroIndex = tilte_a_nonzeroIndex[all_min_indices]
        a_values = tilte_a[all_min_nonzeroIndex]
        # print(f"a_values: {a_values}")
        
        sk_hashes = all_signs[all_min_indices, np.arange(all_signs.shape[1])]
        # print first five corresponding all_min_indices and sk_hashes
        # print(f"all_min_indices[:5]: {all_min_indices[:10]}")
        # print(f"sk_hashes[:5]: {sk_hashes[:10]}")
        
        
        sk_values = np.where(all_ps < np.abs(a_values), 1, 0)
        # print(f"sk_values: {sk_values}")
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
    
    # @staticmethod
    # def vector_rounding(z, L):
    #     # Step 1: Compute the rounded values for all elements in bv_z
    #     tilde_z = np.sign(z) * np.sqrt(np.floor(z**2 * L) / L)
    #     # Step 2: Find the index i* with maximum absolute value in bv_z
    #     i_star = np.argmax(np.abs(z))
    #     # Step 3: Adjust bv_tilde_z[i*] to ensure it's a unit vector (norm = 1)
    #     delta = 1 - np.linalg.norm(tilde_z)**2
    #     tilde_z[i_star] = np.sign(z[i_star]) * np.sqrt(tilde_z[i_star]**2 + delta)
    #     return tilde_z

    # @staticmethod
    # @njit(parallel=True)
    # def sketch_geometric_numba(vec_norm_nonzeroIndex, vec_norm2_repeat, sample_size, seed):
    #     vec_hash = []
    #     for ind_ind in range(vec_norm_nonzeroIndex.shape[0]):
    #         ind = vec_norm_nonzeroIndex[ind_ind]
    #         k = ind + 1
    #         m_range = vec_norm2_repeat[ind]
    #         sub_hash = []
    #         for j in range(sample_size):
    #             random.seed(k * (1000000) + seed * sample_size + j)
    #             m = 1.0
    #             h_ = 1.0
    #             while m <= m_range:
    #                 u = random.uniform(0.0, 1.0)
    #                 h_ *= u
    #                 v = random.uniform(0.0, 1.0)
    #                 g = np.floor(np.log(v) / np.log(1.0 - h_))
    #                 m = m + 1.0 + g
    #             sub_hash.append(h_)
    #         vec_hash.append(sub_hash)
    #     vec_hash = np.array(vec_hash)
    #     return vec_hash
    
    @staticmethod
    @njit(parallel=True)
    def sketch_geometric_numba(vec_norm_nonzeroIndex, vec_norm_repeat, sample_size, seed):
        n = vec_norm_nonzeroIndex.shape[0]
        vec_hash = np.zeros((n, sample_size))  # Preallocated array
        vec_sign = np.zeros((n, sample_size))  # Preallocated array

        for ind_ind in prange(n):
            ind = vec_norm_nonzeroIndex[ind_ind]
            k = ind + 1
            # number of tilda_a value at ind
            m_range = vec_norm_repeat[ind]

            for j in range(sample_size):
                random.seed(k * (1000000) + seed * sample_size + j)
                t = 0
                while t == 0:
                    t = random.uniform(-1, 1)
                s_ = 1 if t > 0 else -1
                m = 1.0
                h_ = 1.0
                while m <= m_range:
                    u = random.uniform(0.0, 1.0)
                    h_ *= u
                    v = random.uniform(0.0, 1.0)
                    g = np.floor(np.log(v) / np.log(1.0 - h_))
                    m = m + 1.0 + g
                
                vec_hash[ind_ind, j] = h_
                vec_sign[ind_ind, j] = s_

        return vec_hash, vec_sign
