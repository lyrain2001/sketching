import numpy as np

class BinaryDataGenerator:
    def __init__(self, length, zero_ratio, overlap_ratio):
        if not (0 <= zero_ratio <= 1):
            raise ValueError("zero_ratio must be between 0 and 1.")
        if not (0 <= overlap_ratio <= 1):
            raise ValueError("overlap_ratio must be between 0 and 1.")
        if not (length > 0):
            raise ValueError("length must be a positive integer.")
        
        self.length = length
        self.zero_count = int(self.length * zero_ratio)
        self.one_count = length - self.zero_count
        self.overlap_count = int(self.length * overlap_ratio)
    
    def generate_vector(self):
        # normal_part is either 0 or 1
        normal_part = np.random.randint(0, 2, self.one_count)
        zero_part = np.zeros(self.zero_count)
        vector = np.concatenate([normal_part, zero_part])
        
        np.random.shuffle(vector)
        return vector.astype(int)
    
    def generate_pair(self):
        # Initialize the first vector
        vector_1 = self.generate_vector()
        
        # Initialize the second vector with overlapping and non-overlapping parts
        overlap_part = vector_1[:self.overlap_count].copy()
        # print("overlap_count:", self.overlap_count)
        non_overlap_part = self.generate_vector()[self.overlap_count:]
        vector_2 = np.concatenate([overlap_part, non_overlap_part])
        
        return vector_1, vector_2


# Example Usage
length = 100
zero_ratio = 0
overlap_ratio = 0.5  # 50% overlap of non-zero elements

generator = BinaryDataGenerator(length, zero_ratio, overlap_ratio)

vector_1, vector_2 = generator.generate_pair()