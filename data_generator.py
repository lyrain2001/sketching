import numpy as np

class DataGenerator:
    def __init__(self, length, zero_ratio, overlap_ratio, outlier_ratio):
        if not (0 <= zero_ratio <= 1):
            raise ValueError("zero_ratio must be between 0 and 1.")
        if not (0 <= overlap_ratio <= 1):
            raise ValueError("overlap_ratio must be between 0 and 1.")
        if not (0 <= outlier_ratio <= 1):
            raise ValueError("outlier_ratio must be between 0 and 1.")
        if not (length > 0):
            raise ValueError("length must be a positive integer.")
        
        self.length = length
        self.zero_count = int(length * zero_ratio)
        self.one_count = length - self.zero_count
        self.overlap_count = int(self.one_count * overlap_ratio)
        self.outlier_count = int(self.one_count * outlier_ratio)
        self.outlier_values = [5, 7, 9]
    
    def generate_vector(self, with_outliers=True):
        normal_count = self.one_count - (self.outlier_count if with_outliers else 0)
        normal_part = np.random.rand(normal_count)
        zero_part = np.zeros(self.zero_count)
        
        if with_outliers:
            outlier_part = np.random.choice(self.outlier_values, self.outlier_count)
            vector = np.concatenate([normal_part, zero_part, outlier_part])
        else:
            vector = np.concatenate([normal_part, zero_part])
        
        np.random.shuffle(vector)
        return vector
    
    def generate_pair(self):
        # Initialize the first vector
        vector_1 = self.generate_vector()
        
        # Initialize the second vector with overlapping and non-overlapping parts
        overlap_part = vector_1[:self.overlap_count].copy()
        non_overlap_part = self.generate_vector(with_outliers=False)[self.overlap_count:]
        vector_2 = np.concatenate([overlap_part, non_overlap_part])
        
        return vector_1, vector_2


# Example Usage
length = 2000
zero_ratio = 0.5
overlap_ratio = 0.5  # 50% overlap of non-zero elements
outlier_ratio = 0.1  # 10% of non-zero elements are outliers

generator = DataGenerator(length, zero_ratio, overlap_ratio, outlier_ratio)

vector_1, vector_2 = generator.generate_pair()

# If you want to print the vectors, uncomment the lines below
# print("Vector 1:", vector_1)
# print("Vector 2:", vector_2)
