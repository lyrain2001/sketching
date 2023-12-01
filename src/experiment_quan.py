import argparse
import numpy as np
import time
import csv

try:
    from .data_generator import *
    from .simHash import *
    from .simHashFunction import *
    from .ups_q import *
except ImportError:
    from data_generator import *
    from simHash import *
    from simHashFunction import *
    from ups_q import *
    

def args_from_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-overlap", "--overlap", default=0.1,
        help="overlap ratio of 2 vectors", type=float)
    parser.add_argument("-outlier", "--outlier", default=0,
        help="outlier ratio of the vector", type=float)
    parser.add_argument("-zeroes", "--zeroes", default=0.8,
        help="zero ratio of the vector", type=float)
    parser.add_argument("-sketch_methods", "--sketch_methods",
        help="sketch methods to run", type=str)
    parser.add_argument("-vector_size", "--vector_size", default=10000,
        help="original vector size", type=int)
    parser.add_argument("-sketch_size", "--sketch_size", default=1000,
        help="expected sketch size", type=int)
    parser.add_argument("-storage_size", "--storage_size", default=0,
        help="expected storage size", type=int)
    parser.add_argument("-iterations", "--iterations", default=1,
        help="number of iterations", type=int)
    parser.add_argument("-log", "--log", default=False,
        help="log the result", type=bool)
    parser.add_argument("-log_name", "--log_name", default="", 
        help= "log name", type=str)
    args = parser.parse_args()
    assert args.sketch_methods is not None, "sketch_methods is missing"
    return args

if __name__ == "__main__":
    args = args_from_parser()
    overlap_ratio, outlier_ratio, zeroes_ratio, sketch_methods, vector_size, sketch_size, storage_size, iterations, log, log_name = args.overlap, args.outlier, args.zeroes, args.sketch_methods, args.vector_size, args.sketch_size, args.storage_size, args.iterations, args.log, args.log_name
    # Initialize the data generator
    errors = []
    time_start = time.time()
    for i in range(iterations):
        generator = DataGenerator(vector_size, zeroes_ratio, overlap_ratio, outlier_ratio)
        vector_a, vector_b = generator.generate_pair()
        seed = int((time.time() * 1000) % 4294967295)  # '4294967295' is the maximum value for a 32-bit integer.
        # generate condition for different sketch methods
        if sketch_methods == "SimHash":
            if storage_size != 0:
                sketch_size = int((storage_size * 64 - 128) / 2)
            sh = SimHash(sketch_size, vector_size)
            sketch_a = sh.sketch(vector_a)
            sketch_b = sh.sketch(vector_b)
        elif sketch_methods == "SimHashFunction":
            if storage_size != 0:
                sketch_size = int((storage_size * 64 - 128) / 2)
            sh = SimHashFunction(sketch_size, vector_size, vector_a, vector_b)
        else:
            raise ValueError("sketch_methods is not valid")
        
        inner_product = vector_a.dot(vector_b)
        
        if sketch_methods != "SimHashFunction":
            inner_product_sketch = sketch_a.inner_product(sketch_b)
        else:
            inner_product_sketch = sh.sketch()
            
        error = np.abs(inner_product - inner_product_sketch) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        errors.append(error)

    time_end = time.time()
    it_time = (time_end - time_start) / iterations
    
    if log:
        with open(log_name, 'a', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=['storage_size', 'mean', 'std', 'time'])
            csvwriter.writerow({"storage_size": storage_size, "mean": np.mean(errors), "std": np.std(errors), "time": it_time})
    
    print("Average relative error: {}".format(np.mean(errors)))
    print("Standard deviation of relative error: {}".format(np.std(errors)))
    print("Time elapsed: {}".format(it_time))
    