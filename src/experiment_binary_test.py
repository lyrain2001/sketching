import argparse
import numpy as np
import time
import csv

try:
    from .binary_generator import *
    from .mh_q import *
    from .simHash import *
except ImportError:
    from binary_generator import *
    from mh_q import *
    from simHash import *
    

def args_from_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-overlap", "--overlap", default=0.1,
        help="overlap ratio of 2 vectors", type=float)
    parser.add_argument("-zeroes", "--zeroes", default=0,
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
    overlap_ratio, zeroes_ratio, sketch_methods, vector_size, sketch_size, storage_size, iterations, log, log_name = args.overlap, args.zeroes, args.sketch_methods, args.vector_size, args.sketch_size, args.storage_size, args.iterations, args.log, args.log_name
    # Initialize the data generator
    errors = []
    exact = []
    est = []
    time_start = time.time()
    for i in range(iterations):
        generator = BinaryDataGenerator(vector_size, zeroes_ratio, overlap_ratio)
        vector_a, vector_b = generator.generate_pair()
        # print("Union size: {}".format(len(set(np.nonzero(vector_a)[0]).union(set(np.nonzero(vector_b)[0])))))
        

        seed = int((time.time() * 1000) % 4294967295)  # '4294967295' is the maximum value for a 32-bit integer.
        # print("iteration: {}".format(i))
        # generate condition for different sketch methods
        if sketch_methods == "QuantizedMinHash":
            sh = QMH(sketch_size, seed)
            sketch_a = sh.sketch(vector_a)
            sketch_b = sh.sketch(vector_b)
            print("sketch created")
        elif sketch_methods == "SimHash":
            sh = SimHash(sketch_size, vector_size, seed)
            sketch_a = sh.sketch(vector_a)
            sketch_b = sh.sketch(vector_b)
        else:
            raise ValueError("sketch_methods is not valid")
        
        inner_product = vector_a.dot(vector_b)
        # print("consine: {}".format(inner_product / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))))
        
        inner_product_sketch = sketch_a.inner_product(sketch_b)
        # inner_product_sketch = np.mean(result)
        print("inner_product: {}".format(inner_product))
        print("inner_product_sketch: {}".format(inner_product_sketch))
        
        error = np.abs(inner_product - inner_product_sketch) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        
        errors.append(error)
        exact.append(inner_product)
        est.append(inner_product_sketch)
        print("Relative error: {}".format(error))
        # print("Inner product of the vector with itself: {}".format(inner_product))
        # print("Inner product of the vector with itself using the {} sketch: {}".format(sketch_methods, inner_product_sketch))
        # print("Relative error: {}".format(np.abs(inner_product - inner_product_sketch) / inner_product))
    time_end = time.time()
    it_time = (time_end - time_start) / iterations
    
    if log:
        with open(log_name, 'a', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=['storage_size', 'mean', 'std', 'time'])
            csvwriter.writerow({"storage_size": storage_size, "mean": np.mean(errors), "std": np.std(errors), "time": it_time})
        
    # store error list in a file, each line is an error
    with open("error.txt", "a") as f:
        for i in range(len(errors)):
            f.write(str(exact[i]) + "\t")
            f.write(str(est[i]) + "\t")
            f.write(str(errors[i]) + "\n")
        f.write("\n")
        f.write("Average relative error: {}\n".format(np.mean(errors)))
        f.write("Standard deviation of relative error: {}\n".format(np.std(errors)))
        f.write("Average time elapsed: {}\n".format(it_time))
        
    print("Average relative error: {}".format(np.mean(errors)))
    print("Standard deviation of relative error: {}".format(np.std(errors)))
    print("Time elapsed: {}".format(it_time))
    