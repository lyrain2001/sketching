import argparse
import numpy as np
import time
import csv

try:
    from .data_generator import *
    from .simHash import *
    from .prioritySampling import *
    from .jl import *
except ImportError:
    from data_generator import *
    from simHash import *
    from prioritySampling import *
    from jl import *
    

def args_from_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-overlap", "--overlap", default=0.1,
        help="overlap ratio of 2 vectors", type=float)
    parser.add_argument("-outlier", "--outlier", default=0,
        help="outlier ratio of the vector", type=float)
    parser.add_argument("-zeroes", "--zeroes", default=0.2,
        help="zero ratio of the vector", type=float)
    parser.add_argument("-sketch_methods", "--sketch_methods",
        help="sketch methods to run", type=str)
    parser.add_argument("-vector_size", "--vector_size", default=10000,
        help="original vector size", type=int)
    parser.add_argument("-sketch_size", "--sketch_size", default=100,
        help="expected sketch size", type=int)
    parser.add_argument("-iterations", "--iterations", default=1,
        help="number of iterations", type=int)
    parser.add_argument("-log", "--log", default=False,
        help="log the result", type=bool)
    args = parser.parse_args()
    assert args.sketch_methods is not None, "sketch_methods is missing"
    return args

if __name__ == "__main__":
    args = args_from_parser()
    overlap_ratio, outlier_ratio, zeroes_ratio, sketch_methods, vector_size, sketch_size, iterations, log = args.overlap, args.outlier, args.zeroes, args.sketch_methods, args.vector_size, args.sketch_size, args.iterations, args.log
    # Initialize the data generator
    generator = DataGenerator(vector_size, zeroes_ratio, overlap_ratio, outlier_ratio)
    errors = []
    time_start = time.time()
    for i in range(iterations):
        vector_a, vector_b = generator.generate_pair()
        seed = 1
        # print("vector_a:", vector_a)
        # print("vector_b:", vector_b)
        # generate condition for different sketch methods
        if sketch_methods == "SimHash":
            sh = SimHash(sketch_size, seed)
            sketch_a = sh.sketch(vector_a)
            sketch_b = sh.sketch(vector_b)
        elif sketch_methods == "PrioritySampling":
            ps = PrioritySampling(sketch_size)
            sketch_a = ps.sketch(vector_a)
            sketch_b = ps.sketch(vector_b)
        elif sketch_methods == "JL":
            jl = JL(sketch_size, seed)
            sketch_a = jl.sketch(vector_a)
            sketch_b = jl.sketch(vector_b)
        else:
            raise ValueError("sketch_methods is not valid")
        
        inner_product = vector_a.dot(vector_b)
        inner_product_sketch = sketch_a.inner_product(sketch_b)
        error = np.abs(inner_product - inner_product_sketch) / inner_product
        errors.append(error)
        # Print the results
        # print("Inner product of the vector with itself: {}".format(inner_product))
        # print("Inner product of the vector with itself using the {} sketch: {}".format(sketch_methods, inner_product_sketch))
        # print("Relative error: {}".format(np.abs(inner_product - inner_product_sketch) / inner_product))
    time_end = time.time()
    
    if log:
        # log name = sketch method + vector size + sketch size + overlap ratio + outlier ratio + zero ratio
        # log_name = "./results/" + sketch_methods + "/" + str(vector_size) + "_" + str(sketch_size) + "_" + str(overlap_ratio) + "_" + str(outlier_ratio) + "_" + str(zeroes_ratio) + ".log"
        # with open(log_name, "a") as f:
        #     # f only writes string
        #     # change np.mean(errors) to str(np.mean(errors))
        #     f.write(str(np.mean(errors)) + "\n")
        #     f.write(str(np.std(errors)) + "\n")
        #     f.write(str(time_end - time_start) + "\n")
        
        # Writing to csv file 
        # csv_name = "./results/" + sketch_methods + "/" + str(vector_size) + "_" + str(overlap_ratio) + "_" + str(outlier_ratio) + "_" + str(zeroes_ratio) + ".csv"
        csv_name = "./results/" + sketch_methods + ".csv"
        
        with open(csv_name, 'a', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=['sketch_size', 'mean', 'std', 'time'])
            csvwriter.writerow({"sketch_size": sketch_size, "mean": np.mean(errors), "std": np.std(errors), "time": time_end - time_start})
        
        # with open(csv_name, 'w', newline='') as csvfile:
        #     fieldnames = ['sketch_size', 'mean', 'std', 'time']
        #     csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     csvwriter.writeheader()
        #     csvwriter.writerows([{"sketch_size": sketch_size, "mean": np.mean(errors), "std": np.std(errors), "time": time_end - time_start}])
    
    print("Average relative error: {}".format(np.mean(errors)))
    print("Standard deviation of relative error: {}".format(np.std(errors)))
    print("Time elapsed: {}".format(time_end - time_start))
    