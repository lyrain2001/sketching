import argparse
import numpy as np
import time
import csv

try:
    from .data_generator import *
    from .simHash import *
    from .prioritySampling import *
    from .jl import *
    from .unweightedPrioritySampling import *
    from .upsSimHash import *
except ImportError:
    from data_generator import *
    from simHash import *
    from prioritySampling import *
    from jl import *
    from unweightedPrioritySampling import *
    from upsSimHash import *
    

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
        # np.savetxt('vector_a.txt', vector_a, fmt='%f')
        # np.savetxt('vector_b.txt', vector_b, fmt='%f')
        seed = int((time.time() * 1000) % 4294967295)  # '4294967295' is the maximum value for a 32-bit integer.
        # print("vector_a:", vector_a)
        # print("vector_b:", vector_b)
        # generate condition for different sketch methods
        if sketch_methods == "SimHash":
            if storage_size != 0:
                sketch_size = int((storage_size * 64 - 128) / 2)
            sh = SimHash(sketch_size, vector_size)
            sketch_a = sh.sketch(vector_a)
            sketch_b = sh.sketch(vector_b)
        elif sketch_methods == "PrioritySampling":
            if storage_size != 0:
                sketch_size = int((storage_size * 64 / 2 - 64)/ (32 + 64))
            ps = PrioritySampling(sketch_size, vector_size)
            sketch_a = ps.sketch(vector_a)
            sketch_b = ps.sketch(vector_b)
        elif sketch_methods == "JL":
            if storage_size != 0:
                sketch_size = int(storage_size)
            jl = JL(sketch_size, seed)
            sketch_a = jl.sketch(vector_a)
            sketch_b = jl.sketch(vector_b)
        elif sketch_methods == "UnweightedPrioritySampling":
            if storage_size != 0:
                sketch_size = int((storage_size * 64 / 2 - 64)/ (32 + 64))
            ups = UnweightedPrioritySampling(sketch_size, vector_size)
            sketch_a = ups.sketch(vector_a)
            sketch_b = ups.sketch(vector_b)
        elif sketch_methods == "UnweightedPrioritySamplingSimHash":
            if storage_size != 0:
                sketch_size = int((storage_size * 64 / 2 - 128)/ (32 + 1))
            upssh = UnweightedPrioritySamplingSimHash(sketch_size, vector_size)
            sketch_a = upssh.sketch(vector_a)
            sketch_b = upssh.sketch(vector_b)
        else:
            raise ValueError("sketch_methods is not valid")
        
        inner_product = vector_a.dot(vector_b)
        # print("consine: {}".format(inner_product / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))))
        # print("inner_product: {}".format(inner_product))
        inner_product_sketch = sketch_a.inner_product(sketch_b)
        # print("inner_product: {}".format(inner_product))
        # print("inner_product_sketch: {}".format(inner_product_sketch))
        
        # error = np.abs(inner_product - inner_product_sketch) / inner_product
        error = np.abs(inner_product - inner_product_sketch) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        
        errors.append(error)
        # Print the results
        # print("Inner product of the vector with itself: {}".format(inner_product))
        # print("Inner product of the vector with itself using the {} sketch: {}".format(sketch_methods, inner_product_sketch))
        # print("Relative error: {}".format(np.abs(inner_product - inner_product_sketch) / inner_product))
    time_end = time.time()
    it_time = (time_end - time_start) / iterations
    
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
        # csv_name = "./results/sketch_" + sketch_methods + ".csv"
        # csv_name = "./results/" + sketch_methods + "/" + str(vector_size) + "_" + str(overlap_ratio) + "_" + str(outlier_ratio) + "_" + str(zeroes_ratio) + ".csv"
        # csv_name = "./results/storage_olp0.01_" + sketch_methods + ".csv"
        # csv_name = log_name + sketch_methods + ".csv"
        
        # with open(csv_name, 'a', newline='') as csvfile:
        #     csvwriter = csv.DictWriter(csvfile, fieldnames=['sketch_size', 'mean', 'std', 'time'])
        #     csvwriter.writerow({"sketch_size": sketch_size, "mean": np.mean(errors), "std": np.std(errors), "time": it_time})
            
        with open(log_name, 'a', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=['storage_size', 'mean', 'std', 'time'])
            csvwriter.writerow({"storage_size": storage_size, "mean": np.mean(errors), "std": np.std(errors), "time": it_time})
        
        # with open(csv_name, 'w', newline='') as csvfile:
        #     fieldnames = ['sketch_size', 'mean', 'std', 'time']
        #     csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     csvwriter.writeheader()
        #     csvwriter.writerows([{"sketch_size": sketch_size, "mean": np.mean(errors), "std": np.std(errors), "time": time_end - time_start}])
    
    print("Average relative error: {}".format(np.mean(errors)))
    print("Standard deviation of relative error: {}".format(np.std(errors)))
    print("Time elapsed: {}".format(it_time))
    