import argparse
import numpy as np
import csv
import struct

try:
    from .data_generator import *
    from .simHash import *
    from .ups_q import *
    from .wmh_q import *    
except ImportError:
    from data_generator import *
    from simHash import *
    from ups_q import *
    from wmh_q import *

def args_from_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-overlap", "--overlap", default=0.1,
        help="overlap ratio of 2 vectors", type=float)
    parser.add_argument("-outlier", "--outlier", default=0,
        help="outlier ratio of the vector", type=float)
    parser.add_argument("-zeroes", "--zeroes", default=0.8,
        help="zero ratio of the vector", type=float)
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
    return args

def main():
    args = args_from_parser()
    overlap_ratio, outlier_ratio, zeroes_ratio, vector_size, sketch_size, storage_size, iterations, log, log_name = args.overlap, args.outlier, args.zeroes, args.vector_size, args.sketch_size, args.storage_size, args.iterations, args.log, args.log_name
    sh_errors = []
    qups_errors = []
    qwmh_errors = []
    
    print("sketch_size: ", sketch_size)
    
    for i in range(iterations):
        generator = DataGenerator(vector_size, zeroes_ratio, overlap_ratio, outlier_ratio)
        vector_a, vector_b = generator.generate_pair()
        seed = int((time.time() * 1000) % 4294967295)  # '4294967295' is the maximum value for a 32-bit integer.
        
        t1 =  time.time()
        
        sh = SimHash(sketch_size, vector_size)
        sh_a = sh.sketch(vector_a)
        sh_b = sh.sketch(vector_b)
        sh_est = sh_a.inner_product(sh_b)
        
        t2 = time.time()
        sh_time = t2 - t1
        
        qups = UnweightedPrioritySamplingSimHash(sketch_size, vector_size)
        qups_a = qups.sketch(vector_a)
        qups_b = qups.sketch(vector_b)
        qups_est = qups_a.inner_product(qups_b)
        
        t3 = time.time()
        qups_time = t3 - t2
        
        qwmh = QWMH_L1(sketch_size, seed)
        qwmh_a = qwmh.sketch(vector_a)
        qwmh_b = qwmh.sketch(vector_b)
        qwmh_est = qwmh_a.inner_product(qwmh_b)
        
        t4 = time.time()
        qwmh_time = t4 - t3
        
        inner_product = vector_a.dot(vector_b)
        sh_error = np.abs(inner_product - sh_est) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        qups_error = np.abs(inner_product - qups_est) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        qwmh_error = np.abs(inner_product - qwmh_est) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        
        sh_errors.append(sh_error)
        qups_errors.append(qups_error)
        qwmh_errors.append(qwmh_error)
            
    with open(log_name + "sh.csv", 'a', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=['sketch_size', 'mean', 'std', 'time'])
            csvwriter.writerow({"sketch_size": sketch_size, "mean": np.mean(sh_errors), "std": np.std(sh_errors), "time": sh_time})
    
    with open(log_name + "qups.csv", 'a', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=['sketch_size', 'mean', 'std', 'time'])
            csvwriter.writerow({"sketch_size": sketch_size, "mean": np.mean(qups_errors), "std": np.std(qups_errors), "time": qups_time})
            
    with open(log_name + "qwmh.csv", 'a', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=['sketch_size', 'mean', 'std', 'time'])
            csvwriter.writerow({"sketch_size": sketch_size, "mean": np.mean(qwmh_errors), "std": np.std(qwmh_errors), "time": qwmh_time})
    
    print("sh_error: ", np.mean(sh_errors), "time: ", sh_time)
    print("qups_error: ", np.mean(qups_errors), "time: ", qups_time)
    print("qwmh_error: ", np.mean(qwmh_errors), "time: ", qwmh_time)


if __name__ == "__main__":
    main()