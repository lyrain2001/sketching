import argparse
import numpy as np
import csv

# try:
#     from .binary_generator import *
#     from .mh_q import *
#     from .simHash import *
#     from .prioritySampling import *
# except ImportError:
from binary_generator import *
from mh_q import *
from simHash import *
from prioritySampling import *
    
def args_from_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-overlap", "--overlap", default=0.1,
        help="overlap ratio of 2 vectors", type=float)
    parser.add_argument("-zeroes", "--zeroes", default=0,
        help="zero ratio of the vector", type=float)
    parser.add_argument("-vector_size", "--vector_size", default=10000,
        help="original vector size", type=int)
    parser.add_argument("-sketch_size", "--sketch_size", default=1000,
        help="expected sketch size", type=int)
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
    overlap_ratio, zeroes_ratio, vector_size, sketch_size, iterations, log, log_name = args.overlap, args.zeroes, args.vector_size, args.sketch_size, args.iterations, args.log, args.log_name
    # Initialize the data generator
    sh_errors = []
    qmh_errors = []
    ups_errors = []
    
    print("sketch_size: ", sketch_size)
    
    for i in range(iterations):
        generator = BinaryDataGenerator(vector_size, zeroes_ratio, overlap_ratio)
        vector_a, vector_b = generator.generate_pair()
        seed = int((time.time() * 1000) % 4294967295)  # '4294967295' is the maximum value for a 32-bit integer.
        
        t1 =  time.time()
        
        sh = SimHash(sketch_size, vector_size, seed)
        sh_a = sh.sketch(vector_a)
        sh_b = sh.sketch(vector_b)
        sh_est = sh_a.inner_product(sh_b)
        
        t2 = time.time()
        sh_time = t2 - t1
        
        qmh = QMH(sketch_size, seed)
        qmh_a = qmh.sketch(vector_a)
        qmh_b = qmh.sketch(vector_b)
        qmh_est = qmh_a.inner_product(qmh_b)
        
        t3 = time.time()
        qmh_time = t3 - t2
        
        ups = UnweightedPrioritySampling(int(sketch_size/1.5), vector_size)
        ups_a = ups.sketch(vector_a)
        ups_b = ups.sketch(vector_b)
        ups_est = ups_a.inner_product(ups_b)
        
        t4 = time.time()
        ups_time = t4 - t3
        
        
        inner_product = vector_a.dot(vector_b)
        sh_error = np.abs(inner_product - sh_est) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        qmh_error = np.abs(inner_product - qmh_est) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        ups_error = np.abs(inner_product - ups_est) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        
        sh_errors.append(sh_error)
        qmh_errors.append(qmh_error)
        ups_errors.append(ups_error)
        # print("sh_error: ", sh_error, "time: ", sh_time)
        # print("qmh_error: ", qmh_error, "time: ", qmh_time)
    
    if log:
        with open(log_name + "sh.csv", 'a', newline='') as csvfile:
                csvwriter = csv.DictWriter(csvfile, fieldnames=['sketch_size', 'mean', 'std', 'time'])
                csvwriter.writerow({"sketch_size": sketch_size, "mean": np.mean(sh_errors), "std": np.std(sh_errors), "time": sh_time})
                
        with open(log_name + "qmh.csv", 'a', newline='') as csvfile:
                csvwriter = csv.DictWriter(csvfile, fieldnames=['sketch_size', 'mean', 'std', 'time'])
                csvwriter.writerow({"sketch_size": sketch_size, "mean": np.mean(qmh_errors), "std": np.std(qmh_errors), "time": qmh_time})
                
        with open(log_name + "ups.csv", 'a', newline='') as csvfile:
                csvwriter = csv.DictWriter(csvfile, fieldnames=['sketch_size', 'mean', 'std', 'time'])
                csvwriter.writerow({"sketch_size": sketch_size, "mean": np.mean(ups_errors), "std": np.std(ups_errors), "time": ups_time})
    
    
    print("sh_error: ", np.mean(sh_errors), "time: ", sh_time)
    print("qmh_error: ", np.mean(qmh_errors), "time: ", qmh_time)
    print("ups_error: ", np.mean(ups_errors), "time: ", ups_time)


if __name__ == "__main__":
    main()