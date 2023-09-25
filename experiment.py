import argparse
import numpy as np

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
    parser.add_argument("-overlap", "--overlap", 
        help="overlap ratio of 2 vectors", type=float)
    parser.add_argument("-outlier", "--outlier", 
        help="outlier ratio of the vector", type=float)
    parser.add_argument("-zeroes", "--zeroes",
        help="zero ratio of the vector", type=float)
    parser.add_argument("-sketch_methods", "--sketch_methods",
        help="sketch methods to run", type=str)
    parser.add_argument("-vector_size", "--vector_size",
        help="original vector size", type=int)
    parser.add_argument("-skech_size", "--skech_size",
        help="expected sketch size", type=int)
    parser.add_argument("-log_name", "--log_name", 
        help="log name of the run", type=str)
    args = parser.parse_args()
    # Check if the required parameter is present
    assert args.log_name is not None, "log_name is missing"
    assert args.sketch_methods is not None, "sketch_methods is missing"
    return args

def vars_from_args(args):
    overlap_ratio = args.overlap or 0.1
    outlier_ratio = args.outlier or 0.1
    zeroes_ratio = args.zeroes or 0.1
    sketch_methods = args.sketch_methods.split("+")
    vector_size = args.vector_size or 1000
    sketch_size = args.skech_size or 100 
    log_name = args.log_name
    print("overlap_ratio:", overlap_ratio)
    print("outlier_ratio:", outlier_ratio)
    print("zeroes_ratio:", zeroes_ratio)
    print("sketch_methods", sketch_methods)
    print("vector_size:", vector_size)
    print("sketch_size:", sketch_size)
    print("log_name:", log_name)
    return overlap_ratio,outlier_ratio,zeroes_ratio,sketch_methods,vector_size,sketch_size,log_name

if __name__ == "__main__":
    args = args_from_parser()
    overlap_ratio,outlier_ratio,zeroes_ratio,sketch_methods,vector_size,sketch_size,log_name = vars_from_args(args)
    # Initialize the data generator
    generator = DataGenerator(vector_size, zeroes_ratio, overlap_ratio, outlier_ratio)
    vector_a, vector_b = generator.generate_pair()
    # print("vector_a:", vector_a)
    # print("vector_b:", vector_b)
    # generate condition for different sketch methods
    if sketch_methods == "simHash":
        sketch_a = SimHash(sketch_size).sketch(vector_a)
        sketch_b = SimHash(sketch_size).sketch(vector_b)
    elif sketch_methods == "prioritySampling":
        sketch_a = PrioritySampling(sketch_size).sketch(vector_a)
        sketch_b = PrioritySampling(sketch_size).sketch(vector_b)
    elif sketch_methods == "JL":
        sketch_a = JL(sketch_size).sketch(vector_a)
        sketch_b = JL(sketch_size).sketch(vector_b)
    
    inner_product = vector_a.dot(vector_b)
    inner_product_sketch = sketch_a.inner_product(sketch_b)
    # Print the results
    print("Inner product of the vector with itself: {}".format(inner_product))
    print("Inner product of the vector with itself using the {} sketch: {}".format(sketch_methods, inner_product_sketch))
    print("Relative error: {}".format(np.abs(inner_product - inner_product_sketch) / inner_product))