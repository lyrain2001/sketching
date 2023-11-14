import argparse
import numpy as np
import csv
import struct

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

    
# def calculate_bit_loss(ps, sh):
#     # Calculate bit difference using XOR
#     bit_loss = np.sum(np.bitwise_xor(ps, sh))
#     return bit_loss
    
def ints_to_bit_array(ints):
    max_value = max(ints)
    bit_length = max_value.bit_length()
    return [[int(digit) for digit in format(num, f'0{bit_length}b')] for num in ints]

def float_to_bit_array(number):
    # Pack the float into 4 bytes using IEEE 754 format (single precision)
    packed = struct.pack('!f', number)
    # Unpack into an integer
    [number] = struct.unpack('!I', packed)
    # Convert the integer to a 32-bit binary representation
    return [int(bit) for bit in f'{number:032b}']

def floats_to_bit_array(floats):
    bit_array = []
    
    for number in floats:
        # Pack the float into 4 bytes using IEEE 754 format (single precision)
        packed = struct.pack('!f', number)
        # Unpack into an integer
        [number] = struct.unpack('!I', packed)
        # Convert the integer to a 32-bit binary representation and append to the bit array
        bit_array.extend([int(bit) for bit in f'{number:032b}'])

    return bit_array

def calculate_original_bit(vector_a, vector_b):
    vector_a_bit = floats_to_bit_array(vector_a)
    vector_b_bit = floats_to_bit_array(vector_b)
    return len(vector_a_bit) + len(vector_b_bit)

def calculate_sh_bit(sh_a, sh_b):
    sh_a_norm = float_to_bit_array(sh_a.norm)
    sh_b_norm = float_to_bit_array(sh_b.norm)
    sh_a_sketch = [0 if x == -1 else 1 for x in sh_a.sk_values]
    sh_b_sketch = [0 if x == -1 else 1 for x in sh_b.sk_values]
    
    return len(sh_a_norm) + len(sh_b_norm) + len(sh_a_sketch) + len(sh_b_sketch)

def calculate_ps_bit(ps_a, ps_b):
    ps_a_norm = float_to_bit_array(ps_a.norm)
    ps_b_norm = float_to_bit_array(ps_b.norm)
    ps_a_tau = float_to_bit_array(ps_a.tau)
    ps_b_tau = float_to_bit_array(ps_b.tau)
    
    ps_a_sketch = floats_to_bit_array(ps_a.sk_values)
    ps_b_sketch = floats_to_bit_array(ps_b.sk_values)
    ps_a_indices = ints_to_bit_array(ps_a.sk_indices)
    ps_b_indices = ints_to_bit_array(ps_b.sk_indices)
    
    return len(ps_a_norm) + len(ps_b_norm) + len(ps_a_tau) + len(ps_b_tau) + len(ps_a_sketch) + len(ps_b_sketch) + len(ps_a_indices) + len(ps_b_indices)


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

if __name__ == "__main__":
    args = args_from_parser()
    overlap_ratio, outlier_ratio, zeroes_ratio, vector_size, sketch_size, storage_size, iterations, log, log_name = args.overlap, args.outlier, args.zeroes, args.vector_size, args.sketch_size, args.storage_size, args.iterations, args.log, args.log_name
    
    bits = np.zeros((iterations, 3))
    for i in range(iterations):
        generator = DataGenerator(vector_size, zeroes_ratio, overlap_ratio, outlier_ratio)
        vector_a, vector_b = generator.generate_pair()
        
        sh = SimHash(sketch_size, vector_size)
        sh_a = sh.sketch(vector_a)
        sh_b = sh.sketch(vector_b)
        sh_bits = calculate_sh_bit(sh_a, sh_b)
        
        ps = PrioritySampling(sketch_size, vector_size)
        ps_a = ps.sketch(vector_a)
        ps_b = ps.sketch(vector_b)
        ps_bits = calculate_ps_bit(ps_a, ps_b)
        
        original_bits = calculate_original_bit(vector_a, vector_b)

        bits[i] = [original_bits, sh_bits, ps_bits]
    
    if log:
        with open(log_name, 'a', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=['sketch_size', 'original_bits', 'sh_bits', 'ps_bits'])
            csvwriter.writerow({'sketch_size': sketch_size, 'original_bits': np.mean(bits[:,0]), 'sh_bits': np.mean(bits[:,1]), 'ps_bits': np.mean(bits[:,2])})

    