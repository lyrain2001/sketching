import argparse
import numpy as np
import scipy.signal
import time

def generate_binary_data(size):
    """Generate binary data."""
    return np.random.randint(0, 2, size)

def generate_normal_data(size):
    """Generate normally distributed data."""
    return np.random.randn(size)

def correlate_and_measure_time(data1, data2, method):
    """Perform cross-correlation and measure time."""
    start_time = time.time()
    scipy.signal.correlate(data1, data2, method=method)
    return time.time() - start_time

def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Cross-correlation efficiency analysis.")
    parser.add_argument("--size", type=int, default=1000, help="Size of the synthetic data.")
    parser.add_argument("--method", choices=['auto', 'direct', 'fft'], default='auto', help="Method for cross-correlation.")
    parser.add_argument("--data_type", choices=['binary', 'normal'], default='binary', help="Type of synthetic data (binary or normal).")

    args = parser.parse_args()

    # Generate synthetic data
    size = args.size
    data_type = args.data_type
    if data_type == 'binary':
        data1 = generate_binary_data(size)
        data2 = generate_binary_data(size)
    else:
        data1 = generate_normal_data(size)
        data2 = generate_normal_data(size)

    # Perform cross-correlation and measure time
    method = args.method
    time_taken = correlate_and_measure_time(data1, data2, method)
    print(f"Method: {method}, Time Taken: {time_taken} seconds")

if __name__ == "__main__":
    main()