import numpy as np
import scipy.signal
import time

def generate_binary_data(size):
    """Generate synthetic binary data."""
    return np.random.randint(0, 2, size)

def correlate_and_measure_time(data1, data2, method):
    """Perform cross-correlation and measure time."""
    start_time = time.time()
    scipy.signal.correlate(data1, data2, method=method)
    return time.time() - start_time

def main():
    # Generate synthetic binary data
    size = 1000  # You can adjust this size
    data1 = generate_binary_data(size)
    data2 = generate_binary_data(size)

    # Methods to test
    methods = ['direct', 'fft']

    # Measure and print time taken by each method
    for method in methods:
        time_taken = correlate_and_measure_time(data1, data2, method)
        print(f"Method: {method}, Time Taken: {time_taken} seconds")

if __name__ == "__main__":
    main()
