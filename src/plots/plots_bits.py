import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def bit_bar(data):
    # Set the style for the plots
    sns.set(style="whitegrid")

    # Bar Chart: Comparing original_bits, sh_bits, and ps_bits for different sketch_sizes
    plt.figure(figsize=(8, 6))
    plt.bar(data['sketch_size'] - 100, data['original_bits'], width=100, label='Original Bits')
    plt.bar(data['sketch_size'], data['sh_bits'], width=100, label='SH Bits')
    plt.bar(data['sketch_size'] + 100, data['ps_bits'], width=100, label='PS Bits')

    plt.xlabel('Sketch Size')
    plt.ylabel('Bits')
    plt.title('Comparison of Bits (0.8 zeroes, [-1, 1], 0.1 olp)')
    plt.legend()

    # Show the plot
    plt.savefig('./results/bit_bar.pdf')

def bit_loss(data):
    # Calculate bit loss as a percentage
    data['sh_bit_loss'] = (1 - data['sh_bits'] / data['original_bits']) * 100
    data['ps_bit_loss'] = (1 - data['ps_bits'] / data['original_bits']) * 100

    # Plotting the bit loss
    plt.figure(figsize=(8, 6))

    plt.plot(data['sketch_size'], data['sh_bit_loss'], label='SH Bit Loss', marker='o')
    plt.plot(data['sketch_size'], data['ps_bit_loss'], label='PS Bit Loss', marker='x')

    plt.xlabel('Sketch Size')
    plt.ylabel('Bit Loss (%)')
    plt.title('Comparison of Bit Loss (0.8 zeroes, [-1, 1], 0.1 olp)')
    plt.legend()

    plt.savefig('./results/bit_loss.pdf')


# Load the data from the CSV file
file_path = './results/bit_loss.csv'
data = pd.read_csv(file_path)
data.columns = ['sketch_size', 'original_bits', 'sh_bits', 'ps_bits']
bit_bar(data)
bit_loss(data)
