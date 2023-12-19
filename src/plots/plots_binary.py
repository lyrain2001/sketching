import pandas as pd
import matplotlib.pyplot as plt

def plot_average_error(sh, qmh, ups):

    plt.figure(figsize=(5,4))
    plt.title('Comparison of Average Relative Error')
    plt.xlabel('Sketch (Storage) Size')
    plt.ylabel('Scaled Average Error')
    plt.grid(False)  # Remove the grid lines

    errorbar_opts = {'elinewidth': 1, 'capsize': 3, 'capthick': 1, 'alpha': 0.9}  # Adjust the alpha for denser look

    x_axis = [1000, 2000, 3000, 4000, 5000]

    plt.xticks(x_axis)

    plt.errorbar(x_axis, qmh.iloc[:,1], label='1-bit MinHash', linestyle='-', marker='s', color='#56B4E9', **errorbar_opts)
    plt.errorbar(x_axis, sh.iloc[:,1], label='SimHash', linestyle='-', marker='o', color='#E69F00', **errorbar_opts)
    plt.errorbar(x_axis, ups.iloc[:,1], label='PS-uniform', linestyle='-', marker='^', color='#009E73', **errorbar_opts)

    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig('../../results/binary_zero/error_zero_1.pdf')


def plot_time(sh, qmh, ups):
    plt.figure(figsize=(5,4))
    plt.title('Comparison of Time (s)')
    plt.xlabel('Sketch (Storage) Size')
    plt.ylabel('Time (s)')
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.grid(False)
    
    x_axis = [1000, 2000, 3000, 4000, 5000]
    
    plt.xticks(x_axis)

    plt.plot(x_axis, qmh.iloc[:,3], label='1-bit MinHash', linestyle='-', marker='s', color='#56B4E9')
    plt.plot(x_axis, sh.iloc[:,3], label='SimHash', linestyle='-', marker='o', color='#E69F00')
    plt.plot(x_axis, ups.iloc[:,3], label='PS-uniform', linestyle='-', marker='^', color='#009E73')
    
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig('../../results/binary_zero/time_zero_1.pdf')


sh = pd.read_csv('../../results/binary_zero/1_sh.csv', header=None)
qmh = pd.read_csv('../../results/binary_zero/1_qmh.csv', header=None)
ups = pd.read_csv('../../results/binary_zero/1_ups.csv', header=None)


plot_average_error(sh, qmh, ups)
plot_time(sh, qmh, ups)