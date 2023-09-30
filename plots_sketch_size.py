import pandas as pd
import matplotlib.pyplot as plt

def plot_average_error(sh, jl, ps):

    plt.figure(figsize=(8,6))
    plt.title('Comparison of Average Relative Error (0.8 zeroes, [-1, 1], 0.01 olp)')
    plt.xlabel('Sketch Size')
    plt.ylabel('Average Relative Error')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    errorbar_opts = {'elinewidth': 2, 'capsize': 5, 'capthick': 2} 
    
    x_axis = [1000, 2000, 3000, 4000, 5000]
    print(sh.iloc[:,1].shape)

    plt.errorbar(x_axis, sh.iloc[:,1], yerr=sh.iloc[:,2], label='SimHash', linestyle='-', marker='o', **errorbar_opts)
    plt.errorbar(x_axis, jl.iloc[:,1], yerr=jl.iloc[:,2], label='JL', linestyle='-', marker='s', **errorbar_opts)
    plt.errorbar(x_axis, ps.iloc[:,1], yerr=ps.iloc[:,2], label='PrioritySampling', linestyle='-', marker='^', **errorbar_opts)

    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig('./results/sketch_error_olp0.01.png')
    
def plot_time(sh, jl, ps):
    plt.figure(figsize=(8,6))
    plt.title('Comparison of Time (s)')
    plt.xlabel('Sketch Size')
    plt.ylabel('Time (s)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    x_axis = [1000, 2000, 3000, 4000, 5000]

    plt.plot(x_axis, sh.iloc[:,3], label='SimHash', linestyle='-', marker='o')
    plt.plot(x_axis, jl.iloc[:,3], label='JL', linestyle='-', marker='s')
    plt.plot(x_axis, ps.iloc[:,3], label='PrioritySampling', linestyle='-', marker='^')

    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig('./results/sketch_time_olp0.01.png')


sh = pd.read_csv('./results/sketch_size_olp0.01_SimHash.csv', header=None)
jl = pd.read_csv('./results/sketch_size_olp0.01_JL.csv', header=None)
ps = pd.read_csv('./results/sketch_size_olp0.01_PrioritySampling.csv', header=None)


plot_average_error(sh, jl, ps)
plot_time(sh, jl, ps)