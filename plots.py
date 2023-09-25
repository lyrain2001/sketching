import pandas as pd
import matplotlib.pyplot as plt

def plot_average_error(sh, jl, ps):

    plt.figure(figsize=(8,6))
    plt.title('Comparison of Average Relative Error')
    plt.xlabel('Sketch Size')
    plt.ylabel('Average Relative Error')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    errorbar_opts = {'elinewidth': 2, 'capsize': 5, 'capthick': 2}  # Adjust as necessary

    plt.errorbar(sh.iloc[:,0], sh.iloc[:,1], yerr=sh.iloc[:,2], label='SimHash', linestyle='-', marker='o', **errorbar_opts)
    plt.errorbar(jl.iloc[:,0], jl.iloc[:,1], yerr=jl.iloc[:,2], label='JL', linestyle='-', marker='s', **errorbar_opts)
    plt.errorbar(ps.iloc[:,0], ps.iloc[:,1], yerr=ps.iloc[:,2], label='PrioritySampling', linestyle='-', marker='^', **errorbar_opts)

    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig('average_error.png')
    
def plot_time(sh, jl, ps):
    plt.figure(figsize=(8,6))
    plt.title('Comparison of Time')
    plt.xlabel('Sketch Size')
    plt.ylabel('Time')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.plot(sh.iloc[:,0], sh.iloc[:,2], label='SimHash', linestyle='-', marker='o')
    plt.plot(jl.iloc[:,0], jl.iloc[:,2], label='JL', linestyle='-', marker='s')
    plt.plot(ps.iloc[:,0], ps.iloc[:,2], label='PrioritySampling', linestyle='-', marker='^')

    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig('time.png')


sh = pd.read_csv('./results/SimHash.csv')
jl = pd.read_csv('./results/JL.csv')
ps = pd.read_csv('./results/PrioritySampling.csv')

plot_average_error(sh, jl, ps)
plot_time(sh, jl, ps)