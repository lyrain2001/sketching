import pandas as pd
import matplotlib.pyplot as plt

def plot_average_error(sh, jl, ps, shm):

    plt.figure(figsize=(8,6))
    plt.title('Comparison of Average Relative Error')
    plt.xlabel('Sketch Size')
    plt.ylabel('Average Relative Error')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    errorbar_opts = {'elinewidth': 2, 'capsize': 5, 'capthick': 2}  # Adjust as necessary

    plt.errorbar(sh.iloc[:,0], sh.iloc[:,1], yerr=sh.iloc[:,2], label='SimHash', linestyle='-', marker='o', **errorbar_opts)
    plt.errorbar(jl.iloc[:,0], jl.iloc[:,1], yerr=jl.iloc[:,2], label='JL', linestyle='-', marker='s', **errorbar_opts)
    plt.errorbar(ps.iloc[:,0], ps.iloc[:,1], yerr=ps.iloc[:,2], label='PrioritySampling', linestyle='-', marker='^', **errorbar_opts)
    plt.errorbar(shm.iloc[:,0], shm.iloc[:,1], yerr=shm.iloc[:,2], label='SimHashM (m = n)', linestyle='-', marker='*', **errorbar_opts)

    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig('./results/sketch_average_error.png')
    
def plot_time(sh, jl, ps, shm):
    plt.figure(figsize=(8,6))
    plt.title('Comparison of Time (s)')
    plt.xlabel('Sketch Size')
    plt.ylabel('Time (s)')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.plot(sh.iloc[:,0], sh.iloc[:,3], label='SimHash', linestyle='-', marker='o')
    plt.plot(jl.iloc[:,0], jl.iloc[:,3], label='JL', linestyle='-', marker='s')
    plt.plot(ps.iloc[:,0], ps.iloc[:,3], label='PrioritySampling', linestyle='-', marker='^')
    plt.plot(shm.iloc[:,0], shm.iloc[:,3], label='SimHashM (m = n)', linestyle='-', marker='*')

    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig('./results/sketch_time.png')


sh = pd.read_csv('./results/sketch_SimHash.csv')
jl = pd.read_csv('./results/sketch_JL.csv')
ps = pd.read_csv('./results/sketch_PrioritySampling.csv')
shm = pd.read_csv('./results/sketch_SimHashM.csv')


plot_average_error(sh, jl, ps, shm)
plot_time(sh, jl, ps, shm)