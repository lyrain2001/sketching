import pandas as pd
import matplotlib.pyplot as plt

def plot_average_error(sh, jl, ps, ups, upssh):

    plt.figure(figsize=(8,6))
    plt.title('Comparison of Average Relative Error (0.8 zeroes, [-1, 1], 0.1 olp)')
    plt.xlabel('Storage Size')
    plt.ylabel('Average Relative Error')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    errorbar_opts = {'elinewidth': 2, 'capsize': 5, 'capthick': 2} 

    plt.errorbar(sh.iloc[:,0], sh.iloc[:,1], yerr=sh.iloc[:,2], label='SimHash', linestyle='-', marker='o', **errorbar_opts)
    plt.errorbar(jl.iloc[:,0], jl.iloc[:,1], yerr=jl.iloc[:,2], label='JL', linestyle='-', marker='s', **errorbar_opts)
    plt.errorbar(ps.iloc[:,0], ps.iloc[:,1], yerr=ps.iloc[:,2], label='PrioritySampling', linestyle='-', marker='^', **errorbar_opts)
    plt.errorbar(ups.iloc[:,0], ups.iloc[:,1], yerr=ups.iloc[:,2], label='UnweightedPrioritySampling', linestyle='-', marker='*', **errorbar_opts)
    plt.errorbar(upssh.iloc[:,0], upssh.iloc[:,1], yerr=upssh.iloc[:,2], label='UnweightedPrioritySamplingSimHash', linestyle='-', marker='x', **errorbar_opts)
    
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig('./results/vec-size-1w/new_error_olp0.1.png')
    
# def plot_time(sh, jl, ps):
#     plt.figure(figsize=(8,6))
#     plt.title('Comparison of Average Time(s) (n = 10000)')
#     plt.xlabel('Storage Size')
#     plt.ylabel('Time (s)')
#     plt.grid(True, linestyle='--', alpha=0.7)

#     plt.plot(sh.iloc[:,0], sh.iloc[:,3], label='SimHash', linestyle='-', marker='o')
#     plt.plot(jl.iloc[:,0], jl.iloc[:,3], label='JL', linestyle='-', marker='s')
#     plt.plot(ps.iloc[:,0], ps.iloc[:,3], label='PrioritySampling', linestyle='-', marker='^')
#     # plt.plot(shm.iloc[:,0], shm.iloc[:,3], label='SimHashM (m = n)', linestyle='-', marker='*')

#     plt.legend(loc='upper right', frameon=True, shadow=True)
#     plt.tight_layout()

#     plt.savefig('./results/storage_time.png')


sh = pd.read_csv('./results/vec-size-1w/olp0.1_SimHash.csv', header=None)
jl = pd.read_csv('./results/vec-size-1w/olp0.1_JL.csv', header=None)
ps = pd.read_csv('./results/vec-size-1w/olp0.1_PrioritySampling.csv', header=None)
ups = pd.read_csv('./results/vec-size-1w/olp0.1_UnweightedPrioritySampling.csv', header=None)
upssh = pd.read_csv('./results/vec-size-1w/olp0.1_UnweightedPrioritySamplingSimHash.csv', header=None)


plot_average_error(sh, jl, ps, ups, upssh)
# plot_time(sh, jl, ps)