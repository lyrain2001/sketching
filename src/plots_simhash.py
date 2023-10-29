import pandas as pd
import matplotlib.pyplot as plt

def plot_average_error(sh, shr, shf):

    plt.figure(figsize=(8,6))
    plt.title('Comparison of Average Relative Error for Simhash(0.8 zeroes, [-1, 1], 0.1 olp)')
    plt.xlabel('Storage Size')
    plt.ylabel('Average Relative Error')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    errorbar_opts = {'elinewidth': 2, 'capsize': 5, 'capthick': 2} 

    plt.errorbar(sh.iloc[:,0], sh.iloc[:,1], yerr=sh.iloc[:,2], label='SimHash', linestyle='-', marker='o', **errorbar_opts)
    plt.errorbar(shr.iloc[:,0], shr.iloc[:,1], yerr=shr.iloc[:,2], label='SimHashRound', linestyle='-', marker='s', **errorbar_opts)
    plt.errorbar(shf.iloc[:,0], shf.iloc[:,1], yerr=shf.iloc[:,2], label='SimHashFunction', linestyle='-', marker='^', **errorbar_opts)
    
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig('./results/new_simhash_results/error.png')
    
def plot_time(sh, shr, shf):
    plt.figure(figsize=(8,6))
    plt.title('Comparison of Average Time(s) (n = 10000)')
    plt.xlabel('Storage Size')
    plt.ylabel('Time (s)')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.plot(sh.iloc[:,0], sh.iloc[:,3], label='SimHash', linestyle='-', marker='o')
    plt.plot(shr.iloc[:,0], shr.iloc[:,3], label='SimHashRound', linestyle='-', marker='s')
    plt.plot(shf.iloc[:,0], shf.iloc[:,3], label='SimHashFunction', linestyle='-', marker='^')
    # plt.plot(shm.iloc[:,0], shm.iloc[:,3], label='SimHashM (m = n)', linestyle='-', marker='*')

    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig('./results/new_simhash_results/time.png')

sh = pd.read_csv('./results/new_simhash_results/SimHash.csv', header=None)
shr = pd.read_csv('./results/new_simhash_results/SimHashRound.csv', header=None)
shf = pd.read_csv('./results/new_simhash_results/SimHashFunction.csv', header=None)


plot_average_error(sh, shr, shf)
plot_time(sh, shr, shf)