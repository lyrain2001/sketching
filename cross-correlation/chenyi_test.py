import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import time
from tqdm import tqdm
import scipy

def load_sessions(output_dir):
    manifest_path = os.path.join(output_dir, "manifest.json")

    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    print(cache.get_all_session_types())

    sessions = cache.get_session_table()
    brain_observatory_type_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]
    # brain_observatory_type_sessions = sessions[sessions["session_type"] == "functional_connectivity"]
    return cache, brain_observatory_type_sessions

def load_histograms(session):
    presentations = session.get_stimulus_table("drifting_gratings")
    # presentations = session.get_stimulus_table("drifting_gratings_contrast")
    presentations = presentations[presentations["contrast"]==0.8]
    presentations = presentations[presentations["orientation"]==90]

    # print(np.unique(presentations["contrast"].values))
    v1units = session.units[session.units["ecephys_structure_acronym"] == 'VISp']
    rlunits = session.units[session.units["ecephys_structure_acronym"] == 'VISrl']
    lmunits = session.units[session.units["ecephys_structure_acronym"] == 'VISl']
    alunits = session.units[session.units["ecephys_structure_acronym"] == 'VISal']
    pmunits = session.units[session.units["ecephys_structure_acronym"] == 'VISpm']
    amunits = session.units[session.units["ecephys_structure_acronym"] == 'VISam']

    time_step = 0.0001
    time_bins = np.arange(-0.1, 0.5 + time_step, time_step)

    v1histograms = session.presentationwise_spike_counts(
        stimulus_presentation_ids=presentations.index.values,  
        bin_edges=time_bins,
        unit_ids=v1units.index.values
    )

    rlhistograms = session.presentationwise_spike_counts(
        stimulus_presentation_ids=presentations.index.values,  
        bin_edges=time_bins,
        unit_ids=rlunits.index.values
    )
    lmhistograms = session.presentationwise_spike_counts(
        stimulus_presentation_ids=presentations.index.values,  
        bin_edges=time_bins,
        unit_ids=lmunits.index.values
    )
    amhistograms = session.presentationwise_spike_counts(
        stimulus_presentation_ids=presentations.index.values,  
        bin_edges=time_bins,
        unit_ids=amunits.index.values
    )

    pmhistograms = session.presentationwise_spike_counts(
        stimulus_presentation_ids=presentations.index.values,  
        bin_edges=time_bins,
        unit_ids=pmunits.index.values
    )
    histograms_list = [v1histograms, rlhistograms, lmhistograms, pmhistograms, amhistograms]

    return histograms_list

def correlate_and_measure_time(Si, Sj, method='auto'):
    """Perform cross-correlation and measure time."""
    start_time = time.time()
    # source: https://github.com/scipy/scipy/blob/v1.11.4/scipy/signal/_signaltools.py#L91-L288
    raw_corr = scipy.signal.correlate(Si, Sj, method=method)
    T = len(Si)
    # print(T, len(raw_corr))
    # # k = tau + T - 1
    # meani = np.mean(Si)
    # meanj = np.mean(Sj)
    # print(np.array(range(len(raw_corr))))
    # normalized_corr = np.array([raw_corr[lag] / (T - abs(lag - T + 1)) for lag in range(len(raw_corr))])
    # normalized_corr = normalized_corr / np.sqrt(meani*meanj)
    # return normalized_corr
    return raw_corr

import numpy as np
from scipy.signal import correlate

def reorganize_data(histogram):
    """
    Reorganize data from stimulus_id x unit_number x timestep to unit_number x stimulus_id x timestep or reverse.
    """
    return np.transpose(histogram, (1, 0, 2))


def create_jittered_spike_trains(histograms, jitter_window_size, time_step):
    all_spike_trains = []
    for stimu_id in np.unique(histograms["stimulus_presentation_id"]):
    
        spike_train = np.array(histograms[histograms["stimulus_presentation_id"] == stimu_id]).squeeze().T
    
        all_spike_trains.append(spike_train)
    concatenated_spike_trains = np.array(all_spike_trains)
    reorganized_spike_train = reorganize_data(concatenated_spike_trains)
    jittered_spike_train = jitter_spikes_across_trials(reorganized_spike_train, jitter_window_size, time_step)
    jittered_spike_train = reorganize_data(jittered_spike_train)

    print(jittered_spike_train.shape)
    return concatenated_spike_trains, jittered_spike_train

def jitter_spikes_across_trials(spike_trains, jitter_window_size, time_step):
    """
    Jitter the spikes across trials within specified jitter window size.
    Each window of size 'jitter_window_size' is shuffled across trials.
    """
    num_units, num_trials, num_steps = spike_trains.shape
    jittered_spike_trains = np.zeros_like(spike_trains)

    # Calculate the number of steps in each jitter window
    steps_per_window = int(jitter_window_size / time_step)

    for unit in range(num_units):
        for window_start in range(0, num_steps, steps_per_window):
            window_end = min(window_start + steps_per_window, num_steps)
            for step in range(window_start, window_end):
                # Shuffle the spikes across trials for each step in the window
                trial_indices = np.random.permutation(num_trials)
                jittered_spike_trains[unit, :, step] = spike_trains[unit, trial_indices, step]

    return jittered_spike_trains


def create_jitter(histograms_list):
    unit_num1 = len(histograms_list[0]["unit_id"].indexes["unit_id"])
    unit_num2 = len(histograms_list[0]["unit_id"].indexes["unit_id"])

    timestep = len(histograms_list[0]["time_relative_to_stimulus_onset"].indexes["time_relative_to_stimulus_onset"])
    trials = len(histograms_list[0]["stimulus_presentation_id"].indexes["stimulus_presentation_id"])
    # correlation_matrix = np.zeros([unit_num1, unit_num2,timestep * 2-1])

    # Assuming histograms1 and histograms2 are your data structures
    jitter_window_size = 25  # 25ms
    time_step = 0.1  # Each time step represents 0.1ms
    histograms_list2 = []
    jittered_histograms_list = []
    for histogram in histograms_list:
        histogram_new, jittered_histogram_new = create_jittered_spike_trains(histogram, jitter_window_size, time_step)
        histograms_list2.append(histogram_new)
        jittered_histograms_list.append(jittered_histogram_new)
    return histograms_list2, jittered_histograms_list

def correlation_computation(histograms_array):
    # for each two histograms, calculate cross-correlation between each unit
    start_time = time.time()
    correlation_list = []
    for p, histograms1 in enumerate(histograms_array):
        for q, histograms2 in enumerate(histograms_array):
            if p < q:
                print(p, q)
                unit_num1 = histograms1.shape[1]
                unit_num2 = histograms2.shape[1]

                # timestep = len(histograms1["time_relative_to_stimulus_onset"].indexes["time_relative_to_stimulus_onset"])
                timestep = histograms1.shape[2]
                trials = histograms1.shape[0]
                correlation_matrix = np.zeros([unit_num1, unit_num2,timestep * 2-1])
                for stimu_id in tqdm(range(trials)):
                    spike_train1 = histograms1[stimu_id]
                    spike_train2 = histograms2[stimu_id]
                    
                    for i in range(spike_train1.shape[0]):
                        for j in range(i+1, spike_train2.shape[0]):
                            
                            result = correlate_and_measure_time(spike_train1[i], spike_train2[j], "auto")
                            correlation_matrix[i,j] += result
                correlation_matrix /= trials
                correlation_list.append(correlation_matrix)

    print(time.time() - start_time)
    return correlation_list

class LSH:
    def __init__(self, n_dimensions, n_hashes):
        self.n_dimensions = n_dimensions
        self.n_hashes = n_hashes
        self.hash_tables = [dict() for _ in range(n_hashes)]
        self.random_vectors = [np.random.randn(n_dimensions) for _ in range(n_hashes)]

    def hash(self, vector, hash_index):
        """Hash a vector into a binary hash using a random projection method."""
        return int(np.dot(vector, self.random_vectors[hash_index]) > 0)

    def add_to_hash_table(self, vector, label, hash_index):
        """Add a vector to a specific hash table."""
        hash_value = self.hash(vector, hash_index)
        if hash_value not in self.hash_tables[hash_index]:
            self.hash_tables[hash_index][hash_value] = []
        self.hash_tables[hash_index][hash_value].append(label)

    def fit(self, data, labels):
        """Fit LSH to data."""
        for i in range(self.n_hashes):
            for label, vector in zip(labels, data):
                self.add_to_hash_table(vector, label, i)

    def query(self, vector):
        """Query LSH with a vector to find nearest neighbors."""
        candidates = set()
        for i in range(self.n_hashes):
            hash_value = self.hash(vector, i)
            if hash_value in self.hash_tables[i]:
                candidates.update(self.hash_tables[i][hash_value])
        return list(candidates)



### lsh method for correlation computation
def correlation_computation_lsh(histograms_array):
    # for each two histograms, calculate cross-correlation between each unit
    start_time = time.time()
    correlation_list = []
    for p, histograms1 in enumerate(histograms_array):
        for q, histograms2 in enumerate(histograms_array):
            if p < q:
                print(p, q)
                unit_num1 = histograms1.shape[1]
                unit_num2 = histograms2.shape[1]

                # timestep = len(histograms1["time_relative_to_stimulus_onset"].indexes["time_relative_to_stimulus_onset"])
                timestep = histograms1.shape[2]
                trials = histograms1.shape[0]
                correlation_matrix = np.zeros([unit_num1, unit_num2,timestep * 2-1])
                for stimu_id in tqdm(range(trials)):
                    spike_train1 = histograms1[stimu_id]
                    spike_train2 = histograms2[stimu_id]
                    
                    

                    for i in range(spike_train1.shape[0]):
                        n_hashes = 5
                        lsh = LSH(timestep, n_hashes)
                        # lsh fit one spike train1 and all spike train2
                        lsh.fit(spike_train2, labels=np.arange(unit_num2))
                        nearest_neighbors = lsh.query(spike_train1[i])
                        # print("nearest neighbors: ",unit_num2 - len(nearest_neighbors))
                        for j in nearest_neighbors:
                            result = correlate_and_measure_time(spike_train1[i], spike_train2[j], "auto")
                            correlation_matrix[i,j] += result
                        # for j in range(i+1, spike_train2.shape[0]):
                            
                        #     result = correlate_and_measure_time(spike_train1[i], spike_train2[j], "auto")
                        #     correlation_matrix[i,j] += result
                correlation_matrix /= trials
                correlation_list.append(correlation_matrix)

    print("Time consumption:", time.time() - start_time)
    return correlation_list

def summary_ccg(normalized_ccg):
    # find peak offset time lag for each unit pair in correction_ccg
    peak_offset_list = []
    dslist = []
    for i in range(len(normalized_ccg)):
        peak_offset = []
        for j in range(normalized_ccg[i].shape[0]):
            for k in range(normalized_ccg[i].shape[1]):
                # calculate std of -100ms to -50ms and 50ms to 100ms
                # print(normalized_ccg[i][j,k][4999:5499])
                tmp = np.concatenate((normalized_ccg[i][j,k][4999:5499], normalized_ccg[i][j,k][6499:6999]))
                std1 = np.std(tmp)

    
                if np.max(normalized_ccg[i][j,k]) > 7*std1 and np.argmax(normalized_ccg[i][j,k]) >= 5899 and np.argmax(normalized_ccg[i][j,k]) <= 6099:
                    
                    peak_offset.append(np.argmax(normalized_ccg[i][j,k])) # find the peak offset time lag
        peak_offset_list.append(peak_offset)
        cpos = np.sum(np.where(np.array(peak_offset) >= 5999, 1, 0))
        cneg = np.sum(np.where(np.array(peak_offset) < 5999, 1, 0))
        ds = (cpos - cneg) / (cpos + cneg)
        dslist.append(ds)
        
    return peak_offset_list, dslist


def plot_ccg(ccg, img_dir, title, length=5):
    significant_count = np.zeros([length,length])
    for i in range(length):
        for j in range(length):
            

            if i<j:
                if i == 0:
                    significant_count[i,j] = np.sum(ccg[j-1][:,:,5899:6099])
                    significant_count[j,i] = np.sum(ccg[j-1][:,:,5899:6099])
                else:
                    # count the number of significant correlation in correlation_list
                    significant_count[i,j] = np.sum(ccg[j-2*i+4][:,:,5899:6099])
                    significant_count[j,i] = np.sum(ccg[j-2*i+4][:,:,5899:6099])




    cmap = colors.LinearSegmentedColormap.from_list("", ["white", "red"])

    plt.figure()
    plt.imshow(significant_count, cmap=cmap)

    # Set x and y ticks
    plt.xticks(np.arange(5), ["V1", "RL", "LM", "PM", "AM"])
    plt.yticks(np.arange(5), ["V1", "RL", "LM", "PM", "AM"])

    # Add colorbar
    plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap=cmap))

    plt.show()
    title = img_dir + title
    plt.savefig(title)



def plot_ds(dslist, peak_offset_list, img_dir, length=5):
    ds_matrix = np.zeros([length, length])
    name_list = ["V1", "RL", "LM", "PM", "AM"]
    for i in range(length):
        for j in range(length):
            

            if i<j:
                if i == 0:
                    ds_matrix[i,j] = -dslist[j-1]
                    ds_matrix[j,i] = dslist[j-1]
                    plt.figure()
                    plt.hist(peak_offset_list[j-1], bins=200, label=f"n={len(peak_offset_list[j-1])} pairs")
                    plt.legend()
                    plt.show()
                    plt.title(f"{name_list[i]}-{name_list[j]}")
                    title = img_dir+f"{name_list[i]}-{name_list[j]}-ccg.png"
                    plt.savefig(title)
                else:
                    # count the number of significant correlation in correlation_list
                    ds_matrix[i,j] = -dslist[j-2*i+4]
                    ds_matrix[j,i] = dslist[j-2*i+4]
                    plt.figure()
                    plt.hist(peak_offset_list[j-2*i+4], bins=200, label=f"n={len(peak_offset_list[j-2*i+4])} pairs")
                    plt.legend()
                    plt.show()
                    plt.title(f"{name_list[i]}-{name_list[j]}")
                    title = img_dir+f"{name_list[i]}-{name_list[j]}-ccg.png"
                    plt.savefig(title)
                
                
            elif i==j:
                ds_matrix[i,j] = 0
                ds_matrix[j,i] = 0



    cmap = colors.LinearSegmentedColormap.from_list("custom_colormap", ["blue", "white", "red"])
    norm = colors.Normalize(vmin=-1, vmax=1)
    plt.figure()
    plt.imshow(ds_matrix, cmap=cmap)

    # Set x and y ticks
    plt.xticks(np.arange(5), ["V1", "RL", "LM", "PM", "AM"])
    plt.yticks(np.arange(5), ["V1", "RL", "LM", "PM", "AM"])

    # Add colorbar
    plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap=cmap))
    plt.title("DS")

    plt.show()
    title = img_dir+"ds.png"
    plt.savefig(title)



def main():
    # load data
    
    print("loading data --- ")
    output_dir = '/scratch/cl7201/neuro_ccg/ecephys_cache_dir/'
    raw_dir = "raw_output/"
    lsh_dir = "lsh_output/"
    cache, brain_observatory_type_sessions = load_sessions(output_dir=output_dir)
    session_id = 715093703
    session = cache.get_session_data(session_id)

    histograms_list = load_histograms(session)
    histograms_list2, jittered_histograms_list = create_jitter(histograms_list)
    
    # compute correlation

    print("compute correlation --- ")
    start_time = time.time()
    correlation = correlation_computation(histograms_list2)
    jittered_correlation = correlation_computation(jittered_histograms_list)
    normalized_ccg = []
    for i in range(len(correlation)):
        normalized_ccg.append(correlation[i] - jittered_correlation[i])
        # print(np.max(normalized_ccg[i]), np.min(normalized_ccg[i]))
    time_consump = time.time() - start_time
    print("Total Time consumption:", time.time() - start_time)
    np.save(f"{raw_dir}normalized_ccg.npy", normalized_ccg)
    np.save(f"{raw_dir}time.npy", time_consump)

    # metrics
    print("compute metrics --- ")
    # img_dir = "/scracth/cl7201/neuro_ccg/chenyi_output/"
    
    peak_offset_list, dslist = summary_ccg(normalized_ccg)
    plot_ds(dslist, peak_offset_list, raw_dir)
    plot_ccg(normalized_ccg, raw_dir, "normalized_ccg.png")
    plot_ccg(correlation, raw_dir, "raw_ccg.png")
    plot_ccg(jittered_correlation, raw_dir, "jittered_ccg.png")

    print("raw ending!")
    
    print("compute lsh correlation --- ")
    start_time = time.time()
    correlation = correlation_computation_lsh(histograms_list2)
    jittered_correlation = correlation_computation_lsh(jittered_histograms_list)
    normalized_ccg = []
    for i in range(len(correlation)):
        normalized_ccg.append(correlation[i] - jittered_correlation[i])
        # print(np.max(normalized_ccg[i]), np.min(normalized_ccg[i]))
    time_consump = time.time() - start_time
    print("Total Time consumption:", time.time() - start_time)
    np.save(f"{lsh_dir}normalized_ccg.npy", normalized_ccg)
    np.save(f"{lsh_dir}time.npy", time_consump)

    # metrics
    print("compute lsh metrics --- ")
    
    peak_offset_list, dslist = summary_ccg(normalized_ccg)
    plot_ds(dslist, peak_offset_list, lsh_dir)
    plot_ccg(normalized_ccg, lsh_dir, "normalized_ccg.png")
    plot_ccg(correlation, lsh_dir, "raw_ccg.png")
    plot_ccg(jittered_correlation, lsh_dir, "jittered_ccg.png")
    






    return 1


if __name__ == '__main__':

    main()