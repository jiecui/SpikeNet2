# get hard-negative samples

# 2025 Richard J. Cui. Modified: Fri 09/12/2025 04:16:14.055411 PM
# $Revision: 0.4 $  $Date: Thu 09/25/2025 09:35:51.999862 AM $
#
# Mayo Clinic Foundation
# Rochester, MN 55901, USA
#
# Email: Cui.Jie@mayo.edu

# imports
import numpy as np
import pandas as pd
import mat73
from tqdm import tqdm
import os
from spikenet2_lib import get_database_root, get_output_root, get_proj_root
from sleeplib.config import Config
from spikenet2_lib import copy_new_files

# constants
# Assuming signal is your original signal stored as a numpy array
# signal = np.array([...])
config = Config()
sampling_rate = config.FQ  # Hz
time_step = 8
window_length = 3  # in seconds
num_points = sampling_rate * window_length
threshold = 0.75  # .8, 0.75, 0.5, 0.4; 1st round set at .8
cluster_len = 3  # 10 8 5 3


# function definition
def hard_mine(df):
    """
    The function `hard_mine` identifies clusters of false positives in a
    DataFrame based on a specified threshold and returns the indices of the
    highest prediction values within each cluster.

    :param df: It seems like the code snippet you provided is a function named
    `hard_mine` that processes a DataFrame to identify clusters of false
    positives based on a threshold value. The function then extracts the index
    of the highest prediction value from each cluster and filters them based on
    a specified number of points
    :return: The `filtered_indices` list containing the indices of the highest
    prediction values from each cluster that meet the specified conditions is
    being returned.
    """

    column_name = df.columns[0]
    # Remove rows of df[column_name] where the type is 'str' and 'nan'
    df = df[~df[column_name].apply(lambda x: isinstance(x, str))].dropna()
    false_positives = df[df[column_name] > threshold]

    # Display the number of false positives and their first few rows
    num_false_positives = len(false_positives)
    false_positives.head(), num_false_positives

    # Find clusters of false positives
    clusters = []
    current_cluster = []

    # Iterate over the false positives
    for idx in false_positives.index:
        # If current_cluster is empty or the current index is consecutive to the last index in current_cluster
        if not current_cluster or idx == current_cluster[-1] + 1:
            current_cluster.append(idx)
        else:
            # If the current index is not consecutive, check if the current cluster is valid (has at least 8 false positives)
            if len(current_cluster) >= cluster_len:
                clusters.append(current_cluster)
            # Reset current_cluster and start a new one
            current_cluster = [idx]

    # Check for the last cluster
    if len(current_cluster) >= cluster_len:
        clusters.append(current_cluster)

    # Display the number of clusters and the indices of the first few clusters
    num_clusters = len(clusters)
    # Extract the index of the highest prediction value from each cluster
    highest_prediction_indices = []
    bonobo_idx = []

    for cluster in clusters:
        max_index = df.loc[cluster][column_name].idxmax()
        highest_prediction_indices.append(max_index)

    filtered_indices = []
    for idx in sorted(highest_prediction_indices):
        if not filtered_indices or (idx - filtered_indices[-1] > num_points):
            filtered_indices.append(idx)
        else:
            if idx - filtered_indices[-1] <= num_points:
                last_idx = filtered_indices.pop()  # Store the popped value if needed
    return filtered_indices


def count_files_in_directory(directory_path):
    items = os.listdir(directory_path)

    return len(
        [item for item in items if os.path.isfile(os.path.join(directory_path, item))]
    )


# round 1?
path_controls = os.path.join(get_proj_root(), "controlset.csv")  # EEG without spikes
path_model = os.path.join(get_output_root(), "models")
path_round2 = os.path.join(path_model, "hardmine_npy_round2")
controls = pd.read_csv(path_controls)
train_controls = controls[controls["Mode"] == "Train"]
# train_controls = controls

for eeg_file in tqdm(train_controls.EEG_index, desc="Hard Mining"):
    # df_path = "your_path/SpikeNet2/Models/SpikeNet2/con_hardmine/" + eeg_file + ".csv"
    df_path = os.path.join(get_output_root(), "models", "hard_mine", eeg_file + ".csv")
    # signal_path = "your_path/Bonobo_data/" + eeg_file + ".mat"
    signal_path = os.path.join(
        get_database_root(), "EEG", "hm_negative_eeg", eeg_file + ".mat"
    )
    df = pd.read_csv(df_path)
    filtered_indices = hard_mine(df)  # get false positive events
    signal = mat73.loadmat(signal_path)["data"]  # 128 sample rate
    signal = signal.transpose(1, 0)
    start_times = filtered_indices  # 16 sample rate

    # Calculate start indices
    start_indices = [int(time * time_step) for time in start_times]

    # Extract signal segments
    segments = [signal[start : start + num_points] for start in start_indices]
    for idx, seg in zip(start_indices, segments):
        # path = str(
        #     "your_path/SpikeNet2/Models/SpikeNet2/hardmine_npy_round2/"
        #     + eeg_file
        #     + "_"
        #     + str(idx)
        #     + ".npy"
        # )
        path = os.path.join(path_round2, eeg_file + "_" + str(idx) + ".npy")
        seg = seg.transpose(1, 0)
        # print(seg.shape)
        np.save(path, seg)

# round 2
# count_files_in_directory("your_path/SpikeNet2/Models/SpikeNet2/hardmine_npy_round2")
count_files_in_directory(path_round2)

# read csv
# df = pd.read_csv("your_path/SpikeNet2/hard_mining_round1.csv")
df = pd.read_csv(os.path.join(get_output_root(), "models", "hardmine_npy_round1.csv"))

# Get the names of all .npy files in the folder
npy_files = [
    f
    # for f in os.listdir("your_path/SpikeNet2/Models/SpikeNet2/hardmine_npy_round2")
    for f in os.listdir(path_round2)
    if f.endswith(".npy")
]

# Extract file name part
event_files = [f[:-4] for f in npy_files]  # remove.npy
eeg_files = [
    f.split("_")[:-1] for f in event_files
]  # # Use '_' as delimiter and remove the last part
eeg_files = ["_".join(f) for f in eeg_files]
# Create a new DataFrame to store filenames and other information
new_data = {
    "event_file": event_files,
    "eeg_file": eeg_files,
    "total_votes_received": [3] * len(event_files),
    "fraction_of_yes": [0] * len(event_files),
    "Mode": ["Train"] * len(event_files),
}

new_df = pd.DataFrame(new_data)

# Add new data to the original DataFrame
df = pd.concat([df, new_df], ignore_index=True)
# df.to_csv("your_path/SpikeNet2/hard_mining_round2.csv", index=False)
df.to_csv(os.path.join(path_model, "hardmine_npy_round2.csv"), index=False)

# copy files to folder train_hard_npy
# check if the folder train_hard_npy exists under path_model
path_npy = os.path.join(path_model, "train_hard_npy")
source_path = config.PATH_FILES_BONOBO

print(f"Copying new files from {source_path} to {path_npy}...")
copy_new_files(source_path, path_npy)

# copy files from hardmine_npy_round2 to path_npy
print(f"Copying new files from {path_round2} to {path_npy}...")
copy_new_files(path_round2, path_npy)

# [EOF]
