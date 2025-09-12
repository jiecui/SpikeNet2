# Convert .mat files to .npy format

# 2025 Richard J. Cui. Modified: Fri 09/12/2025 04:16:14.055411 PM
# $Revision: 0.1 $  $Date: Fri 09/12/2025 04:16:14.055411 PM $
#
# Mayo Clinic Foundation
# Rochester, MN 55901, USA
#
# Email: Cui.Jie@mayo.edu

# import libraries
import os
import glob
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from spikenet2_lib import get_database_root, get_output_root


# defined function to transfer data
def batch_convert_mat_to_npy(input_dir, output_dir=None):
    """
    Batch convert all .mat files in a directory to .npy format.

    Parameters:
    input_dir: Directory containing .mat files
    output_dir: Output directory (optional, defaults to the input directory)
    """
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))

    success_count = 0
    for mat_file in tqdm(mat_files, desc=".mat --> .npy", unit="file"):
        try:
            data = sio.loadmat(mat_file)["data"]

            base_name = os.path.splitext(os.path.basename(mat_file))[0]
            npy_file = os.path.join(output_dir, base_name + ".npy")
            np.save(npy_file, data)

            success_count += 1

        except Exception as e:
            print(f"✗ failed {mat_file}: {e}")

    return success_count


# main
if __name__ == "__main__":
    input_directory = os.path.join(get_database_root(), "Events", "real")
    output_directory = os.path.join(get_output_root(), "Events", "real_npy")

    count = batch_convert_mat_to_npy(input_directory, output_directory)
    print(f"Successfully converted {count} files.")


# [EOF]
