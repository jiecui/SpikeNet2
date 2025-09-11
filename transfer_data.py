# import libraries
import os
import glob
import scipy.io as sio
import numpy as np


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
    for mat_file in mat_files:
        try:
            data = sio.loadmat(mat_file)["data"]

            base_name = os.path.splitext(os.path.basename(mat_file))[0]
            npy_file = os.path.join(output_dir, base_name + ".npy")
            np.save(npy_file, data)
            print(f"✓ converted {base_name}")

            success_count += 1

        except Exception as e:
            print(f"✗ failed {mat_file}: {e}")


# main
if __name__ == "__main__":
    input_directory = (
        "H:\\Documents\\Richard\\Datasets\\spikenet2_datasets\\Events\\real"
    )
    output_directory = (
        "H:\\Documents\\Richard\\Datasets\\spikenet2_datasets\\Events\\real_npy"
    )

    batch_convert_mat_to_npy(input_directory, output_directory)

# [EOF]
