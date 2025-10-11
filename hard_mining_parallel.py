"""
Hard Mining Parallel Processing for EEG Data
Parallel version of hard_mining.py with multi-GPU and CPU support

This module implements parallel hard negative mining for EEG spike detection data.
It supports multiple parallelization modes:
- CPU multiprocessing (recommended for most systems)
- GPU parallel processing (for multi-GPU systems)
- Threading (for I/O-bound operations)
- Auto mode (automatically selects best method)

Author: Modified from original hard_mining.py for parallel processing
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings
import gc
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import mat73

# Import original modules
from spikenet2_lib import get_output_root, copy_new_files, get_database_root
from sleeplib import config

# Constants for hard mining (from original hard_mining.py)
cfg = config.Config()
sampling_rate = cfg.FQ  # Hz
time_step = 8
window_length = 3  # in seconds
num_points = sampling_rate * window_length
threshold = 0.5  # .8, 0.75, 0.5, 0.4; 1st round set at .8
cluster_len = 3  # 10 8 5 3


def hard_mine(df):
    """
    The function `hard_mine` identifies clusters of false positives in a
    DataFrame based on a specified threshold and returns the indices of the
    highest prediction values within each cluster.

    :param df: DataFrame containing prediction results
    :return: The `filtered_indices` list containing the indices of the highest
    prediction values from each cluster that meet the specified conditions.
    """
    column_name = df.columns[0]
    # Remove rows of df[column_name] where the type is 'str' and 'nan'
    df = df[~df[column_name].apply(lambda x: isinstance(x, str))].dropna()
    false_positives = df[df[column_name] > threshold]

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

    # Extract the index of the highest prediction value from each cluster
    highest_prediction_indices = []

    for cluster in clusters:
        max_index = df.loc[cluster][column_name].idxmax()
        highest_prediction_indices.append(max_index)

    filtered_indices = []
    for idx in sorted(highest_prediction_indices):
        if not filtered_indices or (idx - filtered_indices[-1] > num_points):
            filtered_indices.append(idx)
        else:
            if idx - filtered_indices[-1] <= num_points:
                filtered_indices.pop()  # Remove the previous index if too close
    return filtered_indices


def count_files_in_directory(directory_path):
    """Count the number of files in a directory."""
    items = os.listdir(directory_path)
    return len(
        [item for item in items if os.path.isfile(os.path.join(directory_path, item))]
    )


class HardMiningParallel:
    """Parallel processor for hard negative mining operations."""

    def __init__(
        self,
        mode: str = "auto",
        max_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        verbose: bool = True,
        suppress_warnings: bool = True,
    ):
        """
        Initialize the parallel processor.

        Args:
            mode: Processing mode ('auto', 'cpu', 'gpu', 'threading')
            max_workers: Number of workers (None for automatic)
            chunk_size: Chunk size for batching (None for automatic)
            verbose: Enable verbose output
            suppress_warnings: Suppress PyTorch warnings
        """
        self.mode = mode
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.verbose = verbose

        if suppress_warnings:
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*num_workers.*")

        # Detect hardware
        self.num_cpus = mp.cpu_count()
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Set optimal configuration
        self._configure_processing()

        if self.verbose:
            print("📍 HardMiningParallel initialized:")
            print(f"  Mode: {self.actual_mode}")
            print(f"  Workers: {self.workers}")
            print(f"  CPUs: {self.num_cpus}, GPUs: {self.num_gpus}")
            print(f"  Chunk size: {self.chunk_size}")

    def _configure_processing(self):
        """Configure processing parameters based on hardware and mode."""
        # Determine actual processing mode
        if self.mode == "auto":
            if self.num_gpus > 1:
                self.actual_mode = "gpu"
            else:
                self.actual_mode = "cpu"
        else:
            self.actual_mode = self.mode

        # Set number of workers
        if self.max_workers is None:
            if self.actual_mode == "cpu":
                self.workers = min(self.num_cpus, 8)  # Reasonable limit
            elif self.actual_mode == "gpu":
                self.workers = min(self.num_gpus, 4)
            elif self.actual_mode == "threading":
                self.workers = min(self.num_cpus * 2, 16)
            else:
                self.workers = 4
        else:
            self.workers = self.max_workers

        # Set chunk size
        if self.chunk_size is None:
            if self.actual_mode == "gpu":
                self.chunk_size = 2  # Smaller chunks for GPU processing
            else:
                self.chunk_size = max(1, self.workers // 2)

        # Ensure chunk_size is always an integer
        self.chunk_size = int(self.chunk_size)

        # Optimize environment
        self._optimize_environment()

    def _optimize_environment(self):
        """Optimize environment variables for parallel processing."""
        # Optimize PyTorch for multiprocessing
        torch.set_num_threads(1)  # Prevent thread competition

        # Set environment variables for better performance
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    def process_single_eeg(self, args: Tuple[str, str, str, int]) -> Dict[str, Any]:
        """
        Process a single EEG file for hard mining.

        Args:
            args: Tuple of (eeg_file, path_round2, path_hard_mine, device_id)

        Returns:
            Dictionary with processing results
        """
        eeg_file, path_round2, path_hard_mine, device_id = args

        # Initialize device
        device = "cpu"

        try:
            # Set device if GPU mode
            if self.actual_mode == "gpu" and torch.cuda.is_available():
                device = f"cuda:{device_id}"
                torch.cuda.set_device(device_id)
            else:
                device = "cpu"

            # Read CSV prediction
            csv_file = f"{eeg_file}.csv"
            csv_path = os.path.join(path_hard_mine, csv_file)

            if not os.path.exists(csv_path):
                return {
                    "eeg_file": eeg_file,
                    "status": "skipped",
                    "reason": f"CSV file not found: {csv_path}",
                    "segments_saved": 0,
                }

            # Perform hard mining
            df_csv = pd.read_csv(csv_path)
            hard_mined_indices = hard_mine(df_csv)

            if not hard_mined_indices:
                return {
                    "eeg_file": eeg_file,
                    "status": "completed",
                    "reason": "No hard mining results",
                    "segments_saved": 0,
                }

            # Load signal for segment extraction
            signal_path = os.path.join(
                get_database_root(), "EEG", "hm_negative_eeg", f"{eeg_file}.mat"
            )

            if not os.path.exists(signal_path):
                return {
                    "eeg_file": eeg_file,
                    "status": "skipped",
                    "reason": f"Signal file not found: {signal_path}",
                    "segments_saved": 0,
                }

            # Load signal (exactly like original)
            signal = mat73.loadmat(signal_path)["data"]  # 128 sample rate
            signal = signal.transpose(1, 0)
            start_times = hard_mined_indices  # 16 sample rate

            # Calculate start indices (exactly like original)
            start_indices = [int(time * time_step) for time in start_times]

            # Extract signal segments (exactly like original)
            segments = [signal[start : start + num_points] for start in start_indices]

            # Save individual segment files (exactly like original)
            saved_segments = 0
            for start_idx, seg in zip(start_indices, segments):
                try:
                    # Check if segment has correct length and shape
                    if seg.shape[0] == num_points:  # Ensure full segment
                        output_file = os.path.join(
                            path_round2, f"{eeg_file}_{start_idx}.npy"
                        )
                        seg = seg.transpose(
                            1, 0
                        )  # Transpose before saving (like original)
                        np.save(output_file, seg)
                        saved_segments += 1
                except Exception as e:
                    if self.verbose:
                        print(
                            f"Warning: Failed to save segment {start_idx} from {eeg_file}: {e}"
                        )
                    continue

            return {
                "eeg_file": eeg_file,
                "status": "completed",
                "reason": "Successfully processed",
                "segments_saved": saved_segments,
            }

        except Exception as e:
            return {
                "eeg_file": eeg_file,
                "status": "error",
                "reason": f"Processing error: {str(e)}",
                "segments_saved": 0,
            }
        finally:
            # Clean up GPU memory if using GPU
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

    def _create_chunks(self, eeg_files: List[str]) -> List[List[str]]:
        """Create chunks of EEG files for batch processing."""
        chunks = []
        chunk_size = self.chunk_size or 1  # Ensure we have a valid integer
        for i in range(0, len(eeg_files), chunk_size):
            chunks.append(eeg_files[i : i + chunk_size])
        return chunks

    def process_parallel(
        self, train_controls: pd.DataFrame, path_round2: str, path_hard_mine: str
    ) -> List[Dict[str, Any]]:
        """
        Process EEG files in parallel for hard mining.

        Args:
            train_controls: DataFrame with EEG_index column
            path_round2: Output path for hard mining results
            path_hard_mine: Path containing CSV prediction files for hard mining

        Returns:
            List of processing results
        """
        eeg_files = train_controls.EEG_index.tolist()

        if self.verbose:
            print(
                f"🚀 Processing {len(eeg_files)} EEG files using {self.actual_mode} mode with {self.workers} workers"
            )

        if self.actual_mode == "gpu":
            # Distribute files across available GPUs
            args_list = []
            for i, eeg_file in enumerate(eeg_files):
                device_id = i % self.num_gpus
                args_list.append((eeg_file, path_round2, path_hard_mine, device_id))
        else:
            # CPU/threading mode - device_id not used
            args_list = [
                (eeg_file, path_round2, path_hard_mine, 0) for eeg_file in eeg_files
            ]

        results = []

        # Create progress bar
        pbar = tqdm(total=len(eeg_files), desc="Hard Mining Progress")

        try:
            if self.actual_mode == "cpu":
                # CPU multiprocessing
                with ProcessPoolExecutor(max_workers=self.workers) as executor:
                    # Submit all tasks
                    future_to_args = {
                        executor.submit(self.process_single_eeg, args): args
                        for args in args_list
                    }

                    # Process completed tasks
                    for future in as_completed(future_to_args):
                        try:
                            result = future.result()
                            results.append(result)
                            pbar.update(1)

                            if self.verbose and result["status"] == "error":
                                pbar.write(
                                    f"Error processing {result['eeg_file']}: {result['reason']}"
                                )

                        except Exception as e:
                            args = future_to_args[future]
                            results.append(
                                {
                                    "eeg_file": args[0],
                                    "status": "error",
                                    "reason": f"Future exception: {str(e)}",
                                    "segments_saved": 0,
                                }
                            )
                            pbar.update(1)
                            if self.verbose:
                                pbar.write(f"Future error for {args[0]}: {e}")

            elif self.actual_mode == "threading":
                # Threading mode
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    future_to_args = {
                        executor.submit(self.process_single_eeg, args): args
                        for args in args_list
                    }

                    for future in as_completed(future_to_args):
                        try:
                            result = future.result()
                            results.append(result)
                            pbar.update(1)

                            if self.verbose and result["status"] == "error":
                                pbar.write(
                                    f"Error processing {result['eeg_file']}: {result['reason']}"
                                )

                        except Exception as e:
                            args = future_to_args[future]
                            results.append(
                                {
                                    "eeg_file": args[0],
                                    "status": "error",
                                    "reason": f"Thread exception: {str(e)}",
                                    "segments_saved": 0,
                                }
                            )
                            pbar.update(1)
                            if self.verbose:
                                pbar.write(f"Thread error for {args[0]}: {e}")

            elif self.actual_mode == "gpu":
                # GPU parallel processing
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    future_to_args = {
                        executor.submit(self.process_single_eeg, args): args
                        for args in args_list
                    }

                    for future in as_completed(future_to_args):
                        try:
                            result = future.result()
                            results.append(result)
                            pbar.update(1)

                            if self.verbose and result["status"] == "error":
                                pbar.write(
                                    f"Error processing {result['eeg_file']}: {result['reason']}"
                                )

                        except Exception as e:
                            args = future_to_args[future]
                            results.append(
                                {
                                    "eeg_file": args[0],
                                    "status": "error",
                                    "reason": f"GPU exception: {str(e)}",
                                    "segments_saved": 0,
                                }
                            )
                            pbar.update(1)
                            if self.verbose:
                                pbar.write(f"GPU error for {args[0]}: {e}")

            else:
                # Sequential fallback
                for args in args_list:
                    result = self.process_single_eeg(args)
                    results.append(result)
                    pbar.update(1)

                    if self.verbose and result["status"] == "error":
                        pbar.write(
                            f"Error processing {result['eeg_file']}: {result['reason']}"
                        )

        finally:
            pbar.close()

        return results

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of processing results."""
        if not results:
            print("No results to summarize.")
            return

        total_files = len(results)
        completed = sum(1 for r in results if r["status"] == "completed")
        errors = sum(1 for r in results if r["status"] == "error")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        total_segments = sum(r["segments_saved"] for r in results)

        print("\nHard Mining Summary:")
        print(f"  Total files: {total_files}")
        print(f"  Completed: {completed}")
        print(f"  Errors: {errors}")
        print(f"  Skipped: {skipped}")
        print(f"  Total segments saved: {total_segments}")

        if errors > 0 and self.verbose:
            print("\nError details:")
            for r in results:
                if r["status"] == "error":
                    print(f"  {r['eeg_file']}: {r['reason']}")


def main():
    """
    Main function implementing the parallel hard mining workflow.
    """

    print("ℹ️ Hard mining controls:")
    print(
        f"Sampling Rate: {sampling_rate}, Time Step: {time_step}, Num Points: {num_points}, Threshold: {threshold}, Cluster Length: {cluster_len}, Window Length: {window_length}s"
    )

    print("Starting Parallel Hard Mining Process...")
    print(f"Available CPUs: {mp.cpu_count()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

    # Set paths
    path_model = os.path.join(get_output_root(), "models")
    path_round2 = os.path.join(path_model, "hardmine_npy_round2")
    path_hard_mine = os.path.join(path_model, "hard_mine")

    # Create output directory
    os.makedirs(path_round2, exist_ok=True)

    # Load training controls
    train_controls = pd.read_csv("controlset.csv")
    train_controls = train_controls[train_controls.Mode == "Train"]

    print(f"Found {len(train_controls)} training files to process")

    # Initialize parallel processor
    # Available modes: 'auto', 'cpu', 'gpu', 'threading'
    processor = HardMiningParallel(
        mode="auto",  # Let the system choose the best mode
        max_workers=None,  # Use optimal number of workers
        verbose=True,
        suppress_warnings=True,
    )

    # Process files in parallel
    start_time = time.time()
    results = processor.process_parallel(train_controls, path_round2, path_hard_mine)
    end_time = time.time()

    # Print summary
    processor.print_summary(results)
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

    # Count files created (like original)
    print(f"\nFiles in output directory: {count_files_in_directory(path_round2)}")

    # Post-processing: Create CSV and copy files
    print("\nPost-processing...")

    # Read existing CSV
    df = pd.read_csv(
        os.path.join(get_output_root(), "models", "hardmine_npy_round1.csv")
    )

    # Get the names of all .npy files in the folder
    npy_files = [f for f in os.listdir(path_round2) if f.endswith(".npy")]

    if npy_files:
        # Extract file name part
        event_files = [f[:-4] for f in npy_files]  # remove .npy
        eeg_files = [
            f.split("_")[:-1] for f in event_files
        ]  # Use '_' as delimiter and remove the last part
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
        df.to_csv(os.path.join(path_model, "hardmine_npy_round2.csv"), index=False)

        print(f"Created hardmine_npy_round2.csv with {len(new_df)} new entries")
    else:
        print("No .npy files found in output directory")

    # Copy files to training folder
    path_npy = os.path.join(path_model, "train_hard_npy")
    cfg = config.Config()
    source_path = cfg.PATH_FILES_BONOBO

    print(f"Copying new files from {source_path} to {path_npy}...")
    copy_new_files(source_path, path_npy)

    print(f"Copying new files from {path_round2} to {path_npy}...")
    copy_new_files(path_round2, path_npy)
    # after copying, remove all files in path_round2
    [os.remove(os.path.join(path_round2, f)) for f in os.listdir(path_round2)]
    # remove all files in path_model/hard_mine
    [os.remove(os.path.join(path_hard_mine, f)) for f in os.listdir(path_hard_mine)]

    print("🎉 Parallel hard mining process completed successfully!")


if __name__ == "__main__":
    # Enable multiprocessing support
    mp.set_start_method("spawn", force=True)
    main()
