# get predictions on control EEG - Multi-CPU Version

# 2025 Richard J. Cui. Modified: Fri 09/12/2025 04:16:14.055411 PM
# $Revision: 0.5 $  $Date: Tue 10/08/2025 02:00:00.000000 PM $
#
# Mayo Clinic Foundation
# Rochester, MN 55901, USA
#
# Email: Cui.Jie@mayo.edu

# imports
import os
import sys
import logging
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor
import argparse
import multiprocessing as mp
from sleeplib.Resnet_15.model import ResNet
from sleeplib.datasets import ContinousToSnippetDataset
from sleeplib.config import Config
from sleeplib.transforms import extremes_remover
from spikenet2_lib import get_output_root, get_proj_root, get_database_root

# set global logging level
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
# load own code
sys.path.append("../")

def process_single_eeg_file_cpu(args):
    """Process a single EEG file using CPU.

    Args:
        args (tuple): (eeg_file, config_dict) where config_dict contains necessary config values

    Returns:
        str: Status message
    """
    eeg_file, config_dict = args

    try:
        # Create a separate model instance for this process
        model = ResNet.load_from_checkpoint(
            config_dict["checkpoint_path"],
            lr=config_dict["lr"],
            n_channels=config_dict["n_channels"],
        )

        # Create trainer for this process - using CPU
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            fast_dev_run=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        path_eeg = os.path.join(
            config_dict["database_root"], "EEG", "hm_negative_eeg", eeg_file + ".mat"
        )

        # Create montage instance properly  
        from sleeplib.montages import con_combine_montage as montage_func
        montage_instance = montage_func()
        transform_local = transforms.Compose(
            [extremes_remover(signal_max=2000, signal_min=20)]
        )

        Bonobo_con = ContinousToSnippetDataset(
            path_eeg,
            montage=montage_instance,
            transform=transform_local,
            window_size=int(config_dict["window_size"]),
        )

        con_dataloader = DataLoader(
            Bonobo_con,
            batch_size=config_dict["batch_size"],
            shuffle=False,
            num_workers=1,  # Single worker per process to avoid conflicts
        )

        preds = trainer.predict(model, con_dataloader)

        if preds is not None and len(preds) > 0:
            # Convert predictions to numpy arrays robustly
            def _to_numpy(x):
                try:
                    if torch.is_tensor(x):
                        return x.detach().cpu().numpy()
                    if isinstance(x, (list, tuple)):
                        parts = [_to_numpy(p) for p in x]
                        try:
                            return np.concatenate(parts)
                        except Exception:
                            return np.array(parts, dtype=object)
                    if hasattr(x, "numpy"):
                        return x.numpy()
                    return np.array(x)
                except Exception:
                    return np.array(x)

            numpy_preds = [_to_numpy(batch) for batch in preds]
            preds_array = np.concatenate(numpy_preds)
            preds_array = preds_array.astype(float)

            preds_df = pd.DataFrame(preds_array)
            output_path = os.path.join(config_dict["output_dir"], eeg_file + ".csv")
            preds_df.to_csv(output_path, index=False)

            return f"✓ Completed: {eeg_file}"
        else:
            return f"⚠ No predictions for: {eeg_file}"

    except Exception as e:
        return f"✗ Error processing {eeg_file}: {str(e)}"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="EEG Prediction with Multi-CPU Processing")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of CPU workers (default: use all available cores)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

# model checkpoint
model_ckpt = "hardmine-v7.ckpt"

# this holds all the configuration parameters
# load config and show all default parameters
config = Config()
path_model = os.path.join(get_output_root(), "models")
path_hdmin = os.path.join(path_model, "hard_mine")
path_chkpt = os.path.join(path_model, "checkpoint")

def main():
    """Main function with multi-CPU parallel processing."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set multiprocessing start method to 'spawn' for better compatibility
    try:
        mp.set_start_method("spawn", force=True)
        if args.verbose:
            print("🔧 Set multiprocessing to 'spawn' method for compatibility")
    except RuntimeError as e:
        if "context has already been set" not in str(e):
            print(f"⚠️ Warning: Could not set multiprocessing method: {e}")

    # Load controls
    path_controls = os.path.join(get_proj_root(), "controlset.csv")
    controls = pd.read_csv(path_controls)
    eeg_files = controls.EEG_index.tolist()

    print(f"Found {len(eeg_files)} EEG files to process")

    # Check available CPU resources
    num_cpus = os.cpu_count() or 1
    max_workers = args.max_workers if args.max_workers else num_cpus
    
    # Limit to reasonable number to avoid memory issues
    max_workers = min(max_workers, len(eeg_files), 8)

    print(f"Available CPU cores: {num_cpus}")
    print(f"Using {max_workers} worker processes")

    # Prepare arguments for multiprocessing
    config_dict = {
        "checkpoint_path": os.path.join(path_chkpt, model_ckpt),
        "lr": config.LR,
        "n_channels": config.N_CHANNELS,
        "database_root": get_database_root(),
        "window_size": config.WINDOWSIZE,
        "batch_size": config.BATCH_SIZE,
        "output_dir": path_hdmin,
    }

    args_list = [(eeg_file, config_dict) for eeg_file in eeg_files]

    print(f"\n🚀 Starting Multi-CPU processing with {max_workers} processes...")

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_single_eeg_file_cpu, args_list),
                total=len(args_list),
                desc="Processing EEG files",
                unit="file"
            )
        )

    # Print results summary
    print("\n📊 Processing Results:")
    successful = sum(1 for r in results if r.startswith("✓"))
    failed = sum(1 for r in results if r.startswith("✗"))
    warnings = sum(1 for r in results if r.startswith("⚠"))
    
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⚠️  Warnings: {warnings}")
    
    if args.verbose:
        print("\nDetailed results:")
        for result in results:
            print(f"  {result}")

    print("\n🎉 Multi-CPU processing completed!")

if __name__ == "__main__":
    main()

# [EOF]
