# get predictions on control EEG - Parallel Version

# 2025 Richard J. Cui. Modified: Tue 10/07/2025 03:41:54.030530 PM
# $Revision: 0.2 $  $Date: Tue 10/07/2025 04:49:32.169684 PM $
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from sleeplib.Resnet_15.model import ResNet
from sleeplib.datasets import ContinousToSnippetDataset
from sleeplib.config import Config
from sleeplib.montages import (
    con_combine_montage,
)
from sleeplib.transforms import extremes_remover
from spikenet2_lib import get_output_root, get_proj_root, get_database_root

# set global logging level
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
# load own code
sys.path.append("../")

# this holds all the configuration parameters
# load config and show all default parameters
config = Config()
path_model = os.path.join(get_output_root(), "models")
path_hdmin = os.path.join(path_model, "hard_mine")
path_chkpt = os.path.join(path_model, "checkpoint")

# set up dataloader to predict all samples in test dataset
transform_train = transforms.Compose([extremes_remover(signal_max=2000, signal_min=20)])
con_combine_montage = con_combine_montage()


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
            config_dict['checkpoint_path'],
            lr=config_dict['lr'],
            n_channels=config_dict['n_channels'],
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
            config_dict['database_root'], "EEG", "hm_negative_eeg", eeg_file + ".mat"
        )
        
        con_combine_montage_local = con_combine_montage()
        
        Bonobo_con = ContinousToSnippetDataset(
            path_eeg,
            montage=con_combine_montage_local,
            transform=transform_train,
            window_size=int(config_dict['window_size']),
        )
        
        con_dataloader = DataLoader(
            Bonobo_con,
            batch_size=config_dict['batch_size'],
            shuffle=False,
            num_workers=1,  # Reduce workers per process
        )

        preds = trainer.predict(model, con_dataloader)
        
        if preds is not None and len(preds) > 0:
            # Convert predictions to numpy arrays
            numpy_preds = []
            for batch in preds:
                if torch.is_tensor(batch):
                    numpy_preds.append(batch.cpu().numpy())
                else:
                    numpy_preds.append(np.array(batch))
            
            preds_array = np.concatenate(numpy_preds)
            preds_array = preds_array.astype(float)
            
            preds_df = pd.DataFrame(preds_array)
            output_path = os.path.join(config_dict['output_dir'], eeg_file + ".csv")
            preds_df.to_csv(output_path, index=False)
            
            return f"✓ Completed: {eeg_file}"
        else:
            return f"⚠ No predictions for: {eeg_file}"
        
    except Exception as e:
        return f"✗ Error processing {eeg_file}: {str(e)}"


def process_batch_gpu(eeg_files_batch, gpu_id=0):
    """Process a batch of EEG files on a specific GPU.
    
    Args:
        eeg_files_batch (list): List of EEG file names to process
        gpu_id (int): GPU device ID to use
        
    Returns:
        list: List of status messages
    """
    results = []
    
    try:
        # Explicitly set the CUDA device for this process
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)
            print(f"🔧 Setting GPU {gpu_id} for this batch")
        
        # Load model on specific GPU
        model = ResNet.load_from_checkpoint(
            os.path.join(path_chkpt, "hardmine-v5.ckpt"),
            lr=config.LR,
            n_channels=config.N_CHANNELS,
        )
        
        # Move model to specific GPU if available
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            model = model.cuda(gpu_id)
        
        # Create trainer for this GPU
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=[gpu_id] if torch.cuda.is_available() else 1,
            fast_dev_run=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        
        # Add progress bar for this GPU's batch
        for eeg_file in tqdm(eeg_files_batch, desc=f"GPU {gpu_id}", position=gpu_id, leave=False):
            try:
                # Verify we're using the correct GPU
                if torch.cuda.is_available():
                    current_device = torch.cuda.current_device()
                    if current_device != gpu_id:
                        print(f"⚠️  Warning: Expected GPU {gpu_id}, but using GPU {current_device}")
                
                path_eeg = os.path.join(
                    get_database_root(), "EEG", "hm_negative_eeg", eeg_file + ".mat"
                )
                
                Bonobo_con = ContinousToSnippetDataset(
                    path_eeg,
                    montage=con_combine_montage,
                    transform=transform_train,
                    window_size=int(config.WINDOWSIZE),
                )
                
                con_dataloader = DataLoader(
                    Bonobo_con,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,  # Moderate number of workers per GPU
                )

                preds = trainer.predict(model, con_dataloader)
                
                if preds is not None and len(preds) > 0:
                    # Convert predictions to numpy arrays - improved tensor handling
                    numpy_preds = []
                    for batch in preds:
                        if hasattr(batch, 'cpu'):  # PyTorch tensor
                            numpy_preds.append(batch.cpu().numpy())
                        elif hasattr(batch, 'numpy'):  # NumPy-compatible tensor
                            numpy_preds.append(batch.numpy())
                        else:  # Already numpy array or other
                            numpy_preds.append(np.array(batch))
                    
                    preds_array = np.concatenate(numpy_preds)
                    preds_array = preds_array.astype(float)
                    
                    preds_df = pd.DataFrame(preds_array)
                    preds_df.to_csv(os.path.join(path_hdmin, eeg_file + ".csv"), index=False)
                    
                    results.append(f"✓ GPU {gpu_id}: Completed {eeg_file}")
                else:
                    results.append(f"⚠ GPU {gpu_id}: No predictions for {eeg_file}")
                    
            except Exception as e:
                results.append(f"✗ GPU {gpu_id}: Error processing {eeg_file}: {str(e)}")
        
        return results
        
    except Exception as e:
        return [f"✗ GPU {gpu_id}: Batch processing error: {str(e)}"]


def main():
    """Main function with multiple parallelization options."""
    
    # Load controls
    path_controls = os.path.join(get_proj_root(), "controlset.csv")
    controls = pd.read_csv(path_controls)
    eeg_files = controls.EEG_index.tolist()
    
    print(f"Found {len(eeg_files)} EEG files to process")
    
    # Check available resources
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_cpus = os.cpu_count() or 1
    
    print(f"Available resources: {num_gpus} GPUs, {num_cpus} CPU cores")
    
    # Display GPU information if available
    if torch.cuda.is_available() and num_gpus > 0:
        print("🔍 GPU Information:")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check current GPU
        current_gpu = torch.cuda.current_device()
        print(f"📍 Current default GPU: {current_gpu}")
    else:
        print("⚠️  No CUDA GPUs available")
    
    # Option 1: Multi-GPU processing (recommended if you have multiple GPUs)
    if num_gpus > 1:
        print(f"\n🚀 Using Multi-GPU processing with {num_gpus} GPUs")
        
        # Split files among GPUs
        files_per_gpu = len(eeg_files) // num_gpus
        gpu_batches = []
        
        print(f"📁 Total files: {len(eeg_files)}, Files per GPU: {files_per_gpu}")
        
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * files_per_gpu
            if gpu_id == num_gpus - 1:  # Last GPU gets remaining files
                end_idx = len(eeg_files)
            else:
                end_idx = (gpu_id + 1) * files_per_gpu
            
            batch_files = eeg_files[start_idx:end_idx]
            gpu_batches.append((batch_files, gpu_id))
            
            print(f"🎯 GPU {gpu_id}: Processing {len(batch_files)} files (indices {start_idx}-{end_idx-1})")
        
        # Verify all GPUs have work
        total_assigned = sum(len(batch) for batch, _ in gpu_batches)
        print(f"✅ Total files assigned: {total_assigned}/{len(eeg_files)}")
        
        if total_assigned != len(eeg_files):
            print(f"⚠️  Warning: File count mismatch!")
        
        # Process batches in parallel using threads (since each uses different GPU)
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            future_to_gpu = {
                executor.submit(process_batch_gpu, batch, gpu_id): gpu_id 
                for batch, gpu_id in gpu_batches
            }
            
            all_results = []
            completed_gpus = 0
            
            # Create progress bar that we can manually update
            progress_bar = tqdm(total=num_gpus, desc="GPU Processing")
            
            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    progress_bar.set_postfix({"Completed GPU": gpu_id})
                except Exception as e:
                    all_results.append(f"✗ GPU {gpu_id} failed: {str(e)}")
                
                completed_gpus += 1
                progress_bar.update(1)
            
            progress_bar.close()
        
        # Print results
        for result in all_results:
            print(result)
    
    # Option 2: Multi-CPU processing (if no multiple GPUs available)
    elif num_cpus > 1:
        print(f"\n🚀 Using Multi-CPU processing with {min(num_cpus, len(eeg_files))} processes")
        
        # Prepare arguments for multiprocessing
        config_dict = {
            'checkpoint_path': os.path.join(path_chkpt, "hardmine-v5.ckpt"),
            'lr': config.LR,
            'n_channels': config.N_CHANNELS,
            'database_root': get_database_root(),
            'window_size': config.WINDOWSIZE,
            'batch_size': config.BATCH_SIZE,
            'output_dir': path_hdmin,
        }
        
        args_list = [(eeg_file, config_dict) for eeg_file in eeg_files]
        
        # Use ProcessPoolExecutor for better control
        max_workers = min(num_cpus, len(eeg_files), 8)  # Limit to 8 to avoid memory issues
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_eeg_file_cpu, args_list),
                total=len(args_list),
                desc="CPU Processing"
            ))
        
        # Print results
        for result in results:
            print(result)
    
    # Option 3: Single GPU/CPU processing (fallback)
    else:
        print(f"\n🚀 Using single device processing")
        
        # Load model
        model = ResNet.load_from_checkpoint(
            os.path.join(path_chkpt, "hardmine-v5.ckpt"),
            lr=config.LR,
            n_channels=config.N_CHANNELS,
        )
        
        # Create trainer
        trainer = pl.Trainer(
            devices=config.DEVICES if num_gpus > 0 else 1,
            accelerator="gpu" if num_gpus > 0 else "cpu",
            fast_dev_run=False,
            enable_progress_bar=False,
        )
        
        # Process files sequentially with progress bar
        for eeg_file in tqdm(eeg_files, desc="Processing EEG files"):
            path_eeg = os.path.join(
                get_database_root(), "EEG", "hm_negative_eeg", eeg_file + ".mat"
            )
            
            Bonobo_con = ContinousToSnippetDataset(
                path_eeg,
                montage=con_combine_montage,
                transform=transform_train,
                window_size=int(config.WINDOWSIZE),
            )
            
            con_dataloader = DataLoader(
                Bonobo_con,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=num_cpus // 2,
            )

            preds = trainer.predict(model, con_dataloader)
            
            if preds is not None and len(preds) > 0:
                numpy_preds = []
                for batch in preds:
                    if torch.is_tensor(batch):
                        numpy_preds.append(batch.cpu().numpy())
                    else:
                        numpy_preds.append(np.array(batch))
                
                preds_array = np.concatenate(numpy_preds)
                preds_array = preds_array.astype(float)
                
                preds_df = pd.DataFrame(preds_array)
                preds_df.to_csv(os.path.join(path_hdmin, eeg_file + ".csv"), index=False)
                
                print(f"✓ Completed: {eeg_file}")
            else:
                print(f"⚠ No predictions for: {eeg_file}")


if __name__ == "__main__":
    main()

# [EOF]