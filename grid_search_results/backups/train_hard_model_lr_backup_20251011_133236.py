# train model using hard mining dataset

# 2025 Richard J. Cui. Modified: Fri 09/19/2025 03:06:14.957544 PM
# $Revision: 0.4 $  $Date: Thu 10/09/2025 06:10:40.213699 PM $
#
# Mayo Clinic Foundation
# Rochester, MN 55901, USA
#
# Email: Cui.Jie@mayo.edu

# imports
import wandb
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torchvision import transforms
from sleeplib.config import Config
import pickle
import time

# from sleeplib.Resnet_15.model import FineTuning
from sleeplib.Resnet_15.model import ResNet
from sleeplib.datasets import (
    BonoboDataset,
    Hardmine_BonoboDataset,
)
from sleeplib.montages import CDAC_combine_montage
from sleeplib.transforms import (
    cut_and_jitter,
    channel_flip,
    extremes_remover,
)
from spikenet2_lib import get_output_root


# definition of functions
def get_ddp_rank():
    """Get the current DDP rank. Returns 0 if not in DDP mode."""
    try:
        return int(os.environ.get('LOCAL_RANK', 0))
    except (ValueError, TypeError):
        return 0

def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_ddp_rank() == 0

def save_lr_result(optimal_lr, path_lr):
    """Save LR result to file for sharing between DDP processes."""
    lr_file = os.path.join(path_lr, 'optimal_lr.pkl')
    with open(lr_file, 'wb') as f:
        pickle.dump(optimal_lr, f)
    print(f"💾 Saved optimal LR {optimal_lr:.2e} to {lr_file}")

def load_lr_result(path_lr, timeout=60):
    """Load LR result from file, waiting if necessary for main process to complete."""
    lr_file = os.path.join(path_lr, 'optimal_lr.pkl')
    
    # Wait for file to be created by main process
    start_time = time.time()
    while not os.path.exists(lr_file):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for LR result file: {lr_file}")
        time.sleep(1)
        print(f"⏳ Rank {get_ddp_rank()}: Waiting for optimal LR from main process...")
    
    # Load the result
    with open(lr_file, 'rb') as f:
        optimal_lr = pickle.load(f)
    print(f"📥 Rank {get_ddp_rank()}: Loaded optimal LR {optimal_lr:.2e} from main process")
    return optimal_lr

def cleanup_lr_result(path_lr):
    """Clean up temporary LR result file."""
    lr_file = os.path.join(path_lr, 'optimal_lr.pkl')
    try:
        if os.path.exists(lr_file):
            os.remove(lr_file)
            print(f"🧹 Cleaned up LR result file: {lr_file}")
    except Exception as e:
        print(f"⚠️ Could not clean up LR result file: {e}")

def find_optimal_learning_rate_pytorch_lightning(
    model, train_dataloader, val_dataloader, config, path_lr
):
    """
    Use PyTorch Lightning's built-in learning rate finder to find optimal learning rate.
    This function is designed to run ONCE and find optimal LR for all GPUs.

    Args:
        model: The PyTorch Lightning model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        config: Configuration object
        path_lr: Path to directory for LR finder checkpoints and plots

    Returns:
        suggested_lr: Suggested learning rate from LR finder
    """
    print(
        "🔍 Running PyTorch Lightning Learning Rate Finder (single run for all GPUs)..."
    )

    # Ensure the LR finder directory exists
    os.makedirs(path_lr, exist_ok=True)
    print(f"📁 LR finder checkpoints will be saved to: {path_lr}")

    # Force LR finder to run on single device to avoid redundant computation
    # Even in multi-GPU setup, LR finder only needs to run once
    accelerator = "gpu" if config.DEVICES > 0 else "cpu"
    devices = 1  # Always use single device for LR finding

    # Create trainer configuration for LR finding (single device only)
    trainer_kwargs = {
        "devices": devices,
        "accelerator": accelerator,
        "logger": False,  # No logging for LR finding
        "enable_checkpointing": False,  # No checkpoints for LR finding
        "enable_progress_bar": True,
        "max_epochs": 1,
        "default_root_dir": path_lr,  # Set LR finder checkpoint directory
    }

    # No DDP strategy for LR finder - single device only
    lr_trainer = pl.Trainer(**trainer_kwargs)

    # Run the learning rate finder
    print("📡 Running LR finder scan...")

    try:
        # Import the tuner
        from pytorch_lightning.tuner.tuning import Tuner

        tuner = Tuner(lr_trainer)

        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            min_lr=1e-8,
            max_lr=1e-1,
            num_training=100,
            mode="exponential",
            early_stop_threshold=4.0,
        )

        if lr_finder is not None:
            # Get the suggested learning rate
            if hasattr(lr_finder, "suggestion"):
                suggested_lr = lr_finder.suggestion()
            else:
                # Fallback: use the LR with minimum loss
                suggested_lr = lr_finder.results["lr"][
                    lr_finder.results["loss"].index(min(lr_finder.results["loss"]))
                ]

            # Create and save the plot
            if hasattr(lr_finder, "plot"):
                try:
                    fig = lr_finder.plot(suggest=True, show=False)
                    if fig is not None:
                        plot_path = os.path.join(path_lr, "lr_finder_plot.png")
                        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        print(f"📊 LR Finder plot saved to: {plot_path}")
                except Exception as plot_error:
                    print(f"⚠️ Could not save plot: {plot_error}")

            print(f"💡 Best learning rate found: {suggested_lr:.2e}")
            print(f"🎯 Original config LR: {config.LR:.2e}")

            return suggested_lr
        else:
            print("❌ LR Finder returned None, using original LR")
            return config.LR

    except Exception as e:
        print(f"❌ Error running LR finder: {str(e)}")
        print("🔄 Falling back to original learning rate")
        return config.LR


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train model with PyTorch Lightning Learning Rate Finder"
    )
    parser.add_argument(
        "--lr-finder-only",
        action="store_true",
        help="Run only the learning rate finder without training",
    )
    parser.add_argument(
        "--skip-lr-finder",
        action="store_true",
        help="Skip learning rate finder and use config LR directly",
    )
    return parser.parse_args()


# load own code
sys.path.append("../")

# parameters
old_ckpt = "hardmine-v12"
new_ckpt = "hardmine-v13"


# Parse command line arguments
args = parse_arguments()

# define model name and path
# model_path = "your_path/Models/spikenet2"
path_model = os.path.join(get_output_root(), "models")
path_chkpt = os.path.join(get_output_root(), "models", "checkpoint")
path_npy = os.path.join(path_model, "train_hard_npy")
path_lr = os.path.join(path_model, "lr_finder")

# load config and show all default parameters
config = Config()
config.print_config()
combine_montage = CDAC_combine_montage()

# load dataset
# df = pd.read_csv("your_path/SpikeNet2/hard_mining.csv", sep=",")
df_lut = pd.read_csv(config.PATH_LUT_BONOBO, sep=";")  # ; -> ,
df_hm = pd.read_csv(os.path.join(path_model, "hardmine_npy_round2.csv"), sep=",")
# concatenate two dataframes
df = pd.concat([df_lut, df_hm], ignore_index=True)

transform_train_pos = transforms.Compose(
    [
        cut_and_jitter(windowsize=config.WINDOWSIZE, max_offset=0.1, Fq=config.FQ),
        channel_flip(p=0.5),
        extremes_remover(signal_max=2000, signal_min=20),
        # random_channel_zero(p=0.1),
        # noise_adder(s=0.1, p=0.1)
    ]
)

transform_train_neg = transforms.Compose(
    [
        cut_and_jitter(windowsize=config.WINDOWSIZE, max_offset=0.1, Fq=config.FQ),
        channel_flip(p=0.5),
        # random_channel_zero(p=0.1),
        # noise_adder(s=0.1, p=0.1),
        extremes_remover(signal_max=2000, signal_min=20),
    ]
)

transform_val = transforms.Compose(
    [
        cut_and_jitter(windowsize=config.WINDOWSIZE, max_offset=0, Fq=config.FQ),
        extremes_remover(signal_max=2000, signal_min=20),
    ]
)  # ,CDAC_signal_flip(p=0)])

# init datasets
sub_df = df[df["total_votes_received"] > 2]
train_df = sub_df[sub_df["Mode"] == "Train"]
val_df = sub_df[sub_df["Mode"] == "Val"]


# set up dataloaders
Bonobo_train = Hardmine_BonoboDataset(
    train_df,  # config.PATH_FILES_BONOBO,
    path_npy,
    transform=transform_train_pos,
    transform_pos=transform_train_pos,
    transform_neg=transform_train_neg,
    montage=combine_montage,
    window_size=config.WINDOWSIZE,
    num_pos_augmentations=1,  # 2 3 4
)
train_dataloader = DataLoader(
    Bonobo_train,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count() or 1,
)

Bonobo_val = BonoboDataset(
    val_df,
    config.PATH_FILES_BONOBO,
    transform=transform_val,
    montage=combine_montage,
    window_size=config.WINDOWSIZE,
    #  num_pos_augmentations = 1 #1
)
val_dataloader = DataLoader(
    Bonobo_val,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=os.cpu_count() or 1,
)


# Determine learning rate once before training loop
# Only run on main process in DDP mode to avoid redundant computation
rank = get_ddp_rank()
print(f"🏷️ Process rank: {rank}")

if args.skip_lr_finder:
    print(f"⏭️ Skipping LR finder, using config LR: {config.LR:.2e}")
    optimal_lr = config.LR
else:
    if is_main_process():
        # Main process: Find optimal learning rate
        print("\n🔍 Main process: Finding optimal learning rate (running once for all GPUs)...")
        
        # Create a temporary model instance just for LR finding
        lr_finder_model = ResNet.load_from_checkpoint(
            os.path.join(path_chkpt, old_ckpt + ".ckpt"),
            lr=config.LR,
            n_channels=config.N_CHANNELS,
            Focal_loss=False,
        )
        
        optimal_lr = find_optimal_learning_rate_pytorch_lightning(
            lr_finder_model, train_dataloader, val_dataloader, config, path_lr
        )
        
        print(f"🎯 Main process: Optimal LR found: {optimal_lr:.2e}")
        
        # Save result for other processes
        save_lr_result(optimal_lr, path_lr)
        
        # Clean up temporary model
        del lr_finder_model
        
    else:
        # Other processes: Wait for and load result from main process
        print(f"\n⏳ Rank {rank}: Waiting for optimal LR from main process...")
        optimal_lr = load_lr_result(path_lr)
        
    print(f"📋 Rank {rank}: Using optimal LR: {optimal_lr:.2e}")

# If only running LR finder, exit here
if args.lr_finder_only:
    print("✅ Learning rate finder completed. Exiting without training.")
    sys.exit(0)

print(f"\n🚀 Starting training with optimal LR: {optimal_lr:.2e}")

for i in range(1):
    # build model with optimal learning rate (determined once above)
    # model = FineTuning(lr=optimal_lr, n_channels=37, Focal_loss=False)  # False
    model = ResNet.load_from_checkpoint(
        os.path.join(path_chkpt, old_ckpt + ".ckpt"),
        lr=optimal_lr,  # Use the optimal LR found once above
        n_channels=config.N_CHANNELS,
        Focal_loss=False,  # True means loss function will be Focal loss. Otherwise will be BCE loss
    )

    print(f"📦 Model {i+1} loaded with optimal LR: {optimal_lr:.2e}")

    # create a logger
    wandb.init(dir="logging")
    wandb_logger = WandbLogger(
        project="spikenet2", name=f"spikenet2_lr_{optimal_lr:.2e}"
    )

    # create callbacks with early stopping and model checkpoint (saves the best model)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5),
        ModelCheckpoint(dirpath=path_chkpt, filename=new_ckpt, monitor="val_loss"),
    ]
    # create trainer, use fast dev run to test the code
    trainer = pl.Trainer(
        devices=config.DEVICES,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        log_every_n_steps=5,
        min_epochs=200,
        max_epochs=300,
        logger=wandb_logger,
        callbacks=callbacks,
        fast_dev_run=False,
    )
    # train the model
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.finish()

# Clean up LR result file after training (only main process)
if is_main_process():
    cleanup_lr_result(path_lr)

# [EOF]
