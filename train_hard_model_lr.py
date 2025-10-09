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
def find_optimal_learning_rate_pytorch_lightning(
    model, train_dataloader, val_dataloader, config
):
    """
    Use PyTorch Lightning's built-in learning rate finder to find optimal learning rate.

    Args:
        model: The PyTorch Lightning model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        config: Configuration object

    Returns:
        suggested_lr: Suggested learning rate from LR finder
    """
    print("🔍 Running PyTorch Lightning Learning Rate Finder...")

    # Use the same device configuration as main training to respect GPU assignment
    accelerator = "gpu" if config.DEVICES > 0 else "cpu"
    
    # Create trainer configuration
    trainer_kwargs = {
        "devices": config.DEVICES,
        "accelerator": accelerator,
        "logger": False,  # No logging for LR finding
        "enable_checkpointing": False,  # No checkpoints for LR finding
        "enable_progress_bar": True,
        "max_epochs": 1,
    }
    
    # For multi-GPU setup, use the same strategy as main training
    if accelerator == "gpu" and config.DEVICES > 1:
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)
    
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
                        plot_path = os.path.join(
                            get_output_root(), "models", "lr_finder_plot.png"
                        )
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
old_ckpt = "hardmine-v9"
new_ckpt = "hardmine-v10"


# Parse command line arguments
args = parse_arguments()

# define model name and path
# model_path = "your_path/Models/spikenet2"
path_model = os.path.join(get_output_root(), "models")
path_chkpt = os.path.join(get_output_root(), "models", "checkpoint")
path_npy = os.path.join(path_model, "train_hard_npy")

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


for i in range(1):
    # build model
    # model = FineTuning(lr=config.LR, n_channels=37, Focal_loss=False)  # False
    model = ResNet.load_from_checkpoint(
        os.path.join(path_chkpt, old_ckpt + ".ckpt"),
        lr=config.LR,
        n_channels=config.N_CHANNELS,
        Focal_loss=False,  # True means loss function will be Focal loss. Otherwise will be BCE loss
    )

    # Determine learning rate based on command line arguments
    if args.skip_lr_finder:
        print(f"⏭️ Skipping LR finder, using config LR: {config.LR:.2e}")
        optimal_lr = config.LR
    else:
        # Find optimal learning rate using PyTorch Lightning's built-in LR finder
        print("\n🔍 Finding optimal learning rate...")
        optimal_lr = find_optimal_learning_rate_pytorch_lightning(
            model, train_dataloader, val_dataloader, config
        )

        print(f"🎯 Updating model LR from {config.LR:.2e} to {optimal_lr:.2e}")

        # Create new model instance with optimal learning rate
        model = ResNet.load_from_checkpoint(
            os.path.join(path_chkpt, old_ckpt + ".ckpt"),
            lr=optimal_lr,
            n_channels=config.N_CHANNELS,
            Focal_loss=False,
        )

    # If only running LR finder, exit here
    if args.lr_finder_only:
        print("✅ Learning rate finder completed. Exiting without training.")
        sys.exit(0)

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

# [EOF]
