import pandas as pd
import wandb
import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
)
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

# load own code
import sys

sys.path.append("../")
from sleeplib.Resnet_15.model import FineTuning
from sleeplib.datasets import (
    BonoboDataset,
    ContinousToSnippetDataset,
    Hardmine_BonoboDataset,
)
from sleeplib.montages import CDAC_combine_montage, ECG_combine_montage
from sleeplib.transforms import (
    cut_and_jitter,
    channel_flip,
    extremes_remover,
    random_channel_zero,
    noise_adder,
    ECG_channel_flip,
)

# this holds all the configuration parameters
from sleeplib.config import Config
import pickle

# define model name and path
model_path = "your_path/Models/spikenet2"
# load config and show all default parameters
config = Config()
config.print_config()
combine_montage = CDAC_combine_montage()

# load dataset
df = pd.read_csv("your_path/SpikeNet2/hard_mining.csv", sep=",")

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
    train_df,
    config.PATH_FILES_BONOBO,
    transform=transform_train_pos,
    transform_pos=transform_train_pos,
    transform_neg=transform_train_neg,
    montage=combine_montage,
    window_size=config.WINDOWSIZE,
    num_pos_augmentations=1,  # 2 3 4
)
train_dataloader = DataLoader(
    Bonobo_train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()
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
    Bonobo_val, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()
)


for i in range(1):
    # build model
    model = FineTuning(lr=config.LR, n_channels=37, Focal_loss=False)  # False

    # create a logger
    wandb.init(dir="logging")
    wandb_logger = WandbLogger(project="super_awesome_project")

    # create callbacks with early stopping and model checkpoint (saves the best model)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5),
        ModelCheckpoint(
            dirpath=model_path, filename="without_HM_1sec", monitor="val_loss"
        ),
    ]
    # create trainer, use fast dev run to test the code
    trainer = pl.Trainer(
        devices=[1],
        accelerator="gpu",
        min_epochs=30,
        max_epochs=100,
        logger=wandb_logger,
        callbacks=callbacks,
        fast_dev_run=False,
    )
    # train the model
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.finish()

    # [EOF]
