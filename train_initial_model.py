# Train initial model on bonobo dataset

# 2025 Richard J. Cui. Modified: Fri 09/12/2025 04:16:14.055411 PM
# $Revision: 0.1 $  $Date: Fri 09/12/2025 04:16:14.055411 PM $
#
# Mayo Clinic Foundation
# Rochester, MN 55901, USA
#
# Email: Cui.Jie@mayo.edu

# import libraries
import wandb
import os
import sys
import pandas as pd
import pytorch_lightning as pl
from sleeplib.config import Config
from sleeplib.transforms import cut_and_jitter, channel_flip, extremes_remover
from sleeplib.montages import CDAC_combine_montage
from sleeplib.datasets import BonoboDataset
from sleeplib.Resnet_15.model import ResNet
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
from spikenet2_lib import get_output_root

# load own code
sys.path.append("../")

# define model name and path
model_path = os.path.join(get_output_root(), "models")
# load config and show all default parameters
config = Config()
config.print_config()

combine_montage = CDAC_combine_montage()

# load dataset
df = pd.read_csv(config.PATH_LUT_BONOBO, sep=";")  # ; -> ,
# add transformations
transform_train = transforms.Compose(
    [
        cut_and_jitter(windowsize=config.WINDOWSIZE, max_offset=0.1, Fq=config.FQ),
        channel_flip(p=0.5),
        extremes_remover(signal_max=2000, signal_min=2),
    ]
)
transform_val = transforms.Compose(
    [
        cut_and_jitter(windowsize=config.WINDOWSIZE, max_offset=0, Fq=config.FQ),
        extremes_remover(signal_max=2000, signal_min=2),
    ]
)  # ,CDAC_signal_flip(p=0)])


# init datasets
sub_df = df[df["total_votes_received"] > 2]
train_df = sub_df[sub_df["Mode"] == "Train"]
val_df = sub_df[sub_df["Mode"] == "Val"]

# set up dataloaders
Bonobo_train = BonoboDataset(
    train_df,
    config.PATH_FILES_BONOBO,
    transform=transform_train,
    montage=combine_montage,
)

train_dataloader = DataLoader(
    Bonobo_train,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count() or 0,
)

Bonobo_val = BonoboDataset(
    val_df, config.PATH_FILES_BONOBO, transform=transform_val, montage=combine_montage
)

val_dataloader = DataLoader(
    Bonobo_val,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=os.cpu_count() or 0,
)

# ***********
# build model
# ***********
model = ResNet(
    lr=config.LR,
    n_channels=config.N_CHANNELS,
    Focal_loss=True,  # True means loss function will be Focal loss. Otherwise will be BCE loss
)

# ***************
# train the model
# ***************
# create a logger
wandb.init(dir="logging")  # potential conflict between drive letters and UNC paths
wandb_logger = WandbLogger(project="spikenet2_project", name="spikenet2_run")

# create callbacks with early stopping and model checkpoint (saves the best model)
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5),
    ModelCheckpoint(dirpath=model_path, filename="hardmine", monitor="val_loss"),
]
# create trainer, use fast dev run to test the code
trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    min_epochs=30,
    max_epochs=100,
    logger=wandb_logger,
    callbacks=callbacks,
    fast_dev_run=False,
)

trainer.fit(model, train_dataloader, val_dataloader)
wandb.finish()

# [EOF]
