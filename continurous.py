# get predictions on control EEG

# 2025 Richard J. Cui. Modified: Fri 09/12/2025 04:16:14.055411 PM
# $Revision: 0.2 $  $Date: Mon 09/22/2025 04:16:14.055411 PM $
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
from pytorch_lightning.strategies import DDPStrategy
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sleeplib.Resnet_15.model import ResNet
from sleeplib.datasets import ContinousToSnippetDataset
from sleeplib.config import Config
from sleeplib.montages import (
    # CDAC_bipolar_montage,
    # CDAC_common_average_montage,
    # CDAC_combine_montage,
    con_combine_montage,
    # con_ECG_combine_montage,
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
path_hdmin = os.path.join(get_output_root(), "models", "hard_mine")
path_chkpt = os.path.join(get_output_root(), "models", "checkpoint")

# set up dataloader to predict all samples in test dataset
transform_train = transforms.Compose([extremes_remover(signal_max=2000, signal_min=20)])
# con_combine_montage = con_ECG_combine_montage()
con_combine_montage = con_combine_montage()


# load pretrained model
model = ResNet.load_from_checkpoint(
    # "your_path/Models/spikenet2/hardmine.ckpt",
    os.path.join(path_chkpt, "hardmine-v1.ckpt"),
    lr=config.LR,
    n_channels=config.N_CHANNELS,
)
# map_location=torch.device('cpu') add this if running on CPU machine
# init trainer
trainer = pl.Trainer(
    devices="auto",
    accelerator="gpu",
    strategy=DDPStrategy(find_unused_parameters=True),
    fast_dev_run=False,
    enable_progress_bar=False,
)

# store results
path_controls = os.path.join(get_proj_root(), "controlset.csv")
controls = pd.read_csv(path_controls)
# controls = controls[controls['Mode']=='Test']

for eeg_file in tqdm(controls.EEG_index):
    # path = "your_path/continuousEEG/" + eeg_file + ".mat"
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
        batch_size=128,
        shuffle=False,
        num_workers=os.cpu_count() or 0,
    )

    preds = trainer.predict(model, con_dataloader)
    # preds = [np.squeeze(p) for p in preds]  # Ensure each part is 1D

    preds = np.concatenate(preds)
    preds = preds.astype(float)

    preds = pd.DataFrame(preds)
    preds.to_csv(os.path.join(path_hdmin, eeg_file + ".csv"), index=False)

# [EOF]
