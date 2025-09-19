# evaluation model performance

# 2025 Richard J. Cui. Modified: Fri 09/19/2025 03:06:14.957544 PM
# $Revision: 0.1 $  $Date: Fri 09/19/2025 03:06:14.957544 PM $
#
# Mayo Clinic Foundation
# Rochester, MN 55901, USA
#
# Email: Cui.Jie@mayo.edu

# imports
import os
import pickle
import torch
import pickle
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

# from pytorch_lightning.callbacks import modelcheckpoint
from sleeplib.datasets import BonoboDataset

# from sleeplib.resnet_15.model import resnet
from sleeplib.transforms import cut_and_jitter, channel_flip, extremes_remover
from sleeplib.config import Config
from sleeplib.montages import (
    # CDAC_bipolar_montage,
    # CDAC_common_average_montage,
    CDAC_combine_montage,
    # con_combine_montage,
    # con_ECG_combine_montage,
)

# load own code
sys.path.append("../")

# this holds all the configuration parameters

# main
# path_model = "your_path/SpikeNet2/Models/spikenet2/"

# load config file
config = Config()
config.print_config()

# load dataset
# df = pd.read_csv("your_path.csv", sep=",")  # ; -> ,
df = pd.read_csv(config.PATH_LUT_BONOBO, sep=";")  # ; -> ,

# fraction filter
frac_filter = (df["fraction_of_yes"] >= 6 / 8) | (df["fraction_of_yes"] <= 2 / 8)
spike_filter = df["fraction_of_yes"] >= 6 / 8
mode_filter = df["Mode"] == "Test"
extreme_quality_filter = df["total_votes_received"] >= 8
quality_filter = df["total_votes_received"] >= 2

test_df = df[mode_filter]
AUC_df = df[extreme_quality_filter & mode_filter & frac_filter]
spike_df = df[extreme_quality_filter & mode_filter & spike_filter]
print(f"there are {len(AUC_df)} test samples")

print(f"there are {len(spike_df)} spike")
# set up dataloader to predict all samples in test dataset
transform_val = transforms.Compose(
    [
        cut_and_jitter(windowsize=config.WINDOWSIZE, max_offset=0, Fq=config.FQ),
        extremes_remover(signal_max=2000, signal_min=20),
    ]
)  # ,CDAC_signal_flip(p=0)])
combine_montage = CDAC_combine_montage()

test_dataset = BonoboDataset(
    test_df,
    config.PATH_FILES_BONOBO,
    transform=transform_val,
    window_size=config.WINDOWSIZE,
    montage=combine_montage,
)
test_dataloader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=os.cpu_count() or 1
)
for x, y in test_dataloader:
    with torch.no_grad():
        print(x.shape)
        break
