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
import torch
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms

# from pytorch_lightning.callbacks import modelcheckpoint
from sleeplib.datasets import BonoboDataset
from sleeplib.Resnet_15.model import ResNet
from sleeplib.transforms import cut_and_jitter, extremes_remover
from sleeplib.config import Config
from sleeplib.montages import (
    # CDAC_bipolar_montage,
    # CDAC_common_average_montage,
    CDAC_combine_montage,
    # con_combine_montage,
    # con_ECG_combine_montage,
)
from spikenet2_lib import get_output_root

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
path_model = os.path.join(get_output_root(), "models")
path_chkpt = os.path.join(path_model, "checkpoint")

# fraction filter
frac_filter = (df["fraction_of_yes"] >= 6 / 8) | (df["fraction_of_yes"] <= 2 / 8)
spike_filter = df["fraction_of_yes"] >= 6 / 8
mode_filter = df["Mode"] == "Test"  # "Train" "Test" "Val"
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

# load pretrained model
model = ResNet.load_from_checkpoint(
    os.path.join(path_chkpt, "hardmine-v4.ckpt"),
    lr=config.LR,
    n_channels=config.N_CHANNELS,
)
# map_location=torch.device('cpu') add this if running on CPU machine
# init trainer
trainer = pl.Trainer(
    devices=1, accelerator="gpu", fast_dev_run=False, enable_progress_bar=False
)

# predict all samples
preds = trainer.predict(model, test_dataloader)
preds = np.concatenate(preds)  # seems OK

# store results
results = test_df[
    ["event_file", "fraction_of_yes", "total_votes_received", "Mode"]
].copy()
results["preds"] = preds

results.to_csv(path_model + "/predictions.csv", index=False)

# auc
df = pd.read_csv(path_model + "/predictions.csv")

# set up filters for datasets
high_quality_filter = df["total_votes_received"] >= 2
ultra_quality_filter = df["total_votes_received"] >= 8
mode_filter = df["Mode"] == "Test"  # "Train" "Test" "Val"
frac_filter = (df["fraction_of_yes"] >= 6 / 8) | (df["fraction_of_yes"] <= 2 / 8)

# load samples as defined in spikenet paper
AUC_df = df[ultra_quality_filter & mode_filter & frac_filter]

labels = AUC_df.fraction_of_yes.values.round(0).astype(int)

preds = AUC_df.preds
# calculate ROC and AUC
fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)

# plot ROC
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(fpr, tpr, label="ROC curve (AUC = %0.4f)" % roc_auc)
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic (ROC) Curve")
ax.legend()
fig.savefig(os.path.join(path_model, "ROC-v4.png"), dpi=300, bbox_inches="tight")

# [EOF]
