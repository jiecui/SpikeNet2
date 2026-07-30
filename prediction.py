# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: SpikeNet2
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Evaluation of model performance

# %%
# 2025-2026 Richard J. Cui. Modified: Fri 09/19/2025 03:06:14.957544 PM
# $Revision: 0.8 $  $Date: Thu 07/30/2026 12:18:14.990589 PM $
#
# Mayo Clinic Foundation
# Rochester, MN 55901, USA
#
# Email: Cui.Jie@mayo.edu

# %% [markdown]
# ## Import libraries

# %%
import os
import sys

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms

from sleeplib.config import Config

# from pytorch_lightning.callbacks import modelcheckpoint
from sleeplib.datasets import BonoboDataset
from sleeplib.montages import (
    # CDAC_bipolar_montage,
    # CDAC_common_average_montage,
    CDAC_combine_montage,
    # con_combine_montage,
    # con_ECG_combine_montage,
)
from sleeplib.Resnet_15.model import ResNet
from sleeplib.transforms import cut_and_jitter, extremes_remover
from spikenet2_lib import get_output_root

# load own code
sys.path.append("../")

# %% [markdown]
# ## Main

# %% [markdown]
# ### Load config file

# %%
config = Config()
config.print_config()

# %% [markdown]
# ### Load dataset

# %%
# dataset
# -------
# * data
df = pd.read_csv(config.PATH_LUT_BONOBO, sep=";")  # ; -> ,

# * data choices
# data type
test_filter = df["Mode"] == "Test"  # "Train", "Test", "Val"
# data selection
test_df = df[test_filter]  # total test samples

# %% [markdown]
# ### Model prediction

# %%
# model path and checkpoint
# -------------------------
path_model = os.path.join(get_output_root(), "models")
path_chkpt = os.path.join(path_model, "checkpoint")

# set up dataloader to predict all samples in test dataset
# --------------------------------------------------------
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
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,  # os.cpu_count() or 1, 0 for windows
)

# load pretrained model
model = ResNet.load_from_checkpoint(
    os.path.join(path_chkpt, config.MODEL_CHECKPOINT + ".ckpt"),
    lr=config.LR,
    n_channels=config.N_CHANNELS,
    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# init trainer
trainer = pl.Trainer(
    devices=config.DEVICES,
    accelerator="gpu",
    fast_dev_run=False,
    enable_progress_bar=False,
)

# predict all test samples
# ------------------------
preds = trainer.predict(model, test_dataloader)
if preds is None:
    raise ValueError("No predictions were made. Check the model and dataloader.")
preds = np.concatenate(preds)  # seems OK

# store results
results = test_df[
    ["event_file", "fraction_of_yes", "total_votes_received", "Mode"]
].copy()
results["preds"] = preds

# save results to csv
# -------------------
path_preds = os.path.join(path_model, "predictions.csv")
results.to_csv(path_preds, index=False)
print(f"🎉 Predictions saved to {path_preds}")

# %% [markdown]
# ### Performance evaluation

# %%
# load results for performance evaluation
# ---------------------------------------
df = pd.read_csv(path_preds)

# vote fraction (ration of yes votes)
spike_filter = df["fraction_of_yes"] >= 6 / 8
nonspike_filter = df["fraction_of_yes"] <= 2 / 8
AUC_filter = spike_filter | nonspike_filter
# vote quality (number of votes received)
ultra_quality_filter = df["total_votes_received"] >= 8
# test samples for performance evaluation
AUC_df = df[ultra_quality_filter & AUC_filter]
spike_df = df[ultra_quality_filter & spike_filter]
print(f"{len(AUC_df)} out of {len(test_df)} test samples used for AUC evaluation.")
print(
    f"There are {len(spike_df)} spike and {len(AUC_df) - len(spike_df)} non-spike samples."
)

# plots
# -----
# get the labels of ground truth and predictions
labels = AUC_df.fraction_of_yes.values.round(0).astype(int)
preds = AUC_df.preds

# * calculate ROC and ROC-AUC
fpr, tpr, thresholds = roc_curve(labels, preds)
roc_auc = auc(fpr, tpr)

# plot ROC
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:0.4f}")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic (ROC) Curve")
ax.legend()
roc_fname = "ROC-" + config.MODEL_CHECKPOINT + ".pdf"
fig.savefig(
    os.path.join(path_model, roc_fname),
    bbox_inches="tight",
)
print(f"ℹ️ ROC curve saved to {os.path.join(path_model, roc_fname)}")

# * calculate precision-recall curve (PRC) and AUC
precision, recall, thresholds = precision_recall_curve(labels, preds)
prc_auc = auc(recall, precision)

# plot PRC
prevalence = len(spike_df) / len(AUC_df)
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(recall, precision, label=f"PRC curve (AUC = {prc_auc:0.4f})")
ax.plot([0, 1], [prevalence, prevalence], linestyle="--", label="Prevalence")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend()
prc_fname = "PRC-" + config.MODEL_CHECKPOINT + ".pdf"
fig.savefig(
    os.path.join(path_model, prc_fname),
    bbox_inches="tight",
)
print(f"ℹ️ PRC curve saved to {os.path.join(path_model, prc_fname)}")


# [EOF]
