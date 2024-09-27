from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import pickle
from torchvision import transforms
import pytorch_lightning as pl
import torch
from tqdm import tqdm

# load own code
import sys
sys.path.append('../')
from sleeplib.Resnet_15.model import ResNet
from sleeplib.datasets import BonoboDataset, ContinousToSnippetDataset
# this holds all the configuration parameters
from sleeplib.config import Config
import pickle

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms

from sleeplib.datasets import BonoboDataset , ContinousToSnippetDataset
from sleeplib.montages import CDAC_bipolar_montage,CDAC_common_average_montage,CDAC_combine_montage,con_combine_montage, con_ECG_combine_montage
from sleeplib.transforms import cut_and_jitter, channel_flip,extremes_remover
# load config and show all default parameters
config = Config()
path_model = 'your_path/Models/spikenet2/'

# set up dataloader to predict all samples in test dataset
transform_train = transforms.Compose([extremes_remover(signal_max = 2000, signal_min = 20)])
con_combine_montage = con_ECG_combine_montage()


# load pretrained model
model = ResNet.load_from_checkpoint('your_path/Models/spikenet2/hardmine.ckpt',
                                        lr=config.LR,
                                        n_channels=37,
                                       )
                                        #map_location=torch.device('cpu') add this if running on CPU machine
# init trainer
trainer = pl.Trainer(fast_dev_run=False,enable_progress_bar=False,devices = 1,strategy ='ddp')

# store results
path_controls = os.path.join("your_path/Models/spikenet2/controlset.csv")

controls = pd.read_csv(path_controls)
i = 0
#controls = controls[controls['Mode']=='Test']
for eeg_file in tqdm(controls.EEG_index):
    path = 'your_path/continuousEEG/'+eeg_file+'.mat'
    Bonobo_con = ContinousToSnippetDataset(path,montage=con_combine_montage,transform=transform_train,window_size=config.WINDOWSIZE)
    con_dataloader = DataLoader(Bonobo_con, batch_size=128,shuffle=False,num_workers=os.cpu_count())
    
    preds = trainer.predict(model,con_dataloader)
    #preds = [np.squeeze(p) for p in preds]  # Ensure each part is 1D

    preds = np.concatenate(preds)
    preds = preds.astype(float)

    preds = pd.DataFrame(preds)
    preds.to_csv(path_model+'/hard_mine/'+ eeg_file +'.csv',index=False)

    