import pandas as pd 
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader

import sys
sys.path.append('../')
from sleeplib.datasets import BonoboDataset
from sleeplib.transforms import cut_and_jitter, signal_flip, channel_deletion


class BonoboDatamodule(LightningDataModule):
    """
    A custom LightningDataModule for handling multiple datasets in a unified manner.
    This module takes a list of datasets and creates combined loaders for training and validation.

    Inputs:
        - datasets (list): A list of tuples, where each tuple represents a dataset and contains the following elements:
            - df: The DataFrame representing the dataset.
            - path_dataset: The path or identifier of the dataset.
            - batch_size: The batch size for the dataset.

        - num_workers (int): The number of workers to use for data loading (optional, default is 4).

    Outputs:
        - Combined loaders for training and validation data, which combine multiple datasets into a single dataloader.

    Usage Example:
        # Create a list of datasets
        datasets = [(df1, path1, batch_size1), (df2, path2, batch_size2), ...]

        # Create an instance of the BonoboDatamodule
        datamodule = BonoboDatamodule(datasets, num_workers=4)

        # Use the datamodule for training
        trainer.fit(model,datamodule=datamodule)
    """
    def __init__(self,datasets,transform, num_workers = 4):
        super().__init__()
        # init dataloader params
        self.num_workers = num_workers//len(datasets)
        self.transform = transform
        self.datasets = datasets    

    def setup(self,stage:str):
        pass
        
    def train_dataloader(self):
        # return dataloader for all datasets in training mode
        iterables = {}
        # iterate over datasets
        for i,(name,df,path_dataset,batch_size) in enumerate(self.datasets):
            # create dataset and dataloader
            dataset = BonoboDataset(df[df['Mode']=='Train'], path_dataset, transform=self.transform)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
            # add dataloader to iterables for combined loader
            iterables[name] = dataloader 
        combined_loader = CombinedLoader(iterables, mode="min_size")
        return combined_loader
    
    def val_dataloader(self):
        # return dataloader for all datasets in validation mode
        iterables = {}
        # iterate over datasets
        for i,(name,df,path_dataset,batch_size) in enumerate(self.datasets):
            # create dataset and dataloader
            dataset = BonoboDataset(df[df['Mode']=='Val'], path_dataset, transform=self.transform)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
            # add dataloader to iterables for combined loader
            iterables[name] = dataloader 
        combined_loader = CombinedLoader(iterables, mode="sequential")
        return combined_loader
    
