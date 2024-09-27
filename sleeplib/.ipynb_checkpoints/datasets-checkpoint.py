import os 
import numpy as np
import torch
import mat73


import sys
sys.path.append('../')

class BonoboDataset(torch.utils.data.Dataset):
    def __init__(self,df,path_folder, montage=None, transform=None):
        self.df = df
        # set transform
        self.transform = transform
        # set montage
        self.montage = montage
        # set path to bucket
        self.path_folder = path_folder

    def __len__(self):
        return len(self.df)

    def _preprocess(self,signal):
        # convert to desired montage
        if self.montage is not None:
            signal = self.montage(signal)
        # apply transformations
        if self.transform is not None:
            signal = self.transform(signal)
                
        # normalize signal
        signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)

        # convert to torch tensor, the copy is due to torch bug
        signal = torch.FloatTensor(signal.copy())
        
        return signal

    def __getitem__(self, idx):
        # get name and label of the idx-th sample
        event_file = self.df.iloc[idx]['event_file']
        label = self.df.iloc[idx]['fraction_of_yes']
        # load signal of the idx-th sample
        path_signal = os.path.join(self.path_folder,event_file+'.npy')
        signal = np.load(path_signal)
        # preprocess signal
        signal = self._preprocess(signal)
        # return signal          
        return signal,label

class ContinousToSnippetDataset(torch.utils.data.Dataset):
    # Dataset that takes a continous signal and returns snippets of a given length
    # input shape: (n_channels,n_timepoints), output shape: (n_snippets,n_channels,ts)
    def __init__(self,path_signal,montage, transform=None, Fq=128,window_size=10,step=1):
 
        # load signal
        signal = mat73.loadmat(path_signal)['data']
        # move signal to torch
        signal = torch.FloatTensor(signal.astype(np.float32))
        # generate snippets of shape (n_snippets,n_channels,ts)
        self.snippets = signal.unfold(dimension = 1,size = window_size*Fq, step = step *Fq).permute(1,0,2)

        # set transform
        self.transform = transform
        # add montage
        self.montage = montage

    def __len__(self):
        # get item zero of self. snippets, which has shape (n_snippets,n_channels,ts)
        return self.snippets.shape[0]

    def _preprocess(self,signal):
        '''preprocess signal and apply montage, transform and normalization'''

        # apply montage to signal
        signal = self.montage(signal)

        # apply transformations
        if self.transform is not None:
            signal = self.transform(signal)

        # normalize signal
        signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)

        
        # if signal is not a float, convert it
        if signal.dtype != 'float32':
            signal = signal.float()

        return signal

    def __getitem__(self, idx):
        # get the snippet        
        signal = self.snippets[idx,:,:]
        # preprocess signal
        signal = self._preprocess(signal)
        
        # return signal and dummy label, the latter to prevent lightning dataloader from complaining
        return signal,0