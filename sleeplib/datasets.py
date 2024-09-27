import os 
import numpy as np
import torch
import mat73
import scipy
from scipy.interpolate import interp1d

import sys
sys.path.append('../')

class LocationDataset(torch.utils.data.Dataset):
    def __init__(self,df,
                 path_folder, 
                 montage=None, 
                 window_size=1,
                 fq=128,
                 transform=None):
        self.df = df
        # set transform
        self.transform = transform
        # set montage
        self.montage = montage
        # set path to bucket
        self.path_folder = path_folder
        self.num_points = fq * window_size
    def __len__(self):
        return len(self.df)

    def _preprocess(self,signal):
        # convert to desired montage
        if self.montage is not None:
            signal = self.montage(signal)

        # apply transformations
        if self.transform is not None:
            signal = self.transform(signal)

        if signal.shape[-1] != 0:
        # normalize signal
            signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)

        while signal.shape[1] < self.num_points:
        # Pad with zeros if shorter length
            padding = np.zeros((signal.shape[0], 1))
            signal = np.hstack((signal, padding))
        if signal.shape[1] > self.num_points:
        # Truncate if longer length
            signal = signal[:, :self.num_points]
        # convert to torch tensor, the copy is due to torch bug
        signal = torch.FloatTensor(signal.copy())
        
        return signal

    def __getitem__(self, idx):
        try:
            # get name and label of the idx-th sample
            event_file = self.df.iloc[idx]['event_file']
            label = self.df.iloc[idx]['location']
            # load signal of the idx-th sample
            path_signal = os.path.join(self.path_folder,event_file+'.npy')
            signal = np.load(path_signal)
            # preprocess signal
            #print(signal.shape)
            signal = self._preprocess(signal)
            #print(signal.shape)

            # return signal          
            return signal,label
        except Exception as e:
            print(f"Error at index {event_file}: {e}")
            raise

class BonoboDataset(torch.utils.data.Dataset):
    def __init__(self,df,
                 path_folder, 
                 montage=None, 
                 window_size=1,
                 fq=128,
                 transform=None):
        self.df = df
        # set transform
        self.transform = transform
        # set montage
        self.montage = montage
        # set path to bucket
        self.path_folder = path_folder
        self.num_points = fq * window_size
    def __len__(self):
        return len(self.df)

    def _preprocess(self,signal):
        # convert to desired montage
        if self.montage is not None:
            signal = self.montage(signal)

        # apply transformations
        if self.transform is not None:
            signal = self.transform(signal)

        if signal.shape[-1] != 0:
        # normalize signal
            signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)

        while signal.shape[1] < self.num_points:
        # Pad with zeros if shorter length
            padding = np.zeros((signal.shape[0], 1))
            signal = np.hstack((signal, padding))
        if signal.shape[1] > self.num_points:
        # Truncate if longer length
            signal = signal[:, :self.num_points]
        # convert to torch tensor, the copy is due to torch bug
        signal = torch.FloatTensor(signal.copy())
        
        return signal

    def __getitem__(self, idx):
        try:
            # get name and label of the idx-th sample
            event_file = self.df.iloc[idx]['event_file']
            label = self.df.iloc[idx]['fraction_of_yes']
            # load signal of the idx-th sample
            path_signal = os.path.join(self.path_folder,event_file+'.npy')
            signal = np.load(path_signal)
            # preprocess signal
            #print(signal.shape)
            signal = self._preprocess(signal)
            #print(signal.shape)

            # return signal          
            return signal,label
        except Exception as e:
            print(f"Error at index {event_file}: {e}")
            raise

class Hardmine_BonoboDataset(torch.utils.data.Dataset):
    def __init__(self,df,
                 path_folder, 
                 montage=None, 
                 transform_neg=None,
                 transform=None,
                 transform_pos=None,
                 window_size=1,
                 fq = 128,
                 num_pos_augmentations=4 # 2, 3, 4
                 ):
        self.df = df
        # set transform
        self.transform_neg = transform_neg
        self.transform_pos = transform_pos
        self.transform = transform
        # set montage
        self.montage = montage
        # set path to bucket
        self.path_folder = path_folder
        self.num_pos_augmentations = num_pos_augmentations
        self.label_filter = self.df['fraction_of_yes'] >= 0.75
        self.mode_filter = self.df['Mode'] == 'Train'
        self.num_points = fq * window_size
    def __len__(self):
        num_positives = len(self.df[self.label_filter & self.mode_filter])
        num_negatives = len(self.df) - num_positives
        return num_negatives + num_positives * self.num_pos_augmentations
        #return len(self.df)

    def _preprocess(self,signal,label):
        # convert to desired montage
        if self.montage is not None:
            signal = self.montage(signal)

        # apply transformations
        if self.transform is not None:
            signal = self.transform(signal)        

        if self.transform_pos is not None:
            if label >= 0.1:
                signal = self.transform_pos(signal)

        if self.transform_neg is not None:
            if label < 0.1:
                signal = self.transform_neg(signal)

        if signal.shape[-1] != 0:
        # normalize signal
            signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)

        while signal.shape[1] < self.num_points:
        # Pad with zeros if shorter length
            padding = np.zeros((signal.shape[0], 1))
            signal = np.hstack((signal, padding))
        if signal.shape[1] > self.num_points:
        # Truncate if longer length
            signal = signal[:, :self.num_points]
        # convert to torch tensor, the copy is due to torch bug
        signal = torch.FloatTensor(signal.copy())
        
        return signal

    def __getitem__(self, idx):
        try:
            # If the index is greater than the length of the original dataframe,
            # it's an augmented positive sample
            if idx >= len(self.df):
                idx = (idx - len(self.df)) % len(self.df[self.label_filter & self.mode_filter])
            # get name and label of the idx-th sample
            event_file = self.df.iloc[idx]['event_file']
            label = self.df.iloc[idx]['fraction_of_yes']
            # load signal of the idx-th sample
            path_signal = os.path.join(self.path_folder,event_file+'.npy')
            signal = np.load(path_signal)
            # preprocess signal
            #print(signal.shape)
            signal = self._preprocess(signal,label)
            #print(signal.shape)

            # return signal          
            return signal,label
        except Exception as e:
            print(f"Error at index {event_file}: {e}")
            raise

class SpikeDetectionDataset(torch.utils.data.Dataset):
    def __init__(self,df,
                 path_folder, 
                 montage=None, 
                 transform=None):
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
        #chan_score = self.df.iloc[idx]['Fp1']
        chan_columns = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        chan_score = {chan: self.df.iloc[idx][chan] for chan in chan_columns}
        # load signal of the idx-th sample
        path_signal = os.path.join(self.path_folder,event_file+'.npy')
        signal = np.load(path_signal)
        # preprocess signal
        signal = self._preprocess(signal)
        # return signal          
        return signal,label,chan_score

class ContinousToSnippetDataset(torch.utils.data.Dataset):
    # Dataset that takes a continous signal and returns snippets of a given length
    # input shape: (n_channels,n_timepoints), output shape: (n_snippets,n_channels,ts)
    def __init__(self,
                 path_signal,
                 montage=None, 
                 transform=None,
                 Fq=128,
                 window_size=2,
                 step=32):
 
        # load signal
        signal = mat73.loadmat(path_signal)['data']
        # move signal to torch
        signal = torch.FloatTensor(signal.astype(np.float32))
        # generate snippets of shape (n_snippets,n_channels,ts)
        self.snippets = signal.unfold(dimension = 1,size = window_size*Fq, step = step).permute(1,0,2)
        # set transform
        self.transform = transform
        # set montage
        self.montage = montage


    def __len__(self):
        # get item zero of self. snippets, which has shape (n_snippets,n_channels,ts)
        return self.snippets.shape[0]

    def _preprocess(self,signal):
        '''preprocess signal and apply montage, transform and normalization'''

        if self.montage is not None:
            signal = self.montage(signal)

        # apply transformations
        if self.transform is not None:
            signal = self.transform(signal)

        # normalize signal
        signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)

        
        signal = torch.FloatTensor(signal.copy())

        return signal

    def __getitem__(self, idx):
        # get the snippet        
        signal = self.snippets[idx,:,:]
        # preprocess signal
        signal = self._preprocess(signal)
        
        # return signal and dummy label, the latter to prevent lightning dataloader from complaining
        return signal,0

class ECG2BonoboDataset(torch.utils.data.Dataset):
    def __init__(self,df,
                 path_folder, 
                 montage=None, 
                 transform=None):
        self.df = df
        # set transform
        self.transform = transform
        # set montage
        self.montage = montage
        # set path to bucket
        self.path_folder = path_folder
        self.fs_out = 1000 #2 * 500Hz

    def __len__(self):
        return len(self.df)
    
    def resample_unequal(self, ts, fs_in, fs_out):
        if fs_in == 0 or len(ts) == 0:
            return ts
        t = ts.shape[1] / fs_in
        fs_in, fs_out = int(fs_in), int(fs_out)
    
        if fs_out == fs_in:
            return ts
        if 2 * fs_out == fs_in:
            return ts[:, ::2]
    
        # 插值后的新长度为 fs_out
        resampled_ts = np.zeros((ts.shape[0], fs_out))
        x_old = np.linspace(0, t, num=ts.shape[1], endpoint=True)  # 包含原始序列的终点
        x_new = np.linspace(0, t, num=int(fs_out), endpoint=True) 
        for i in range(ts.shape[0]):
            y_old = ts[i, :]
            f = interp1d(x_old, y_old, kind='linear')
            resampled_ts[i, :] = f(x_new)
    
        return resampled_ts

    def _preprocess(self,signal):
        # convert to desired montage
        if self.montage is not None:
            signal = self.montage(signal)

        # apply transformations
        if self.transform is not None:
            signal = self.transform(signal)

        if signal.shape[-1] != 0:
        # normalize signal
            signal = signal / (np.quantile(np.abs(signal), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)
            #signal = 2 * ((signal - np.min(signal)) / (np.max(signal) - np.min(signal))) - 1
        while signal.shape[1] < 256:
        # Pad with zeros if shorter length
            padding = np.zeros((signal.shape[0], 1))
            signal = np.hstack((signal, padding))
        if signal.shape[1] > 256:
        # Truncate if longer length
            signal = signal[:, :256]
        # convert to torch tensor, the copy is due to torch bug
        signal = torch.FloatTensor(signal.copy())
        
        return signal

    def __getitem__(self, idx):
        try:
            # get name and label of the idx-th sample
            event_file = self.df.iloc[idx]['event_file']
            label = self.df.iloc[idx]['fraction_of_yes']
            sample_rate = 128
            # load signal of the idx-th sample
            path_signal = os.path.join(self.path_folder,event_file+'.npy')
            signal = np.load(path_signal)
            
            # preprocess signal
            #print(signal.shape)
            signal = self._preprocess(signal)
            signal = self.resample_unequal(signal, sample_rate, self.fs_out)
            signal = torch.FloatTensor(signal.copy())
            #print(signal.shape)

            # return signal          
            return signal,label
        except Exception as e:
            print(f"Error at index {event_file}: {e}")
            raise
