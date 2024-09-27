import numpy as np
import torch
# this class is used to convert a signal from monopolar to bipolar montage, using the CDAC data convention
#input 20 channels, 19 monopolar, 1 EKG
#output 37 channels, all montage
class CDAC_bipolar_montage():
    def __init__(self):
        mono_channels    = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']
        bipolar_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']#18

        self.bipolar_ids = np.array([[mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])

    def __call__(self, signal):
        # Bipolar Montage
        bipolar_signal = signal[self.bipolar_ids[:, 0]] - signal[self.bipolar_ids[:, 1]]

        return bipolar_signal
   
class CDAC_common_average_montage():
    def __init__(self):
        mono_channels    = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']
        channel_average = ['Fp1-avg','F3-avg','C3-avg','P3-avg','F7-avg','T3-avg','T5-avg','O1-avg','Fz-avg','Cz-avg','Pz-avg','Fp2-avg','F4-avg','C4-avg','P4-avg','F8-avg','T4-avg','T6-avg','O2-avg']#19


        self.average_ids = [mono_channels.index(ch.split('-')[0]) for ch in channel_average]

    def __call__(self, signal):
        # Common Average Montage
        common_average_signal = signal[self.average_ids] - np.mean(signal, axis=0)

        return common_average_signal

class ECG_combine_montage():
    def __init__(self):
        mono_channels    = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']
        bipolar_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
        channel_average = ['Fp1-avg','F3-avg','C3-avg','P3-avg','F7-avg','T3-avg','T5-avg','O1-avg','Fz-avg','Cz-avg','Pz-avg','Fp2-avg','F4-avg','C4-avg','P4-avg','F8-avg','T4-avg','T6-avg','O2-avg']

        self.bipolar_ids = np.array([[mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])

        self.average_ids = [mono_channels.index(ch.split('-')[0]) for ch in channel_average]

    def __call__(self, signal):
        bipolar_signal = signal[self.bipolar_ids[:, 0]] - signal[self.bipolar_ids[:, 1]]
        common_average_signal = signal[self.average_ids] - np.mean(signal[self.average_ids], axis=0, keepdims=True)

        ECG_signal = signal[-1, :]

        combined_signal = np.vstack([bipolar_signal, common_average_signal, ECG_signal])

        return combined_signal

class CDAC_noECG_montage():
    def __init__(self):
        self.mono_channels = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

    def __call__(self, signal):
        # Assuming signal is of shape (channels, time), and channels are in the same order as in self.mono_channels
        all_channels = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2','ECG']
        channel_indices = [all_channels.index(ch) for ch in self.mono_channels if ch in all_channels]
        eeg_signal = signal[channel_indices]

        return eeg_signal

class CDAC_combine_montage():
    def __init__(self):
        mono_channels    = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']
        bipolar_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']#18
        channel_average = ['Fp1-avg','F3-avg','C3-avg','P3-avg','F7-avg','T3-avg','T5-avg','O1-avg','Fz-avg','Cz-avg','Pz-avg','Fp2-avg','F4-avg','C4-avg','P4-avg','F8-avg','T4-avg','T6-avg','O2-avg']#19

        self.bipolar_ids = np.array([[mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])

        self.average_ids = [mono_channels.index(ch.split('-')[0]) for ch in channel_average]

    def __call__(self, signal):
        bipolar_signal = signal[self.bipolar_ids[:, 0]] - signal[self.bipolar_ids[:, 1]]
        #print(signal.shape)
        common_average_signal = signal[self.average_ids] - np.mean(signal[self.average_ids], axis=0, keepdims=True)

        #ECG_signal = signal[19, :]

        combined_signal = np.vstack([bipolar_signal, common_average_signal])

        return combined_signal
    
class con_combine_montage():
    def __init__(self):
        mono_channels    = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']
        bipolar_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']#18
        channel_average = ['Fp1-avg','F3-avg','C3-avg','P3-avg','F7-avg','T3-avg','T5-avg','O1-avg','Fz-avg','Cz-avg','Pz-avg','Fp2-avg','F4-avg','C4-avg','P4-avg','F8-avg','T4-avg','T6-avg','O2-avg']#19

        self.bipolar_ids = np.array([[mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])

        self.average_ids = [mono_channels.index(ch.split('-')[0]) for ch in channel_average]

    def __call__(self, signal):
        bipolar_signal = signal[self.bipolar_ids[:, 0]] - signal[self.bipolar_ids[:, 1]]
        
        common_average_signal = signal[self.average_ids] - torch.mean(signal[self.average_ids], dim=0, keepdim=True)

        combined_signal = np.vstack([bipolar_signal, common_average_signal])

        return combined_signal    
    
class con_ECG_combine_montage():
    def __init__(self):
        mono_channels    = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']
        bipolar_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']#18
        channel_average = ['Fp1-avg','F3-avg','C3-avg','P3-avg','F7-avg','T3-avg','T5-avg','O1-avg','Fz-avg','Cz-avg','Pz-avg','Fp2-avg','F4-avg','C4-avg','P4-avg','F8-avg','T4-avg','T6-avg','O2-avg']#19

        self.bipolar_ids = np.array([[mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])

        self.average_ids = [mono_channels.index(ch.split('-')[0]) for ch in channel_average]

    def __call__(self, signal):
        bipolar_signal = signal[self.bipolar_ids[:, 0]] - signal[self.bipolar_ids[:, 1]]
        
        common_average_signal = signal[self.average_ids] - torch.mean(signal[self.average_ids], dim=0, keepdim=True)

        ECG_signal = signal[-1, :]

        combined_signal = np.vstack([bipolar_signal, common_average_signal, ECG_signal])

        return combined_signal          