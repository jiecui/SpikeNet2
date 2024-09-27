import numpy as np

# cuts out a window of size windowsize from the signal, centered around the middle of the signal and shifted by a random offset
#shape goes from (channels,timesteps) -> (channels,windowsize*Fq)
class cut_and_jitter():
    def __init__(self,windowsize,max_offset,Fq):
        self.windowsize = windowsize*Fq
        self.max_offset = max_offset*Fq

    def __call__(self,signal):
        # get index of center
        center = signal.shape[1]//2
        # get index of window start
        start = center - (self.windowsize)//2
        # shift by up -1 or 1 x offsetsize
        start = start + int(np.random.uniform(-1, 1)*self.max_offset)
        return signal[:,start:start+self.windowsize]

# write a pytorch transform to flip the monopolar signal, specific for CDAC data storage convention
# input shape (n_channels=19,n_timepoints) = output shape
class CDAC_monopolar_signal_flip():
    def __init__(self, p=0.5):
        self.p = p
        # original order of mono_channels   = [‘FP1’,‘F3’,‘C3’,‘P3’,‘F7’,‘T3’,‘T5’,‘O1’,
        #                                      ‘FZ’,‘CZ’,‘PZ’,
        #                                      ‘FP2’,‘F4’,‘C4’,‘P4’,‘F8’,‘T4’,‘T6’,‘O2’]
        # flip channels on the sides, keep the three middle ones in place
        self.flipped_order = np.concatenate([np.arange(11,19),np.array([8,9,10]),np.arange(0,8)])
    def __call__(self, signal):
        if np.random.random() < self.p:
            signal = signal[self.flipped_order,:]
        return signal

# write a pytorch transform to flip the monopolar signal, specific for CDAC data storage convention\
# input shape (n_channels=18,n_timepoints) = output shape
class CDAC_bipolar_signal_flip():
    def __init__(self, p=0.5):
        self.p = p
        # original order of biploar channels := ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
        #                                        'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
        #                                        'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        #                                        'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2'
        #                                        'Fz-Cz', 'Cz-Pz'        
        #                                         ]
        # flip first two and last two rows of channel, keep 'Fz-Cz', 'Cz-Pz' in place        
        self.flipped_order = np.concatenate([np.arange(4,8),np.arange(0,4),np.arange(12,16),np.arange(8,12),np.array([16,17])])
    def __call__(self, signal):
        if np.random.random() < self.p:
            signal = signal[self.flipped_order,:]
        return signal

class delete_channels():
    # replaces n_deletes channels with zeros
    def __init__(self,n_channels, n_deletes='random', channels_to_delete='random'):
        self.n_channels = n_channels
        # n_deletes: number of channels to delete, can be an integer or 'random'
        self.n_deletes = n_deletes
        # channels_to_delete: list of channels to delete, can be a list of integers or 'random'
        self.channels_to_delete = channels_to_delete

    def __call__(self,signal):
        # use given n_deletes or generate a random in int range 0, n_channels using a uniform distribution
        if self.n_deletes != 'random':
            n_deletes = self.n_deletes
        elif self.n_deletes == 'random':
            n_deletes = np.random.randint(0,self.n_channels)
            
        # use given list or generate a list containing n random numbers in range 0, n_channels
        if self.channels_to_delete != 'random':
            channels_to_delete = self.channels_to_delete
        elif self.channels_to_delete == 'random':
            channels_to_delete = np.random.choice(self.n_channels, n_deletes, replace=False)

        # replace desired channels with zeros
        signal[channels_to_delete,:] = 0
        return signal