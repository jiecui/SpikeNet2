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

class ECG_channel_flip():
    def __init__(self, p):
        self.p = p

        # Predefine the channels
        monopolar_channels = ['FP1','F3','C3','P3','F7','T3','T5','O1',
                              'FZ','CZ','PZ',
                              'FP2','F4','C4','P4','F8','T4','T6','O2']
        bipolar_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
                            'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                            'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
                            'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
                            'Fz-Cz', 'Cz-Pz']
        channel_average = ['Fp1-avg','F3-avg','C3-avg','P3-avg','F7-avg','T3-avg','T5-avg','O1-avg','Fz-avg','Cz-avg','Pz-avg','Fp2-avg','F4-avg','C4-avg','P4-avg','F8-avg','T4-avg','T6-avg','O2-avg']#19

        channels_flipped = []
        

        # Handle monopolar montage
        channels_flipped.extend(self._bipolar_flipper(bipolar_channels))
        # Handle bipolar montage
        channels_flipped.extend(self._monopolar_flipper(monopolar_channels))
        
        # Consolidate all the channels
        all_channels = bipolar_channels + monopolar_channels
        # Convert strings to indices, this is later applied to the signal
        self.flipped_order = [all_channels.index(flipped_channel) for flipped_channel in channels_flipped]

        # print output to check

    def _flip_channel(self,channel):
        '''
        this function flips a channel string, e.g. Fp1 -> Fp2, F3 -> F4, etc.
        Central channels are kept in place, e.g. Fz -> Fz
        '''
        loc = channel[-1]
        # keep central channels in place
        if loc.lower() == 'z':
            return channel
        # flip all other channels
        else: loc = int(loc)
        if loc % 2 ==1: # +1 all uneven channels. Fp1 -> Fp2, F3 -> F4, etc.
            loc +=1
        else: # -1 all even channels. Fp2 -> Fp1, F4 -> F3,etc.
            loc -= 1
        # recompose channel string
        channel = channel[:-1] + str(loc)
        return channel

    def _monopolar_flipper(self,channels):
        ''' flip all channels in a list of monopolar channels'''
        channels_flipped = []
        for channel in channels:
            channel = self._flip_channel(channel)
            channels_flipped.append(channel)

        return channels_flipped
       
    def _bipolar_flipper(self,channels):
        ''' for both channels in bipolar channel, flip them separately
        keeps zentral in place '''
        channels_flipped = []
        for bipolar_channel in channels:
            numbers = [int(x) for x in bipolar_channel if x.isdigit()]
            # dont flip if channel does not contain numbers (e.g. Fz-Cz, etc.)
            # dont flip if sum of channel numbers is uneven (e.g. Fp1-F3, C3-C4, etc.)
            if (sum(numbers)==0) | (sum(numbers)%2==1):
                channels_flipped.append(bipolar_channel)
            else:
                channel1, channel2 = bipolar_channel.split('-')
                channel1, channel2 = self._flip_channel(channel1), self._flip_channel(channel2)
                channels_flipped.append(channel1+'-'+channel2) # recomposebipolar channel
        #print(channels_flipped)    
        return channels_flipped
   
    def __call__(self, signal):
        #  ECG 
        ECG_channel_data = signal[-1, :].reshape(1, -1)  # 

        if np.random.random() < self.p:
            signal = signal[:-1][self.flipped_order, :]
            signal = np.vstack([signal, ECG_channel_data])
        else:
            signal = np.vstack([signal[:-1], ECG_channel_data])

        return signal

class channel_flip():
    def __init__(self, p):
        self.p = p

        # Predefine the channels
        monopolar_channels = ['FP1','F3','C3','P3','F7','T3','T5','O1',
                              'FZ','CZ','PZ',
                              'FP2','F4','C4','P4','F8','T4','T6','O2']
        bipolar_channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
                            'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
                            'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
                            'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
                            'Fz-Cz', 'Cz-Pz']
        channel_average = ['Fp1-avg','F3-avg','C3-avg','P3-avg','F7-avg','T3-avg','T5-avg','O1-avg','Fz-avg','Cz-avg','Pz-avg','Fp2-avg','F4-avg','C4-avg','P4-avg','F8-avg','T4-avg','T6-avg','O2-avg']#19

        channels_flipped = []
        

        # Handle monopolar montage
        channels_flipped.extend(self._bipolar_flipper(bipolar_channels))
        # Handle bipolar montage
        channels_flipped.extend(self._monopolar_flipper(monopolar_channels))
        
        # Consolidate all the channels
        all_channels = bipolar_channels + monopolar_channels
        # Convert strings to indices, this is later applied to the signal
        self.flipped_order = [all_channels.index(flipped_channel) for flipped_channel in channels_flipped]

        # print output to check

    def _flip_channel(self,channel):
        '''
        this function flips a channel string, e.g. Fp1 -> Fp2, F3 -> F4, etc.
        Central channels are kept in place, e.g. Fz -> Fz
        '''
        loc = channel[-1]
        # keep central channels in place
        if loc.lower() == 'z':
            return channel
        # flip all other channels
        else: loc = int(loc)
        if loc % 2 ==1: # +1 all uneven channels. Fp1 -> Fp2, F3 -> F4, etc.
            loc +=1
        else: # -1 all even channels. Fp2 -> Fp1, F4 -> F3,etc.
            loc -= 1
        # recompose channel string
        channel = channel[:-1] + str(loc)
        return channel

    def _monopolar_flipper(self,channels):
        ''' flip all channels in a list of monopolar channels'''
        channels_flipped = []
        for channel in channels:
            channel = self._flip_channel(channel)
            channels_flipped.append(channel)

        return channels_flipped
       
    def _bipolar_flipper(self,channels):
        ''' for both channels in bipolar channel, flip them separately
        keeps zentral in place '''
        channels_flipped = []
        for bipolar_channel in channels:
            numbers = [int(x) for x in bipolar_channel if x.isdigit()]
            # dont flip if channel does not contain numbers (e.g. Fz-Cz, etc.)
            # dont flip if sum of channel numbers is uneven (e.g. Fp1-F3, C3-C4, etc.)
            if (sum(numbers)==0) | (sum(numbers)%2==1):
                channels_flipped.append(bipolar_channel)
            else:
                channel1, channel2 = bipolar_channel.split('-')
                channel1, channel2 = self._flip_channel(channel1), self._flip_channel(channel2)
                channels_flipped.append(channel1+'-'+channel2) # recomposebipolar channel
        #print(channels_flipped)    
        return channels_flipped
   
    def __call__(self, signal):
        # if random number is smaller than p, flip channels
        if np.random.random() < self.p:
            signal = signal[self.flipped_order,:]
            #print(signal.shape)
        return signal
# write a pytorch transform to flip the monopolar signal, specific for CDAC data storage convention
# input shape (n_channels=19,n_timepoints) = output shape
class CDAC_monopolar_signal_flip():
    def __init__(self, p):
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
    def __init__(self, p):
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

class extremes_remover():
    def __init__(self,signal_max = 2000, signal_min = 20):
        self.signal_max = signal_max #microV
        self.signal_min = signal_min #microV

    def __call__(self,signal):
        for channel in range(signal.shape[0]):
            if signal[channel, :].size > 0:
            # get peak to peak            
              PP = signal[channel,:].max() - signal[channel,:].min()
              # blank out artifact channel wise
              if (PP<self.signal_min) | (PP>self.signal_max):
                  signal[channel,:] = 0
            else:
                signal[:,:] = 0
                pass
        return signal
'''
class extremes_remover():
    def __init__(self,signal_max = 2000, signal_min = 2):
        self.signal_max = signal_max #mV
        self.signal_min = signal_min #mV
    def __call__(self,signal):
        for channel in range(signal.shape[0]):
            too_large = (np.abs(signal[channel,:]).max()>self.signal_max).any()
            too_small = (np.abs(signal[channel,:]).max()<self.signal_min).any()
            if too_large:
                signal[channel,:] = 0
            if too_small:
                signal[channel,:] = 0    
        return signal
'''
class random_channel_zero():
    # with certain probability, zero out a channel (no limit to how many)
    def __init__(self,p=0.1):
        self.p = p
        pass
    def __call__(self, signal):
        # zero out a random channel
        # chans will be 1 with prob 1-p, and 0 with prob p
        chans = 1.0*(np.random.rand(signal.shape[0])>self.p)
        # we do matrix multiply to zero out some channels
        signal = chans[:, np.newaxis] * signal
        
        return signal

class noise_adder():
    def __init__(self, s=0.1, p=0.1):
        self.s = s
        self.p = p

    def __call__(self, signal):
        if np.random.rand() < self.p:
            return signal + np.random.randn(*signal.shape) * self.s
        else:
            return signal
