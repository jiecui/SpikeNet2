# config.py

# 2025 Richard J. Cui. Modified: Fri 09/12/2025 04:16:14.055411 PM
# $Revision: 0.3 $  $Date: Fri 09/26/2025 02:34:19.732870 PM $
#
# Mayo Clinic Foundation
# Rochester, MN 55901, USA
#
# Email: Cui.Jie@mayo.edu

import os
from typing import List, Union
from dataclasses import dataclass
from spikenet2_lib import get_database_root, get_output_root, get_proj_root


@dataclass
class Config:

    # Data params
    # 'your_path' is the path of your project
    PATH_FILES_BONOBO: str = os.path.join(get_output_root(), "Events", "real_npy")
    PATH_LUT_BONOBO: str = os.path.join(get_proj_root(), "lut_labelled_20230628.csv")
    PATH_CONTINOUS_EEG: str = os.path.join(
        get_database_root(), "EEG", "hm_negative_eeg"
    )

    FQ: int = 128  # Hz

    # Preprocessing
    MONTAGE: str = "combine"
    WINDOWSIZE: int = 1  # 2 seconds (cut length of EEG signals)

    # Model parameters
    N_CHANNELS: int = 37  # 19+18

    # training parameters
    BATCH_SIZE: int = 256  # test 128
    LR: float = 1e-4  # test 1e-4

    # accelerator
    DEVICES: Union[List[int], str, int] = "auto"

    def print_config(self):
        print("THIS CONFIG FILE CONTAINS THE FOLLOWING PARAMETERS :\n")
        for key, value in self.__dict__.items():
            print(key, value)


# [EOF]
