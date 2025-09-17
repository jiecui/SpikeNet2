create the enviornment
----------------------
conda create -n SpikeNet2 python=3.10
conda activate SpikeNet2
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements_gcp.txt
pip install ipython

test
----
# in python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.backends.cudnn.version())  # Should match cuDNN 8.9.x
