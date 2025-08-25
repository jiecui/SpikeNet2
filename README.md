# SpikeNet2


This is the official implementation of our paper "Expert-Level Detection of Epilepsy Markers in EEG on Short and Long Timescales".


## Dependencies

To clone all files:


```
git clone https://github.com/bdsp-core/SpikeNet2.git
```



To install Python dependencies:


```
conda create -n SpikeNet2 python=3.10
conda activate SpikeNet2
pip install -r requirements.txt
```



## Data 

[Spikenet-2](https://bdsp.io/content/spikenet/2.0/)

## Preparation

First, you need to download the EEG data above. Then run the jupyter notebook to transfer the '.mat' files into '.npy' files.

```
transfer_data.ipynb
```

Next, configure your file '/sleeplib/config.py'. Fill in your path into 'your_path'.

## Running Training

Run the following command to perform initial training of SpikeNet2.  

```
python train_initial_model.py
```


After we get the initial model,  it can be performed on control EEG dataset and get thousands of hard negative samples. Run the following command to perform the model on control EEG and get the predictions of EEG.

```
python continurous.py
```



Next, to get thousands of hard negative samples, we can run the code to get them.

```
hard_mining.ipynb
```



Then we get the new samples to run another round of hard mining.

```
python train_hard_model.py
```



If you want to check the model performance, please run the code

```
prediction.ipynb
```



## Citation
If you found our work useful in your research, please consider citing our works at:
> ```
>@article{li2025expert,
>  title={Expert-Level Detection of Epilepsy Markers in EEG on Short and Long Timescales},
>  author={Li, Jun and Goldenholz, Daniel M and Alkofer, Moritz and Sun, Chenxi and Nascimento, Fabio A and Halford, Jonathan J and Dean, Brian C and Galanti, Mattia and Struck, Aaron F and Greenblatt, Adam S and others},
>  journal={NEJM AI},
>  volume={2},
>  number={7},
>  pages={AIoa2401221},
>  year={2025},
>  publisher={Massachusetts Medical Society}
>}
> ```
