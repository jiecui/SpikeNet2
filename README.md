# SpikeNet2



This repository contains code to train a deep learning model on EEG spikes and evaluate this model's performance on event-level EEG tasks.





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

will be released on bdsp.io



### Model Checkpoints

Model checkpoints of  SpikeNet2 can be found on bdsp.io.





## Running Training

Run the following command to perform initial training of SpikeNet2.  

```
python train_initial_model.py
```



After we get the initial model,  it can be performed on control EEG dataset and get thousands of hard negative samples. Run the following command to perform the model on control EEG and get the predictions of EEG.

```python continurous.py
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

...
