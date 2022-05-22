# Gaussian GANs

## Requirement

 - cuda 11.1
 - pytorch 1.8.1

 <sub> pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html 
  </sub>

## Usage 
**Preinstall: wandb is used in the montoring process, thus go to [wandb official website](wandb.ai) and conduct the initial settings**

   - Wandb is recommended, because its visualization on image is better than tensorboard.

   - If you are not familiar with the wandb and would like to use tensorboard instead, we  provide tensorboard monitor in our code as well. 
   But, you have to manully delete the wandb parts in our code. 

### Simulation part 

1. (Optional) Training the Gaussian GANs to transform the Gaussian Random field (GRF) into strandard multivariate Gaussian distribution;
GRF contains **pointwise transformation and truncation**.

```
cd ./train-terminal/
python SimuInverseGAN_train.py
```
   - The well-trained models are provided in the file ```train-terminal/save/```, and you can use them without training new Gaussian GANs (around 2hrs).
   - If your would like to reproduce the paper, you need to use three combinations of arguments: 

     - ```--truncate True```
     - ```--transform True```
     - ```--transform True \ --truncate True```


2. Training the generative model (GANs) under the task of GRF, while the well-trained Gaussian GANs would serve as a monitor to keep the trace of goodness of generated samples during the training process;
```
python Simu_train.py
```
<sub> Note: the whole process would be automatically monitored by visualization tools and numeric metrics; </sub>

3. Check the results on wandb account
   



### Real dataset

For the Running dataset (same as Jumping):

1. (Optional) Training the Gaussian GANs to transform the true data into strandard multivariate Gaussian distribution;

```
cd ./train-terminal/
python RunningInverseGAN_train.py
```

Note: The well-trained models are provided in the file ```train-terminal/save/```, and you can use them without training new Gaussian GANs (around 2hrs). 


2. Training the generative model (GANs) under the task of the multivariate time series, while the well-trained Gaussian GANs would serve as a monitor to keep the trace of goodness of generated samples during the training process;
```
python RunningGAN_Train.py
```


## Description 
 - **Dataset**
 The way of downloading dataset is contained the folder ```./dataset/```, which would be automatically downloaded and simulated when training the model.
 
 - **Models**
 The architecture of models contained the ```./models/```
 
 - **Visualization Tools**
 The heatmap for correlation matrix, QQ-plot, and PCA/t-SNE are all contained in the ```./utils```
 
 ## TO DO
 
 - [ ] Do the simulation verifcation on Gaussian Random Field (GRF) 
