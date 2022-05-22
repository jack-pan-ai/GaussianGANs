# Gaussian GANs

## Requirement

 - cuda 11.1
 - pytorch 1.8.1

 <sub> pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html 
  </sub>

## Usage 
**Preinstall: wandb is used in the montoring process, thus go to [wandb official website](wandb.ai) and conduct the initial settings**

For the Running dataset:

 - (Optional) Firstly, training a generative model to generate the multivariate time series;
```
cd ./train-terminal/
python RunningGAN_Train.py
```
The well-trained models are provided in the file ```train-terminal/save/```

 - Secondly, traing the Gaussian GANs to transform the true data into multivariate Gaussian distribution;
```
python RunningInverseGAN_train.py
```
<sub> Note: the whole process would be automatically monitored by visualization tools and numeric metrics; </sub>

- Thirdly, check the results on wandb account

## Description 
 - **Dataset**
 The way of downloading dataset is contained the folder ```./dataset/```, which would be automatically downloaded and simulated when training the model.
 
 - **Models**
 The architecture of models contained the ```./models/```
 
 - **Visualization Tools**
 The heatmap for correlation matrix, QQ-plot, and PCA/t-SNE are all contained in the ```./utils```
 
 ## TO DO
 
 - [ ] Do the simulation verifcation on Gaussian Random Field (GRF) 
