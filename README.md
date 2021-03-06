# Optical flow prediction using Generative Adversarial Neural Networks
<p align="center"> Alex Mathai(BITS Pilani)    |   Vidit Jain(BITS Pilani) </p>

# TL;DR
This repository contains scripts for training a Generative Adversarial Framework for Optical Flow Prediction. This is an ongoing project. We have modelled optical flow prediction 
in this way, so that this framework can later on be used for the purpose of novelty detection.

# Environment
To easily replicate my environment, please clone the **environment.yml** file using **Conda**.

# Data
We have used the **Fall Detection Dataset** which can be found on this [website](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html). It is divided into two further sections - **Fall** and **Non-Fall**.

The **Fall** section contains videos of people falling.

The **Non-Fall** section contains videos of people completing some task. They do not fall at any point in time.

We have already provided a [link](https://drive.google.com/drive/folders/1GWpGggmkS2F_YrXTPhRWttQUzF321dLX?usp=sharing) to the preprocessed data, so there is no need of downloading the dataset.

## Steps
1. Download the folders ```test_data```, ```train_data``` and move all the data to the ```Code/train_data``` and ```Code/test_data``` folders.
2. Download the ```adl_dataset.hdf5``` and the ```adl_dataset_test.hdf5``` files and put them in the ```Code``` folder.

# Optical Flow
The optical flow between frames helps to focus on the the movement of the objects we are interested in. A simple example is given below.
<p align="middle">
  <img src="/Images/optical_flow.png" width="600"/>
</p>

On the left is the result of super-imposing 2 successive frames. On the right is the optical flow of the tennis player that highlights her movement.

# Preprocessing using [PwC Net](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch)
We pass successive frames from the **Fall Dataset** to **PwC Net** in order to get optical-flows. If you have followed the steps in the **Data** section then the optical flows can be located in the ```Code/train_data``` and ```Code/test_data``` folders.

# Architecture
In this section we explain to you the flow of the data through the networks and the basic network components.
We create stacks of 10 optical flows (from 20 video frames) that were produced by PwC-Net. For each stack, we slice out the last optical flow and feed in the beginning 9 optical flows through the generator. We then make the generator predict the 10th optical flow. 

## Generator U-Net
<p align="middle">
  <img src="/Images/generator.png" width="600"/>
</p>

We concatenate the predicted optical flow with the previous 9 optical flows. We then pass this stack to the discriminator and ask the discriminator whether it thinks if this predicted optical flow is real or fake. This is the basic adversarial framework.

## Discriminator Convolution Network
<p align="middle">
  <img src="/Images/discriminator.png" width="600"/>
</p>

The adversarial training of this framework (detailed in the section below) helps in the prediction of optical flows to be very accurate.

# Loss Functions

## Generator
<p align="middle">
  <img src="/Images/generator_eqn.gif" width="300"/>
</p>

In the generator equation, the first term is simply the mean squared error of pixel values in the predicted flow. The second term is the loss for not being able to fool the discriminator.

## Discriminator
<p align="middle">
  <img src="/Images/discriminator_eqn.gif" width="300"/>
</p>

In the discriminator equation, the first term is the loss for not being able to predict the actual optical flows as real. The second term is the loss for not being able to predict the generated optical flow as fake.

# Code Structure
```
Code
├──  test_data
├──  train_data
├──  parameters
├──  create_test_adl_dataset.ipynb
├──  create_train_adl_dataset.ipynb
├──  flow_gans.ipynb
├──  Model U-Net for Optical Flow Estimation.ipynb
```

There are four important files in the Code folder

1. create_train_adl_dataset.ipynb
2. create_test_adl_dataset.ipynb
3. flow_gans.ipynb
4. Model U-Net for Optical Flow Estimation.ipynb

If you have already downloaded the data from the link that was previously provided, then you do NOT need to run the 1st three jupyter notebooks.

### create_train_adl_dataset.ipynb
This script creates the training dataset - an efficient ".hdf5" file named "adl_dataset". This combines the preprocessed frames in the ```Code/train_data``` folder which were already passed through PwC Net.

### create_test_adl_dataset.ipynb
This script creates the testing dataset - an efficient ".hdf5" file named "adl_dataset_test". This combines the preprocessed frames in the ```Code/test_data``` folder which were already passed through PwC Net.

### Model U-Net for Optical Flow Estimation.ipynb
A temporary file that includes the architectures of the generator and the discriminator. If this is just what you are looking for, then you can use the architectures for your research purposes.

### flow_gans.ipynb
This script contains the training code for GANs and also predicts and saves some optical flow frames from the training_data.


<!-- Step 1: Create a new conda environment using the environment.yml file

Step 2 : Please click the link : https://drive.google.com/drive/folders/1GWpGggmkS2F_YrXTPhRWttQUzF321dLX?usp=sharing
Download all the subfolders and place them within the "Code" folder.

Step 3: All files can be run by "jupyter notebook <filename.ipynb>"

	A) The output of "jupyter notebook adl_dataset_create.ipynb" in the terminal will give you "adl_dataset.hdf5" which you have already downloaded from google drive

	B) The output of "jupyter notebook adl_dataset_test.ipynb" in the terminal will give you "adl_dataset_test.hdf5" which you have already downloaded from google drive
	
	C) Finally Run "jupyter notebook flow_gans.ipynb" 
	
Please Note the ".npy" files in the train_data and test_data folder have been created by the script "pwcnet_predict_from_img_pairs.ipynb" located in the tfoptflow.
"pwcnet_predict_from_img_pairs.ipynb" is the only file we created, all other files have been provided by Nvidia.
-->

# Results

## Training Data Samples
Left Images are the **Ground Truth** and the right images are the **Predictions**
<p align="middle">
  <img src="/Images/ex1_ground_truth.png" width="300" hspace="50"/>  
  <img src="/Images/ex1_predicted.png" width="300"/>
</p>
<p align="middle">
  <img src="/Images/ex2_ground_truth.png" width="300" hspace="50"/>  
  <img src="/Images/ex2_predicted.png" width="300"/>
</p>
