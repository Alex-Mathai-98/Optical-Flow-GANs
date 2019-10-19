# Optical flow prediction using Generative Adversarial Neural Networks
<p align="center"> Alex Mathai(BITS Pilani)    |   Vidit Jain(BITS Pilani) </p>

# TL;DR
This repository contains scripts for training a Generative Adversarial Framework for Optical Flow Prediction. This is an ongoing project. We have modelled optical flow prediction 
in this way, so that this framework can later on be used for the purpose of novelty detection.

# Environment
To easily replicate my environment, please clone the **environment.yml** file using **Conda**.

There are four important files in the Code folder

A) create_train_adl_dataset.ipynb
B) create_test_adl_dataset.ipynb
C) flow_gans.ipynb
D) Model U-Net for Optical Flow Estimation.ipynb

A) create_train_adl_dataset.ipynb
This script creates an efficient ".hdf5" file named "adl_dataset". This combines the preprocessed frames in the train_data folder which have been passed through PwC Net.

B) create_test_adl_dataset.ipynb
This script creates an efficient ".hdf5" file named "adl_dataset". This combines the preprocessed frames in the test_data folder which have been passed through PwC Net.

C) Model U-Net for Optical Flow Estimation.ipynb
A temporary file that includes the architectures of the generator and the discriminator.

D) flow_gans.ipynb
This script contains the training code for GANs and also predicts and saves some optical flow frames from the training_data.


Step 1: Create a new conda environment using the environment.yml file

Step 2 : Please click the link : https://drive.google.com/drive/folders/1GWpGggmkS2F_YrXTPhRWttQUzF321dLX?usp=sharing
Download all the subfolders and place them within the "Code" folder.

Step 3: All files can be run by "jupyter notebook <filename.ipynb>"

	A) The output of "jupyter notebook adl_dataset_create.ipynb" in the terminal will give you "adl_dataset.hdf5" which you have already downloaded from google drive

	B) The output of "jupyter notebook adl_dataset_test.ipynb" in the terminal will give you "adl_dataset_test.hdf5" which you have already downloaded from google drive
	
	C) Finally Run "jupyter notebook flow_gans.ipynb"
	
	
Please Note the ".npy" files in the train_data and test_data folder have been created by the script "pwcnet_predict_from_img_pairs.ipynb" located in the tfoptflow.
"pwcnet_predict_from_img_pairs.ipynb" is the only file we created, all other files have been provided by Nvidia.
