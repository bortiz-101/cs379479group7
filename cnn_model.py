import os
import pandas as pd
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dotenv import load_dotenv

#Environmental variables are set in .env make sure to update with your paths
load_dotenv()
cwd = os.getenv('CWD')
data_path = os.getenv('DATA_PATH')
images_path = os.getenv('IMAGES_PATH')
train_hdf5 = os.getenv('TRAIN_HDF5')
train_csv = os.getenv('TRAIN_CSV')

#DO NOT UNCOMMENT UNTIL FINAL TESTING
#test_hdf5 = os.getenv('TEST_HDF5')
#test_csv = os.getenv('TEST_CSV')



#MacOS GPU Support
if torch.backends.mps.is_available():
    device = torch.device("mps")
#NVIDIA Support
elif torch.cuda.is_available():
    device = torch.device("cuda")
#No GPU
else:
    device=torch.device("cpu")


#Hyper-Parameters
epochs = 4
batch_size = 32
learning_rate = 0.001






