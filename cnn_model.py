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

#Initialize the train and test data
train_data = pd.read_csv(os.path.join(data_path, 'train-metadata.csv'))
test_data = pd.read_csv(os.path.join(data_path, 'test-metadata.csv'))

# Filling in missing values for numerical and caegorical columns
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
train_data.fillna(train_data.mode().iloc[0], inplace=True)
test_data.fillna(test_data.median(numeric_only=True), inplace=True)
test_data.fillna(test_data.mode().iloc[0], inplace=True)

# One hot encoding for the categorical columns
train_data = pd.get_dummies(train_data, columns=['sex', 'anatom_site_general'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['sex', 'anatom_site_general'], drop_first=True)

# TODO Add columns that are irrelevant to the cnn here, if all are needed, resolve.
X = train_data.drop(columns=['target'])
y = train_data['target']

# Stratified split of training and test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#Normalize numerical features due to benign values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_data.drop(columns=['isic_id', 'patient_id']))





