import os
import pandas as pd
import cv2
import torch
import torchvision


# Load dataset
dirname = os.path.dirname(__file__)
data_path = os.path.join(dirname, 'data')
images_path = os.listdir(data_path + "/train-image")


