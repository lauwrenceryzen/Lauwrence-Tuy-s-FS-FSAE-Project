# Lauwrence TUY
# Neural Network to identify face emotion recognition

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# keep da model consistent
SEED = 6767
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# set da parameters
data_dir = "processed_data"
image_size = 128
batch_size = 32
num_epochs = 5
learning_rate = 0.01
seed = 6767
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

# integrate da transformers 
train_tfms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# load dataset, 
