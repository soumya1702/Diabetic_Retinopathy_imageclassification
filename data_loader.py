#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
import os
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as T
from torchvision.transforms import ToTensor, Normalize, Compose

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()  


# In[7]:


#dataframe with labels
trainLabels_df = pd.read_csv("/N/slate/syennapu/trainLabels.csv")
#printing the top labels
trainLabels_df.head()


# In[8]:


distribution = trainLabels_df["level"].value_counts()
distribution


# In[9]:


distribution.plot(kind='bar')
plt.title("distribution of labels")
plt.xlabel("labels")
plt.ylabel("number of images")
plt.show()


# In[10]:


class diabetic(Dataset):
    """diabetic retinopathy dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, f"{self.labels_df.iloc[idx, 0]}.jpeg")
        try:
            image = io.imread(img_name)
        except FileNotFoundError:
            print(f"File not found: {img_name}. Skipping.")
            return None, None
        # Convert to PIL Image
        image = Image.fromarray(image)
        label = self.labels_df.iloc[idx, 1]
        
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


# In[11]:


# Define the transformations
transform = transforms.Compose(
    [
        # This transforms takes a np.array or a PIL image of integers
        # in the range 0-255 and transforms it to a float tensor in the
        transforms.Resize((128, 128)),
        # range 0.0 - 1.0
        transforms.ToTensor(),
        # This then renormalize the tensor to be between -1.0 and 1.0,
        # which is a better range for modern activation functions like
        # Relu
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5)),
    ]
)

# Create the dataset with transformations applied
train_dataset = diabetic(
    csv_file='/N/slate/syennapu/trainLabels.csv',
    root_dir='/N/slate/syennapu/diabetic/train',
    transform=transform
)
# Create the dataset with transformations applied
test_dataset = diabetic(
    csv_file='/N/slate/syennapu/trainLabels.csv',
    root_dir='/N/slate/syennapu/test',
    transform=transform
)


# In[12]:


import multiprocessing
num_workers = multiprocessing.cpu_count()

# how many samples per batch to load. You can experiment
# with this parameter to try to improve performances
batch_size = 32


# In[13]:


# Create the dataset with transformations applied
train_dataset = diabetic(
    csv_file='/N/slate/syennapu/trainLabels.csv',
    root_dir='/N/slate/syennapu/diabetic/train',
    transform=transform
)

# Create the dataset with transformations applied
test_dataset = diabetic(
    csv_file='/N/slate/syennapu/trainLabels.csv',
    root_dir='/N/slate/syennapu/test',
    transform=transform
)
# Split in train and validation
# NOTE: we set the generator with a fixed random seed for reproducibility
train_len = int(len(train_dataset))
test_len = int(len(test_dataset) * 0.5)
val_len = len(test_dataset) - test_len
test_subset, val_subset = torch.utils.data.random_split(
    test_dataset, [test_len, val_len], generator=torch.Generator().manual_seed(42)
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
)
val_loader = torch.utils.data.DataLoader(
    dataset=val_subset, shuffle=False, batch_size=batch_size, num_workers=num_workers
)

# Get test data
test_loader = torch.utils.data.DataLoader(
    test_subset, shuffle=False, batch_size=batch_size, num_workers=num_workers
)


# In[ ]:




