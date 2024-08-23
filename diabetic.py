#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#dataframe with labels
trainLabels_df = pd.read_csv("/N/slate/syennapu/trainLabels.csv")
#printing the top labels
print(trainLabels_df.head())
distribution = trainLabels_df["level"].value_counts()
print(distribution)


# In[2]:


import matplotlib.pyplot as plt
distribution.plot(kind='bar')
plt.title("distribution of labels")
plt.xlabel("labels")
plt.ylabel("number of iamges")
plt.show()


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt


# In[4]:


import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


# In[27]:


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


# In[28]:


import torchvision.transforms as T
from torchvision.transforms import ToTensor, Normalize, Compose

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


# In[29]:


import multiprocessing
num_workers = multiprocessing.cpu_count()

# how many samples per batch to load. You can experiment
# with this parameter to try to improve performances
batch_size = 32


# In[30]:


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
print(f"Using {train_len} examples for training")
print(f"Using {test_len} examples for testing and {val_len} for validation")
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
print(f"Using {len(test_subset)} for testing")


# In[9]:


import torch
import matplotlib.pyplot as plt
import numpy as np

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a batch of data
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Move images and labels to the device
images, labels = images.to(device), labels.to(device)

# Move images to CPU for visualization
images = images.cpu().numpy()

# Plot the images in the batch
fig, subs = plt.subplots(2, 8, figsize=(16, 4))
for idx, sub in zip(np.arange(16), subs.flatten()):
    img = np.transpose(images[idx], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    sub.imshow(img)
    sub.set_title(str(int(labels[idx].item())))
    sub.axis("off")
plt.show()


# In[31]:


from PIL import Image
import numpy as np

# Load a sample image from your dataset
sample_image_path = '/N/slate/syennapu/diabetic/train/10_left.jpeg'  # Replace with the path to a sample image
sample_image = Image.open(sample_image_path)

# Get the dimensions of the image
width, height = sample_image.size
print(f"Image Dimensions: Width = {width}, Height = {height}")

# If you want to see the number of channels (e.g., RGB = 3 channels)
sample_image_array = np.array(sample_image)
num_channels = sample_image_array.shape[2] if len(sample_image_array.shape) == 3 else 1
print(f"Number of Channels: {num_channels}")


# In[32]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pooling to ensure the output size is fixed
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Output will be 512x7x7
        
        # Adjusted input size to match the output of the adaptive pooling layer
        self.fc1 = nn.Linear(in_features=512*7*7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout(x)
        
        x = self.adaptive_pool(x)  # Apply adaptive pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x

# Initialize the NN
model = Net()

# Print the model
print(model)


# In[33]:


if torch.cuda.is_available():
    model.cuda()


# In[36]:


from torch import nn

criterion = nn.CrossEntropyLoss()


# In[37]:


import torch.optim

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)


# In[20]:


# number of epochs to train the model
n_epochs = 50

# Set model to training mode
# (this changes the behavior of some layers, like Dropout)
model.train()

# Loop over the epochs
for epoch in range(n_epochs):

    # monitor training loss
    train_loss = 0.0

    # Loop over all the dataset using the training
    # dataloader
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # Convert target to LongTensor
        target = target.long()

        # forward pass: 
        # compute predictions
        output = model(data)

        # calculate the loss which compare the model
        # output for the current batch with the relative
        # ground truth (the target)
        loss = criterion(output, target)

        # backward pass: 
        # compute gradient of the loss with respect to 
        # model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()

        # update running training loss
        train_loss += loss.item()*data.size(0)

    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)
    # Print the average loss for this epoch
    print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}')


# In[44]:


import torch
import matplotlib.pyplot as plt
import numpy as np

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a batch of data
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Move images and labels to the device
images, labels = images.to(device), labels.to(device)

# Move images to CPU for visualization
images = images.cpu().numpy()

# Plot the images in the batch
fig, subs = plt.subplots(2, 8, figsize=(16, 4))
for idx, sub in zip(np.arange(16), subs.flatten()):
    img = np.transpose(images[idx], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    sub.imshow(img)
    sub.set_title(str(int(labels[idx].item())))
    sub.axis("off")
plt.show()


# In[43]:


# number of epochs to train the model
n_epochs = 50

# Set model to training mode
# (this changes the behavior of some layers, like Dropout)
model.eval()

# Loop over the epochs
for epoch in range(n_epochs):
    valid_loss = 0.0   # Optional when not using Model Specific layer
    for data, target in val_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        
        output = model(data)
        loss = criterion(output, target)
        valid_loss = loss.item() * data.size(0)

    print(f'Epoch {e+1} \t\t Training Loss: {train_loss_avg:.4f} \t\t Validation Loss: {valid_loss_avg:.4f}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased ({min_valid_loss:.6f} --> {valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss

        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')


# In[ ]:




