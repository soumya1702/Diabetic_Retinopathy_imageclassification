#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

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


# In[18]:


criterion = nn.CrossEntropyLoss()


# In[19]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)


# In[ ]:




