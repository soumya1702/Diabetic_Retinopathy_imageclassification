#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


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
    f'Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}'

