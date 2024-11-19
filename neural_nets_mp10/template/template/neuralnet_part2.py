# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10 Part2. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        Parameters:
        lrate (float): Learning rate for the model.
        loss_fn (callable): A loss function defined as follows:
            Parameters:
                yhat (Tensor): An (N, out_size) Tensor.
                y (Tensor): An (N,) Tensor.
            Returns:
                Tensor: A scalar Tensor that is the mean loss.
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # For Part 1, the network should have the following architecture (in terms of hidden units):
        # in_size -> h -> out_size, where 1 <= h <= 256
        h = 128 # Start at half


        # TODO Define the network architecture (layers) based on these specifications.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 7 * 7, h) # First layer
        self.fc2 = nn.Linear(h, out_size) # Output layer

        self.activation = nn.LeakyReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=lrate)
        # raise NotImplementedError("You need to write this part!")
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """
        # TODO Implement the forward pass.

        # Reshape
        x = x.view(-1, 3, 31, 31)

        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
        raise NotImplementedError("You need to write this part!")
        return torch.ones(x.shape[0], 1)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """
        self.optimizer.zero_grad()  # Reset gradient

        # Calculate loss
        yhat = self.forward(x)      
        loss_value = self.loss_fn(yhat, y)  

        # Return backpropogation
        loss_value.backward()        
        self.optimizer.step()    
        return loss_value.item()         # Or just use .item() to convert to python float. It will automatically detach and move to cpu.

        raise NotImplementedError("You need to write this part!")
        # Important, detach and move to cpu before converting to numpy and then to python float.
        # Or just use .item() to convert to python float. It will automatically detach and move to cpu.
        return 0.0



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """
    Creates and trains a NeuralNet object 'net'. Use net.step() to train the neural net
    and net(x) to evaluate the neural net.

    Parameters:
    train_set (Tensor): An (N, in_size) Tensor representing the training data.
    train_labels (Tensor): An (N,) Tensor representing the training labels.
    dev_set (Tensor): An (M,) Tensor representing the development set.
    epochs (int): The number of training epochs.
    batch_size (int, optional): The size of each training batch. Defaults to 100.

    This method must work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values if your initial choice does not work well.
    For Part 1, we recommend setting the learning rate to 0.01.

    Returns:
    list: A list of floats containing the total loss for every epoch.
        Ensure that len(losses) == epochs.
    numpy.ndarray: An (M,) NumPy array (dtype=np.int64) of estimated class labels (0,1,2, or 3) for the development set (model predictions).
    NeuralNet: A NeuralNet object.
    """

    train_mean = train_set.mean(dim=0)
    train_std = train_set.std(dim=0)
    train_set = (train_set - train_mean) / train_std
    dev_set = (dev_set - train_mean) / train_std

    in_size = train_set.shape[1]
    out_size = len(torch.unique(train_labels))
    lrate = 0.001

    criterion = nn.CrossEntropyLoss()
    net = NeuralNet(lrate, criterion, in_size, out_size)

    train_dataset = get_dataset_from_arrays(train_set, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch['features'], batch['labels']
            loss = net.step(inputs, targets)
            running_loss += loss

        val_loss = running_loss / len(train_loader)
        train_losses.append(val_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {val_loss:.4f}')

    with torch.no_grad():
        dev_set_output = net(dev_set)
        predicted_labels = torch.argmax(dev_set_output, dim=1)

    return train_losses, predicted_labels.numpy(), net

    raise NotImplementedError("You need to write this part!")
    # Important, don't forget to detach losses and model predictions and convert them to the right return types.
    return [],[],None
