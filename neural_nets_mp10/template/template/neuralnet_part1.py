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
This is the main entry point for MP10 Part1. You should only modify code within this file.
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

        self.fc1 = nn.Linear(in_size, h)  # First layer
        self.fc2 = nn.Linear(h, out_size)  # Output layer
        self.optimizer = optim.SGD(self.parameters(), lr=lrate)  # Using SGD as the optimizer
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """
        # TODO Implement the forward pass.
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = self.fc2(x)           # Output layer (no activation needed here for CrossEntropyLoss)
        return x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """
    
        # raise NotImplementedError("You need to write this part!")
        # Important, detach and move to cpu before converting to numpy and then to python float.

        self.optimizer.zero_grad()  # Reset gradient

        # Calculate loss
        yhat = self.forward(x)      
        loss_value = self.loss_fn(yhat, y)  

        # Return backpropogation
        loss_value.backward()        
        self.optimizer.step()    
        return loss_value.item()         # Or just use .item() to convert to python float. It will automatically detach and move to cpu.

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
    # raise NotImplementedError("You need to write this part!")
    # Important, don't forget to detach losses and model predictions and convert them to the right return types.
    in_size = train_set.shape[1]
    out_size = len(torch.unique(train_labels))

    train_mean = train_set.mean(dim=0)
    train_std = train_set.std(dim=0)
    train_set = (train_set - train_mean) / train_std
    dev_set = (dev_set - train_mean) / train_std

    criterion = nn.CrossEntropyLoss()

    # Instantiate the model
    lrate = 0.01
    net = NeuralNet(lrate, criterion, in_size, out_size)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Create the dataset
    train_dataset = get_dataset_from_arrays(train_set, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []

    for epoch in range(0, epochs):
        running_loss = 0.0
        # Iterate over the DataLoader for training data
        for batch in train_loader:
            inputs, targets = batch['features'], batch['labels']  # Get inputs
            loss = net.step(inputs, targets)  # Perform a step in training
            running_loss += loss

        val_loss = running_loss / len(train_loader)
        train_losses.append(val_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {val_loss:.4f}')

    with torch.set_grad_enabled(False):
        dev_set_output = net(dev_set)
        predicted_labels = torch.argmax(dev_set_output, dim=1)

    return train_losses, predicted_labels.numpy(), net

    return [],[],None
