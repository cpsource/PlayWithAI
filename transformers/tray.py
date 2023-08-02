#!/home/pagec/venv/bin/python3

# tray.py - group numbers, predict next group

test_mode = False
skip_array = None
learning_rate = 1e-1
model_name = None
reloaded_flag = False
discount_array_flag = False
discount_array = []

import sys

# Remove the first entry (current working directory) from sys.path
#sys.path = sys.path[1:]
# Append the current working directory at the end of sys.path
#sys.path.append("")

# Now the current working directory will be searched last

if False:
    # so if we want to load from current directory first on each import, we do
    import sys

    # Add the desired path to sys.path temporarily
    sys.path.insert(0, "/path/to/module_directory")

    # Import the module
    import module_name

    # Remove the temporarily added path from sys.path
    sys.path.pop(0)

# Onward

#from datetime import datetime
#import yfinance as yf
#import mplfinance as mpf
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
#import torch.optim as optim
#from torch.optim.lr_scheduler import ReduceLROnPlateau
#import sqlite3
import squish as squ
import cmd_lin as cmd

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
# runs slower with cuda
#device = "cpu"
print(f"Device: {device}")

# set print options
torch.set_printoptions(threshold=5000)
np.set_printoptions(threshold=sys.maxsize)

# create our neural network

# we are going to sqish the data from 0 to 13. Each row has 5 elements.
# We are going 100 deep
#
# so it's 14 x 5 x 100 in (7000)
# and 14 x 5 out          (70)
#

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.l0 = nn.Linear(5041, 5041) # each next layer will be 30% of previous layer
        #self.l0s = nn.Sigmoid()
        self.l1 = nn.Linear(7000, 2100) # each next layer will be 30% of previous layer and multiple of 70
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(2100,140)
        self.l4 = nn.Sigmoid()
        self.l5 = nn.Linear(140, 70)
#        self.l6 = nn.ReLU()
        self.l6 = nn.Softmax(dim=1) # This will be column #1 result

    def forward(self, x):
        #pred_0 = self.l0(x)
        #pred_0s = self.l0s(pred_0)
        pred_1 = self.l1(x)
        pred_2 = self.l2(pred_1)
        pred_3 = self.l3(pred_2)
        pred_4 = self.l4(pred_3)
        pred_5 = self.l5(pred_4)
        logits = pred_6 = self.l6(pred_5)
        #return logits
        return logits

# make an instance of our network on device
model = NeuralNetwork().to(device)

# Create the optimizer
# we can also try lr=0.001, momentum=0.9
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9) # or -2 ???
print(f"creating optimizer with lr = {learning_rate}")
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # or -2 ???
#optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

#lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)

#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()

# Check if {model_name} exists and if so, load it. Set to eval mode
# see also: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
def attempt_reload():
    '''
    attempt to reload a pre-trained model
    '''
    global model
    global optimizer
    global epoch
    global loss
    global model_name
    global reloaded_flag
    
    reloaded_flag = False
    epoch = 0
    model_name = f"models/tray-{cmd.our_game}.model"
    if os.path.exists(model_name):
        reloaded_flag = True
        print(f"Reloading pre-trained model {model_name}")
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Reloaded epoch = {epoch}")
        loss = checkpoint['loss']
        learning_rate = checkpoint['learning_rate']
        print(f"Restored last learning_rate = {learning_rate}")
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # or -2 ???
        print("model to eval mode")
        model.eval()
    return

        
# Main Training Loop
def train(model, X, y, loss_fn, optimizer):

    #print(f"train: y = {y}")
    
    # set the model to training mode
    model.train()

    # Forward Pass
    y_hat = model(X)

    # Calculate loss
    loss = loss_fn(y, y_hat)

    #print(f"y_hat = {y_hat}\nloss = {loss}\n")

    # Zero gradients from last time
    optimizer.zero_grad()

    # Backpropigate Loss (calculates new gradients)
    loss.backward()

    # Update the parameters
    optimizer.step()

    # Update learning rate scheduler
    #lr_scheduler.step(loss)
    
    # Take model out of training mode
    #model.eval()
    
    #loss, current = loss.item()
    #print(f"loss: {loss}\n")
 
    return loss

if __name__ == "__main__":
    # get command line info
    cmd.give_help(sys.argv)
    cmd.set_our_game(sys.argv)
    #print(cmd.our_game)
    discount_array_flag, discount_array = cmd.is_discount(sys.argv)
    if discount_array_flag:
        print(f"Using discount. {discount_array}")
        
    # must have a model name
    if cmd.our_game is None:
        print("You must specify a model name of the form -g mm/pb")
        exit(0)

    # reload model if possible
    attempt_reload()
