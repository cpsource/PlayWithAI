#!/home/pagec/venv/bin/python3

# etray.py - no grouping of numbers, predict next group
# etray-120.py - run 120 deep instead of 100 in etray.py

# note on probability
#  P(A and B) = P(A) * P(B)
#  P(A or B) = P(A) + P(B) - P(A and B)

test_mode = False
skip_array = None
learning_rate = 1e-2
model_name = None
reloaded_flag = False
discount_array_flag = False
discount_array = []
ts_array = []
our_depth = None
model = None
optimizer = None
loss_fn = None

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

def softmax_np(x):
  """Calculates the softmax of an np vector.

  Args:
    x: A NumPy array.

  Returns:
    A NumPy array of the same shape as x, with the softmax of each element.
  """

  exp_x = np.exp(x)
  #print(f"x = {x},exp_x = {exp_x}")
  #exit(0)
  sum_exp_x = np.sum(exp_x)
  return exp_x / sum_exp_x

# create our neural network

# we are going to sqish the data from 0 to 13. Each row has 5 elements.
# We are going 100 deep
#
# so it's 71 x 5 x 100 in (35,500)
# and 71 x 5 out          (   355)
#
# so it's 71 x 5 x our_depth in (42,600)
# and 71 x 5 out          (   355)

class NeuralNetwork(nn.Module):
    def __init__(self, k1, k2, k3, k4):
        super().__init__()
        
        #self.l0 = nn.Linear(5041, 5041) # each next layer will be 30% of previous layer
        #self.l0s = nn.Sigmoid()
        self.l1 = nn.Linear(k1, k2)
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(k2, k3)
        self.l4 = nn.Sigmoid()
        self.l5 = nn.Linear(k3, k4)
#        self.l6 = nn.ReLU()
#        self.l6 = nn.Softmax(dim=1) # This will be column #1 result

    def forward(self, x):
        #pred_0 = self.l0(x)
        #pred_0s = self.l0s(pred_0)
        pred_1 = self.l1(x)
        pred_2 = self.l2(pred_1)
        pred_3 = self.l3(pred_2)
        pred_4 = self.l4(pred_3)
        logits = pred_5 = self.l5(pred_4)
#        logits = pred_6 = self.l6(pred_5)
        #return logits
        return logits

def initialize_model():
    global our_depth
    global model
    global optimizer
    global loss_fn

    # make an instance of our network on device
    k1 = 71*5*our_depth[0]
    k2 = 71*4*our_depth[1]
    k3 = 71*3*our_depth[2]
    k4 = 71*5*our_depth[3]

    print(f"Initializing Model with {k1} {k2} {k3} {k4}")
    
    model = NeuralNetwork(k1,k2,k3,k4).to(device)

    # Create the optimizer
    # we can also try lr=0.001, momentum=0.9
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9) # or -2 ???
    print(f"creating optimizer with lr = {learning_rate}")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # or -2 ???
    #optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)

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

#@torch.compile
def single_pass(model, loss_fn, optimizer, cnt, ts_array):
    global our_depth
    idx = cnt-2 - 500
    
    while idx < (cnt-1):
        #print(f"sp idx = {idx}")
        
        # build y
        y = ts_array[idx]
        y_hot = squ.one_hot_no_squish(y)
        # cvt y to Y for torch
        Y = torch.as_tensor(y_hot).float().cuda()

        # build x
        tmp = []
        for i in range(idx-our_depth[0], idx):
            if i == idx:
                print("can't consider y")
                exit(0)
            if i >= cnt:
                print(f"out of range, i = {i}, idx = {idx}")
                exit(0)
            tmp.append(ts_array[i])
        x = []
        for i in tmp:
            for j in i:
                x.append(j)
        x_hot = squ.one_hot_no_squish(x)
        # cvt x to X for torch 
        X = torch.as_tensor(x_hot).float().cuda()

        # train
        loss = train(model,
                     X,
                     Y,
                     loss_fn,
                     optimizer)

        # onward
        idx += 1
    # done
    return loss

def is_in_skip_array(n):
    if skip_array is None:
        return False
    for i in skip_array:
        if n == i:
            return True
    return False

# after the fifth call to display
# this guy will be
# [[] [] [] [] []]

column_probabilities = []

def display(idx,a,y):
    p1 = []

    print(f"Column: {idx+1}")
    
    for i in range(0,71):
        p1.append(a[idx*71+i])
    #print(f"p1 before softmax {p1}")
    p1 = softmax_np(np.array(p1))
    p1 -= 0.01387
    #print(f"p1 after softmax {p1}")
    
    indices          = np.argsort(p1)
    indices_reversed = indices[::-1]

    #print(indices)
    #print(indices_reversed)
    #exit(0)
    
    print(f"Balls in descending order: {indices_reversed}")
    total_probability = 0.0
    j = i = 0
    probability_array = []
    while True:
        #if is_in_skip_array(indices_reversed[i]):
        #    i += 1
        #    continue
        total_probability += p1[indices_reversed[i]]
        probability_array.append(p1[indices_reversed[i]])

        distance = abs(y[idx] - indices_reversed[i])
        
        print(f"#{i+1} Ball : {indices_reversed[i]:2d}, probability {p1[indices_reversed[i]]:.5f} , distance = {distance}, total {total_probability:.5f}")
        i += 1
        j += 1
        if j >= 71:
            break

    column_probabilities.append(probability_array)

'''
  Test and display a prediction
'''
def test_and_display(model, cnt, ts_array):
    global our_depth
    model.eval()
    # lets test against the last one
    idx = cnt - 1

    # build y
    y = ts_array[idx]
    y_hot = squ.one_hot_no_squish(y)
    # cvt y to Y for torch
    Y = torch.as_tensor(y_hot).float().cuda()

    # build x
    tmp = []
    for i in range(idx-our_depth[0], idx):
        if i >= cnt:
            print(f"out of range, i = {i}, idx = {idx}")
            exit(0)
        tmp.append(ts_array[i])
    x = []
    for i in tmp:
        for j in i:
            x.append(j)
    x_hot = squ.one_hot_no_squish(x)
    # cvt x to X for torch 
    X = torch.as_tensor(x_hot).float().cuda()

    # Test
    y_hat = model(X).cpu()
    #print(y_hat)
    y_hat_detached = y_hat.detach()
    a = y_hat_detached_np = y_hat_detached.numpy()[0]

    # display each of the five balls
    display(0,a,y)
    display(1,a,y)
    display(2,a,y)
    display(3,a,y)
    display(4,a,y)

    #print(f"column_probabilities = {column_probabilities}")
    
def save_model(epoch,model,optimizer,loss,learning_rate,model_name):
    print(f"Saving Model {model_name}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'learning_rate' : learning_rate,
    }, model_name)
    return

if __name__ == "__main__":

    # lets run a test of the numbering
    if False:
        y = [1,70,3,20,2]
        print(f"len(y) = {len(y)},y = {y}")
        yhat = squ.one_hot_no_squish(y)[0]
        print(f"len(yhat) = {len(yhat)}, yhat before display {yhat}")
        display(4,yhat,y)
        exit(0)
        
    # get command line info
    cmd.give_help(sys.argv)
    cmd.set_our_game(sys.argv)
    our_depth = cmd.set_our_depth(sys.argv)
    initialize_model()
    test_mode = cmd.is_test(sys.argv)

    model_name = f"models/etray-{our_depth[0]}-{cmd.our_game}.model"
    if cmd.is_zero(sys.argv):
        if os.path.exists(model_name):
            print(f"Removing model {model_name}")
            os.unlink(model_name)
            
    #print(cmd.our_game)
    discount_array_flag, discount_array = cmd.is_discount(sys.argv)
    if discount_array_flag:
        print(f"Using discount. {discount_array}")
        
    # must have a model name
    if cmd.our_game is None:
        print("You must specify a model name of the form -g mm/pb")
        exit(0)

    # load in csv file
    ifile = f"data/{cmd.our_game}.csv"
    print(f"Using datafile {ifile}")
    cnt, ts_array = squ.read_file_line_by_line_readline(ifile)
    print(cnt)

    idx = cnt - 1 # this is the last record in ts_array of the form [[],...[]]
    
    # reload model if possible
    attempt_reload()

    if reloaded_flag:
        # stop at 800 epochs
        if not test_mode and epoch >= 800:
            print("At 800 epoch limit, exiting")
            exit(0)
            
        # train for another 100 epochs
        model.train()
        old_epochs = epoch
        epochs = epoch + 101
        print(f"New Epochs: {epochs}")
    else:
        # number of epochs to execute
        old_epochs = 0
        epochs = 101

    old_loss = 1.0
    first_save_flag = False

    print(f"old_epochs = {old_epochs}, epochs = {epochs}")

    loss_cnt = 0
    max_loss_cnt = 35
    minimum_delta = 1.5e-08
    
    for epoch in range(old_epochs,epochs):

        if test_mode:
            continue

        if False and (epoch == 20 or epoch == 80):
            learning_rate /= 10.0
            set_lr(model,learning_rate)

        # do a single pass through ts_array
        loss = single_pass(model, loss_fn, optimizer, cnt, ts_array)

        #print(loss)
        #exit(0)
        
        if loss < old_loss:
            status = "better"
        else:
            status = "worse"
            loss_cnt += 1
            if loss_cnt > max_loss_cnt:
                print(f"Oops, loss_cnt = {loss_cnt} exceeds {max_loss_cnt}. No model save. We are exiting.")
                exit(2)

        delta = loss-old_loss
        print(f"epoch: {epoch}, loss = {loss}, delta = {delta}, status = {status}")
        if abs(delta) < minimum_delta:
            print(f"Good, delta = {abs(delta)} less than minimum threshold of {minimum_delta}. We are exiting.")
            if loss_cnt == 0:
                save_model(old_epochs,model,optimizer,loss,learning_rate,model_name)
            exit(1)
        old_loss = loss

        # save every 10
        if not epoch % 10:
            if not first_save_flag:
                first_save_flag = True
            else:
                loss_cnt = 0
                save_model(epoch,model,optimizer,loss,learning_rate,model_name)

    if False:
        # now save model
        save_model(epoch,model,optimizer,loss,learning_rate,model_name)
    
    # do a single pass
    #loss = single_pass(model, loss_fn, optimizer, cnt, ts_array)

    # now lets test
    model.eval()
    test_and_display(model, cnt, ts_array)

    # write out column_probabilities for third*.py
    filename = f"probs_{cmd.our_game}.py"
    print(f"Writing {filename}")
    with open(filename, "w+") as f:
        f.write("column_probabilities = %s\n" % column_probabilities)
    f.close()
