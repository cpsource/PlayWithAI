#!/home/pagec/venv/bin/python3

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
#import sqlite3
import cmd_lin as cmd
import operator
import squish as squ

# reloaded from disk
reloaded_flag = False
# test only ( no training )
test_mode = False
# what's our model name
model_name = ""
# what's our learning rate
learning_rate = 1e-3
# winning numbers
winning_numbers = []
# total worse count during training
total_worse_count = 0
# our model
model = None
# optimizer
optimizer = None
# loss function
loss_fn = None
# column 1 to 6, 6 being the pb
my_col = 6
# in check mode
is_check = False
# force a cnt
my_cnt_flag = False
my_cnt = 0
# the laragest ball we are playing
max_ball = None
# our depth array = [our-depth , our-back, model-sizes...]
our_depth = [30, 500, 2100, 630, 189, 40]

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

# set new learning rate
def set_lr(model,lr):
    global optimizer
    print(f"setting optimizer with lr = {lr}")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) # or -2 ???

def extract_second_to_last_column(data):
  """Extracts the second to last column from a string of data.

  Args:
    data: A string of data.

  Returns:
    The second to last column from the data.
  """

  columns = data.split(",")
  second_to_last_column = columns[-2]
  return int(second_to_last_column)

def read_file_line_by_line_readline(filename):
  """Reads a file line by line using readline.

  Args:
    filename: The name of the file to read.

  Returns:
    A list of the lines in the file.
  """

  with open(filename, "r") as f:
    ts_array = []
    line_number = 1

    # skip some at front
    if cmd.our_game == 'mm':
        for i in range(1029):
            f.readline()
            line_number += 1
            
    while True:
      line = f.readline()
      if line == "":
        break
      if line[0] == '#':
          continue
      x = extract_second_to_last_column(line)
      if cmd.our_game == 'mm':
          if x > 25:
              print(f"mm Warning at line {line_number}: {x} gt 25, adjusting")
              x %= 26
      else:
          if x > 39:
              print(f"pb Warning at line {line_number}: {x} gt 39, adjusting")
              x %= 39 
        
      ts_array.append(x)

      # onward
      line_number += 1
      
  f.close()
  return ts_array

def scale_tensor(tensor):
    # Compute the maximum and minimum values of the tensor
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    
    # Scale the tensor from -1 to +1
    scaled_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1
    
    return scaled_tensor

import my_class as net

def initialize_model(k1,k2,k3,k4):
    global our_depth
    global model
    global optimizer
    global loss_fn
    
    # make an instance of our network on device
    #k1 = 2100
    #k2 = 630
    #k3 = 189
    #k4 = 40

    print(f"Initializing Model with {k1} {k2} {k3} {k4}")
    
    model = net.NeuralNetwork(k1,k2,k3,k4).to(device)

    # Create the optimizer
    # we can also try lr=0.001, momentum=0.9
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9) # or -2 ???
    print(f"creating optimizer with lr = {learning_rate}")
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # or -2 ???
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    #lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()

#
# Check if second-to-last.model exists and if so, load it. Set to eval mode
# see also: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
#
def attempt_reload():
    '''
    attempt to reload a model
    '''
    global model
    global optimizer
    global epoch
    global loss
    global model_name
    global reloaded_flag
    global learning_rate

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
        model.eval()
        # - or -
        #model.train()

    # done
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

    # Take model out of training mode
    #model.eval()
    
    #loss, current = loss.item()
    #print(f"loss: {loss}\n")
 
    return loss

def softmax_np(x):
  """Calculates the softmax of a vector.

  Args:
    x: A NumPy array.

  Returns:
    A NumPy array of the same shape as x, with the softmax of each element.
  """

  exp_x = np.exp(x)
  sum_exp_x = np.sum(exp_x)
  return exp_x / sum_exp_x

def softmax(tensor):
  """Computes the softmax function on a tensor.

  Args:
    tensor: The tensor to compute the softmax function on.

  Returns:
    The softmaxed tensor.
  """

  # Get the exponents of the tensor.
  exponents = torch.exp(tensor)

  # Sum the exponents.
  sum_of_exponents = torch.sum(exponents, dim=1, keepdim=True)

  # Divide each exponent by the sum of the exponents.
  softmaxed_tensor = exponents / sum_of_exponents

  # Return the softmaxed tensor.
  return softmaxed_tensor

def test_and_display(model, cnt, X, Y, ts_array, total_worse_count, is_check, winning_numbers,max_ball):
    ball = 0
    
    # lets test - a head game, this.
    if is_check:
        idx_y = cnt - 2
        idx_x = cnt - 3
        idx_z = cnt - 1

        z_oh = squ.one_hot_no_squish_max_ball(Y[idx_z],max_ball)
        ball = operator.indexOf(z_oh[0],1.0)
        
        #print(ball, z_oh)
        
    else:
        idx_y = cnt - 1
        idx_x = cnt - 2
        idx_z = 0
        if len(winning_numbers) > 0:
            ball = winning_numbers[5]
        else:
            print("Error: no way to determine ball as --win is 0 length")
            exit(0)

    y_oh = squ.one_hot_no_squish_max_ball(Y[idx_y],max_ball)
    #y_oh = one_hot_encode_array_39(np.array(Y[idx_y])).reshape(1,40)
    #print(y_oh)
    x_oh = squ.one_hot_no_squish_max_ball(X[idx_x],max_ball)
    #x_oh = one_hot_encode_array_69(np.array(X[idx_x])).reshape(1,2100)
    #print(x_oh)
            
    # convert to tensors
    #y_oh_t = torch.tensor(y_oh, dtype=torch.float32, device=device)
    x_oh_t = torch.tensor(x_oh, dtype=torch.float32, device=device)

    # run prepared data through the model
    y_hat = model(x_oh_t).cpu()
    y_hat_detached = y_hat.detach()
    a = y_hat_detached_np = y_hat_detached.numpy()[0]

    # prepare model's guess for display and analysis
    indices = np.argsort(a)
    indices_reversed = indices[::-1]
    print(f"Balls in descending probability: {indices_reversed}")

    # find index of ball in indices_reversed. It shows how far off we are
    error_distance = operator.indexOf(indices_reversed,ball)
    print(f"Error Distance: {error_distance+1} for ball {ball}")
    
    #total_probability = 0.0
    #for i in (0,1,2,3,4,5,6,7,8,9):
        #total_probability += a[indices_reversed[i]]
        #print(f"#{i+1} pick : {indices_reversed[i]:2d}, probability {a[indices_reversed[i]]:.5f} , total {total_probability:.5f}")

    #print(f"(actual) Y[{idx}] = {Y[idx][0]}")
    #print(f"y_oh  = {y_oh}")
    #print(f"y_hat_detached_np = {y_hat_detached_np}")

    # stats
    print(f"Total Worse Count: {total_worse_count}")
    
def save_model(epoch,model,optimizer,loss,learning_rate,model_name):
    print(f"Saving Model {model_name}, epoch = {epoch}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'learning_rate' : learning_rate,
    }, model_name)
    return
    
if __name__ == "__main__":
    cmd.give_help(sys.argv)
    our_depth = cmd.get_our_depth(sys.argv)
    is_check = cmd.is_check_mode(sys.argv)
    cmd.set_my_col(sys.argv)
    winning_numbers = cmd.get_winning_numbers(sys.argv)
    cmd.set_our_game(sys.argv)
    my_cnt_flag, my_cnt = cmd.set_cnt(sys.argv)
    if cmd.is_test(sys.argv):
        print('Running in test mode')
        test_mode = True
    else:
        print('Running in training mode')


    # read in correct data file
    ifile = f"data/{cmd.our_game}.csv"
    print(f"Using datafile {ifile}")
    ts_array = read_file_line_by_line_readline(ifile)
    #print(max(ts_array))
    #print(type(ts_array))
    #print(ts_array)
    #print(len)

    if False:
        # stop at 400 epochs
        if not test_mode and epoch >= 400:
            print("At 400 epoch limit, exiting")
            exit(0)

    # determine the largest ball we are playing
    max_ball = ts_array[0]
    for i in ts_array:
        if i > max_ball:
            max_ball = i
    print(f"max_ball = {max_ball}")

    #print(ts_array)
    #exit(0)
    
    # track number of each ball (strange way to get an array of 0's)
    ball_count_array = [0]*(max_ball+1)
    
    tmp = len(ts_array)
    idx = 0
    X = []
    Y = []
    for i in range(0, tmp-our_depth[0]):
        x = []
        y = []
        for j in range(0, our_depth[0]):
            x.append(ts_array[j+idx])
        y.append(ts_array[idx+our_depth[0]])

        # now that x,y arrays are built, add them to X,Y
        X.append(x)
        Y.append(y)
        
        # onward
        idx += 1

    # display ball_count_array
    #for idx,z in enumerate(ball_count_array):
        #print(f"{idx}, {z}")

    #print(len(X))
    #exit(0)
    
    # initialize our model
    initialize_model((max_ball+1)*our_depth[0], our_depth[3], our_depth[4], (max_ball+1))

    model_name = f"models/second-to-last-{cmd.our_game}.model"
    print(f"Model Name: {model_name}")
    
    if cmd.is_zero(sys.argv):
        if os.path.exists(model_name):
            print(f"Removing model {model_name}")
            os.unlink(model_name)

    # attempt to reload pre-trained model from disk
    attempt_reload()
    
    # what is the top of Y that we can consider ?
    cnt = 0
    for i in Y:
        cnt += 1
    if my_cnt_flag:
        cnt += my_cnt
        print(f"Number of elements in Y forced (cnt): {cnt}")
    else:
        print(f"Number of elements in Y (cnt): {cnt}")        

    if reloaded_flag:
        # prepare to train for another 100 epochs
        model.train()
        old_epochs = epoch + 1
        epochs = epoch + 101
        print(f"New Epochs: {epochs}")
    else:
        # number of epochs to execute
        epochs = 201
        old_epochs = 0

    old_loss = 1.0

    first_save_flag = False

    print(f"test_mode = {test_mode}, old_epochs = {old_epochs}, epochs = {epochs}")
    #exit(0)

    if is_check:
        top = cnt - 2
    else:
        top = cnt - 1
        
    for epoch in range(old_epochs,epochs):

        # (this code is just stupid)
        if test_mode:
            continue

        if epoch == 200:
            learning_rate /= 10.0
            set_lr(model,learning_rate)
        
        idx = top + our_depth[1]
        while idx < top:
            # show our handywork
            #print(X[0],Y[0])
            y_oh = squ.one_hot_no_squish_max_ball(Y[idx],max_ball)
            #print(y_oh)
            x_oh = squ.one_hot_no_squish_max_ball(X[idx],max_ball)

            # only do once
            if epoch == old_epochs:
                for z in X[idx]:
                    ball_count_array[z] += 1
                
            #print(x_oh)
            
            # convert to tensors
            y_oh_t = torch.tensor(y_oh, dtype=torch.float32, device=device)
            x_oh_t = torch.tensor(x_oh, dtype=torch.float32, device=device)
            
            # train
            loss = train(model, x_oh_t, y_oh_t, loss_fn, optimizer)
            if False and (idx >= (cnt-30)):
                # train a bit more on the last one
                for tmp in (1,2,3):
                    loss = train(model, x_oh_t, y_oh_t, loss_fn, optimizer)
            # onward
            idx += 1

        if loss < old_loss:
            status = "better"
        else:
            total_worse_count += 1
            status = "worse"

        print(f"epoch: {epoch}, loss = {loss}, delta = {loss-old_loss}, status = {status}")
        #print(f"Epoch: {epoch} : {loss} : {status}")
        old_loss = loss

        if not epoch % 10:
            if not first_save_flag:
                first_save_flag = True
            else:
                # now save model
                save_model(epoch,model,optimizer,loss,learning_rate,model_name)

    # display
    for i,x in enumerate(ball_count_array):
        print(f"ball = {i}, count = {x}")
        
    # now lets test
    model.eval()
    test_and_display(model, cnt, X, Y, ts_array, total_worse_count, is_check, winning_numbers, max_ball)

    # finis
