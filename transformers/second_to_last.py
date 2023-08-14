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
import inspect

# reloaded from disk
reloaded_flag = False
# test only ( no training )
test_mode = False
# what's our model name
model_name = ""
# what's our learning rate
learning_rate = 1e-2
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
# column 1 to 6, 6 being the pb, the default if no --col N used in command line
my_col = 6
# in check mode
is_check = False
# force a cnt
my_prev_play_flag = False
my_prev_play = 0
# our depth array = [our-depth , our-back, model-sizes...]
our_depth = [30, 500, 2100, 630, 189, 40]
# maximum ball in play
max_ball_expected = None

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

def extract_column(data,col):
  """Extracts the second to last column from a string of data.

  Args:
    data: A string of data.

  Returns:
    The second to last column from the data.
  """

  columns = data.split(",")

  val = columns[col]

  #print(col, val, columns)
  #exit(0)
  
  return int(val)

def read_file_line_by_line_readline(filename, my_col):
  """Reads a file line by line using readline.

  Args:
    filename: The name of the file to read.

  Returns:
    A list of the lines in the file.
  """
  global max_ball_expected

  adjust_col = [0,-7,-6,-5,-4,-3,-2]

  #print(my_col,adjust_col[my_col])
  #exit(0)

  # get max expected ball from web sites
  if cmd.our_game == 'mm':
      if my_col == 6:
          max_ball_expected = 25
      else:
          max_ball_expected = 70
  else:
      if my_col == 6:
          max_ball_expected = 26
      else:
          max_ball_expected = 69
  
  with open(filename, "r") as f:
    ts_array = []
    line_number = 1

    # skip some at front from older games
    if cmd.our_game == 'mm':
        for i in range(1449):
            f.readline()
            line_number += 1

    while True:
      line = f.readline()
      if line == "":
        break
      if line[0] == '#':
          line_number += 1
          continue

      # we only use in sorted order, so get all balls
      # then sort them
      
      m = []
      m.append(extract_column(line,adjust_col[1]))
      m.append(extract_column(line,adjust_col[2]))
      m.append(extract_column(line,adjust_col[3]))
      m.append(extract_column(line,adjust_col[4]))
      m.append(extract_column(line,adjust_col[5]))
      m.sort()
      
      x = m[my_col-1]

      if x <= max_ball_expected:
          ts_array.append(x)
      else:
          print(f"Ball {x} rejected at line {line_number} as out of bounds [1..{max_ball_expected}]")

      # onward
      line_number += 1
      continue
      
  f.close()

  #print(ts_array)

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

# Training Loop
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

def test_and_display(model, top, X, Y, ts_array, total_worse_count, is_check, winning_numbers,max_ball_expected, my_prev_play):
    ball = 0
    
    # lets test
    idx = len(ts_array) + my_prev_play
    print(f"my_prev_play = {my_prev_play}, idx = {idx}, len(ts_array) = {len(ts_array)}")
    if is_check is True:
        print(f"idx = {idx}")
        ball = ts_array[idx]
        print(f"Testing for ball in ts_array[{idx}] = {ball}")
        x = []
        Xtmp = []
        for j in range(idx-our_depth[0], idx):
            x.append(ts_array[j])
        Xtmp.append(x)
        x_oh = squ.one_hot_no_squish_max_ball(Xtmp[0],max_ball_expected)        

        #print(f"{len(Xtmp)}, Xtmp = {Xtmp}")
        #exit(0)

    else:
        
        #print("is_check is False")
        # build array deepth back from the most recent play
        x = []
        Xtmp = []
        tmp = len(ts_array)
        for j in range(tmp-our_depth[0], tmp):
            x.append(ts_array[j])
        Xtmp.append(x)
        #print(Xtmp)
        #print(X[len(X)-1])
        #print(f"len(ts_array) = {len(ts_array)}, ts_array[] = {ts_array[len(ts_array)-1]}")
        #print(idx,our_depth[0], len(X), X[len(X)-1], Xtmp)
        #exit(0)
        x_oh = squ.one_hot_no_squish_max_ball(Xtmp[0],max_ball_expected)        

        if len(winning_numbers) > 0:
            ball = winning_numbers[5]
        else:
            print("Error: no way to determine ball as --win is 0 length")
            ball = 0

    # convert to tensors
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
    my_col = cmd.set_my_col(sys.argv)
    winning_numbers = cmd.get_winning_numbers(sys.argv)
    cmd.set_our_game(sys.argv)
    my_prev_play_flag, my_prev_play = cmd.set_prev_play(sys.argv)
    if cmd.is_test(sys.argv):
        print('Running in test mode')
        test_mode = True
    else:
        print('Running in training mode')

    # read in correct data file
    ifile = f"data/{cmd.our_game}.csv"
    print(f"Using datafile {ifile}")

    # get a column array. The last element will be the last play
    ts_array = read_file_line_by_line_readline(ifile,my_col)
    #print(max(ts_array))
    #print(type(ts_array))
    #print(ts_array)
    #print(len)
    #exit(0)
    
    if False:
        # stop at 400 epochs
        if not test_mode and epoch >= 400:
            print("At 400 epoch limit, exiting")
            exit(0)

    # track number of each ball (strange way to get an array of 0's)
    ball_count_array = [0]*(max_ball_expected+1)

    # now build X and Y. X will be the width of data presented,
    # and Y will be the actual value. Note that Y will be +1 later
    # than X.

    # Note that our_depth[0] is the width of the game
    
    ts_array_len = len(ts_array)
    idx_x = 0
    idx_y = idx_x + our_depth[0]
    
    X = []
    Y = []

    while idx_y < ts_array_len:
        x = []
        y = []
        for j in range(idx_x, idx_x+our_depth[0]):
            x.append(ts_array[j])
        y.append(ts_array[idx_y])

        # track ball usage
        ball_count_array[ts_array[idx_y]] += 1
        
        # now that x,y arrays are built, add them to X,Y
        X.append(x)
        Y.append(y)
        
        # onward
        idx_x += 1
        idx_y += 1

    # display our handy work
    print(f"Max Plays to Train: {len(X)}")
    #for idx,z in enumerate(ball_count_array):
        #print(f"{idx}, {z}")
    #print(X)
    #print(Y)
    #exit(0)
    
    # initialize our model
    initialize_model((max_ball_expected+1)*our_depth[0], our_depth[3], our_depth[4], (max_ball_expected+1))

    model_name = f"models/second-to-last-{cmd.our_game}-{my_col}.model"
    print(f"Model Name: {model_name}")

    # zero old .model if necessary
    if cmd.is_zero(sys.argv):
        if os.path.exists(model_name):
            print(f"Removing model {model_name}")
            os.unlink(model_name)

    # attempt to reload pre-trained model from disk
    attempt_reload()

    if len(Y) != len(X):
        print(f"476: something wrong with Y and X")
        exit(0)
    #frame = inspect.currentframe()
    #print(f"{frame.f_lineno}: len(Y) = {len(Y)}")
    #exit(0)

    # what is the top of Y that we can consider ?
    top = len(Y)
    if my_prev_play_flag:
        top += my_prev_play
        print(f"Number of elements in Y forced (top): {top}")
    else:
        print(f"Number of elements in Y (top): {top}")        

    if reloaded_flag:
        # prepare to train for another 100 epochs
        model.train()
        old_epochs = epoch + 1
        epochs = epoch + 101
        print(f"New Epochs: {epochs}")
    else:
        # number of epochs to execute
        epochs = 101
        old_epochs = 0
    old_loss = 1.0
    first_save_flag = False
    print(f"test_mode = {test_mode}, old_epochs = {old_epochs}, epochs = {epochs}")
    #exit(0)

    # train on 1 less play if we want to check
    if is_check:
        top -= 2
        
    for epoch in range(old_epochs,epochs):

        # (this code is just stupid)
        if test_mode:
            continue

        if False and (epoch == 200):
            learning_rate /= 10.0
            set_lr(model,learning_rate)

        # our_depth[1] is how far back we want to start training
        if our_depth[1] != 0:
            # train from this depth
            idx = top + our_depth[1]
        else:
            # train from beginning
            idx = 0
            
        while idx < top:
            # get elements
            y_oh = squ.one_hot_no_squish_max_ball(Y[idx],max_ball_expected)
            x_oh = squ.one_hot_no_squish_max_ball(X[idx],max_ball_expected)
            
            # convert to tensors
            y_oh_t = torch.tensor(y_oh, dtype=torch.float32, device=device)
            x_oh_t = torch.tensor(x_oh, dtype=torch.float32, device=device)
            
            # train
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
    if False:
        for i,x in enumerate(ball_count_array):
            print(f"ball = {i}, count = {x}")
        
    # now lets test
    model.eval()
    test_and_display(model, top, X, Y, ts_array, total_worse_count, is_check, winning_numbers, max_ball_expected, my_prev_play)

    # finis
