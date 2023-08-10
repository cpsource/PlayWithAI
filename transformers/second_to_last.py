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

# say which game we are playing
our_game = "pb" # or pb, with pb being the default
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

def one_hot_encode_array_with_pytorch_zz(array):
  """One-hot encodes an array with PyTorch.

  Args:
    array: The array to one-hot encode.

  Returns:
    A one-hot encoded array.
  """

  #max_value = max(array) + 1
  max_value = 40
  one_hot_encoded_array = torch.zeros(max_value, dtype=torch.float32)
  one_hot_encoded_array[array] = 1
  return one_hot_encoded_array

def one_hot_encode_array_39(array):
  """One-hot encodes an array with PyTorch.

  Args:
    array: The array to one-hot encode.

  Returns:
    A one-hot encoded array.
  """

  tmp = np.zeros((array.size, 39 + 1))
  tmp[np.arange(array.size), array] = 1
  
  # Flatten the array.
  flat_array = tmp.flatten()

  # Convert the flattened array to a list.
  list_array = flat_array.tolist()

  # Return the list representation of the array.
  return np.array(list_array)

def one_hot_encode_array_69(array):
  """One-hot encodes an array with PyTorch.

  Args:
    array: The array to one-hot encode.

  Returns:
    A one-hot encoded array.
  """

  tmp = np.zeros((array.size, 69 + 1))
  tmp[np.arange(array.size), array] = 1
  
  # Flatten the array.
  flat_array = tmp.flatten()

  # Convert the flattened array to a list.
  list_array = flat_array.tolist()

  # Return the list representation of the array.
  return np.array(list_array)

def break_up_integer_into_array(integer):
  """Breaks up an integer into an array of integers.

  Args:
    integer: An integer from 1 to 26.

  Returns:
    An array of integers where the i'th index to array is 1 if i is the integer
    input, and 0 otherwise.
  """

  array = [0] * 26
  if integer <= 26:
    array[integer - 1] = 1
  return array


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
    if our_game == 'mm':
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
      if our_game == 'mm':
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

class NeuralNetwork(nn.Module):
    def __init__(self, k1, k2, k3, k4):
        super().__init__()
        self.l1 = nn.Linear(k1, k2)
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(k2,k3)
        self.l4 = nn.Sigmoid()
        self.l5 = nn.Linear(k3, k4)
#        self.l6 = nn.ReLU()
        self.l6 = nn.Softmax(dim=1)

    def forward(self, x):
        pred_1 = self.l1(x)
        pred_2 = self.l2(pred_1)
        pred_3 = self.l3(pred_2)
        pred_4 = self.l4(pred_3)
        pred_5 = self.l5(pred_4)
        logits = pred_6 = self.l6(pred_5)
        #return logits
        return logits


def initialize_model():
    global our_depth
    global model
    global optimizer
    global loss_fn
    
    # make an instance of our network on device
    k1 = 2100
    k2 = 630
    k3 = 189
    k4 = 40

    print(f"Initializing Model with {k1} {k2} {k3} {k4}")
    
    model = NeuralNetwork(k1,k2,k3,k4).to(device)

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
        print("Reloading pre-trained model")
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

def test_and_display(model, cnt, X, Y, ts_array, total_worse_count):

    # lets test, get last row
    idx = cnt - 1

    y_oh = one_hot_encode_array_39(np.array(Y[idx])).reshape(1,40)
    #print(y_oh)
    x_oh = one_hot_encode_array_69(np.array(X[idx])).reshape(1,2100)
    #print(x_oh)
            
    # convert to tensors
    #y_oh_t = torch.tensor(y_oh, dtype=torch.float32, device=device)
    x_oh_t = torch.tensor(x_oh, dtype=torch.float32, device=device)

    # run prepared data through the model
    y_hat = model(x_oh_t).cpu()
    y_hat_detached = y_hat.detach()
    a = y_hat_detached_np = y_hat_detached.numpy()[0]

    indices = np.argsort(a)
    indices_reversed = indices[::-1]
    print(f"Balls in descending probability: {indices_reversed}")
    total_probability = 0.0
    for i in (0,1,2,3,4,5,6,7,8,9):
        total_probability += a[indices_reversed[i]]
        print(f"#{i+1} pick : {indices_reversed[i]:2d}, probability {a[indices_reversed[i]]:.5f} , total {total_probability:.5f}")

    #print(f"(actual) Y[{idx}] = {Y[idx][0]}")
    #print(f"y_oh  = {y_oh}")
    #print(f"y_hat_detached_np = {y_hat_detached_np}")

    # stats
    print(f"Total Worse Count: {total_worse_count}")
    
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
    cmd.give_help(sys.argv)
    cmd.set_my_col(sys.argv)
    winning_numbers = cmd.get_winning_numbers(sys.argv)
    cmd.set_our_game(sys.argv)
    if cmd.is_test(sys.argv):
        print('Running in test mode')
        test_mode = True
    else:
        print('Running in training mode')

    # initialize our model
    initialize_model()

    model_name = f"models/second-to-last-{our_game}.model"
    print(f"Model Name: {model_name}")
    if cmd.is_zero(sys.argv):
        if os.path.exists(model_name):
            print(f"Removing model {model_name}")
            os.unlink(model_name)

    # attempt to reload pre-trained model from disk
    attempt_reload()

    # read in correct data file
    ifile = f"data/{our_game}.csv"
    print(f"Using datafile {ifile}")
    ts_array = read_file_line_by_line_readline(ifile)
    #print(max(ts_array))
    #print(type(ts_array))
    #print(ts_array)
    #print(len)

    # stop at 400 epochs
    if not test_mode and epoch >= 400:
        print("At 400 epoch limit, exiting")
        exit(0)

    #largest = find_largest_integer_in_array(ts_array)
    #print(f"Largest integer: {largest}")
    #exit(0)

    len = len(ts_array)
    training_set_size = 30
    idx = 0
    X = []
    Y = []
    for i in range(0, len-training_set_size):
        x = []
        y = []
        for j in range(0, training_set_size):
            x.append(ts_array[j+idx])
        y.append(ts_array[idx+training_set_size])

        # now that x,y arrays are built, add them to X,Y
        X.append(x)
        Y.append(y)
        
        # onward
        idx += 1
        
    # just do one training call

    # how many elements ???
    cnt = 0
    for i in Y:
        cnt += 1
    print(f"Number of elements in Y: {cnt}")

    if reloaded_flag:
        # prepare to train for another 100 epochs
        model.train()
        old_epochs = epoch
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
    
    for epoch in range(old_epochs,epochs):

        # (this code is just stupid)
        if test_mode:
            continue

        if epoch == 200:
            learning_rate /= 10.0
            set_lr(model,learning_rate)
        
        idx = 0
        while idx < cnt: # ??? Note: we leave the last one for test
            # show our handywork
            #print(X[0],Y[0])
            y_oh = one_hot_encode_array_39(np.array(Y[idx])).reshape(1,40)
            #print(y_oh)
            x_oh = one_hot_encode_array_69(np.array(X[idx])).reshape(1,2100)
            #print(x_oh)
            
            # convert to tensors
            y_oh_t = torch.tensor(y_oh, dtype=torch.float32, device=device)
            x_oh_t = torch.tensor(x_oh, dtype=torch.float32, device=device)
            
            # train
            loss = train(model, x_oh_t, y_oh_t, loss_fn, optimizer)
            if idx >= (cnt-30):
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
                save_model(old_epochs,model,optimizer,loss,learning_rate,model_name)

    # now lets test
    model.eval()
    test_and_display(model, cnt, X, Y, ts_array, total_worse_count)

    # finis
