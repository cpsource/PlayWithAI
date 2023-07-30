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

def append_integer_to_array(array, integer):
  """Appends an integer to an array.

  Args:
    array: An array.
    integer: An integer.

  Returns:
    The array with the integer appended.
  """

  array.append(integer)
  return array

def read_file_line_by_line_readline(filename):
  """Reads a file line by line using readline.

  Args:
    filename: The name of the file to read.

  Returns:
    A list of the lines in the file.
  """

  with open(filename, "r") as f:
    ts_array = []
    while True:
      line = f.readline()
      if line == "":
        break
      x = extract_second_to_last_column(line)

      if x > 39:
        print(f"Warning: {x} gt 39, adjusting")
        x %= 39 
        
      ts_array.append(x)
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
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2100, 630)
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(630,189)
        self.l4 = nn.Sigmoid()
        self.l5 = nn.Linear(189, 40)
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

# make an instance of our network on device
model = NeuralNetwork().to(device)

# Create the optimizer
# we can also try lr=0.001, momentum=0.9
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # or -2 ???

# Check if second-to-last.model exists and if so, load it. Set to eval mode
# see also: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

reloaded_flag = False
epoch = None
model_name = "second-to-last.model"
if os.path.exists(model_name):
    reloaded_flag = True
    print("Reloading pre-trained model")
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Reloaded epoch = {epoch}")
    loss = checkpoint['loss']
    model.eval()
    # - or -
    #model.train()

#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()

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

if __name__ == "__main__":
    ts_array = read_file_line_by_line_readline("data/pb.csv")
    #print(max(ts_array))
    #print(type(ts_array))
    #print(ts_array)
    #print(len)

    #largest = find_largest_integer_in_array(ts_array)
    #print(f"Largest integer: {largest}")
    #exit(0)

    len = len(ts_array)
    training_set_size = 30
    idx = 0
    z = 3
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
        
        # stop and test here
        #print(idx, x, y)
        #print(break_up_integer_into_array(y))
        #z -= 1
        #if not z:
        #    exit(0)

    # just do one training call

    # how many elements ???
    cnt = 0
    for i in Y:
        cnt += 1
    print(f"Number of elements: {cnt}")
    
    if True and reloaded_flag:
        # lets test for a bit
        idx = cnt - 1

        y_oh = one_hot_encode_array_39(np.array(Y[idx])).reshape(1,40)
        #print(y_oh)
        x_oh = one_hot_encode_array_69(np.array(X[idx])).reshape(1,2100)
        #print(x_oh)
            
        # convert to tensors
        #y_oh_t = torch.tensor(y_oh, dtype=torch.float32, device=device)
        x_oh_t = torch.tensor(x_oh, dtype=torch.float32, device=device)

        y_hat = model(x_oh_t).cpu()
        y_hat_detached = y_hat.detach()
        a = y_hat_detached_np = y_hat_detached.numpy()[0]
        # drop last couple
        dropped = 0.0
        for tmpidx, tmpval in enumerate(a):
            if tmpidx == Y[idx][0] or tmpidx == Y[idx-1][0]:
                dropped += a[tmpidx]
                a[tmpidx] = 0.0
        # recalculate softmax
        #a = softmax_np(a)
        # make sure we add up to to one hundred percent
        sum = np.sum(a) + dropped

        print(f"Sum: {sum}")
        if False:
            # Enumerate
            for index, element in enumerate(a):
                # Print the index and element
                print(f"Index: {index}, Element: {element}")

        indices = np.argsort(a)
        indices_reversed = indices[::-1]
        print(f"Powerballs in descending order: {indices_reversed}")
        total_probability = 0.0
        for i in (0,1,2,3,4,5,6,7,8,9):
            total_probability += a[indices_reversed[i]]
            print(f"#{i+1} pick : {indices_reversed[i]:2d}, probability {a[indices_reversed[i]]:.5f} , total {total_probability:.5f}")

        #m = nn.Softmax(dim=0)
        #y_hat_sm = m(y_hat)

        print(f"(actual) Y[{idx}] = {Y[idx][0]}")
        #print(f"y_oh  = {y_oh}")
        #print(f"y_hat_detached_np = {y_hat_detached_np}")

        exit(0)

    if reloaded_flag:
        # train for another 500 epochs
        model.train()
        old_epochs = epoch
        epochs = epoch + 101
        print(f"New Epochs: {epochs}")
    else:
        # number of epochs to execute
        epochs = 501
        old_epochs = 0

    old_loss = 1.0

    #exit(0)

    first_save_flag = False
    
    for epoch in range(old_epochs,epochs):
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
            status = "worse"
        old_loss = loss
        
        print(f"Epoch: {epoch} : {loss} : {status}")
        if not epoch % 100:
            if not first_save_flag:
                first_save_flag = True
            else:
                # now save model
                print("Saving Model")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, "second-to-last.model")
