#!/home/pagec/venv/bin/python3

# Col #2
my_col = 2

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

'''
  Methinks, for this column, lets try
  an array of the first column 30 deep into the past
  and an array of 4 wide from 2->5 for the current
  element. Note y will be the last element in
  the first array.

  We have numbers from 1 to 69, so we'll use 70
  wide for one-hot storage.

  2380 comes from
    70 * 70 = 4900
    70 *  4 =  280
              ----
              5180

  The model will eventually end up as a transformer,
  as currently col2 doesn't consider the results of
  col1.

'''
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
#        self.l0 = nn.Linear(5180, 5180) # each next layer will be 30% of previous layer
        self.l1 = nn.Linear(5180, 1554) # each next layer will be 30% of previous layer
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(1554,140)
        self.l4 = nn.Sigmoid()
        self.l5 = nn.Linear(140, 70)
#        self.l6 = nn.ReLU()
        self.l6 = nn.Softmax(dim=1) # This will be column #1 result

    def forward(self, x):
 #       pred_0 = self.l0(x)
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
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9) # or -2 ???
#optimizer = torch.optim.Adam([var1, var2], lr=0.0001)

#lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)

# Check if second-to-last.model exists and if so, load it. Set to eval mode
# see also: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

reloaded_flag = False
epoch = None
model_name = f"third-col{my_col}-70.model"
if os.path.exists(model_name):
    reloaded_flag = True
    print(f"Reloading pre-trained model {model_name}")
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Reloaded epoch = {epoch}")
    loss = checkpoint['loss']
    model.eval()
    # - or -
    #model.train()

print(model.state_dict())
#print(model.get_seed())
#random_seed = model.state_dict()["random_seed"]
#print(f"random_seed = {random_seed}")

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

    # Update learning rate scheduler
    #lr_scheduler.step(loss)
    
    # Take model out of training mode
    #model.eval()
    
    #loss, current = loss.item()
    #print(f"loss: {loss}\n")
 
    return loss

def extract_numbers(data):
  """Extracts -7 -> -3 and sorts them

  Args:
    data: A string of data.

  Returns:
    A numpy array in sorted order.
  """

  columns = data.split(",")

  if False:
      print(int(columns[-7]))
      print(int(columns[-6]))
      print(int(columns[-5]))
      print(int(columns[-4]))
      print(int(columns[-3]))
    
  tmp = np.array([int(columns[-7]),
                     int(columns[-6]),
                     int(columns[-5]),
                     int(columns[-4]),
                     int(columns[-3])])

  #result = result[result[:,1].argsort()]
  tmp = np.sort(tmp)
  result = tmp
  #for idx, value in enumerate(tmp):
  #  result.append((idx+1,value))
  return result.tolist()

def read_file_line_by_line_readline(filename):
  """Reads a file line by line using readline.

  Args:
    filename: The name of the file to read.

  Returns:
    A list of the lines in the file.
  """

  ts_array = []
  with open(filename, "r") as f:
    while True:
      line = f.readline()
      if line == "":
        break
      x = extract_numbers(line)
      ts_array.append(x)
  f.close()
  return ts_array

def single_pass(model, loss_fn, optimizer, cnt, ts_array):
    global my_col
    idx = 70 # lets start here as it's easier to build our x
    while idx < cnt:
        # build x
        x = []
        for i in range(-69, 1):
            #print(f"adding - {i+idx}: {i} - ts_array[{i+idx}]: {ts_array[i+idx]}");
            x.append(ts_array[i+idx][my_col-1])
        # now add in 0->4 but skip my_col
        for i in (0,1,2,3,4):
            if (my_col-1) == i:
                continue
            x.append(ts_array[idx][i])
        #print(f"len(x) = {len(x)}")
        #exit(0)
        
        # build y
        y = [ts_array[idx][(my_col-1)]]
        #print(f"len(y) = {len(y)}")
        #print(y)

        # now we must one-hot y
        max_value = 70
        one_hot_encoded_y = torch.zeros(max_value, dtype=torch.float32)
        one_hot_encoded_y[y[0]] = 1.0
        one_hot_encoded_y = one_hot_encoded_y.unsqueeze(0)
        one_hot_encoded_y = one_hot_encoded_y.to(device)    
        #print(f"len(y) = {len(one_hot_encoded_y)}")
    
        # now we must one-hot x
        max_value = 70*(4+70)
        one_hot_encoded_x = torch.zeros(max_value, dtype=torch.float32)
        for index, value in enumerate(x):
            #print(index,value)
            one_hot_encoded_x[index*70 + value] = 1.0
        one_hot_encoded_x = one_hot_encoded_x.unsqueeze(0)
        one_hot_encoded_x = one_hot_encoded_x.to(device)    
        #print(f"len(x) = {len(one_hot_encoded_x)}")

        # train
        loss = train(model,
                     one_hot_encoded_x,
                     one_hot_encoded_y,
                     loss_fn,
                     optimizer)

        # onward
        idx += 1
    # done
    return loss

if __name__ == "__main__":

    # load in csv file
    ts_array = read_file_line_by_line_readline('pb.csv')
    # Note: ts_array elements are sorted small to large
    #print(ts_array)

    # get count of ts_array
    cnt = 0
    for i in ts_array:
        cnt += 1
    print(f"Cnt: {cnt}")

    if reloaded_flag:
        # train for another 100 epochs
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

    for epoch in range(old_epochs,epochs):

        continue

        # do a single pass through ts_array
        loss = single_pass(model, loss_fn, optimizer, cnt, ts_array)

        if loss < old_loss:
            status = "better"
        else:
            status = "worse"
        old_loss = loss

        print(f"epoch: {epoch}, loss = {loss}, {status}")

        # save every 100
        if not epoch % 100:
            if not first_save_flag:
                first_save_flag = True
            else:
                print("Saving Model")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, model_name)

    if False:
        # now save model
        print("Saving Model")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_name)

    # now lets test
    model.eval()
    # build x
    x = []
    idx = cnt - 1
    for i in range(-69, 1):
        #print(f"adding - {i+idx}: {i} - ts_array[{i+idx}]: {ts_array[i+idx]}");
        x.append(ts_array[i+idx][1])
    # now add in 1->4
    for i in (0,2,3,4):
        x.append(ts_array[idx][i])
        #print(f"len(x) = {len(x)}")
        #exit(0)

    #print(x)
    #exit(0)
    
    # now we must one-hot x
    max_value = 70*(4+70)
    one_hot_encoded_x = torch.zeros(max_value, dtype=torch.float32)
    for index, value in enumerate(x):
        #print(index,value)
        one_hot_encoded_x[index*70 + value] = 1.0
    one_hot_encoded_x = one_hot_encoded_x.unsqueeze(0)
    one_hot_encoded_x = one_hot_encoded_x.to(device)    
    #print(f"len(x) = {len(one_hot_encoded_x)}")
            
    # build y
    y = [ts_array[idx][0]]

    # now see where we are
    # Forward Pass
    y_hat = model(one_hot_encoded_x)

    print(y_hat)
    
