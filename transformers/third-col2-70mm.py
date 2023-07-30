#!/home/pagec/venv/bin/python3

# Col #2 - of (1,2,3,4,5)
my_col = 2
test_mode = False
skip_array = None
learning_rate = 1e-1
model_name = ""
reloaded_flag = False

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

# say which game we are playing
our_game = "mm" # or pb, with mm being the default

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
        self.l0 = nn.Linear(5041, 5041) # each next layer will be 30% of previous layer
        self.l0s = nn.Sigmoid()
        self.l1 = nn.Linear(5041, 2100) # each next layer will be 30% of previous layer and multiple of 70
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(2100,140)
        self.l4 = nn.Sigmoid()
        self.l5 = nn.Linear(140, 71)
#        self.l6 = nn.ReLU()
        self.l6 = nn.Softmax(dim=1) # This will be column #1 result

    def forward(self, x):
        pred_0 = self.l0(x)
        pred_0s = self.l0s(pred_0)
        pred_1 = self.l1(pred_0s)
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

# Check if second-to-last.model exists and if so, load it. Set to eval mode
# see also: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

def set_lr(model,lr):
    global optimizer
    print(f"setting optimizer with lr = {lr}")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) # or -2 ???

def set_our_game(array):
    '''
    set our_game - must be one of pb or mm
    '''
    global our_game
    flag = False
    for item in array:
        if flag:
            if not (item == 'mm' or item == 'pb'):
                print("--game must be either mm or pb")
                exit(0)
            our_game = item
            break
        if '-g' == item or '--game' == item:
            flag = True
            continue
    print(f"Our Game is {our_game}")
    return

def set_my_col(array):
    global my_col
    res = 1 # default column in the set (1,2,3,4,5)
    flag = False
    for item in array:
        if flag:
            my_col = int(item)
            if my_col > 5 or my_col < 1:
                print("--col must be between 1 and 5")
                exit(0)
            return
        if '-c' == item or '--col' == item:
            flag = True
            continue
    my_col = res
    return

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
    
    set_my_col(sys.argv)
    print(f"Using column {my_col}")
    set_our_game(sys.argv)

    reloaded_flag = False
    epoch = 0
    model_name = f"models/third-col{my_col}-70{our_game}.model"
    if os.path.exists(model_name):
        reloaded_flag = True
        print(f"Reloading pre-trained model {model_name}")
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Reloaded epoch = {epoch}")
        loss = checkpoint['loss']
        #loss_fn = nn.CrossEntropyLoss()
        try:
            learning_rate = checkpoint['learning_rate']
            print(f"Restored last learning_rate = {learning_rate}")
        except Exception as e:
            learning_rate = 0.001
            print(f"Exception restoring learning_rate. Set to {learning_rate}")
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # or -2 ???
            #print(e)
        else:
            print("model to eval mode")
            model.eval()
            # - or -
            #model.train()

            #print(model.state_dict())
            #print(model.get_seed())
            #random_seed = model.state_dict()["random_seed"]
            #print(f"random_seed = {random_seed}")
    return

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

'''
  Test and display a prediction
'''
def test_and_display(model, cnt, ts_array, my_col):
    model.eval()
    if True:
        # lets test against the last one
        idx = cnt - 1

        while True:
            # build x of the form -70 -> -1
            x = []
            for i in range(-70, 1):
                x.append(ts_array[i+idx][my_col-1])

            # build y
            y = [ts_array[idx][(my_col-1)]]

            # now we must one-hot y
            max_value = 71
            one_hot_encoded_y = torch.zeros(max_value, dtype=torch.float32)
            one_hot_encoded_y[y[0]] = 1.0
            one_hot_encoded_y = one_hot_encoded_y.unsqueeze(0)
            one_hot_encoded_y = one_hot_encoded_y.to(device)    

            # now we must one-hot x
            max_value = 71*71
            one_hot_encoded_x = torch.zeros(max_value, dtype=torch.float32)
            for index, value in enumerate(x):
                #print(index,value)
                one_hot_encoded_x[index*71 + value] = 1.0
            one_hot_encoded_x = one_hot_encoded_x.unsqueeze(0)
            one_hot_encoded_x = one_hot_encoded_x.to(device)    
            #print(f"len(x) = {len(one_hot_encoded_x)}")

            # call the model to make the prediction
            
            y_hat = model(one_hot_encoded_x).cpu()
            y_hat_detached = y_hat.detach()
            a = y_hat_detached_np = y_hat_detached.numpy()[0]

            # drop last couple ??? - let's let the human do this for now
            if False:
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
            print(f"Balls in descending order: {indices_reversed}")
            total_probability = 0.0
            j = i = 0
            while True:
                if is_in_skip_array(indices_reversed[i]):
                    i += 1
                    continue
                total_probability += a[indices_reversed[i]]
                print(f"#{i+1} pick : {indices_reversed[i]:2d}, probability {a[indices_reversed[i]]:.5f} , total {total_probability:.5f}")
                i += 1
                j += 1
                if j >= 10:
                    break
            break
            
def single_pass(model, loss_fn, optimizer, cnt, ts_array):
    global my_col
    idx = 71 # lets start here as it's easier to build our x
    #idx = 1505 # lets start here as it's easier to build our x
    while idx < cnt:
        # build x of the form -71 -> -1
        x = []
        for i in range(-71, 0):
            #print(f"adding - {i+idx}: {i} - ts_array[{i+idx}]: {ts_array[i+idx]}");
            x.append(ts_array[i+idx][my_col-1])
        #print(f"len(x) = {len(x)}")
        #exit(0)
        
        # build y
        y = [ts_array[idx][(my_col-1)]]

        #print(f"idx: {idx}, my_col-1 = {my_col-1}, ts_array = {ts_array[idx]}")
        
        if False:
            # show our handywork
            print(f"idx: {idx}")
            print(f"len(x) = {len(x)}")
            print(x)
            print(f"len(y) = {len(y)}")
            print(y)
            print(ts_array)
        
        # debug
        #idx += 1
        #continue
    
        # now we must one-hot y
        max_value = 71
        if y[0] >= 70:
            y[0] = 70
        one_hot_encoded_y = torch.zeros(max_value, dtype=torch.float32)
        one_hot_encoded_y[y[0]] = 1.0
        one_hot_encoded_y = one_hot_encoded_y.unsqueeze(0)
        one_hot_encoded_y = one_hot_encoded_y.to(device)    
        #print(f"len(y) = {len(one_hot_encoded_y)}")
    
        # now we must one-hot x
        max_value = 71*71
        one_hot_encoded_x = torch.zeros(max_value, dtype=torch.float32)
        for index, value in enumerate(x):
            #print(index,value)
            if value > 70:
                value = 70
            one_hot_encoded_x[index*71 + value] = 1.0
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

def is_test(array):
    '''
    Return True if we have a command line switch -t or --test
    '''
    for item in array:
        if '--test' == item or '-t' == item:
            return True
    return False

def string_to_nparray(string):
  """Converts a string of the form `[1,2,3,4]` to an nparray.

  Args:
    string: A string of the form `[1,2,3,4]`.

  Returns:
    An nparray.
  """

  array_data = []
  for item in string[1:-1].split(","):
    array_data.append(int(item))
  return np.array(array_data)

def is_s(array):
    '''
    Get skip array if present
    '''
    res = np.array([])
    flag = False
    for item in array:
        if flag:
            res = string_to_nparray(item)
            return True, res
        if '-s' == item or '--skip' == item:
            flag = True
            continue
    return False, res

def is_in_skip_array(n):
    if skip_array is None:
        return False
    for i in skip_array:
        if n == i:
            return True
    return False

def give_help(array):
    '''
    Give help if asked. Exit afterwards.
    '''
    for item in array:
        if '--help' == item or '-h' == item:
            print("Usage:")
            print("  --help - give this help message then exit")
            print("  --col n - set column to n in the range of 1 to 5")
            print("  --game mm/pb - set the game. Defaults to mm")
            print("  --test - run in test mode (no training)")
            print("  --skip '[0,...]' - skip these balls as they are impossible")
            exit(0)
    return
    
if __name__ == "__main__":
    give_help(sys.argv)
    attempt_reload()
    s_flag, skip_array = is_s(sys.argv)
    if s_flag:
        print(f"Using skip array {skip_array}")
    if is_test(sys.argv):
        print('Running in test mode')
        test_mode = True
    else:
        print('Running in training mode')

    # load in csv file
    ifile = f"data/{our_game}.csv"
    print(f"Using datafile {ifile}")
    ts_array = read_file_line_by_line_readline(ifile)
    # Note: ts_array elements are sorted small to large
    #print(ts_array)

    # get count of ts_array
    cnt = 0
    for i in ts_array:
        cnt += 1
    print(f"Cnt: {cnt}")

    if reloaded_flag:
        # stop at 400 epochs
        if epoch >= 400:
            print("At 400 epoch limit, exiting")
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

    for epoch in range(old_epochs,epochs):

        if test_mode:
            continue

        if epoch == 20 or epoch == 80:
            learning_rate /= 10.0
            set_lr(model,learning_rate)

        # do a single pass through ts_array
        loss = single_pass(model, loss_fn, optimizer, cnt, ts_array)

        if loss < old_loss:
            status = "better"
        else:
            status = "worse"

        print(f"epoch: {epoch}, loss = {loss}, delta = {loss-old_loss}, status = {status}")

        old_loss = loss

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
                    'learning_rate' : learning_rate,
                }, model_name)

    if False:
        # now save model
        print("Saving Model")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'learning_rate' : learning_rate,
        }, model_name)

    # now lets test
    model.eval()
    test_and_display(model, cnt, ts_array, my_col)
