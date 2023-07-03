#
# This guy uses our database.db to train a model
#
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

from datetime import datetime
import yfinance as yf
import mplfinance as mpf
import numpy as np
import torch
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
import sqlite3

import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
# runs slower with cuda
device = "cpu"

np.set_printoptions(threshold=sys.maxsize)

def get_column_index(cursor,column_name):
  # get column index
  column_index = None
  for i, desc in enumerate(cursor.description):
      if desc[0] == column_name:
          column_index = i
          break
  # Get the schema field names.
  #schema_field_names = [desc[0] for desc in cursor.description]
  #print(schema_field_names)
  # done
  return column_index

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
        self.l1 = nn.Linear(390, 80)
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(80, 3)
        self.l4 = nn.ReLU()
#        self.l4 = nn.Softmax(dim=1)

#        
# this works too, but harder to see internals        
#        self.linear_relu_stack = nn.Sequential(
#            self.l1 = nn.Linear(5, 3),
#            self.l2 = nn.Sigmoid(),
#            self.l3 = nn.Linear(3, 1),
#            self.l4 = nn.ReLU()
#        )

# Note: imput shape of x must be [batch-size,in_features]
# If we break up X into x's before we call forward, it takes
# three or four times longer

    def forward(self, x):
        #x = self.flatten(x)
        #logits = self.linear_relu_stack(x)

        #print(f"Start of Forward, x = {x}\n")

        #show_model("Start of Forward",self,x)

        pred_1 = self.l1(x)

        #show_model("After nn.Linear(5,3)",self,pred_1)
        
        pred_2 = self.l2(pred_1)

        #show_model("After nn.Sigmoid()", self,pred_2)
        
        pred_3 = self.l3(pred_2)

        #show_model("After nn.Linuear(3,1) (logits)",self,pred_3)
                
        logits = self.l4(pred_3)

        #show_model("After nn.ReLU (logits)", self,logits)

        #if not show_model_off:
            #print(f"End of Forward\n")

        #return logits
        return logits

# make an insteance of our network on device
model = NeuralNetwork().to(device)

#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()

# Create the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# Main Training Loop
def train(model, X, y, loss_fn, optimizer):

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
    model.eval()
    
    #loss, current = loss.item()
    #print(f"loss: {loss}\n")

    return loss

# tip of the hat to ChatGPT
def make_array_390(arr):
    if arr.size < 390:
        last_value = arr[-1]  # Get the last value of the array
        num_elements_to_append = 390 - arr.size
        arr = np.append(arr, np.full(num_elements_to_append, last_value))
    return arr

#
# Main Loop
#
def main():
    # setup cuda
    print(f"Using {device} device")

    # get database
    conn = sqlite3.connect("database.db")
    # Select all records from the table.
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM my_table")
    results = cursor.fetchall()

    # get column indexes
    closes_column_index = get_column_index(cursor,'closes')
    y1_column_index = get_column_index(cursor,'y1')
    y2_column_index = get_column_index(cursor,'y2')
    y3_column_index = get_column_index(cursor,'y3')

    #print(y1_column_index,y2_column_index,y3_column_index)
    
    # number of epochs to execute
    epochs = 20001
    for epoch in range(epochs):
        # step through sql database and train model
        for row in results:
            y1 = row[y1_column_index]
            y2 = row[y2_column_index]
            y3 = row[y3_column_index]
            y = np.array([[y1,y2,y3]])
            #print(y)
            #exit(0)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            #print(y.size())
            #exit(0)

            X = pickle.loads(row[closes_column_index])
            # Make sure 390
            X = make_array_390(X)
            # Scale X
            X = scale_tensor(X)
            # Convert to an np array for efficiency
            X = np.array([X])
            # Convert to Tensor on device from np
            X = torch.tensor(X,dtype=torch.float32,device=device)
            #print(X.size())
            #exit(0)
            
            # Send to model
            loss = train(model, X, y, loss_fn, optimizer)

            #
            # Test
            #
            # Take model out of training mode
            model.eval()

        # log every now and again
        if not epoch % 100:
            print(f"Epoch {epoch}, loss = {loss}\n-------------")
        
            #
            #for idx in range(0,4):
            #        # test
            #        x = X[idx]
            #        y_pred = model(x)
            #        print(f"x = {x}, y = {y[idx]}, y_pred = {y_pred}\n")
            #
            #x = torch.tensor([0.5, 0.1,0,0,0])
            #y_pred = model(x)
            #print(f"x = {x}, y = {y[0]}, y_pred = {y_pred}\n")
            
            #exit(0)
            
            #                  [0.3, 0.2,0,1,0],
            #                  [0.7, 0.9,0,0,1],
            #                  [0.8, 0.1, 1,0,0]], dtype=torch.float32, device=device)
            
            #print("Done!")

            #
            # Print (from Bard)
            #
            
            # show final model info
            #show_model("After Done",model)
            
            # Print the module's buffers
            #for name, buf in model.named_buffers():
            #    print(name, buf.size())

            #    # Print the module's internals
            #    print(model)

            #    if False:
            #
            # Plot
            #
            # plt.figure(figsize = (12, 6))
            # 
            # plt.subplot(121)
            # plt.stem(n,X[2], 'b', \
                # markerfmt=" ", basefmt="-b")
            # plt.xlabel('Freq (Hz)')
            # plt.ylabel('FFT Amplitude |X(freq)|')
            # 
            # if False:
            # plt.subplot(122)
            # plt.plot(n, X, 'r')
            # plt.xlabel('Minuite')
            # plt.ylabel('Price')
            # plt.tight_layout()
            # 
            # plt.show()
    sqlite3.close()
    
if __name__ == "__main__":
    main()
