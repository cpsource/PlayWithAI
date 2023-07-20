import sys
# Remove the first entry (current working directory) from sys.path
sys.path = sys.path[1:]
# Append the current working directory at the end of sys.path
sys.path.append("")

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
print(f"Using {device} device")

# Define the path to the CSV file.
csv_file_path = "./WMT.csv"

np.set_printoptions(threshold=sys.maxsize)

def gen_bads(start_value,num_steps,step):
    # Generate the PyTorch tensor
    tensor = np.arange(start_value, start_value - (num_steps * step), -step)
    tensor = tensor[1:]
    #print(tensor)
    #exit(0)
    if len(tensor) == 387:
        tensor = np.append(tensor,tensor[386])
    return tensor

def scale_tensor(tensor):
    # Compute the maximum and minimum values of the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    
    # Scale the tensor from -1 to +1
    scaled_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1
    
    return scaled_tensor

file_path = 'wmt-1m.pkl'

if os.path.exists(file_path):
    print("Loading from cache")
    with open(file_path,"rb") as f:
        d = pickle.load(f)
else:
    print("Loading from Yahoo")
    wmt = yf.Ticker('WMT')
    d = wmt.history(start="2023-06-18",end="2023-06-23",interval='1m')
    with open(file_path,"wb") as f:
        pickle.dump(d,f)

# Take the days
d1 = d[:388]
d2 = d[388:388*2]
d3 = d[388*2:-4]
#print(d1.index[387])

#
# Manage our test case
#

# Slice off the Close
X1 = d1['Close']
X2 = d2['Close']
X3 = d3['Close']
X1 = torch.tensor(X1,dtype=torch.float32, device=device)
X2 = torch.tensor(X2,dtype=torch.float32, device=device)
X3 = torch.tensor(X3,dtype=torch.float32, device=device)
X4 = torch.tensor(gen_bads(150,388,0.1),dtype=torch.float32, device=device)
X5 = torch.tensor(gen_bads(150,388,0.11),dtype=torch.float32, device=device)

#print(X5)
#exit(0)

X = torch.cat((X1.unsqueeze(0),
               X2.unsqueeze(0),
               X3.unsqueeze(0),
               X4.unsqueeze(0),
               X5.unsqueeze(0)),
               dim = 0)

#print(X)
#exit(0)

N = len(X[0])
n = np.arange(N)

# Come up with the answer
y = torch.tensor([[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0]], dtype=torch.float32, device=device)

#print(X.shape, y.shape)
#exit(0)

# Create a DataPipe to read the CSV file.
#np_data = np.loadtxt(csv_file_path, dtype=np.float32, delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6))


#exit(0)

# Scale
X = scale_tensor(X)
#N = len(X)
#n = np.arange(N)
print(X)
print(f"shape of X = {X.shape}")
print(y)
print(f"shape of y = {y.shape}")

#exit(0)

show_model_off = 1
# From Bard, show internals of model
def show_model(when,model,v=None):
    if show_model_off:
        return
    print(f"When: {when}\n")
    if v is not None:
        print(v)
    # Print the module's parameters
    for name, param in model.named_parameters():
        print("      ",name, param.size(),param.shape,param)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(388, 50)
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(50, 3)
        self.l4 = nn.Softmax(dim=1)
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
#print(model)

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
#
# Main Loop
#

# number of epochs to execute
epochs = 20001
for epoch in range(epochs):
    loss = train(model, X, y, loss_fn, optimizer)
    if not epoch % 1000:
        print(f"Epoch {epoch}, loss = {loss}\n-------------------------------")
#    for idx in range(0,4):
#        train(model, X[idx].to(device), y[idx].to(device), loss_fn, optimizer)

#
# Test
#
# Take model out of training mode
model.eval()

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

print("Done!")

#
# Print (from Bard)
#

# show final model info
show_model("After Done",model)

# Print the module's buffers
for name, buf in model.named_buffers():
    print(name, buf.size())

# Print the module's internals
print(model)

#
# Plot
#
plt.figure(figsize = (12, 6))

plt.subplot(121)
plt.stem(n,X[2], 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')

if False:
    plt.subplot(122)
    plt.plot(n, X, 'r')
    plt.xlabel('Minuite')
    plt.ylabel('Price')
    plt.tight_layout()

plt.show()

exit(0)

if __name__ == "__main__":
    x = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 100))
    y, f = fourier_transform(x)

    plt.plot(f, np.abs(y))
    plt.show()
