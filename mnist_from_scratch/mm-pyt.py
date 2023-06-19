# redo , converting to pyTorch
# where.from : https://mmuratarat.github.io/2020-01-09/backpropagation

import os
import torch
from torch import nn
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = "cpu"
print(f"Using {device} device")

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

# define X and y data        
X = torch.tensor([[0.5, 0.1,1,0,0],
                  [0.3, 0.2,0,1,0],
                  [0.7, 0.9,0,0,1],
                  [0.8, 0.1, 1,0,0]], dtype=torch.float32, device=device)

y = torch.tensor([[.1], [.6], [.4], [.1]], dtype=torch.float32, device=device)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(5, 3)
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(3, 1)
        #self.l4 = nn.ReLU()

#        
# this works too, but harder to see internals        
#        self.linear_relu_stack = nn.Sequential(
#            self.l1 = nn.Linear(5, 3),
#            self.l2 = nn.Sigmoid(),
#            self.l3 = nn.Linear(3, 1),
#            self.l4 = nn.ReLU()
#        )

    def forward(self, x):
        #x = self.flatten(x)
        #logits = self.linear_relu_stack(x)

        #print(f"Start of Forward, x = {x}\n")

        show_model("Start of Forward",self,x)

        pred_1 = self.l1(x)

        show_model("After nn.Linear(5,3)",self,pred_1)
        
        pred_2 = self.l2(pred_1)

        show_model("After nn.Sigmoid()", self,pred_2)
        
        pred_3 = self.l3(pred_2)

        show_model("After nn.Linuear(3,1) (logits)",self,pred_3)
                
        #logits = self.l4(pred_3)

        #show_model("After nn.ReLU (logits)", self,logits)

        if not show_model_off:
            print(f"End of Forward\n")

        #return logits
        return pred_3

# make an insteance of our network on device
model = NeuralNetwork().to(device)
print(model)

#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()

# Create the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# Main Training Loop
def train(model, x, y, loss_fn, optimizer):

    # set the model to training mode
    model.train()

    # Forward Pass
    y_hat = model(x)

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

#
# Main Loop
#

# number of epochs to execute
epochs = 10000
for epoch in range(epochs):
    if not epoch % 100:
        print(f"Epoch {epoch+1}\n-------------------------------")
    for idx in range(0,4):
        train(model, X[idx], y[idx], loss_fn, optimizer)

#
# Test
#
# Take model out of training mode
model.eval()

for idx in range(0,4):
        # test
        x = X[idx] # torch.tensor([0.5, 0.1,1,0,0])
        y_pred = model(x)
        print(f"x = {x}, y = {y[idx]}, y_pred = {y_pred}\n")

exit(0)

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
