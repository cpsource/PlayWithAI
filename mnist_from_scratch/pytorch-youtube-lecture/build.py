import time
import argparse
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# parse command line

parser = argparse.ArgumentParser(description="PyTorch Example")
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torcdh.device('cpu')
    
#############################################
# Get Device for Training
# -----------------------
# We want to be able to train our model on a hardware accelerator like the GPU or MPS,
# if available. Let's check to see if `torch.cuda <https://pytorch.org/docs/stable/notes/cuda.html>`_
# or `torch.backends.mps <https://pytorch.org/docs/stable/notes/mps.html>`_ are available, otherwise we use the CPU.

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
# How many gpu's
print(f"torch.cuda.device_count()= {torch.cuda.device_count()}\n")

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias
print(X)
print(y)
print(X[:10])

# create a train/test splot
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split],y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,
                train_labels,
                c="b",
                s=4,
                label="Training Data")
    
    plt.scatter(test_data,
                test_labels,
                c='g',
                s=4,
                label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data,
                     predictions,
                     c='r',
                     s=4,
                     label='Predictions')
    plt.legend(prop={'size': 14})
    
#plot_predictions()
#plt.show()

# Build model

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
        
# torch.optim - optimizer algorithms
torch.manual_seed(42)

# Create instance
model_0 = LinearRegressionModel()

# Check out parameters
print(list(model_0.parameters()))
print(model_0.state_dict())

# Make predictions with model
# Note: inference_mode turns off requires_grad
with torch.inference_mode():
    y_preds = model_0(X_test)

print(X_test,y_preds)

#plot_predictions(predictions=y_preds)
#plt.show()

# need loss function

loss_fn = nn.L1Loss()

# need optimize

optimizer = torch.optim.SGD(model_0.parameters(),
                            lr=0.001)
                            
# training loop

epochs = 5000
for epoch in range(epochs):
    # set model to training mode
    model_0.train()

    # forward pass
    y_pred = model_0(X_train)

    # calculate the loss -predictions first, labels second
    loss = loss_fn(y_pred, y_train)

    # optimizer zero grad
    optimizer.zero_grad()

    #perform backprop on the loss with respect to the parameters
    loss.backward()
    
    # turn off gradient tracking
    model_0.eval()

    # step the optimizer (perform gradient descent)
    optimizer.step()
    
    model_0.eval() # turn off gradient tracking

    print(f"Epoch: {epoch}, loss = {loss}\n")
    print(list(model_0.parameters()))
    print(model_0.state_dict())

y_pred_np = np.array(y_pred.detach())
#print(y_pred_np)
#print(X_train)

plt.figure(figsize=(10,7))
plt.scatter(X_train,
            y_train,
            c="b",
            s=4,
            label="Training Data")
plt.scatter(X_train,
            y_pred_np,
            c='g',
            s=4,
            label="Testing Data")

#    plot_predictions(predictions=y_pred_np)
plt.show()

