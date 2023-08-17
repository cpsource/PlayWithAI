import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, k1, k2, k3, k4):
        super().__init__()
        self.l1 = nn.Linear(k1, k2)
        self.l2 = nn.Sigmoid()
        self.l5 = nn.Linear(k2, k4)
#        self.l6 = nn.ReLU()
#        self.l6 = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l5(x)
        #logits = x = self.l6(x)
        #return logits
        return x

def __init__(self):
    print("Loading Neural Network")
    
