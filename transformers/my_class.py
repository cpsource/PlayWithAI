import torch
from torch import nn

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

def __init__(self):
    print("Loading Neural Network")
    
