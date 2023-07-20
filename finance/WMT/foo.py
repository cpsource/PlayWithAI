import torch
from torch import nn

m = nn.Softmax(dim=1)
n = torch.tensor([1,2,3])
output = m(n) # y_pred)
print(output, y_pred)
