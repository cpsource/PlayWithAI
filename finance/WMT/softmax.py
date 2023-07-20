import sys
# Remove the first entry (current working directory) from sys.path
sys.path = sys.path[1:]
# Append the current working directory at the end of sys.path
sys.path.append("")

import torch
import torch.nn.functional as F

# Assuming you have a PyTorch tensor "output" representing the network output
output = torch.tensor([2.0, 1.0])  # Example tensor, replace with your network's output

# Apply softmax function
probabilities = F.softmax(output, dim=0)

print(probabilities)
