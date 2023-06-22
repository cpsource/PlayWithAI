import json
import torch

# Load the JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

# Convert the JSON data to a list of lists
data = [[float(v) for v in d.values()] for d in data]

# Convert the list of lists to a PyTorch tensor
tensor = torch.tensor(data)

# Print the tensor
print(tensor)
