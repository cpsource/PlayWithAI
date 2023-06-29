import torch

# Create two single-dimensional tensors
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])

# Concatenate the tensors along the specified dimension (0 for rows, 1 for columns)
concatenated_tensor = torch.cat((tensor1.unsqueeze(0), tensor2.unsqueeze(0)), dim=0)

# Print the concatenated tensor
print(concatenated_tensor)
