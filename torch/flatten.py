import torch

# demonstrate flatten
print("Demonstrate the flatten function")

# Create an input tensor
input = torch.randn(1, 2, 3)

print(input)

# Flatten the input tensor
output = torch.flatten(input)

# Print the output tensor
print(output)
