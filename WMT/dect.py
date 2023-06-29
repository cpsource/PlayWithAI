import numpy as np

# Define the starting value and the number of steps
start_value = 150
num_steps = 388
step = 0.2134

# Generate the PyTorch tensor
tensor = np.arange(start_value, start_value - (num_steps * step), -step)

# Print the tensor
print(tensor.shape)
print(tensor)
