import numpy as np

# Create a numpy vector
vector = np.array([0, 1, 2, 3, 3])

# Get the number of unique values in the vector
num_classes = len(np.unique(vector))
print(f"np.unique = {np.unique(vector)}, len = {num_classes}\n")

# Create a one-hot array
one_hot_array = np.zeros((len(np.unique(vector)), num_classes))

if 0:
    for i in range(len(vector)):
        one_hot_array[i][vector[i]] = 1
else:
    # Set the elements of the one-hot array to 1 at the corresponding indices of the vector
    idx = 0
    for i, v in enumerate(np.unique(vector)):
        print(i,v)
        one_hot_array[i][idx] = 1
        idx += 1

# Print the one-hot array
print(one_hot_array)
