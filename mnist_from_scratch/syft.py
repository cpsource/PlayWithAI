import numpy as np

# Create a NumPy array with the categorical data
data = np.array(['red', 'green', 'blue'])

# Create a dictionary that maps each category value to its index
category_to_index = {
    'red': 0,
    'green': 1,
    'blue': 2
}

# Create a new NumPy array for the one-hot encoded data
one_hot_encoded = np.zeros((data.size, len(category_to_index)))
print(f"initial one_hot_encoded = {one_hot_encoded}\n")

# Set the value to 1 in the one-hot encoded array if the original category value is present
for i, value in enumerate(data):
    print(f"i = {i}, value = {value}, category_to_index[value] = {category_to_index[value]}\n")
    one_hot_encoded[i, category_to_index[value]] = 1

# Print the one-hot encoded data
print(f"data              = {data}, data.size = {data.size}\n")
print(f"category_to_index = {category_to_index}, len(category_to_index) = {len(category_to_index)}\n")
print(f"one_hot_encoded   = {one_hot_encoded}\n")
