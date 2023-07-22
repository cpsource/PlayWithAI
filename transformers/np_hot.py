import numpy as np

def convert_array_to_list(array):
  """Converts an array to a list.

  Args:
    array: The array to convert.

  Returns:
    The list representation of the array.
  """

  tmp = np.zeros((array.size, 26 + 1))
  tmp[np.arange(array.size), a] = 1
  
  # Flatten the array.
  flat_array = tmp.flatten()

  # Convert the flattened array to a list.
  list_array = flat_array.tolist()

  # Return the list representation of the array.
  return list_array

a = np.array([1, 22, 26])
b = convert_array_to_list(a)

print(b)
