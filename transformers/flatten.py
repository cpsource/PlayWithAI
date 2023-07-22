import numpy as np

def convert_array_to_list(array):
  """Converts an array to a list.

  Args:
    array: The array to convert.

  Returns:
    The list representation of the array.
  """

  # Flatten the array.
  flat_array = array.flatten()

  # Convert the flattened array to a list.
  list_array = flat_array.tolist()

  # Return the list representation of the array.
  return list_array

if __name__ == "__main__":
  # Create an array.
  array = np.array([[0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])

  # Convert the array to a list.
  list_array = convert_array_to_list(array)

  # Print the list representation of the array.
  print(list_array)
