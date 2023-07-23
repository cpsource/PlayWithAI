import numpy as np

def sort_array(array):
  """Sorts a NumPy array of integers in ascending order.

  Args:
    array: The NumPy array to sort.

  Returns:
    A sorted NumPy array.
  """

  sorted_array = np.sort(array)
  return sorted_array

if __name__ == "__main__":
  array = np.array([10, 5, 2, 7, 3, 1, 8, 6, 4])
  sorted_array = sort_array(array)
  print(sorted_array)
