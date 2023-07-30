def find_largest_integer_in_array(array):
  """Finds the largest integer in an array.

  Args:
    array: An array of integers.

  Returns:
    The largest integer in the array.
  """

  largest_integer = array[0]
  for i in range(1, len(array)):
    if array[i] > largest_integer:
      largest_integer = array[i]
  return largest_integer


if __name__ == "__main__":
  array = [10, 5, 2, 7, 3, 1, 8, 9, 6, 4]
  largest_integer = find_largest_integer_in_array(array)
  print(largest_integer)
