def find_index_of_largest_number(array):
  """Finds the index to the largest number in an array.

  Args:
    array: The array to search.

  Returns:
    The index to the largest number in the array.
  """

  largest_number = array[0]
  largest_index = 0
  for index, number in enumerate(array):
    if number > largest_number:
      largest_number = number
      largest_index = index
  return largest_index


if __name__ == "__main__":
  array = [1, 2, 3, 4, 5]
  index = find_index_of_largest_number(array)
  print(array)
  print(index)
