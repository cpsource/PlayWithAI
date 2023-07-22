def sort_array(array):
  """Sorts an array of numbers in ascending order.

  Args:
    array: An array of numbers.

  Returns:
    The sorted array.
  """

  for i in range(len(array)):
    for j in range(i + 1, len(array)):
      if array[i] > array[j]:
        array[i], array[j] = array[j], array[i]
  return array


if __name__ == "__main__":
  array = [10, 5, 2, 7, 3, 1, 8, 9, 6, 4]
  sorted_array = sort_array(array)
  print(sorted_array)
