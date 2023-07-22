def break_up_integer_into_array(integer):
  """Breaks up an integer into an array of integers.

  Args:
    integer: An integer from 1 to 26.

  Returns:
    An array of integers where the i'th index to array is 1 if i is the integer
    input, and 0 otherwise.
  """

  array = [0] * 26
  if integer <= 26:
    array[integer - 1] = 1
  return array


if __name__ == "__main__":
  integer = 1
  array = break_up_integer_into_array(integer)
  print(array)
