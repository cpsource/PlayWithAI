import numpy as np

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


def extract_second_to_last_column(data):
  """Extracts the second to last column from a string of data.

  Args:
    data: A string of data.

  Returns:
    The second to last column from the data.
  """

  columns = data.split(",")
  second_to_last_column = columns[-2]
  return int(second_to_last_column)

def append_integer_to_array(array, integer):
  """Appends an integer to an array.

  Args:
    array: An array.
    integer: An integer.

  Returns:
    The array with the integer appended.
  """

  array.append(integer)
  return array

def read_file_line_by_line_readline(filename):
  """Reads a file line by line using readline.

  Args:
    filename: The name of the file to read.

  Returns:
    A list of the lines in the file.
  """

  with open(filename, "r") as f:
    ts_array = []
    while True:
      line = f.readline()
      if line == "":
        break
      x = extract_second_to_last_column(line) 
      ts_array.append(x)
  return ts_array

if __name__ == "__main__":
    ts_array = read_file_line_by_line_readline("pb.csv")
    print(max(ts_array))
    print(type(ts_array))
    print(ts_array)
    len = len(ts_array)
    print(len)

    exit(0)

    largest = find_largest_integer_in_array(ts_array)
    print(f"Largest integer: {largest}")
    exit(0)

    training_set_size = 30
    idx = 0
    z = 3
    for i in range(0, len-training_set_size):
        x = []
        y = []
        for j in range(0, training_set_size):
            x.append(ts_array[j+idx])
        y.append(ts_array[idx+training_set_size])
        idx += 1 
        # stop and test here
        print(idx, x, y)
        print(break_up_integer_into_array(y))
        #z -= 1
        if not z:
            exit(0)


