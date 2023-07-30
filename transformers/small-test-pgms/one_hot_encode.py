import numpy as np
import torch

def one_hot_encode(integer):
  """One-hot encodes an integer.

  Args:
    integer: The integer to one-hot encode.

  Returns:
    The one-hot encoded array.

  Note:
    scikit-learn, tensorflow, pytorchy, keras, h2o, spark ml, nltk, and pandas all have
    one-hot encoders built-in.

  """

  modulo = 26
  if integer > modulo:
      integer %= modulo
  array = [0] * (modulo+1)
  array[integer] = 1
  return array

def convert_integer_array_to_numpy_array_of_float32(integer_array):
  """Converts an integer array to a NumPy array of float32.

  Args:
    integer_array: The integer array to convert.

  Returns:
    A NumPy array of float32.
  """

  numpy_array = np.array(integer_array, dtype=np.float32)
  return numpy_array

if __name__ == "__main__":
  integer = 1
  array = one_hot_encode(integer)
  print(array)

  integer = 26
  array = convert_integer_array_to_numpy_array_of_float32(one_hot_encode(integer))
  print(array)

  integer = 27
  array = one_hot_encode(integer)
  print(array)
  device = torch.device("cpu") # or cuda
  y = torch.tensor(array,dtype=torch.float32,device=device)
  print(y)

  if False:
    integer_array = [1, 2, 3, 4, 5]
    numpy_array = convert_integer_array_to_numpy_array_of_float32(integer_array)
    print(numpy_array)
    print(torch.tensor(integer_array,dtype=torch.float32))

