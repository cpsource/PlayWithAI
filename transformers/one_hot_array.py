import torch

def one_hot_array(array):
  """One-hot encodes an array of integers.

  Args:
    array: The array to one-hot encode.

  Returns:
    The one-hot encoded array.
  """

  # Create a tensor with the same shape as the input array.
  tensor = torch.zeros(array.shape)

  # Set the element at the index of the input array to 1.
  for i, value in enumerate(array):
    tensor[i][value - 1] = 1

  # Return the one-hot encoded array.
  return tensor

if __name__ == "__main__":
  # Create an array of integers between 1 and 60.
  array = torch.arange(1, 61)

  # One-hot encode the array.
  one_hot_array = one_hot_array(array)

  # Print the one-hot encoded array.
  print(one_hot_array)
