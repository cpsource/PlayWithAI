import torch

def one_hot_encode_array_with_pytorch(array):
  """One-hot encodes an array with PyTorch.

  Args:
    array: The array to one-hot encode.

  Returns:
    A one-hot encoded array.
  """

  max_value = max(array) + 1
  max_value = 40
  one_hot_encoded_array = torch.zeros(max_value, dtype=torch.float32)
  one_hot_encoded_array[array] = 1
  return one_hot_encoded_array


if __name__ == "__main__":
  array = [21, 14, 8]
  one_hot_encoded_array = one_hot_encode_array_with_pytorch(array)
  print(one_hot_encoded_array)
