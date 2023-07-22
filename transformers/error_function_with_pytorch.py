import torch

def error_function_with_pytorch(softmax_arrays, labels):
  """Calculates the error for a set of softmax arrays and labels.

  Args:
    softmax_arrays: An array of softmax arrays.
    labels: An array of labels.

  Returns:
    The error for the softmax arrays and labels.
  """

  error = 0
  for softmax_array, label in zip(softmax_arrays, labels):
    print(f"{label} {softmax_array[label]}")
    error += -torch.log(softmax_array[label])
  return error


if __name__ == "__main__":
  softmax_arrays = torch.rand(10, 10)
  print(softmax_arrays)
  #labels = torch.randint(0, 10, (10,))
  labels = [0,1,2,3,4,5,6,7,8,9]
  print(labels)
  error = error_function_with_pytorch(softmax_arrays, labels)
  print(error)
