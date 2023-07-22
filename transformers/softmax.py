import torch
import torch.nn as nn

def call_nn_softmax_on_tensor():
  """Calls nn.softmax on a tensor.

  Args:
    No arguments.

  Returns:
    No return value.
  """

  # Create a tensor.
  tensor = torch.randn(3, 4)

  # Call nn.softmax on the tensor.
  softmaxed_tensor = nn.Softmax(tensor)

  # Print the softmaxed tensor.
  print(softmaxed_tensor)

if __name__ == "__main__":
  # Run the function.
  call_nn_softmax_on_tensor()
