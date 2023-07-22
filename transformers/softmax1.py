import torch

def softmax(tensor):
  """Computes the softmax function on a tensor.

  Args:
    tensor: The tensor to compute the softmax function on.

  Returns:
    The softmaxed tensor.
  """

  # Get the exponents of the tensor.
  exponents = torch.exp(tensor)

  # Sum the exponents.
  sum_of_exponents = torch.sum(exponents, dim=1, keepdim=True)

  # Divide each exponent by the sum of the exponents.
  softmaxed_tensor = exponents / sum_of_exponents

  # Return the softmaxed tensor.
  return softmaxed_tensor

if __name__ == "__main__":
  # Create a tensor.
  tensor = torch.randn(3, 4)

  # Compute the softmax of the tensor.
  softmaxed_tensor = softmax(tensor)

  # Print the softmaxed tensor.
  print(softmaxed_tensor)
