import torch

def compliment_tensor(tensor):
  """Compliments a tensor in PyTorch."""
  #tensor = tensor > 0.5
  #tensor = torch.where(tensor, torch.ones_like(tensor), torch.zeros_like(tensor))
  #tensor[tensor == 1.0] = 0.0
  #tensor[tensor == 0.0] = 1.0
  #tensor = tensor.bitwise_not()
  tensor = torch.where(tensor == 1.0, torch.zeros_like(tensor), torch.ones_like(tensor))
  return tensor


if __name__ == "__main__":
  # Create a tensor
  tensor = torch.rand(5, 1)
  tensor[0] = 0.0
  tensor[1] = 0.0
  tensor[2] = 1.0
  tensor[3] = 0.0
  tensor[4] = 0.0
  
  # Compliment the tensor
  tensor = compliment_tensor(tensor)

  # Print the tensor
  print(tensor)
