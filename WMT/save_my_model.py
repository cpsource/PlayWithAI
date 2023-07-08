import torch

def save_model(model, filename):
  """Saves a PyTorch model to a file.

  Args:
    model: The PyTorch model to save.
    filename: The filename to save the model to.
  """

  state_dict = model.state_dict()
  torch.save(state_dict, filename)

  print(f"Model saved to {filename}")

