import torch

def restore_model(filename):
  """Restores a PyTorch model from a file.

  Args:
    filename: The filename to restore the model from.

  Returns:
    The restored PyTorch model.
  """

  state_dict = torch.load(filename)
  model = MyModel()
  model.load_state_dict(state_dict)

  return model

def save_model(model, filename):
  """Saves a PyTorch model to a file.

  Args:
    model: The PyTorch model to save.
    filename: The filename to save the model to.
  """

  state_dict = model.state_dict()
  torch.save(state_dict, filename)

  print(f"Model saved to {filename}")

