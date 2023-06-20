# Humm - modified for pytorch

import torch

def permutations(tensor):
  """
  Lists all permutations of the given tensor.

  Args:
    tensor: The tensor to permute.

  Returns:
    A list of all permutations of the tensor.
  """

  global permutations
  if len(tensor) == 0:
    return [[]]
  else:
    permutations_without_first = permutations(tensor[1:])
    permutations = []
    for permutation in permutations_without_first:
      for i in range(len(permutation) + 1):
        new_permutation = permutation[:i] + [tensor[0].item()] + permutation[i:]
        permutations.append(new_permutation)
    return permutations

def main():
  """
  Main function.
  """

  tensor = torch.tensor([1,2,3,4])
  print(f"tensor[0] = {tensor[0]}, type = {type(tensor[0])}\n")
  p = permutations(tensor)
  a = []
  for permutation in p:
    print(f"permutation = {permutation}")
    a += [permutation]
  b = torch.tensor(a)
  print(b)

if __name__ == "__main__":
  main()

