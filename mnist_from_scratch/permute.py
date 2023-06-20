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
        new_permutation = permutation[:i] + [tensor[0]] + permutation[i:]
        permutations.append(new_permutation)
    return permutations


def main():
  """
  Main function.
  """

  tensor = [1,2,3,4]
  p = permutations(tensor)
  for permutation in p:
    print(permutation)


if __name__ == "__main__":
  main()

