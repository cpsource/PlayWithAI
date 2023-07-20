import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def stretch_sigmoid(x, stretch_factor):
  return sigmoid(x / stretch_factor)

print(sigmoid(0),stretch_sigmoid(0, 2))
# Output: 0.5

print(sigmoid(1),stretch_sigmoid(1, 2))
# Output: 0.7310585786300049

print(sigmoid(-1),stretch_sigmoid(-1, 2))
# Output: 0.2689414213699951
