#!/home/pagec/venv/bin/python

import numpy as np
import random

squish_array = [0] * 72

idx = 1
num = 1
for i in range(0,14):
    for j in (0,1,2,3,4):
        squish_array[idx] = num
        idx += 1
    num += 1
squish_array[idx] = num - 1

def create_random_array(size, min_value, max_value):
  """Creates a random array of numbers in the given range."""
  array = []
  for i in range(size):
    array.append(random.randint(min_value, max_value))
  return array

array = create_random_array(100, 1, 72)

def squish(array):
    res = []
    for i in array:
        res.append(squish_array[i])
    return res

if __name__ == "__main__":
    print(squish_array)
    tst = create_random_array(100,1,70)
    tst.append(71)
    print(tst)
    res = squish(tst)
    print(res)
    
