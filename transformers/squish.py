#!/home/pagec/venv/bin/python

import numpy as np
import random

# Create an initialize a conversion array. 1->71 goes in,
# and 0->13 come out
squish_array = [0] * 72
idx = 1
num = 0
for i in range(0,14):
    for j in (0,1,2,3,4):
        squish_array[idx] = num
        idx += 1
    num += 1
squish_array[idx] = num - 1

# used for test only
if False:
    def create_random_array(size, min_value, max_value):
        """Creates a random array of numbers in the given range."""
        array = []
        for i in range(size):
            array.append(random.randint(min_value, max_value))
            return array

def squish(array):
    res = []
    for i in array:
        res.append(squish_array[i])
    return np.array(res)

# first squish then one-hot.
# The advantage here is that only 14 bits are
# needed per number
def one_hot_squish(array):
    a = squish(array)
    tmp = np.zeros((a.size, 13 + 1))
    tmp[np.arange(a.size), a] = 1
  
    # Flatten the array.
    flat_array = tmp.flatten()

    # Convert the flattened array to a list.
    list_array = flat_array.tolist()

    # Return the list representation of the array.
    return np.array(list_array,dtype=np.float32)
    
if __name__ == "__main__":
    if False:
        array = create_random_array(100, 1, 72)
        print(squish_array)
        tst = create_random_array(100,1,70)
        tst.append(71)
        print(tst)
        res = squish(tst)
        print(res)

    tst = np.array([1,71])
    res = one_hot_squish(tst)

    # should print [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
    # 0. 0. 0. 1.] 28 <class 'numpy.float32'>

    print(res,len(res),type(res[0]))
