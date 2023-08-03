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
    return res

# first squish then one-hot.
# The advantage here is that only 14 bits are
# needed per number
def one_hot_squish(array):
    a = []
    for x in array:
        a.append(x)

    # get length of a
    cnt = 0
    for i in a:
        cnt += 1

    tmp = [[0] * (cnt*14)]

    for idx, val in enumerate(a):
        tmp[0][idx*14 + squish_array[val]] = 1

    # Return the list representation of the array.
    return tmp

def read_file_line_by_line_readline(filename):
  """Reads a file line by line using readline.

  Args:
    filename: The name of the file to read.

  Returns:
    A list of the lines in the file.
  """
  cnt = 0
  ts_array = []
  with open(filename, "r") as f:
    while True:
      line = f.readline()
      if line == "":
        break
      x = extract_numbers(line)
      ts_array.append(x)
  f.close()
  cnt = len(ts_array)
  return cnt, ts_array

def extract_numbers(data):
  """Extracts -7 -> -3 and sorts them

  Args:
    data: A string of data.

  Returns:
    A numpy array in sorted order.
  """

  columns = data.split(",")

  if False:
      print(int(columns[-7]))
      print(int(columns[-6]))
      print(int(columns[-5]))
      print(int(columns[-4]))
      print(int(columns[-3]))
    
  tmp = np.array([int(columns[-7]),
                     int(columns[-6]),
                     int(columns[-5]),
                     int(columns[-4]),
                     int(columns[-3])])

  #result = result[result[:,1].argsort()]
  tmp = np.sort(tmp)
  result = tmp
  #for idx, value in enumerate(tmp):
  #  result.append((idx+1,value))
  return result.tolist()

if __name__ == "__main__":
    tst = [1,71]
    res = one_hot_squish(tst)
    print(res)
    exit(0)
    
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
