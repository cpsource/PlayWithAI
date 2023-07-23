import numpy as np
import matplotlib.pyplot as plt

groups = None

def group_array(array, group_size):
  global groups
  result = []
  # build a lookup table one time if necessary
  if groups is None:
    groups = np.zeros(69,dtype=np.int32)
    idx = 0
    group_number = 1
    while True:
      ctr = group_size
      while idx < 69 and ctr > 0:
        groups[idx] = group_number
        idx += 1
        ctr -= 1
      group_number += 1
      if idx >= 69:
        break
    print(groups)
  for v in array:
    result.append(groups[v-1])
  #return np.array(result)
  return result

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

def read_file_line_by_line_readline(filename):
  """Reads a file line by line using readline.

  Args:
    filename: The name of the file to read.

  Returns:
    A list of the lines in the file.
  """

  ts_array = []
  with open(filename, "r") as f:
    while True:
      line = f.readline()
      if line == "":
        break
      x = extract_numbers(line)
      ts_array.append(x)
  f.close()
  return ts_array

def get_cnt(ts_cvt,tmp):
  cnt = 0
  for v in ts_cvt:
    if v == tmp:
      cnt += 1
  return cnt

if __name__ == "__main__":
  if False:
    t1 = 'Powerball,7,15,2023,57,43,2,55,9,18,2'
    t2 = 'Powerball,7,17,2023,17,8,9,41,5,21,4'
    t3 = 'Powerball,7,19,2023,7,13,10,24,11,24,2'

    en = extract_numbers(t1)
    result = group_array(en,5)
    print(f"{t1} - {extract_numbers(t1)} - result = {result}")

    en = extract_numbers(t2)
    result = group_array(en,5)
    print(f"{t2} - {extract_numbers(t2)} - result = {result}")

    print(f"{t3} - {extract_numbers(t3)}")

  ts_array = read_file_line_by_line_readline('pb.csv')
  #print(ts_array)
  ts_cvt = []
  for v in ts_array:
    tmp = group_array(v,5)
    ts_cvt.append(tmp)
  idx = 0
  for v in ts_cvt:
    cnt = get_cnt(ts_cvt,v)
    if cnt > 1:
      print(f"Idx: {idx} - Cnt: {cnt} - v = {v}")
    idx += 1
  x = []
  y = []
  for v in ts_array:
    tmp = np.array(v)
    mean = np.mean(tmp)
    std_dev = np.std(tmp)
    x.append(mean)
    y.append(std_dev)
  plt.scatter(x,y)
  plt.title("Mean vs Std_Dev")
  plt.xlabel("mean")
  plt.ylabel("std_dev")
  plt.show()
  
