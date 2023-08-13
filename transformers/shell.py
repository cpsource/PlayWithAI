#!/home/pagec/venv/bin/python3

import subprocess
import re
import pickle
import signal
import cmd_lin as cmd
import sys
import frame

'''
  Col    Meaning
  --------------
  1      distance
  2      ball #
  3      number of occurances of ball #
  4      game offset, 0,-1,-2 etc
  5      width of game  30, 35, etc
  6      depth of game, -100 etc
  7      neural network sizes
  8      neural network sizes
  9      neural network sizes
 10      neural network sizes
'''

# which column
my_col = 6

signal_flag = False
def signal_handler(signum, frame):
  global signal_flag
  signal_flag = True
  print("Control-C pressed! Exit will happen after the current shell command in progress.")
  
def find_numbers(string):
  """Finds and returns the two numbers in the specified string as integers."""

  pattern = r"Error Distance: (\d+) for ball (\d+)"
  match = re.search(pattern, string)
#  print(match.group(1))
#  numbers = [int(match) for match in matches]
  return [int(match.group(1)),int(match.group(2))]

def shell_off_program(program_name):
  """Shells off the specified program and returns the printout."""

  process = subprocess.Popen(program_name, shell=True, stdout=subprocess.PIPE)
  output, _ = process.communicate()
  return output.decode("utf-8")

# of the form: ball = n, count = (count)
def find_count(output,regex):
  ball_regex = re.compile(regex)
  match = ball_regex.search(output)
  return int(match.group(1))

if __name__ == "__main__":
  signal.signal(signal.SIGINT, signal_handler)
  frame = inspect.currentframe()
  my_col = cmd.set_my_col(sys.argv)

  results = []
  game = 'mm'

    # get max expected ball (from game web sites)
  if cmd.our_game == 'mm':
      if my_col == 6:
          max_ball_expected = 25
      else:
          max_ball_expected = 70
  else:
      if my_col == 6:
          max_ball_expected = 26
      else:
          max_ball_expected = 69

  # try some 1->5 balls
  l = 265
  depth = 70
  for i in range(1,6):
    for j in range(-5,1):
      for k in range(2,7):
        ktmp = int((depth * max_ball_expected)*(k/10))
        #k = 220
        pgm = f'./second_to_last.py --game mm --col {i} --cnt {j} --depth "[{depth}, -{l}, 0, {ktmp}, 180, 0]" --check --zero'
        #pgm = f'./second_to_last.py --game mm --check --cnt {j} --depth "[{i},-{l},0,{k},189,0]" --zero'
        print(pgm)
        output = shell_off_program(pgm)

        # now parse printout to gather stats
        numbers = find_numbers(output)
        r = f"ball = {numbers[1]}, count = (\d+)"
        count = find_count(output,r)
        numbers.append(count)
        numbers.append(j)
        numbers.append(i)

        # Note: What kind of crazy person invented this goo ?
        m = r'Our Depth is \[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'      
        rx = re.compile(m)
        match = rx.search(output)
        numbers.append(int(match.group(2)))
      
        m = r"Initializing Model with (\d+) (\d+) (\d+) (\d+)"
        rx = re.compile(m)
        match = rx.search(output)
        numbers.append(int(match.group(1)))
        numbers.append(int(match.group(2)))
        numbers.append(int(match.group(3)))
        numbers.append(int(match.group(4)))

        # show what we've found
        print(f"{frame.f_lineno}: {numbers}")

        # store in grand results for later
        results.append(numbers)

        if signal_flag:
          break
      if signal_flag:
        break
    if signal_flag:
      break

  # store our results for later
  
  pfn = f"shell-{game}-{my_col}.pkl" # Pickle File Name
  print(f"Writing results to {pfn}")
  with open(pfn, "wb+") as f:
    pickle.dump(results,f)
  f.close()

  # done
  
  print(results)
  exit(0)
  
  # investigate effect of depth vs distance
  l = 265
  for i in range(20, 140, 10):
    for j in range(-10,1):
      k = int((i * 26)*.3)
      pgm = f'./second_to_last.py --game mm --check --cnt {j} --depth "[{i},-{l},0,{k},189,0]" --zero'
      print(pgm)
      output = shell_off_program(pgm)
      numbers = find_numbers(output)
      r = f"ball = {numbers[1]}, count = (\d+)"
      count = find_count(output,r)
      numbers.append(count)
      numbers.append(j)
      numbers.append(i)

      # Note: What kind of crazy person invented this goo ?
      m = r'Our Depth is \[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'      
      rx = re.compile(m)
      match = rx.search(output)
      numbers.append(int(match.group(2)))
      
      m = r"Initializing Model with (\d+) (\d+) (\d+) (\d+)"
      rx = re.compile(m)
      match = rx.search(output)
      numbers.append(int(match.group(1)))
      numbers.append(int(match.group(2)))
      numbers.append(int(match.group(3)))
      numbers.append(int(match.group(4)))

      # show what we've found
      print(numbers)

      # store in grand results for later
      results.append(numbers)

      if signal_flag:
        break
    if signal_flag:
      break
  
  pfn = f"shell-{game}-{my_col}.pkl" # Pickle File Name
  print(f"Writing results to {pfn}")
  with open(pfn, "wb+") as f:
    pickle.dump(results,f)
  f.close()

  print(results)
