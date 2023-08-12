import subprocess
import re
import pickle

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

  results = []
  game = 'mm'
  
  # investigate effect of depth vs distance
  for i in range(25,36):
    for j in range(-10,1):
      pgm = f'./second_to_last.py --game mm --check --cnt {j} --depth "[{i},-350,0,930,189,0]" --zero'
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
      #break
    #break
  
  pfn = f"shell-{game}.pkl" # Pickle File Name
  print(f"Writing results to {pfn}")
  with open(pfn, "wb+") as f:
    pickle.dump(results,f)
  f.close()

  print(results)
