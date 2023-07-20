def check_input_format(input_string):
  """Checks if the input string is of the form digit,digit,digit.

  Args:
    input_string: The input string to check.

  Returns:
    A tuple of the three numbers if the input string is valid, else an error.
  """
  if len(input_string) != 3:
    return None

  sum = 0
  for i in range(3):
    if not input_string[i].isdigit():
      return None
    else:
      sum += int(input_string[i])
  if sum != 1:
    print("Numbers not accepted")
    return None
  
  return int(input_string[0]), int(input_string[1]), int(input_string[2])

if __name__ == "__main__":
  input_string = "010"
  numbers = check_input_format(input_string)
  if numbers:
    numbers = numbers + (42,)
    print(numbers)
  input_string = 'r'
  numbers = check_input_format(input_string)
  if not numbers:
    print("None")
  input_string = 'rat'
  numbers = check_input_format(input_string)
  if not numbers:
    print("None")
  input_string = '110'
  numbers = check_input_format(input_string)
  if not numbers:
    print("None")

