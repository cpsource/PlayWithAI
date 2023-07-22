def print_numbers_in_two_spaces_each(start_number, end_number):
  """Prints numbers in two spaces each between start_number and end_number.

  Args:
    start_number: The starting number.
    end_number: The ending number.

  Returns:
    No return value.
  """

  # Print numbers in two spaces each between start_number and end_number.
  for number in range(start_number, end_number + 1):
    print(f"{' ' * 2}{number}")

if __name__ == "__main__":
  # Run the function with different numbers.
  print_numbers_in_two_spaces_each(1, 99)
