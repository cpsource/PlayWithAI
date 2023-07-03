def process_input(input_str,id):
    # Split the input string into individual numbers
    numbers = input_str.split(',')

    # Check if the input contains exactly three numbers
    if len(numbers) != 3:
        print("Invalid input format. Please provide three comma-separated numbers.")
        return None

    try:
        # Convert the numbers to integers
        number1 = int(numbers[0])
        number2 = int(numbers[1])
        number3 = int(numbers[2])

        if number1 + number2 + number3 != 1:
            print("Invalid input format. Please provide the sum = 1.")
            return None
        
        # Return the numbers as a tuple
        return (number1, number2, number3, id)
    
    except ValueError:
        print("Invalid input format. Please provide three comma-separated numbers.")
        return None


def get_input(id):
    # cheezy do/while for python3
    while True:
        # Test the subroutine
        input_str = input("Enter three numbers (comma-separated): ")
        result = process_input(input_str,id)
        # Condition to check if the loop should continue
        if result is not None:
            break
    return result

if __name__ == "__main__":
  result = get_input(5)
  if result is not None:
      print(f"The numbers in tuple form: {result}")

