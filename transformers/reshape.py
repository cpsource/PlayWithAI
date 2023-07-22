import numpy as np

def convert_array_with_2100_columns_and_1_row_to_array_with_1_row_and_2100_columns(array):
  """Converts an np array with 2100 columns and 1 row to an array with 1 row and 2100 columns.

  Args:
    array: The np array to convert.

  Returns:
    The np array with the columns converted to rows.
  """

  # Reshape the array.
  reshaped_array = array.reshape(1, array.shape[0])

  # Return the array with the columns converted to rows.
  return reshaped_array

if __name__ == "__main__":
  # Create an array.
  array = np.arange(2100)
  array = array.reshape(1, 2100)

  # Convert the columns to rows.
  converted_array = convert_array_with_2100_columns_and_1_row_to_array_with_1_row_and_2100_columns(array)

  # Print the converted array.
  print(converted_array)
