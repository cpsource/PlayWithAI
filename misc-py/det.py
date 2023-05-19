import numpy as np

def determinant(matrix):
  """
  Finds the determinant of a matrix.

  Args:
    matrix: A NumPy array representing a matrix.

  Returns:
    The determinant of the matrix.
  """

  if len(matrix.shape) != 2:
    raise ValueError("The matrix must be 2D.")

  if matrix.shape[0] != matrix.shape[1]:
    raise ValueError("The matrix must be square.")

  # Initialize the determinant to 1.
  determinant = 1

  # Loop over the rows of the matrix.
  for row in range(matrix.shape[0]):
    # Find the sign of the current row.
    sign = (-1) ** row

    # Calculate the determinant of the submatrix formed by removing the current row and column.
    submatrix = np.delete(matrix, row, 0)
    submatrix = np.delete(submatrix, row, 1)

    # Multiply the determinant of the submatrix by the sign of the current row.
    determinant *= sign * np.linalg.det(submatrix)

  return determinant

  # Create a 2D NumPy array with 3 rows and 4 columns
array = np.array([[50, 29],
                   [30,44]])

# Print the array
print(array)

print(determinant(array))

det = np.linalg.det(array)
print(int(det))
