# where.from : https://en.wikipedia.org/wiki/Matrix_multiplication
# stupid example : https://www.youtube.com/watch?v=2spTnAiQg4M

import numpy as np

a = [[1,2],[3,4]]
b = [[5,6],[7,8]]

#print(dir(np))

c = np.matmul(a,b)
print(a)
print(b)
print(c)

if 0:
    #In mathematics, particularly in linear algebra, matrix multiplication is a binary operation that produces a matrix from two matrices. For matrix multiplication, the number of columns in the first matrix must be equal to the number of rows in the second matrix. The resulting matrix, known as the matrix product, has the number of rows of the first and the number of columns of the second matrix. The product of matrices A and B is denoted as AB.[1]

    a = [1,2,3,4,5]
    b = [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]
    c = np.matmul(a,b)
    print(c)
