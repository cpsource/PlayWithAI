import numpy as np

result = np.array([0,0,0])

a = np.array([[1,2,3],[4,5,6]])

result += a[0]
result += a[1]

print(result)
#print(a[1])
#print(a.shape)

