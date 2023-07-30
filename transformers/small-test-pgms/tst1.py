import numpy as np

a = np.array([0.0044, 0.0315, 0.0307, 0.0335, 0.0378, 0.0342, 0.0347, 0.0335, 0.0334,
         0.0294, 0.0342, 0.0369, 0.0294, 0.0341, 0.0370, 0.0287, 0.0300, 0.0328,
         0.0446, 0.0348, 0.0300, 0.0301, 0.0287, 0.0307, 0.0445, 0.0368, 0.0349,
         0.0123, 0.0088, 0.0176, 0.0092, 0.0074, 0.0097, 0.0136, 0.0078, 0.0098,
         0.0062, 0.0051, 0.0052, 0.0059])

sum = np.sum(a)
print(sum)

# Enumerate
for index, element in enumerate(a):
    # Print the index and element
    print(f"Index: {index}, Element: {element}")

indices = np.argsort(a)
indices_reversed = indices[::-1]
print(indices_reversed)
