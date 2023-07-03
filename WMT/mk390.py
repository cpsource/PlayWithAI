import numpy as np

def make_array_390(arr):
    if arr.size < 5:
        last_value = arr[-1]  # Get the last value of the array
        num_elements_to_append = 5 - arr.size
        arr = np.append(arr, np.full(num_elements_to_append, last_value))
    return arr

a = np.array([1,2,3])
print(make_array_390(a))

