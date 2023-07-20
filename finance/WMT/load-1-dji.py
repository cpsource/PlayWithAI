import torch
import numpy as np

# Define the path to the CSV file.
csv_file_path = "./^dji.csv"

# Create a DataPipe to read the CSV file.
np_data = np.loadtxt(csv_file_path, dtype=np.float32, delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6))

print(np_data)
