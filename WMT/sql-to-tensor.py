import sqlite3
import torch
import numpy as np

# Connect to the SQLite database
conn = sqlite3.connect('tensor_data.db')
cursor = conn.cursor()

# Retrieve the tensor data from the table
cursor.execute('SELECT data FROM tensor WHERE id = ?', (1,))
row = cursor.fetchone()

if row is not None:
    # Extract the tensor bytes from the result
    tensor_bytes = row[0]

    # Convert the tensor bytes back to a NumPy array and make a writable copy
    tensor_array = np.frombuffer(tensor_bytes, dtype=np.float32).copy()

    # Convert the NumPy array to a PyTorch tensor
    tensor = torch.from_numpy(tensor_array)

    # Print the retrieved tensor
    print('Retrieved Tensor:')
    print(tensor)
else:
    print('No tensor found with the specified ID.')

# Close the connection
conn.close()

