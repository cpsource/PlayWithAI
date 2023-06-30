import sqlite3
import torch

# Connect to the SQLite database
conn = sqlite3.connect('tensor_data.db')
cursor = conn.cursor()

# Create a table to store the tensor data
cursor.execute('''CREATE TABLE IF NOT EXISTS tensor (
                    id INTEGER PRIMARY KEY,
                    data BLOB)''')

# Sample PyTorch tensor
tensor = torch.tensor([1.0, 2.0, 3.0])

# Convert the tensor to bytes
tensor_bytes = tensor.numpy().tobytes()

# Insert the tensor data into the table
cursor.execute('INSERT INTO tensor (data) VALUES (?)', (memoryview(tensor_bytes),))

# Commit the changes and close the connection
conn.commit()
conn.close()

