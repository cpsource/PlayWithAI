import torch
import pickle
import sqlite3

# Create a PyTorch tensor
tensor_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Serialize the tensor to a byte array
tensor_pickle = pickle.dumps(tensor_data)

# Connect to the SQLite database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Create a table to store the tensor
cursor.execute('''CREATE TABLE IF NOT EXISTS tensors
                  (id INTEGER PRIMARY KEY AUTOINCREMENT, data BLOB)''')

# Insert the tensor data into the database
cursor.execute('INSERT INTO tensors (data) VALUES (?)', (sqlite3.Binary(tensor_pickle),))
conn.commit()

# Close the database connection
conn.close()

