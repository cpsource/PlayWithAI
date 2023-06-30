#
# The problem with this approach is you have to know the data format to retrieve
# the data, else you get garbage. For example, store an array of ints, and read back
# an array of garbage floats.
#
# I've overcome this by first encoding into pickle, then storing it. Advantage, pickle
# stores metadata that is used on retrieval so you get back the type of a thing
# you stored
#
# And, BTW, ChatGPT seems to suffer from the same types of problems that Bard
# has with coding. Still, I would rate ChatGPT slightly better than Bard.
#

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

