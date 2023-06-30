import torch
import pickle
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Retrieve the serialized tensor from the database
cursor.execute('SELECT data FROM tensors WHERE id = ?', (1,))  # Assuming you want to retrieve the tensor with ID 1
row = cursor.fetchone()
tensor_pickle = row[0]

# Deserialize the tensor from the pickle data
tensor_data = pickle.loads(tensor_pickle)
print(tensor_data)

# Convert the deserialized data to a PyTorch tensor
#tensor = torch.tensor(tensor_data).clone().detach()

# Close the database connection
conn.close()

# Print the PyTorch tensor
#print(tensor)

