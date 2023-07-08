# test any y1,2,3 = 0 against a pre-built model

import sqlite3
import numpy as np
import pickle
import matplotlib.pyplot as plt
import get_ys
import time

import torch
from torch import nn
import os
import sqlite3

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
# runs slower with cuda
device = "cpu"

def update_record(conn, id, field, value):
  """Updates the value of the given field for the record with the given ID.

  Args:
    conn: The database connection.
    id: The ID of the record to update.
    field: The name of the field to update.
    value: The new value for the field.
  """

  cursor = conn.cursor()
  query = f"UPDATE my_table SET {field} = {value} WHERE id = {id}"
  cursor.execute(query)
  conn.commit()

def get_column_index(cursor,column_name):
  # get column index
  column_index = None
  for i, desc in enumerate(cursor.description):
      if desc[0] == column_name:
          column_index = i
          break
  # Get the schema field names.
  #schema_field_names = [desc[0] for desc in cursor.description]
  #print(schema_field_names)
  # done
  return column_index

# tip of the hat to ChatGPT
def make_array_390(arr):
    if arr.size < 390:
        last_value = arr[-1]  # Get the last value of the array
        num_elements_to_append = 390 - arr.size
        arr = np.append(arr, np.full(num_elements_to_append, last_value))
    return arr

def restore_model(filename):
  """Restores a PyTorch model from a file.

  Args:
    filename: The filename to restore the model from.

  Returns:
    The restored PyTorch model.
  """

  # make an insteance of our network on device
  model = NeuralNetwork().to(device)
  # reload
  model.load_state_dict(torch.load(filename))
  # take out of training mode
  model.eval()

  return model

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(390, 100)
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(100,50)
        self.l4 = nn.Sigmoid()
        self.l5 = nn.Linear(50, 3)
        #self.l6 = nn.ReLU()
        self.l6 = nn.Softmax(dim=1)

#        
# this works too, but harder to see internals        
#        self.linear_relu_stack = nn.Sequential(
#            self.l1 = nn.Linear(5, 3),
#            self.l2 = nn.Sigmoid(),
#            self.l3 = nn.Linear(3, 1),
#            self.l4 = nn.ReLU()
#        )

# Note: imput shape of x must be [batch-size,in_features]
# If we break up X into x's before we call forward, it takes
# three or four times longer

    def forward(self, x):
        #x = self.flatten(x)
        #logits = self.linear_relu_stack(x)

        #print(f"Start of Forward, x = {x}\n")

        #show_model("Start of Forward",self,x)

        pred_1 = self.l1(x)
        pred_2 = self.l2(pred_1)
        pred_3 = self.l3(pred_2)
        pred_4 = self.l4(pred_3)
        pred_5 = self.l5(pred_4)
        logits = pred_6 = self.l6(pred_5)

        #show_model("After nn.Linuear(3,1) (logits)",self,pred_3)
                
        #logits = self.l4(pred_3)

        #show_model("After nn.ReLU (logits)", self,logits)

        #if not show_model_off:
            #print(f"End of Forward\n")

        #return logits
        return logits

# make an insteance of our network on device
#model = NeuralNetwork().to(device)
# restore model
model = restore_model("load-3-wmt.model")

#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()

# Create the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# Take model out of training mode
#model.eval()

# tip of the hat to ChatGPT
def make_array_390(arr):
    if arr.size < 390:
        last_value = arr[-1]  # Get the last value of the array
        num_elements_to_append = 390 - arr.size
        arr = np.append(arr, np.full(num_elements_to_append, last_value))
    return arr

def main():
  """The main function."""

  conn = sqlite3.connect("database.db")

  # Select all records from the table.
  cursor = conn.cursor()

  cursor.execute("SELECT * FROM my_table where y1 = 0 and y2 = 0 and y3 = 0")
  results = cursor.fetchall()

  # get column index
  column_name = 'closes'
  closes_column_index = get_column_index(cursor,column_name)
  ticker_name = 'ticker'
  ticker_column_index = get_column_index(cursor,ticker_name)
  
  # Step through the results one at a time and update the name field.
  for row in results:
    id = row[0]
    X = pickle.loads(row[closes_column_index])

    #print(X.shape)
    X = make_array_390(X)
    
    N = len(X)
    n = np.arange(N)
    
    #print(f"id = {id}, X1 = {X1}")
    #update_record(conn, id, "name", "John Doe")

    # Print some info
    min = np.min(X)
    max = np.max(X)
    spread = max-min

    # ask model for prediction
    x = torch.tensor(X, dtype=torch.float32, device=device)
    y_pred = model(x)
    
    print(f"ticker = {row[ticker_column_index]}, y_pred = {y_pred}, min = {min:.2f}, max = {max:.2f}, spread = {spread:.2f}")
    
    #
    # Plot
    #
    plt.figure(figsize = (12, 6))

    if True:
      #plt.subplot(122)
      plt.plot(n, X, 'r')
      plt.xlabel('Minuite')
      plt.ylabel('Price')
      plt.tight_layout()
      
      plt.show(block=False)
      plt.pause(5)
      plt.close()

    if 0:
      # ask
      result = get_ys.get_input(id)
      # update record
      cursor.execute("UPDATE my_table SET y1 = ?, y2 = ?, y3 = ? WHERE id = ?",
                     result)
      conn.commit()

  conn.close()

if __name__ == "__main__":
  main()
