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

import input_string as iss

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
# runs slower with cuda
device = "cpu"

# in  result = [8.5443e-02, 7.8509e-07, 9.1456e-01]
# out result = [0        ,  0         ,          1]
def normalize_ys(result):
    # show starting point
    #print(result)
    # start somewhere
    max = result[0]
    idx = 0
    # find largest
    for i in range(1,len(result)):
        if result[i] > max:
            max = result[i]
            idx = i
    # currect the list
    for i in range(0,len(result)):
        if i == idx:
            result[i] = 1
        else:
            result[i] = 0
    # done
    return result

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

# tip of the hat to ChatGPT
def make_array_390(arr):
    if arr.size < 390:
        last_value = arr[-1]  # Get the last value of the array
        num_elements_to_append = 390 - arr.size
        arr = np.append(arr, np.full(num_elements_to_append, last_value))
    return arr

def scale_tensor(tensor):
    # Compute the maximum and minimum values of the tensor
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    
    # Scale the tensor from -1 to +1
    scaled_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1
    
    return scaled_tensor
  
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(390, 100)
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(100,50)
        self.l4 = nn.Sigmoid()
        self.l5 = nn.Linear(50, 3)
        #self.l6 = nn.ReLU()
        self.l6 = nn.Softmax(dim=0)
 
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

        #show_model("Start of Forward",self,x)

        pred_1 = self.l1(x)
        pred_2 = self.l2(pred_1)
        pred_3 = self.l3(pred_2)
        pred_4 = self.l4(pred_3)
        pred_5 = self.l5(pred_4)
        logits = pred_6 = self.l6(pred_5)        

        return logits

# restore model

# make an instance of our network on device
#model = NeuralNetwork().to(device)
#model = NeuralNetwork()
# reload
#model.load_state_dict(torch.load("load-3-wmt.model"))
# take out of training mode
#model.eval()

model = NeuralNetwork()
#optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load("load-3-wmt.model")
model.load_state_dict(checkpoint['model_state_dict'])
# Create the optimizer - note lr is learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()
# - or -
#model.train()

#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()

# Create the optimizer - note lr is learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

def ff(number, decimal_places=4):
  """Formats a floating point number with a specified number of decimal places.

  Args:
    number: The floating point number to format.
    decimal_places: The number of decimal places to use.

  Returns:
    A formatted string representation of the floating point number.
  """

  format_string = "%." + str(decimal_places) + "f"
  formatted_number = format_string % number
  return formatted_number

def main():
  """The main function."""

  conn = sqlite3.connect("database.db")

  # Select all records from the table.
  cursor = conn.cursor()

  cursor.execute("SELECT * FROM my_table where y1 = 0 and y2 = 0 and y3 = 0")

  # step through all records one at a time
  while True:
      row = cursor.fetchone()
      if row is None:
          break
      # Do something with the row.
      
      # results = cursor.fetchall()
      
      # get column index
      column_name = 'closes'
      closes_column_index = get_column_index(cursor,column_name)
      ticker_name = 'ticker'
      ticker_column_index = get_column_index(cursor,ticker_name)
      
      # Step through the results one at a time and update the name field.
      # for row in results:
      id = row[0]
      
      X = pickle.loads(row[closes_column_index])
      
      # make sure we are 390 big
      X = make_array_390(X)
      
      N = len(X)
      n = np.arange(N)
      
      #print(f"id = {id}, X1 = {X1}")
      #update_record(conn, id, "name", "John Doe")
      
      # Collect some info
      min = np.min(X)
      max = np.max(X)
      spread = max-min
      
      # scale from -1 to +1
      x = scale_tensor(X)
      
      #print(f"x = {x}")
      #exit(0)
      
      # ask model for prediction
      x = torch.tensor(x, dtype=torch.float32, device=device)
      y_pred = model(x)
      y_normal = normalize_ys(y_pred)

      # keep displaying until we decide what to do
      new_vals = None
      while True:

          print(f"ticker = {row[ticker_column_index]}, y_normal = {y_normal}:.1f, min = {min:.2f}, max = {max:.2f}, spread = {spread:.2f}")
          
          #
          # Plot for 5 seconds
          #
          plt.figure(figsize = (12, 6))
          #plt.subplot(122)
          plt.plot(n, X, 'r')
          plt.xlabel('Minuite')
          plt.ylabel('Price')
          plt.tight_layout()
          
          plt.show(block=False)
          plt.pause(5)
          plt.close()

          # get command from operator
          input_str = input("Cmd: 000, r - repeat, c - continue, u - update, w - write, q - quit ")
          vals = iss.check_input_format(input_str)
          if vals :
              new_vals = vals
              print(f"New Values Accepted: {new_vals}")
              continue
          if input_str == 'w':
              # write out our 390 values on console for later testing
              print("X = [")
              x_len = int(len(X)/10)
              print(type(x_len),x_len)
              for idx in range(x_len):
                  print(f"{ff(X[idx*10+0])},{ff(X[idx*10+1])},{ff(X[idx*10+2])},{ff(X[idx*10+3])},{ff(X[idx*10+4])},{ff(X[idx*10+5])},{ff(X[idx*10+6])},{ff(X[idx*10+7])},{ff(X[idx*10+8])},{ff(X[idx*10+9])},")
              print("]")
              continue
          if input_str == 'r':
              continue
          if input_str == 'c':
              break
          if input_str == 'u':
              if new_vals:
                  # use operator entered values
                  new_vals = new_vals + (id,)
                  # lets update the record
                  cursor.execute("UPDATE my_table SET y1 = ?, y2 = ?, y3 = ? WHERE id = ?",
                                 new_vals)
                  conn.commit()
                  print(f"Record Updated with {new_vals}")
                  cursor.execute("SELECT * FROM my_table where y1 = 0 and y2 = 0 and y3 = 0")
              else:
                  # use machine picked values
                  new_vals = (int(y_normal[0].item()),
                              int(y_normal[1].item()),
                              int(y_normal[2].item())
                              ,id)
                  # lets update the record
                  cursor.execute("UPDATE my_table SET y1 = ?, y2 = ?, y3 = ? WHERE id = ?",
                                 new_vals)
                  conn.commit()
                  print(f"Record Updated with {new_vals}")
                  cursor.execute("SELECT * FROM my_table where y1 = 0 and y2 = 0 and y3 = 0")
                  # continue
              break
          if input_str == 'q':
              conn.close()
              exit(0)
          print("Unknown command, we'll continue")
          break

  conn.close()

if __name__ == "__main__":
  main()
