import sqlite3
import numpy as np
import pickle
import matplotlib.pyplot as plt
import get_ys
import time

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
  #print(closes_column_index)
  #print(type(results), len(results))
  ticker_name = 'ticker'
  ticker_column_index = get_column_index(cursor,ticker_name)
  
  # Step through the results one at a time and update the name field.
  for row in results:
    id = row[0]
    X = pickle.loads(row[closes_column_index])

    N = len(X)
    n = np.arange(N)
    
    #print(f"id = {id}, X1 = {X1}")
    #update_record(conn, id, "name", "John Doe")

    # Print some info
    min = np.min(X)
    max = np.max(X)
    spread = max-min
    print(f"ticker = {row[ticker_column_index]}, min = {min:.2f}, max = {max:.2f}, spread = {spread:.2f}")
    
    #
    # Plot
    #
    plt.figure(figsize = (12, 6))

    if False:
      plt.subplot(121)
      plt.stem(n,X[2], 'b', \
               markerfmt=" ", basefmt="-b")
      plt.xlabel('Freq (Hz)')
      plt.ylabel('FFT Amplitude |X(freq)|')

    if True:
      #plt.subplot(122)
      plt.plot(n, X, 'r')
      plt.xlabel('Minuite')
      plt.ylabel('Price')
      plt.tight_layout()
      
      plt.show(block=False)
      plt.pause(5)
      plt.close()
      
    # ask
    result = get_ys.get_input(id)

    # update record
    cursor.execute("UPDATE my_table SET y1 = ?, y2 = ?, y3 = ? WHERE id = ?",
                   result)
    conn.commit()

  conn.close()

if __name__ == "__main__":
  main()
