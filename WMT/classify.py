import sqlite3
import numpy as np
import pickle

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

  # Step through the results one at a time and update the name field.
  for row in results:
    id = row[0]
    X1 = pickle.loads(row[closes_column_index])

    print(f"id = {id}, X1 = {X1}")
    #update_record(conn, id, "name", "John Doe")

    exit(0)
    
  conn.close()

if __name__ == "__main__":
  main()

