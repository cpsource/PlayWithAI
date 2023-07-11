def step_through_cursor(cursor):
  """Steps through a cursor one record at a time.

  Args:
    cursor: The cursor to step through.
  """
  while True:
    row = cursor.fetchone()
    if row is None:
      break

    # Do something with the row.
    print(row)


if __name__ == "__main__":
  connection = sqlite3.connect("my_database.db")
  cursor = connection.cursor()

  # Execute a query.
  cursor.execute("SELECT * FROM my_table")

  # Step through the cursor one record at a time.
  step_through_cursor(cursor)

  # Close the connection.
  connection.close()

