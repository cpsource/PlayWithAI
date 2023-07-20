import sqlite3

def check_record(conn, date, ticker):
  """
  Returns true if the record with the given date and field exists.

  Args:
    conn: The SQLite connection.
    date: The date of the record.
    field: The field of the record.
  """

  sql = "SELECT * FROM my_table WHERE DATE(datetime_column) = ? AND ticker = ?"
  cursor = conn.cursor()
  cursor.execute(sql, (date, ticker))
  return cursor.fetchone() is not None

if __name__ == "__main__":
  conn = sqlite3.connect("database.db")
  if check_record(conn, "2023-06-28", "AAPL"):
    print("True")
  else:
    print("False")

