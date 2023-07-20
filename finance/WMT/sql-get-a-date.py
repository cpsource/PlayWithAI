import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Specify the target date
target_date = '2023-06-30 09:18:37'

# Execute the query to retrieve the record for the target date
cursor.execute('SELECT * FROM my_table WHERE DATETIME(datetime_column) = ?', (target_date,))

# Fetch the record
record = cursor.fetchall()

# Print the retrieved record
if record:
    print(record)
else:
    print("No record found for the specified date.")

# Close the database connection
conn.close()

