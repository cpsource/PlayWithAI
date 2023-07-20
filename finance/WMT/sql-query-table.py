#
# You can also do this from the console with sqlite3 as follows:
#  sqlite3 mydb.db
#  .tables
#  .schema table
#  .exit

import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Execute the SQL query to fetch the schema for the "tensors" table
cursor.execute("PRAGMA table_info(tensors)")

# Fetch all the rows containing the schema information
schema_rows = cursor.fetchall()

# Print the schema details
for row in schema_rows:
    column_name = row[1]
    column_type = row[2]
    print(f"database.db:tensors:{column_name}: {column_type}")

# Close the database connection
conn.close()

