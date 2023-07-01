import sqlite3

# Connect to the database
conn = sqlite3.connect("my_database.sqlite")

# Create a cursor
cur = conn.cursor()

# Position the cursor to write a new record
cur.execute("INSERT INTO my_table (name, age) VALUES ('John Doe', 30)")

# Commit the changes
conn.commit()

# Close the cursor and the connection
cur.close()
conn.close()

