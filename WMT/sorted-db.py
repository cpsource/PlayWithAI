import sqlite3
from datetime import datetime

# Connect to the SQLite database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

try:
    # Create the table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS my_table (
                    id INTEGER PRIMARY KEY,
                    datetime_column TEXT)''')

    # Add a new column for sorting by datetime
    cursor.execute('''ALTER TABLE my_table
                    ADD COLUMN datetime_sort INTEGER''')

    # Update the new column with the sorted values
    cursor.execute('''UPDATE my_table
                    SET datetime_sort = strftime('%Y%m%d%H%M%S', datetime_column)''')

    # Create an index on the datetime_sort column
    cursor.execute('''CREATE INDEX IF NOT EXISTS idx_datetime_sort
                    ON my_table (datetime_sort)''')

except sqlite3.Error as e:
    print("Error occurred while executing",e)

print("... continuing")

# Get the current date and time
current_datetime = datetime.now()

# Convert the datetime to a string representation
datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

# Insert the record into the database
cursor.execute('INSERT INTO my_table (datetime_column) VALUES (?)', (datetime_str,))

# Commit the changes and close the database connection
conn.commit()
conn.close()

