import sqlite3
from datetime import datetime

import os

import add_column_closes as acc
import add_column_moon_phase as acmp
import add_column_ticker as act
import add_column_ys as acy

def remove_file(file_name):
  """Removes the file with the given name."""
  if os.path.exists(file_name):
    print(f"Removing {file_name}")
    os.remove(file_name)
  else:
    print(f"File '{file_name}' does not exist.")

def create_database(file_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(file_name)
    cursor = conn.cursor()

    try:
        # Create the table if it doesn't exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS my_table (
        id INTEGER PRIMARY KEY,
        datetime_column TEXT)''')
        
        # Add a new column for sorting by datetime
        #cursor.execute('''ALTER TABLE my_table ADD COLUMN datetime_sort INTEGER''')
        
        # Update the new column with the sorted values
        #cursor.execute('''UPDATE my_table SET datetime_column = strftime('%Y%m%d%H%M%S', datetime_column)''')
        
        # Create an index on the datetime_sort column
        #cursor.execute('''CREATE INDEX IF NOT EXISTS idx_datetime_sort ON my_table (datetime_sort)''')

    except sqlite3.Error as e:
        print("Error occurred while executing",e)
    # Commit the changes and close the database connection
    conn.commit()
    conn.close()

#print("... continuing")

# Get the current date and time
#current_datetime = datetime.now()

# Convert the datetime to a string representation
#datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

# Insert the record into the database
#cursor.execute('INSERT INTO my_table (datetime_column) VALUES (?)', (datetime_str,))

if __name__ == "__main__":
    file_name = "database.db"
    remove_file(file_name)
    create_database(file_name)
    act.add_column_ticker()
    acmp.add_column_moon_phase()
    acy.add_column_ys()
    acc.add_column_closes()

print("Done")    
