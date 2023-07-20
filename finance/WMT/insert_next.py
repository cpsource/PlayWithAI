import sqlite3

def insert_record(data):
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    try:
        # Insert the record into the table
        cursor.execute("INSERT INTO your_table (column1, column2, column3) VALUES (?, ?, ?)", data)

        # Commit the changes
        conn.commit()
        print("Record inserted successfully.")

        # Fetch the next row to advance the cursor
        cursor.fetchone()
    except sqlite3.Error as e:
        print("Error occurred while inserting the record:", e)

    # Close the database connection
    conn.close()

# Example usage
record_data = ('Value 1', 'Value 2', 'Value 3')
insert_record(record_data)
