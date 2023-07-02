import sqlite3

def add_column_closes():
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    try:
        # Add the new column to the table
        cursor.execute("ALTER TABLE my_table ADD COLUMN closes BLOB")

        # Commit the changes
        conn.commit()
        print("Column 'closes' added successfully.")
    except sqlite3.Error as e:
        print("Error occurred while adding the column:", e)

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    # Call the subroutine to add the column
    add_column_closes()
