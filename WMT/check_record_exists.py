import sqlite3

def check_record_exists(date_to_check):
    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    try:
        # Execute a SELECT query to check if the record exists
        cursor.execute("SELECT COUNT(*) FROM my_table WHERE datetime_column = ?", (date_to_check,))
        count = cursor.fetchone()[0]

        # Return True if the record exists, False otherwise
        if count > 0:
            return True
        else:
            return False
    except sqlite3.Error as e:
        print("Error occurred while checking the record:", e)

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    # Example usage
    date_to_check = '2023-06-30'
    exists = check_record_exists(date_to_check)
    print(exists)
