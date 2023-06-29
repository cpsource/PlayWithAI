import pickle

def save_data_to_disk(data, filename):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data saved to {filename} successfully.")
    except IOError:
        print("Error: Unable to save data to disk.")

def restore_data_from_disk(filename):
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        print(f"Data restored from {filename} successfully.")
        return data
    except IOError:
        print("Error: Unable to restore data from disk.")

# Example usage:
data = [1, 2, 3, 4, 5]
filename = 'data.pkl'

# Save data to disk
save_data_to_disk(data, filename)

# Restore data from disk
restored_data = restore_data_from_disk(filename)

print("Restored data:", restored_data)

