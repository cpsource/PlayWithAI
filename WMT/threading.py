import sys
# Remove the first entry (current working directory) from sys.path
sys.path = sys.path[1:]
# Append the current working directory at the end of sys.path
sys.path.append("")

# Now the current working directory will be searched last

if False:
    # so if we want to load from current directory first on each import, we do
    import sys

    # Add the desired path to sys.path temporarily
    sys.path.insert(0, "/path/to/module_directory")

    # Import the module
    import module_name

    # Remove the temporarily added path from sys.path
    sys.path.pop(0)

# Onward

import threading
import time

# Define a simple function to be executed in a thread
def thread_function(name):
    print(f"Thread '{name}' starting...")
    time.sleep(2)  # Simulate some work being done
    print(f"Thread '{name}' finishing...")

# Create two threads
thread1 = threading.Thread(target=thread_function, args=("Thread 1",))
thread2 = threading.Thread(target=thread_function, args=("Thread 2",))

# Start the threads
thread1.start()
thread2.start()

# Wait for the threads to finish
thread1.join()
thread2.join()

print("All threads have finished.")

