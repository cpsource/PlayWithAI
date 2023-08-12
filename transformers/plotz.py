# plot indicated axis

import sys

# Remove the first entry (current working directory) from sys.path
#sys.path = sys.path[1:]
# Append the current working directory at the end of sys.path
#sys.path.append("")

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

#import numpy as np
import matplotlib.pyplot as plt
import pickle

# build an array from a particular column. If
# column is 0, just use the index
def extract_col(results,n):
    res = []
    if n == 0:
        i = 1
        for run in results:
            res.append(i)
            i += 1
    else:
        for run in results:
            res.append(run[n-1])
    return res

if __name__ == "__main__":
    game = 'mm'
    results = None

    x = []
    y = []

    # process command line
    #cmd.give_help(sys.argv)

    # load database and show
    pfn = f"shell-{game}.pkl" # Pickle File Name
    print(f"Reading results from {pfn}")
    with open(pfn, "rb") as f:
        results = pickle.load(f)
    f.close()
    # show
    #print(results)

    col_x = 1
    col_y = 5

    x = extract_col(results,col_x)
    y = extract_col(results,col_y)
    title = "Dist vs Game"
    x_axis = "Distance"
    y_axis = "Game"
    
    # Plot the scatter plot
    plt.scatter(x, y)

    # Add a title
    plt.title(title)

    # Add labels to the x-axis and y-axis
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    # Show the plot
    plt.show()
    
