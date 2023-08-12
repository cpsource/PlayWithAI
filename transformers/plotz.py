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
import numpy as np

meaning = [0,
           "distance",
           "ball #",
           "ball # occurances",
           "game offset, 0,-1,-2 etc",
           "width of game 30, etc",
           "depth of game, -100 etc",
           "neural network sizes",
           "neural network sizes",
           "neural network sizes",
           " neural network sizes"]

def add_unique_integer(array, integer):
  """Adds an integer to an array only if that integer is unique.

  Args:
    array: The array to add the integer to.
    integer: The integer to add to the array.

  Returns:
    True if the integer was added to the array, False otherwise.
  """

  if integer in array:
    return False

  array.append(integer)
  return True

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

# build an array from a particular column but only if unique. If
# column is 0, just use the index
def extract_col_unique(results,n):
    res = []
    if n == 0:
        i = 1
        for run in results:
            res.append(i)
            i += 1
    else:
        for run in results:
            add_unique_integer(res,run[n-1])
    return res

# get an index
def get_index(array,n):
    for idx, val in enumerate(array):
        if n == val:
            return idx
    print(f"Error: value {n} not found in array {array}")
    exit(0)
    
if __name__ == "__main__":
    my_col = None
    
    if len(sys.argv) < 5:
        print("Error: missing arguments: ./plotz mm/pb col x y")
        exit(0)
    game   = sys.argv[1]
    my_col = int(sys.argv[2])
    col_x  = int(sys.argv[3])
    col_y  = int(sys.argv[4])

    if not (game == 'mm' or game == 'pb'):
        print("Error: game must be mm or pb")
        exit(0)
    if my_col < 1 or my_col > 6:
        print("Error: column out of range [1..6]")
        exit(0)
        
    results = None

    x = []
    y = []

    # process command line
    #cmd.give_help(sys.argv)

    # load database and show
    pfn = f"shell-{game}-{my_col}.pkl" # Pickle File Name
    print(f"Reading results from {pfn}")
    with open(pfn, "rb") as f:
        results = pickle.load(f)
    f.close()
    # show
    #print(results)

    x = extract_col(results,col_x)
    y = extract_col(results,col_y)
    title = f"Col {col_x} vs {col_y}"
    x_axis = meaning[col_x]
    y_axis = meaning[col_y]
    
    # Plot the scatter plot
    plt.scatter(x, y)

    # Add a title
    plt.title(title)

    # Add labels to the x-axis and y-axis
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    # Show the plot
    #plt.show()

    # check stddev of col 1 againt unique col 5
    tst_col = 5
    col_one = 1
    uniques = extract_col_unique(results,tst_col)
    sums = np.zeros(len(uniques), dtype=np.float32)
    mean = np.zeros(len(uniques), dtype=np.float32)
    ctr  = np.zeros(len(uniques), dtype=np.float32)
    cnt = 0
    for result in results:
        #print(uniques,result)
        idx = get_index(uniques,result[tst_col-1])
        sums[idx] += result[col_one-1]*result[col_one-1]
        mean[idx] += result[col_one-1]
        ctr[idx]  += 1
        #print(f"added {result[col_one-1]}")
        cnt += 1
    sums /= ctr
    mean /= ctr
    sums = np.sqrt(sums)
    x = np.argsort(sums)
    #print(x)
    for i in x:
        print(f"{meaning[tst_col]}: {uniques[i]}, Mean: %.2f, Standard Deviation of Distance: %.2f"
              % (mean[i], sums[i]))
