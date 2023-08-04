import os
import pickle

#
# there are 10 balls per column recorded of the form
#  ball#, prob, tray-prob
#
sums = [[[ 1, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]],
[[ 2, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]],
[[ 3, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]],
[[ 4, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]],
[[ 5, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]]]

def init_sums(game):
    global sums
    sums_file_name = f"sums_{game}.pkl"
    if os.path.exists(sums_file_name):
        print(f"Reading {sums_file_name}")
        with open(sums_file_name, "rb") as f:
            sums = pickle.load(f)
    else:
        print(f"Writing {sums_file_name}")
        with open(sums_file_name, "wb+") as f:
            pickle.dump(sums,f)
    f.close()

def replace_col(game,col,array):
    global sums
    res = []
    
    sums_file_name = f"sums_{game}.pkl"

    for idx,val in enumerate(sums):
        if idx == (col-1):
            res.append(array)
        else:
            res.append(val)

    print(f"Updating {sums_file_name}")
    with open(sums_file_name, "wb+") as f:
            pickle.dump(res,f)
    f.close()
            
if __name__ == "__main__":
    init_sums('mm')
    print(sums)
    tst = [[ 1, 2, 3],[ 4, 5, 6],[ 7, 8, 9],[ 10, 11, 12],[ 13, 14, 15],[ 16, 17, 18],[ 19, 20, 21],[ 22, 23, 24],[ 25, 26, 27],[ 28, 29, 30]]
    
    replace_col('mm',4,tst)

    tst = [[ 3, 2, 3],[ 4, 5, 6],[ 7, 8, 9],[ 30, 33, 32],[ 33, 34, 35],[ 36, 37, 38],[ 39, 20, 23],[ 22, 23, 24],[ 25, 26, 27],[ 28, 29, 30]]

    replace_col('mm',1,tst)
