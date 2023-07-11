import numpy as np

#result = [8.5443e-02, 7.8509e-07, 9.1456e-01]
#result = [0        ,  0         ,          1]

def normalize_ys(result):
    # show starting point
    #print(result)
    # start somewhere
    max = result[0]
    idx = 0
    # find largest
    for i in range(1,len(result)):
        if result[i] > max:
            max = result[i]
            idx = i
    # currect the list
    for i in range(0, len(result)):
        if i == idx:
            result[i] = 1
        else:
            result[i] = 0
    # done
    return result

if __name__ == "__main__":
    result = [8.5443e-02, 7.8509e-07, 9.1456e-01]
    result1 = normalize_ys(result)
    if result != result1:
        print("routine failed")
    else:
        print("routine passed")        
    print(result)
    
