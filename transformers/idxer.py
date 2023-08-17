'''
  Calculate all things idx
'''

idxer_width = None
idxer_top = None
idxer_depth = None
idxer_idx_lowest = None
idxer_my_prev_play = None

def idxer_init(w,t,d,mpp):
    global idxer_width
    global idxer_depth
    global idxer_top
    global idxer_idx_lowest
    global idxer_my_prev_play
    
    idxer_width = w
    idxer_top = t
    idxer_depth = d
    idxer_my_prev_play = mpp
    if idxer_my_prev_play > 0:
        print("Error: my_prev_play must be 0 or lower");
        exit(0)
    idxer_idx_lowest = t + d
    #print(idxer_idx_lowest,w)
    if idxer_idx_lowest < w:
        idxer_idx_lowest = w

def idxer_get_lowest():
    global idxer_idx_lowest

    return idxer_idx_lowest

def idxer_get_top():
    global idxer_top
    global idxer_my_prev_play
    
    return idxer_top + idxer_my_prev_play # actually subtracts

def get_x(ts_array, idx):
    global idxer_top

    ball = None
    x = []

    if idxer_my_prev_play != 0 and idx == idxer_get_top():
        ball = ts_array[idx]
        idx -= 1
    else:
        if idx == idxer_top:
            # run top-1 through model. We don't know the last ball, so just display results
            ball = 0
            idx -= 1
        else:
            ball = ts_array[idx]
            idx -= 1

    # get x

    #print(f"top = {top}, len(ts_array) = {len(ts_array)}")
    x = []
    for j in range(idx-idxer_width+1, idx+1):
        x.append(ts_array[j])

    # return
    return ball, x

if __name__ == "__main__":
    # do some testing
    ts_array = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    width = 5
    top = len(ts_array)
    depth = -150
    x = None
    
    # init
    idxer_init(width,top,depth)

    # our ts_array
    print(ts_array)
    
    # get top
    x = get_x(ts_array,top)
    print(x)

    # get top -1
    x = get_x(ts_array,top-1)
    print(x)

    # get lowest
    x = get_x(ts_array,idxer_get_lowest())
    print(x)
    
