import numpy as np
import math

# set a depth
our_depth = [100, 34, 12, 1]
def set_our_depth(array):
    global our_depth
    depth_array = [0]*5
    flag = False
    for item in array:
        if flag:
            our_depth[0] = int(item)
            break
        if '-d' == item or '--depth' == item:
            flag = True
            continue
    if flag:
        # calculate rest of our_depth
        our_depth[1] = int(math.ceil(our_depth[0]/3))
        our_depth[2] = int(math.ceil(our_depth[1]/3))
        our_depth[3] = 1

    if False:
        our_depth[0] = 100
        our_depth[1] = 3
        our_depth[2] = 2
        our_depth[3] = 1
    
    print(f"Our Depth is {our_depth}")
    return our_depth
    
# say which game we are playing
our_game = "mm" # or pb, with mm being the default

def set_our_game(array):
    '''
    set our_game - must be one of pb or mm
    '''
    global our_game
    flag = False
    for item in array:
        if flag:
            if not (item == 'mm' or item == 'pb'):
                print("--game must be either mm or pb")
                exit(0)
            our_game = item
            break
        if '-g' == item or '--game' == item:
            flag = True
            continue
    print(f"Our Game is {our_game}")
    return

def is_discount(array):
    for item in array:
        if '-d' == item or '--discount' == item:
            return True, np.arange(1.0, 0.0, -0.01)
    return False, np.array([])

def is_test(array):
    '''
    Return True if we have a command line switch -t or --test
    '''
    for item in array:
        if '--test' == item or '-t' == item:
            return True
    return False

def is_zero(array):
    for item in array:
        if '--zero' == item or '-z' == item:
            return True
    return False
    
def give_help(array):
    '''
    Give help if asked. Exit afterwards.
    '''
    for item in array:
        if '--help' == item or '-h' == item:
            print("Usage:")
            print("  --help - give this help message then exit")
            #print("  --col n - set column to n in the range of 1 to 5")
            print("  --game mm/pb - set the game. Defaults to mm")
            print("  --depth N - set depth. Defaults to 100")
            print("  --test - run in test mode (no training)")
            print("  --discount - use discount_array for one-hot")            
            #print("  --skip '[0,...]' - skip these balls as they are impossible")
            print("  --zero - unlink the model befor starting")
            exit(0)
    return
