import numpy as np
import math
import sys

def set_cnt(array):
    my_cnt = 0
    flag = False
    for item in array:
        if flag:
            my_cnt = int(item)
            break
        if '--cnt' == item:
            flag = True
            continue
    if flag:
        if my_cnt > 0:
            print("Error: cnt can't be greater than 0")
            exit(0)
        print(f"My Cnt set to {my_cnt}")
    return flag, my_cnt

def set_my_col(array):
    res = None
    flag = False
    for item in array:
        if flag:
            res = int(item)
            if res > 6 or res < 1:
                print("--col must be between 1 and 6")
                exit(0)
            break
        if '-c' == item or '--col' == item:
            flag = True
            continue
    if not flag:
        res = 6
        print(f"My Column set to default {res}")
    else:
        print(f"My Column set to {res}")
    return res

# set a depth
#our_depth = [30,our-back,2100,630,189,40]
def get_our_depth(array):
    tmp = None
    res = []
    flag = False
    for item in array:
        if flag:
            tmp = item
            break
        if '-d' == item or '--depth' == item:
            flag = True
            continue

    # now convert string to an int array
    if tmp is not None:
        for number in tmp[1:-1].split(","):
            res.append(int(number))

    print(f"Our Depth is {res}")
    return res

    if False and flag:
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

# get winning numbers
def get_winning_numbers(array):
    winning_numbers = None
    res = []
    flag = False
    for item in array:
        if flag:
            winning_numbers = item
            break
        if '-w' == item or '--win' == item:
            flag = True
            continue
    # now convert string to an int array
    if winning_numbers is not None:
        for number in winning_numbers[1:-1].split(","):
            res.append(int(number))

    if flag:
        if len(res) != 6:
           print(f"Error: invalid winning numbers array {res}")
           res = []
        else:
            print(f"Our Winning Numbers are {res}")
    return res
    

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
    return our_game

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

def is_check_mode(array):
    for item in array:
        if '--check' == item or '-ch' == item:
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
            print("  --col n - set column to n in the range of 1 to 6")
            print("  --game mm/pb - set the game. Defaults to mm")
            print("  --depth N - set depth. Defaults to 100")
            print("  --test - run in test mode (no training)")
            print("  --discount - use discount_array for one-hot")            
            #print("  --skip '[0,...]' - skip these balls as they are impossible")
            print("  --zero - unlink the model befor starting")
            print("  --win [a,b,c,d,e,f] - return winning numbers array")
            print("  --check - run in check mode")
            exit(0)
    return

if __name__ == "__main__":

    n = get_winning_numbers(sys.argv)
