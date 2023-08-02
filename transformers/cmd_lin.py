import numpy as np

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
            print("  --test - run in test mode (no training)")
            print("  --discount - use discount_array for one-hot")            
            #print("  --skip '[0,...]' - skip these balls as they are impossible")
            exit(0)
    return
