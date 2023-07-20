import sys,os

case1 = False

if case1:
    try:
        # Code that you are trying to execute
        print(10/0)
    except Exception as e:
        # Code that you want to run if an error occurs
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, f"line = {exc_tb.tb_lineno}")

# Bard's cut
try:
    print(10 / 0)
except ZeroDivisionError as e:
    print(e.__str__())
    print(e.args)
    print(e.__repr__())
    #print(e.cause)
    #print(e.__traceback__)
