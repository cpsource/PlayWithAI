class bmoc:
    def __init__(self,message=None):
        if message is not None:
            print("bmoc init", message)
# Note, sinch bmoc is a super class, Arg will contain self.msg
            self.msg = message
        else:
            self.msg = "no message given"

class Arg(bmoc):
    def __init__(self, message=None):
        super().__init__(message)
#        self.msg = message

# define what is returned on a call to
#  a = Arg('cat')
#  print(a)
# I think it is short for 'represent'
#
    def __repr__(self):
        return f'<Arg message={self.msg}>'

a = Arg("Cat")
print(a)

b = Arg()
print(b)

