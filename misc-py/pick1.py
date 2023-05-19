import pickle

class NewClass:
    def __init__(self, data):
        print(data)
        self.data = data
    def foo(self):
        print("Hello World")

# Create an object of NewClass
new_class = NewClass([[1,2,3],[5,6,7]])
print(dir(new_class))
print(type(new_class))

# Serialize and deserialize
pickled_data = pickle.dumps(new_class)
reconstructed = pickle.loads(pickled_data)

# Verify
print("Data from reconstructed object:", reconstructed.data)
reconstructed.foo()
