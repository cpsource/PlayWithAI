import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        # only backpropagate on the first 5 neurons in the second layer
#        x = x[:, :5]
        x = self.fc2(x)
        return x

model = MyModel()

# set the requires_grad attribute of the first 5 neurons in the second layer to False
if False:
    for i in range(5):
        model.fc2.weight[i].requires_grad = False
        model.fc2.bias[i].requires_grad = False

# forward pass
x = torch.randn(10, 10)
y = model(x)

# backward pass
loss = torch.sum(y**2)
loss.backward()

# check the gradients
for param in model.parameters():
    if param.grad is not None:
        print(param.name, param.grad)
