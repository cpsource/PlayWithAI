import torch

class MyModule(torch.nn.Module):

    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, input):
        return input * 2

n = MyModule()
print(n)
print(dir(n))

x = n.forward(torch.tensor([[1,2],[2,3]]))
print(x)

#The forward() method is called during the forward pass of the learning process.# The forward pass is the process of passing the input data through the neural n#etwork to get the output. The forward pass is called for each training example #during the training process.
#
#The forward pass is called in the following steps:
#
#The input data is loaded into the neural network.
#The forward() method is called for each layer in the neural network.
#The output of the last layer is calculated.
#The loss is calculated between the output and the ground truth.
#The backward pass is called to update the weights of the neural network.
#The forward pass is an important part of the learning process. It is responsibl#e for calculating the output of the neural network for a given input. The outpu#t of the forward pass is used to calculate the loss, which is used to update th#e weights of the neural network.
#
#Here are some of the benefits of calling the forward() method during the learni#ng process:
#
#It allows the neural network to learn from the input data.
#It allows the neural network to improve its accuracy over time.
#It allows the neural network to generalize to new data.
#By calling the forward() method during the learning process, the neural network# can learn from the input data and improve its accuracy over time. This allows the neural network to generalize to new data and make accurate predictions.
