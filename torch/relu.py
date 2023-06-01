import torch

#PyTorch supports a wide variety of activation functions, including:
#
#ReLU: The rectified linear unit (ReLU) is a simple but powerful activation function that is commonly used in deep learning models. It works by setting all negative values to 0 and leaving all positive values unchanged.
#Sigmoid: The sigmoid function is a non-linear function that is commonly used in classification problems. It outputs a value between 0 and 1, which can be interpreted as the probability of a sample belonging to a particular class.
#Tanh: The hyperbolic tangent (tanh) function is another non-linear function that is commonly used in classification problems. It outputs a value between -1 and 1, which can be interpreted as the confidence of a sample belonging to a particular class.
#Softmax: The softmax function is a non-linear function that is commonly used in multi-class classification problems. It outputs a vector of probabilities that sum to 1, which can be interpreted as the probability of a sample belonging to each of the possible classes.
#Leaky ReLU: The leaky ReLU is a variant of the ReLU function that allows for a small gradient to flow for negative inputs. This can help to prevent the vanishing gradient problem.
#ELU: The exponential linear unit (ELU) is another variant of the ReLU function that has a negative exponential 'tail' for negative inputs. This can help to prevent the vanishing gradient problem and improve the stability of the model.
#PReLU: The parametric ReLU is a variant of the ReLU function that has a learnable parameter that controls the slope for negative inputs. This can help to improve the performance of the model.

#The choice of activation function depends on the specific task that you are trying to solve. For example, ReLU is a good choice for classification problems, while softmax is a good choice for multi-class classification problems.

#You can experiment with different activation functions to see which one works best for your particular problem.

print("Demonstrate ReLU")

# Create an instance of the nn.ReLU() module
#relu = torch.nn.ReLU()
relu = torch.nn.Sigmoid()

# Create an input tensor
input = torch.randn(1, 10)
print('input tensor',input)

# Forward pass the input tensor through the function
output = relu(input)

# Print the output tensor
print('output tensor',output)

