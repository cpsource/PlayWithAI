import numpy as np

#Softmax is a function that converts a vector of real numbers into a vector of probabilities. The probabilities are normalized so that they sum to 1. This means that the softmax function can be used to represent a probability distribution over a set of possible outcomes.

#The softmax function is often used as the activation function in the output layer of a neural network. The output layer of a neural network is responsible for predicting the class of an input data point. The softmax function ensures that the output of the neural network is a probability distribution, which can then be used to make a prediction.

#For example, suppose we have a neural network that is trained to classify images of cats and dogs. The output layer of the neural network will have two neurons, one for cats and one for dogs. The softmax function will be applied to the output of these neurons, which will result in a probability distribution over the two classes. The neuron with the highest probability will be the class that the neural network predicts.

#The softmax function is a powerful tool that can be used to represent probability distributions in neural networks. It is a versatile function that can be used for a variety of tasks, such as classification, regression, and ranking.

# Note: Preserves the number of columns, but only one row is returned

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

print()
catdog = np.array([ 7, 3, 5, int(8.1) ])
print(f"catdog.size = {catdog.size}\n")
print(f"catdog.max() = {catdog.max()}\n")
print(f"catdog          = {catdog}\n")
print(f"softmax(catdog) = {softmax(catdog)}\n")
print(f"one_hot(catdog) = {one_hot(catdog)}\n")
