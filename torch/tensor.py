# where.from - https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

import torch
import numpy as np

# create and display a tensor
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(f"torch.tensor = {x_data}\n")

# create tensor from numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"torch.from_numpy {x_np}\n")

# tensors retain properties
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# shape is a tuple of tensor dimensions
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# show the shape of this np array
array = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(array.shape)

# why the trailing comma
list1=[1,2]
list2=[1,2,]
print(list1,list2)

# tensor attributes
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# try slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

print(f"y1 = {y1}\n")
print(f"y2 = {y2}\n")
print(f"y3 = {y3}\n")

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(f"z1 = {z1}\n")
print(f"z2 = {z2}\n")
print(f"z3 = {z3}\n")

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# bridge to NumPy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

# Bard expouses on features vs labels
#In AI, a feature is a piece of data that is used to train a machine learning model. A label is the desired output of the model.
#
#For example, if you are training a model to predict whether an image contains a cat or a dog, the features would be the pixels in the image, and the# label would be either "cat" or "dog".
#
#Features are typically numerical, while labels can be numerical or categorical.
#
#Features are used to train the model, and the model learns to associate certain features with certain labels. Once the model is trained, it can be u#sed to predict labels for new data.
#
#Here are some of the key differences between features and labels:
#
#Features are used to train the model, while labels are the desired output of the model.
#Features are typically numerical, while labels can be numerical or categorical.
#Features are used to describe the data, while labels are used to classify the data.
#It is important to understand the difference between features and labels in order to build effective machine learning models.
#
#Here are some additional details about features and labels:
#
#Features: Features are the inputs to a machine learning model. They are the data that the model uses to learn how to make predictions. Features can #be numerical, categorical, or text data.
#Labels: Labels are the outputs of a machine learning model. They are the desired results that the model is trying to predict. Labels can be numerica#l, categorical, or text data.
#It is important to choose features that are relevant to the problem that you are trying to solve. The features should also be representative of the #data that you are using to train the model. If the features are not relevant or representative, the model may not be able to learn how to make accur#ate predictions.
#
#It is also important to choose labels that are accurate and consistent. The labels should be the same for each data point. If the labels are not acc#urate or consistent, the model may not be able to learn how to make accurate predictions.
#
#Features and labels are two of the most important concepts in machine learning. By understanding the difference between features and labels, you can# build more effective machine learning models.
