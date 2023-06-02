#!/home/pagec/scraper/venv/bin/python3

#TRANSFORMS
#Data does not always come in its final processed form that is required for training machine learning algorithms. We use transforms to perform some manipulation of the data and make it suitable for training.
#
#All TorchVision datasets have two parameters -transform to modify the features and target_transform to modify the labels - that accept callables containing the transformation logic. The torchvision.transforms module offers several commonly-used transforms out of the box.
#
#The FashionMNIST features are in PIL Image format, and the labels are integers. For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors. To make these transformations, we use ToTensor and Lambda.
#
#A one-hot encoded tensor is a tensor that represents a categorical variable. Each category in the variable is represented by a separate dimension in the tensor, and each element in that dimension is either 0 or 1, indicating whether the category is present or not.
#
#For example, if we have a categorical variable with three categories, "red", "green", and "blue", we can represent it as a one-hot encoded tensor with three dimensions. The first dimension would represent the category "red", the second dimension would represent the category "green", and the third dimension would represent the category "blue". Each element in each dimension would be either 0 or 1, indicating whether the category is present or not.

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

#Lambda Transforms
#Lambda transforms apply any user-defined lambda function. Here, we define a function to turn the integer into a one-hot encoded tensor. It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls scatter_ which assigns a value=1 on the index as given by the label y.
#
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
print(f"target_transform = {target_transform}\n")

print(f"ds = {ds.test_labels}\n")
print(dir(ds))

