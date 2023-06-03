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

# our feature is between 0 and 9 in the PIL database
# so let's convert it to one-hot
print('one-hot encoded tensors')
for i in [0,1,2,3,4,5,6,7,8,9]:
    print(i,' = ',target_transform(i))

# comment this out as there are a zillion methods in a tensor    
#print(dir(target_transform(0)))


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
