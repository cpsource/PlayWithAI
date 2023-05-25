# bard did privide this example, but on it's first try, it forgot the numpy import.
# Later on, I asked for an explanation of clf.fit and it came up with
# this comment.
#
#The two arguments to clf.fit are the training data and the target variable. The training data is a NumPy
#array that contains the features of the data. The target variable is a NumPy
#array that contains the labels of the data.
#
#The clf.fit() method uses the training data to build a decision tree model. The model is then used toi
#predict the target variable for new data.
#
#In the example code, the training data is stored in the variable X_train and the target variable is storedi
#in the variable y_train. The clf.fit() method is called with these two var#iables as arguments.
#
#The clf.fit() method returns the decision tree model. The model can then be used to predict
#the target variable for new data.
#
#Here is a more detailed explanation of the two arguments to clf.fit:
#
#X_train: This is a NumPy array that contains the features of the training data. The features are the
#independent variables in the model.
#
#y_train: This is a NumPy array that contains the labels of the training data. The labels are the
#dependent variables in the model.
#
#The clf.fit() method uses the training data to build a decision tree model. The model is then used to
#predict the target variable for new data.
#
#
#A DecisionTreeClassifier decides clf.predict by recursively partitioning the feature space until
#each leaf node contains only data points of a single class. The decision tree is built by repeatedly
#choosing the feature and threshold that best splits the data. The best split is the one that minimizes
#the entropy of the data after the split. Entropy is a measure of how mixed the data is. The lower
#the entropy, the more pure the data is.
#
#Once the decision tree is built, it can be used to predict the class of a new data point by
#starting at the root node and following the branches down to a leaf node. The class of the data
#point is the class of the leaf node.
#
#Here is an example of a decision tree for the Iris dataset:
#
#
#Root
#    |
#    |----Sepal Length < 5.1
#        |
#        |----Petal Length < 3.5
#            |
#            |----Iris-Setosa
#        |
#        |----Petal Length >= 3.5
#            |
#            |----Iris-Versicolor
#    |
#    |----Sepal Length >= 5.1
#        |
#        |----Iris-Virginica
#        
#
#Gini impurity is a measure of how mixed the data is at a given node in a decision tree. It is calculated by taking the sum of the
#probabilities of each class at the node. The lower the Gini impurity, the more pure the data is.
#
#When a decision tree is displayed, the Gini impurity of each node is shown next to the node. This helps you to understand how well
#the tree is able to separate the data into different classes.
#
#For example, if a node has a Gini impurity of 0.5, it means that the data at the node is equally divided between the two classes.
#This means that the tree is not very good at separating the data into different classes.
#
#On the other hand, if a node has a Gini impurity of 0, it means that all of the data at the node belongs to the same class.
#This means that the tree is very good at separating the data into different classes.
#
#Gini impurity is a useful measure for understanding how well a decision tree is able to learn from the data. It can also be used to tune
#the hyperpara#meters of the tree, such as the depth of the tree and the number of features to consider at each node.

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn import tree

# Import the training data
X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_train = np.array([0, 1, 2])

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Fit the classifier to the training data
clf.fit(X_train, y_train)

##############################################################################
# We can compare the above output to the plot of the decision tree.

tree.plot_tree(clf)
plt.show()

# Predict the target variable for new data
X_test = np.array([[10, 11, 12]])
y_pred = clf.predict(X_test)

print('X_train')
print(X_train)
print('y_train')
print(y_train)
print('X_test')
print(X_test)
#print('[10,11,12]')
print('y_pred')
print(y_pred)

print('')
print('second try')
r = list(range(1,11))
for a in r:
    for b in r:
        for c in r:
            question = np.array([[a,b,c]])
            y_pred = clf.predict(question)
            print(a,b,c,question,y_pred)

X_test = np.array([[3,6,6]])
y_pred = clf.predict(X_test)
print(X_test,y_pred)
