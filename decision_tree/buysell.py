import numpy as np
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn import tree

# transaction cost
xaction_cost = 3.5

# Build the training data

# cost , sell , qty
X_train = np.array([])
# 0 - sell, 1 - hold, 2 - buy
y_train = np.array([])
i = 40
while i <= 60:
    i += 1
    j = 40
    while j <= 60:
        j += 1
        k = 1
        while k <= 10:
            k += 1
            X_train = np.append(X_train,[i,j,k])

            profit= (j - i)*k
            if profit > xaction_cost :
                buysell = 0
            else:
                if profit < 0:
                    buysell = 1
                else:
                    buysell = 2

            y_train = np.append(y_train,buysell)

X_train = X_train.reshape(4410,3)
# Create a decision tree classifier
clf = DecisionTreeClassifier()

print(X_train)
print(y_train)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

##############################################################################
# We can compare the above output to the plot of the decision tree.

tree.plot_tree(clf)
plt.show()

# Predict the target variable for new data
X_test = np.array([[45, 59, 10],[41,41,1],[50,40,10]])
y_pred = clf.predict(X_test)

print('X_train')
print(X_train)
print('y_train')
print(y_train)
print('X_test')
print(X_test)
print('y_pred')
print(y_pred)


