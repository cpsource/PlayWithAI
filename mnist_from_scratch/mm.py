import numpy as np
#import pandas as pd
#import tensorflow as tf
#import math

#tf.random.set_seed(seed=42)

# define to 1 to trace
do_print = 0

# define the learning rate
alpha = learning_rate = 0.5

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    tmp = sigmoid(z)
    return tmp * (1.0-tmp)

def row_col(z):
    shape = z.shape
    #print("type of shape = ",type(shape))
    #print("two eles      = ",shape[0],shape[-1])
    return shape[0], shape[-1]

num_data = np.array([[0.5, 0.1], [0.3, 0.2], [0.7, 0.9],[0.8, 0.1]])
#(4, 2)
# array([[0.5, 0.1],
#        [0.3, 0.2],
#        [0.7, 0.9],
#        [0.8, 0.1]])

if do_print:
    rows, cols = row_col(num_data)
    print(f"num_data = {num_data}, rows = {rows}, cols = {cols}\n")

cat_data = np.array([[0], [1], [2], [0]])
#(4, 1)
if do_print:
    rows, cols = row_col(cat_data)
    print(f"cat_data = {cat_data}, rows = {rows}, cols = {cols}\n")

# X_hot is the four trials. We have 2 neurons, and three dummies which we use for the hot values of y
X_hot = np.array([[0.5, 0.1,1,0,0], [0.3, 0.2,0,1,0], [0.7, 0.9,0,0,1],[0.8, 0.1, 1,0,0]])
if do_print:
    rows, cols = row_col(X_hot)
    print(f"X_hot =\n{X_hot}, rows = {rows}, cols = {cols}\n")
    
one_hot_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
#(4, 3)
if do_print:
    rows, cols = row_col(one_hot_cat)
    print(f"one_hot_cat = {one_hot_cat}, rows = {rows}, cols = {cols}\n")

target = np.array([[0.1], [0.6], [0.4], [0.1]])
# (4, 1)
if do_print:
    rows, cols = row_col(target)
    print(f"target = {target}, rows = {rows}, cols = {cols}\n")

y = np.array([[.1], [.6], [.4], [.1]])
# (1,4)
if do_print:
    rows, cols = row_col(y)
    print(f"y = {y}, rows = {rows}, cols = {cols}\n")

#First_Layer/Hidden_layer_weights
W_fc1 = np.array([[0.19, 0.55, 0.76],[0.33, 0.16, 0.97],[0.4 , 0.35, 0.7 ],[0.51, 0.85, 0.85],[0.54, 0.49, 0.57]])
if do_print:
    rows,cols = row_col(W_fc1)
    print(f"W_fc1 =\n{W_fc1}, rows = {rows}, cols = {cols}\n")

#First_Layer/Biases
b_fc1 = np.array([[0.1], [0.1], [0.1]])
if do_print:
    rows,cols = row_col(b_fc1)
    print(f"b_fc1 =\n{b_fc1}, row = {rows}, cols = {cols}\n")

#
# Second Layer
#
#Output_Layer/Output_layer_weights
W_fc2 = np.array([[ 0.10],[ 0.03],[-0.17]])
if 1 or do_print:
    rows,cols = row_col(W_fc2)
    print(f"W_fc2 =\n{W_fc2}, row = {rows}, cols = {cols}\n")

b_fc2 = np.array([[0.1]])
if do_print:
    rows,cols = row_col(b_fc2)
    print(f"b_fc2 =\n{b_fc2}, row = {rows}, cols = {cols}\n")

#
# Forward Pass
#

# h1
trial = 0
h1_idx = 0
net_h1 = 0.0
for x in [0,1,2,3,4]:
    net_h1 += X_hot[trial,x] * W_fc1[x,h1_idx]
net_h1 += 1 * b_fc1[h1_idx]
out_h1 = sigmoid(net_h1)
print(net_h1,out_h1)

# h2
trial = 0
h2_idx = 1
net_h2 = 0.0
for x in [0,1,2,3,4]:
    #print(f"x = {x}, X_hot = {X_hot[trial,x]}, W_fcl = {W_fc1[x,h2_idx]}\n")
    net_h2 += X_hot[trial,x] * W_fc1[x,h2_idx]
net_h2 += 1 * b_fc1[h2_idx]
out_h2 = sigmoid(net_h2)
print(net_h2,out_h2)

# h3
trial = 0
h3_idx = 2
net_h3 = 0.0
for x in [0,1,2,3,4]:
    #print(f"x = {x}, X_hot = {X_hot[trial,x]}, W_fcl = {W_fc1[x,h3_idx]}\n")
    net_h3 += X_hot[trial,x] * W_fc1[x,h3_idx]
net_h3 += 1 * b_fc1[h3_idx]
out_h3 = sigmoid(net_h3)
print(net_h3,out_h3)

#
# Output
#

output = out_h1*W_fc2[0] + out_h2*W_fc2[1] + out_h3*W_fc2[2] + 1*b_fc2[0]
print(f"output = {output}\n")

#
# Now calculate total error by root mean square
#
error = (y[trial] - output)**2
print(f"error = {error}\n")

# Good ! We agree with the presentation

#
# The Backward Pass
#

# first, lets see how we should adjust b_fc2[0]
#
# We calculate the negative gradient of the function using the chain rule
#
#
# the chain rule says
# partial(error)/partial(b2) = partial(error)/partial(output) TIMES partial(output)/partial(b2)
#
# since error = (y - output)**2
#
# partial(error)/partial(output) = -2*(y - output)
#
# Note: The minus comes as we need to move in the negative of the gradient
#
# and
#
# partial(output)/partial(b2) = 1
#
b2_adjust = -2.0 * (y[trial] - output)
print(f"b2_adjust = {b2_adjust}\n")
b2_plus = b_fc2[0] - alpha * b2_adjust
print(f"b2_plus (b2 becomes this on next iteration) = {b2_plus}\n")

# do we continuously update the model as we go, or wait for the end?

derror_d16 = out_h1 * b2_adjust
o16p = W_fc2[0] - alpha*derror_d16

derror_d17 = out_h2 * b2_adjust
o17p = W_fc2[1] - alpha*derror_d17

derror_d18 = out_h3 * b2_adjust
o18p = W_fc2[2] - alpha*derror_d18

print(derror_d16,o16p)
print(derror_d17,o17p)
print(derror_d18,o18p)

# Note oXp are the new o16,o17,o18

# Now, From Output Layer Neuron to Bias of Hidden Layer

