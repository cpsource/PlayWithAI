# where.from - https://mmuratarat.github.io/2020-01-09/backpropagation
#
# Note: We had to add .compat.v1. all over the place among
# other things as this source is out-of-date. However, the
# text that goes along with this is pretty decent.
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#Ignore the warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(seed=42)

import math

num_data = np.array([[0.5, 0.1], [0.3, 0.2], [0.7, 0.9],[0.8, 0.1]])
#(4, 2)
# array([[0.5, 0.1],
#        [0.3, 0.2],
#        [0.7, 0.9],
#        [0.8, 0.1]])

cat_data = np.array([[0], [1], [2], [0]])
#(4, 1)

one_hot_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
#(4, 3)

target = np.array([[0.1], [0.6], [0.4], [0.1]])
# (4, 1)

#def get_trainable_variables(graph=tf.get_default_graph()):
#    return [v for v in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
def get_trainable_variables(graph=tf.compat.v1.get_default_graph()):
    return [v for v in graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)]

def miniBatch(x, y, batchSize):
    numObs  = x.shape[0]
    batches = [] 
    batchNum = math.floor(numObs / batchSize)
    
    if numObs % batchSize == 0:
        for i in range(batchNum):
            xBatch = x[i * batchSize:(i + 1) * batchSize, :]
            yBatch = y[i * batchSize:(i + 1) * batchSize, :]
            batches.append((xBatch, yBatch))
    else:
        for i in range(batchNum):
            xBatch = x[i * batchSize:(i + 1) * batchSize, :]
            yBatch = y[i * batchSize:(i + 1) * batchSize, :]
            batches.append((xBatch, yBatch))
        xBatch = x[batchNum * batchSize:, :]
        yBatch = y[batchNum * batchSize:, :]
        batches.append((xBatch, yBatch))
    return batches

data = np.concatenate((num_data, one_hot_cat), axis = 1)
#(4, 5)

n_features = data.shape[1]
n_outputs = target.shape[1]
n_hidden = 3

tf.compat.v1.reset_default_graph()

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('Placeholders'):
        X = tf.compat.v1.placeholder('float', shape=[None, n_features])
        #<tf.Tensor 'Placeholder:0' shape=(?, 5) dtype=float32>
        y = tf.compat.v1.placeholder('float', shape=[None, n_outputs])
        #<tf.Tensor 'Placeholder_1:0' shape=(?, 1) dtype=float32>

    with tf.name_scope("First_Layer"):
        W_fc1 = tf.compat.v1.get_variable('First_Layer/Hidden_layer_weights', initializer=tf.constant(np.array([[0.19, 0.55, 0.76],[0.33, 0.16, 0.97],[0.4 , 0.35, 0.7 ],[0.51, 0.85, 0.85],[0.54, 0.49, 0.57]]), dtype=tf.float32))
        #<tf.Variable 'First_Layer/Variable:0' shape=(5, 3) dtype=float32_ref>
        b_fc1 = tf.compat.v1.get_variable('First_Layer/Biases', initializer=tf.constant(np.array([0.1, 0.1, 0.1]), dtype=tf.float32))
        #<tf.Variable 'First_Layer/Variable_1:0' shape=(3,) dtype=float32_ref>
        h_fc1 = tf.nn.sigmoid(tf.matmul(X, W_fc1) + b_fc1)
        #<tf.Tensor 'First_Layer/Relu:0' shape=(?, 3) dtype=float32>

    with tf.name_scope("Output_Layer"):
        W_fc2 = tf.compat.v1.get_variable('Output_Layer/Output_layer_weights', initializer=tf.constant(np.array([[ 0.10],[ 0.03],[-0.17]]), dtype=tf.float32))
        # <tf.Variable 'Output_Layer/Variable:0' shape=(3, 1) dtype=float32_ref>
        b_fc2 = tf.compat.v1.get_variable('Output_Layer/Biases', initializer=tf.constant(np.array([0.1]), dtype=tf.float32))
        # <tf.Variable 'Output_Layer/Variable_1:0' shape=(1,) dtype=float32_ref>
        y_pred = tf.cast(tf.matmul(h_fc1, W_fc2) + b_fc2, dtype = tf.float32)
        #<tf.Tensor 'Output_Layer/add:0' shape=(?, 1) dtype=float32>

    with tf.name_scope("Loss"):
        loss = tf.compat.v1.losses.mean_squared_error(labels = y, predictions = y_pred)

    with tf.name_scope('Train'):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
        grads_and_vars = optimizer.compute_gradients(loss)
        trainer = optimizer.apply_gradients(grads_and_vars)

    # [(<tf.Tensor 'Train/gradients/First_Layer/MatMul_grad/tuple/control_dependency_1:0' shape=(5, 3) dtype=float32>,
    #   <tf.Variable 'First_Layer/Variable:0' shape=(5, 3) dtype=float32_ref>),
    #  (<tf.Tensor 'Train/gradients/First_Layer/add_grad/tuple/control_dependency_1:0' shape=(3,) dtype=float32>,
    #   <tf.Variable 'First_Layer/Variable_1:0' shape=(3,) dtype=float32_ref>),
    #  (<tf.Tensor 'Train/gradients/Output_Layer/MatMul_grad/tuple/control_dependency_1:0' shape=(3, 1) dtype=float32>,
    #   <tf.Variable 'Output_Layer/Variable:0' shape=(3, 1) dtype=float32_ref>),
    #  (<tf.Tensor 'Train/gradients/Output_Layer/add_grad/tuple/control_dependency_1:0' shape=(1,) dtype=float32>,
    #   <tf.Variable 'Output_Layer/Variable_1:0' shape=(1,) dtype=float32_ref>)]

    with tf.name_scope("Init"):
        global_variables_init = tf.compat.v1.global_variables_initializer()
        
get_trainable_variables(graph=graph)
# [<tf.Variable 'First_Layer/Hidden_layer_weights:0' shape=(5, 3) dtype=float32_ref>,
#  <tf.Variable 'First_Layer/Biases:0' shape=(3,) dtype=float32_ref>,
#  <tf.Variable 'Output_Layer/Output_layer_weights:0' shape=(3, 1) dtype=float32_ref>,
#  <tf.Variable 'Output_Layer/Biases:0' shape=(1,) dtype=float32_ref>]

with tf.compat.v1.Session(graph=graph) as sess:
    global_variables_init.run()
    tf.compat.v1.get_default_graph().finalize()
    print("Initialized")
    
    print ("Variables before training")
    old_var = {}
    for var in tf.compat.v1.global_variables():
        old_var[var.name] = sess.run(var)
        #print (var.name, sess.run(var))
    print(old_var)
    print('\n\n')
    
    miniBatches = miniBatch(data, target, batchSize = 1)
    total_batch = len(miniBatches) 
    i=1
    for batch in miniBatches:
        print('\n{}-observation\n'.format(i))
        xBatch = batch[0]
        yBatch = batch[1]
        _, loss_val, h_fc1_val, grads_and_vars_val, y_pred_val = sess.run([trainer, loss, h_fc1, grads_and_vars, y_pred], feed_dict={X: xBatch, y: yBatch})
        print('Loss: {}'.format(loss_val))
        print('Prediction: {}'.format(y_pred_val))
        print('Hidden layer forward prop:{}'.format(h_fc1_val))
        print('\n\n')
        print(grads_and_vars_val)
        i += 1
    print("Optimization Finished!")   
    print('\n\n')
    print ("Variables after training")
    new_var = {}
    for var in tf.compat.v1.global_variables():
        new_var[var.name] = sess.run(var)
    print(new_var)
        
# Initialized
# Variables before training
# {'First_Layer/Hidden_layer_weights:0': array([[0.19, 0.55, 0.76],
#        [0.33, 0.16, 0.97],
#        [0.4 , 0.35, 0.7 ],
#        [0.51, 0.85, 0.85],
#        [0.54, 0.49, 0.57]], dtype=float32), 'First_Layer/Biases:0': array([0.1, 0.1, 0.1], dtype=float32), 'Output_Layer/Output_layer_weights:0': array([[ 0.1 ],
#        [ 0.03],
#        [-0.17]], dtype=float32), 'Output_Layer/Biases:0': array([0.1], dtype=float32)}




# 1-observation

# Loss: 0.002247666008770466
# Prediction: [[0.05259044]]
# Hidden layer forward prop:[[0.65203583 0.6772145  0.78193873]]



# [(array([[-1.0756523e-03, -3.1090478e-04,  1.3742511e-03],
#        [-2.1513047e-04, -6.2180960e-05,  2.7485023e-04],
#        [-2.1513046e-03, -6.2180957e-04,  2.7485022e-03],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00]], dtype=float32), array([[0.19053783, 0.55015546, 0.75931287],
#        [0.33010757, 0.16003108, 0.9698626 ],
#        [0.40107566, 0.3503109 , 0.69862574],
#        [0.51      , 0.85      , 0.85      ],
#        [0.54      , 0.49      , 0.57      ]], dtype=float32)), (array([-0.0021513 , -0.00062181,  0.0027485 ], dtype=float32), array([0.10107566, 0.10031091, 0.09862575], dtype=float32)), (array([[-0.06182546],
#        [-0.06421288],
#        [-0.07414274]], dtype=float32), array([[ 0.13091274],
#        [ 0.06210644],
#        [-0.13292864]], dtype=float32)), (array([-0.09481911], dtype=float32), array([0.14740956], dtype=float32))]

# 2-observation

# Loss: 0.17892064154148102
# Prediction: [[0.17700991]]
# Hidden layer forward prop:[[0.67573905 0.7590291  0.7974435 ]]



# [(array([[-0.0072801 , -0.00288298,  0.00544937],
#        [-0.0048534 , -0.00192198,  0.00363291],
#        [ 0.        ,  0.        ,  0.        ],
#        [-0.02426698, -0.00960992,  0.01816456],
#        [ 0.        ,  0.        ,  0.        ]], dtype=float32), array([[0.19417787, 0.55159694, 0.75658816],
#        [0.33253425, 0.16099207, 0.9680461 ],
#        [0.40107566, 0.3503109 , 0.69862574],
#        [0.52213347, 0.854805  , 0.84091777],
#        [0.54      , 0.49      , 0.57      ]], dtype=float32)), (array([-0.02426698, -0.00960992,  0.01816456], dtype=float32), array([0.11320915, 0.10511587, 0.08954347], dtype=float32)), (array([[-0.5716619 ],
#        [-0.6421236 ],
#        [-0.67462146]], dtype=float32), array([[0.4167437 ],
#        [0.38316822],
#        [0.20438209]], dtype=float32)), (array([-0.8459802], dtype=float32), array([0.57039964], dtype=float32))]

# 3-observation

# Loss: 0.907796323299408
# Prediction: [[1.3527834]]
# Hidden layer forward prop:[[0.748083   0.7551234  0.88699394]]



# [(array([[0.10476074, 0.09450983, 0.02732672],
#        [0.13469239, 0.12151264, 0.03513435],
#        [0.        , 0.        , 0.        ],
#        [0.        , 0.        , 0.        ],
#        [0.1496582 , 0.13501404, 0.03903817]], dtype=float32), array([[0.1417975 , 0.504342  , 0.7429248 ],
#        [0.26518807, 0.10023575, 0.950479  ],
#        [0.40107566, 0.3503109 , 0.69862574],
#        [0.52213347, 0.854805  , 0.84091777],
#        [0.46517092, 0.42249298, 0.5504809 ]], dtype=float32)), (array([0.1496582 , 0.13501404, 0.03903817], dtype=float32), array([0.03838005, 0.03760885, 0.07002439], dtype=float32)), (array([[1.4255222],
#        [1.4389381],
#        [1.6902263]], dtype=float32), array([[-0.2960174 ],
#        [-0.33630085],
#        [-0.6407311 ]], dtype=float32)), (array([1.9055669], dtype=float32), array([-0.38238382], dtype=float32))]

# 4-observation

# Loss: 2.02787184715271
# Prediction: [[-1.3240336]]
# Hidden layer forward prop:[[0.6409322  0.69027746 0.8112324 ]]



# [(array([[0.15521942, 0.16381918, 0.22355728],
#        [0.01940243, 0.0204774 , 0.02794466],
#        [0.19402426, 0.20477396, 0.2794466 ],
#        [0.        , 0.        , 0.        ],
#        [0.        , 0.        , 0.        ]], dtype=float32), array([[0.06418779, 0.42243242, 0.6311462 ],
#        [0.25548685, 0.08999705, 0.9365066 ],
#        [0.30406353, 0.24792391, 0.55890244],
#        [0.52213347, 0.854805  , 0.84091777],
#        [0.46517092, 0.42249298, 0.5504809 ]], dtype=float32)), (array([0.19402426, 0.20477396, 0.2794466 ], dtype=float32), array([-0.05863208, -0.06477813, -0.06969891], dtype=float32)), (array([[-1.825418 ],
#        [-1.9659567],
#        [-2.3104444]], dtype=float32), array([[0.6166916],
#        [0.6466775],
#        [0.5144911]], dtype=float32)), (array([-2.8480673], dtype=float32), array([1.0416498], dtype=float32))]
# Optimization Finished!



# Variables after training
# {'First_Layer/Hidden_layer_weights:0': array([[0.06418779, 0.42243242, 0.6311462 ],
#        [0.25548685, 0.08999705, 0.9365066 ],
#        [0.30406353, 0.24792391, 0.55890244],
#        [0.52213347, 0.854805  , 0.84091777],
#        [0.46517092, 0.42249298, 0.5504809 ]], dtype=float32), 'First_Layer/Biases:0': array([-0.05863208, -0.06477813, -0.06969891], dtype=float32), 'Output_Layer/Output_layer_weights:0': array([[0.6166916],
#        [0.6466775],
#        [0.5144911]], dtype=float32), 'Output_Layer/Biases:0': array([1.0416498], dtype=float32)}
