import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def softmax(x):
  """Computes the softmax function on a given input tensor."""

  # Calculate the exponential of each element in the input tensor.
  exp_x = tf.exp(x)

  # Calculate the sum of the exponentials.
  sum_exp_x = tf.reduce_sum(exp_x)

  # Return the softmax of the input tensor.
  return exp_x / sum_exp_x

#m = tf.constant([[1,2,3,4],[5,6,7,8]])
m = [1.1,2.2,3.3,4.4]
print('softmax returns',softmax(m))
