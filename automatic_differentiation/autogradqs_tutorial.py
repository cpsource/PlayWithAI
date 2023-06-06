# where.from: https://github.com/pytorch/tutorials/blob/main/beginner_source/basics/autogradqs_tutorial.py
#             https://torchtutorialstaging.z5.web.core.windows.net/beginner/basics/autogradqs_tutorial.html

"""
`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
**Autograd** ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

Automatic Differentiation with ``torch.autograd``
=======================================

When training neural networks, the most frequently used algorithm is
**back propagation**. In this algorithm, parameters (model weights) are
adjusted according to the **gradient** of the loss function with respect
to the given parameter.

To compute those gradients, PyTorch has a built-in differentiation engine
called ``torch.autograd``. It supports automatic computation of gradient for any
computational graph.

Consider the simplest one-layer neural network, with input ``x``,
parameters ``w`` and ``b``, and some loss function. It can be defined in
PyTorch in the following manner:
"""

import torch
import numpy as np

x = torch.ones(5)   # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(f"type = {type(loss)}, loss = {loss}\n")

#
# x is a 1,5
# w is a 5,3
# since columns in first matrix equals rows in second, we can multiply
# and we will get a 1,3 matrix out
#

if 0:
    # lets do a stupid case
    x1 = [1,1,1,1,1]
    y1 = [0,0,0]
    w1 = [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
    z1 = np.matmul(x1,w1)
    print(f"z1 (computed) = {z1}")
    print(f"y1 (actual  ) = {y1}")

    xdet = x.detach() # get rid of requires_grad to be True
    wdet = w.detach() # same as above
    x1 = np.array(xdet) # convert to numpy array
    w1 = np.array(wdet) # same as above

    z1 = np.matmul(x1,w1)
    print(f"z1 (example) = {z1}")

    exit(0)

#The requires_grad argument for torch.randn determines whether the output tensor will be tracked by the autograd engine. If requires_grad is set to True, then the output tensor will be able to backpropagate gradients through it. This means that if you use the output tensor in an operation that computes a gradient, then the gradient will be calculated with respect to the random numbers in the output tensor.

#For example, the following code creates a random tensor with requires_grad set to True. Then, it computes the mean of the tensor and backpropagates the gradient through the mean.

# From Bard
# The w matrix is a 5,3 matrix because the input tensor x is a 5-dimensional vector, and the expected output tensor y is a 3-dimensional vector. The matmul function multiplies two tensors together, and the output tensor will have the same number of dimensions as the first tensor. In this case, the output tensor z will have 5 dimensions, so the w matrix must have 5 dimensions as well.
#
#The w matrix is a random matrix, and the b vector is a random vector. These random values are used to train the model. The model will learn to adjust the values of w and b so that the output tensor z is as close to the expected output tensor y as possible.
#
#The binary_cross_entropy_with_logits function calculates the loss between the output tensor z and the expected output tensor y. The loss function is used to measure how well the model is performing. The model will learn to adjust the values of w and b so that the loss is minimized.
#
#The requires_grad keyword tells PyTorch to track the gradients of the w and b tensors. The gradients are used to update the values of w and b during training.
#

if 0:
    xt = torch.randn(10, requires_grad=True)
    mean = xt.mean()
    mb = mean.backward()
    #mb = None
    print(f"xt      = {xt}\n")
    print(f"mean    = {mean}\n")
    print(f"mean.backward() called\n")
    print(f"xt.grad = {xt.grad}\n")
    print(f"mb      = {mb}\n")

#As you can see, the gradient of the random tensor is a vector of the same size as the tensor, with each element representing the partial derivative of the mean with respect to the corresponding element of the tensor.

#In general, you should only set requires_grad to True for tensors that represent parameters in a neural network. This is because the gradients for these tensors will be used to update the parameters during training. For other tensors, such as tensors that represent input data or intermediate results, you should set requires_grad to False to avoid unnecessary computation.

#The mean.backward() method computes the gradient of the mean tensor with respect to its inputs. This is done by using the chain rule to propagate the gradients from the output of the mean tensor back to its inputs.

#For example, the following code creates a tensor of random numbers and computes its mean. Then, it backpropagates the gradient of the mean through the tensor.

# Onward
if 0:
    print(b)
    print(type(b))
    #print(dir(b))
    exit(0)

#In machine learning, logits are the unnormalized predictions of a model. They are the output of the last layer of a neural network before the softmax function is applied. The softmax function then converts the logits into probabilities.

#Logits are used in classification tasks, where the goal is to predict the class of an input. For example, a model that classifies images of cats and dogs would have two output classes, one for cats and one for dogs. The logits would represent the probability that the input image belongs to each class.

#Logits can also be used in regression tasks, where the goal is to predict a continuous value. For example, a model that predicts the price of a house would have one output value, the predicted price of the house. The logits would represent the log-odds of the house being worth a certain price.

#Logits are a powerful tool for machine learning. They can be used to solve a wide variety of problems, including classification, regression, and ranking.

#Here are some additional details about logits:

#Logits are typically represented as a vector of numbers. The number of elements in the vector is equal to the number of output classes.
#The values of the logits can be positive or negative. A positive value indicates that the input is more likely to belong to the class corresponding to the logit. A negative value indicates that the input is less likely to belong to the class.
#The softmax function is used to convert the logits into probabilities. The softmax function normalizes the logits so that they sum to 1. This ensures that the probabilities represent the relative likelihood of the input belonging to each class.

######################################################################
# Tensors, Functions and Computational graph
# ------------------------------------------
#
# This code defines the following **computational graph**:
#
# .. figure:: /_static/img/basics/comp-graph.png
#    :alt:
#
# In this network, ``w`` and ``b`` are **parameters**, which we need to
# optimize. Thus, we need to be able to compute the gradients of loss
# function with respect to those variables. In order to do that, we set
# the ``requires_grad`` property of those tensors.

#######################################################################
# .. note:: You can set the value of ``requires_grad`` when creating a
#           tensor, or later by using ``x.requires_grad_(True)`` method.

#######################################################################
# A function that we apply to tensors to construct computational graph is
# in fact an object of class ``Function``. This object knows how to
# compute the function in the *forward* direction, and also how to compute
# its derivative during the *backward propagation* step. A reference to
# the backward propagation function is stored in ``grad_fn`` property of a
# tensor. You can find more information of ``Function`` `in the
# documentation <https://pytorch.org/docs/stable/autograd.html#function>`__.
#

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

#exit(0)

######################################################################
# Computing Gradients
# -------------------
#
# To optimize weights of parameters in the neural network, we need to
# compute the derivatives of our loss function with respect to parameters,
# namely, we need :math:`\frac{\partial loss}{\partial w}` and
# :math:`\frac{\partial loss}{\partial b}` under some fixed values of
# ``x`` and ``y``. To compute those derivatives, we call
# ``loss.backward()``, and then retrieve the values from ``w.grad`` and
# ``b.grad``:
#

loss.backward()
print(f"w.grad = {w.grad}\n")
print(f"b.grad = {b.grad}\n")

#exit(0)

######################################################################
# .. note::
#   - We can only obtain the ``grad`` properties for the leaf
#     nodes of the computational graph, which have ``requires_grad`` property
#     set to ``True``. For all other nodes in our graph, gradients will not be
#     available.
#   - We can only perform gradient calculations using
#     ``backward`` once on a given graph, for performance reasons. If we need
#     to do several ``backward`` calls on the same graph, we need to pass
#     ``retain_graph=True`` to the ``backward`` call.
#


######################################################################
# Disabling Gradient Tracking
# ---------------------------
#
# By default, all tensors with ``requires_grad=True`` are tracking their
# computational history and support gradient computation. However, there
# are some cases when we do not need to do that, for example, when we have
# trained the model and just want to apply it to some input data, i.e. we
# only want to do *forward* computations through the network. We can stop
# tracking computations by surrounding our computation code with
# ``torch.no_grad()`` block:
#

z = torch.matmul(x, w)+b
print(f"z.requires_grad) = {z.requires_grad}\n")

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(f"z.requires_grad) = {z.requires_grad}\n")


######################################################################
# Another way to achieve the same result is to use the ``detach()`` method
# on the tensor:
#

z = torch.matmul(x, w)+b
z_det = z.detach()
print(f"z_det.requires_grad) = {z_det.requires_grad}\n")

#exit(0)

######################################################################
# There are reasons you might want to disable gradient tracking:
#   - To mark some parameters in your neural network as **frozen parameters**.
#   - To **speed up computations** when you are only doing forward pass, because computations on tensors that do
#     not track gradients would be more efficient.


######################################################################

######################################################################
# More on Computational Graphs
# ----------------------------
# Conceptually, autograd keeps a record of data (tensors) and all executed
# operations (along with the resulting new tensors) in a directed acyclic
# graph (DAG) consisting of
# `Function <https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function>`__
# objects. In this DAG, leaves are the input tensors, roots are the output
# tensors. By tracing this graph from roots to leaves, you can
# automatically compute the gradients using the chain rule.
#
# In a forward pass, autograd does two things simultaneously:
#
# - run the requested operation to compute a resulting tensor
# - maintain the operation’s *gradient function* in the DAG.
#
# The backward pass kicks off when ``.backward()`` is called on the DAG
# root. ``autograd`` then:
#
# - computes the gradients from each ``.grad_fn``,
# - accumulates them in the respective tensor’s ``.grad`` attribute
# - using the chain rule, propagates all the way to the leaf tensors.
#
# .. note::
#   **DAGs are dynamic in PyTorch**
#   An important thing to note is that the graph is recreated from scratch; after each
#   ``.backward()`` call, autograd starts populating a new graph. This is
#   exactly what allows you to use control flow statements in your model;
#   you can change the shape, size and operations at every iteration if
#   needed.

######################################################################
# Optional Reading: Tensor Gradients and Jacobian Products
# --------------------------------------
#
# In many cases, we have a scalar loss function, and we need to compute
# the gradient with respect to some parameters. However, there are cases
# when the output function is an arbitrary tensor. In this case, PyTorch
# allows you to compute so-called **Jacobian product**, and not the actual
# gradient.
#
# For a vector function :math:`\vec{y}=f(\vec{x})`, where
# :math:`\vec{x}=\langle x_1,\dots,x_n\rangle` and
# :math:`\vec{y}=\langle y_1,\dots,y_m\rangle`, a gradient of
# :math:`\vec{y}` with respect to :math:`\vec{x}` is given by **Jacobian
# matrix**:
#
# .. math::
#
#
#    J=\left(\begin{array}{ccc}
#       \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
#       \vdots & \ddots & \vdots\\
#       \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#       \end{array}\right)
#
# Instead of computing the Jacobian matrix itself, PyTorch allows you to
# compute **Jacobian Product** :math:`v^T\cdot J` for a given input vector
# :math:`v=(v_1 \dots v_m)`. This is achieved by calling ``backward`` with
# :math:`v` as an argument. The size of :math:`v` should be the same as
# the size of the original tensor, with respect to which we want to
# compute the product:
#

print(f"Last Example: Tensor Gradients and Jacobian Products\n")

inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")


######################################################################
# Notice that when we call ``backward`` for the second time with the same
# argument, the value of the gradient is different. This happens because
# when doing ``backward`` propagation, PyTorch **accumulates the
# gradients**, i.e. the value of computed gradients is added to the
# ``grad`` property of all leaf nodes of computational graph. If you want
# to compute the proper gradients, you need to zero out the ``grad``
# property before. In real-life training an *optimizer* helps us to do
# this.

######################################################################
# .. note:: Previously we were calling ``backward()`` function without
#           parameters. This is essentially equivalent to calling
#           ``backward(torch.tensor(1.0))``, which is a useful way to compute the
#           gradients in case of a scalar-valued function, such as loss during
#           neural network training.
#

######################################################################
# --------------
#

#################################################################
# Further Reading
# ~~~~~~~~~~~~~~~~~
# - `Autograd Mechanics <https://pytorch.org/docs/stable/notes/autograd.html>`_
