# where.from - pytorch.org/tutorials/intermediate/jacobians_hessians.html

'''

In AI, the Hessian matrix is a square matrix that contains all the second-order partial derivative information of a scalar function. It is named after German mathematician Carl Friedrich Gauss's student, Ludwig Otto Hesse.

The Hessian matrix can be used to calculate the local curvature of a function, which can be used to determine the stability of a solution to an optimization problem. It can also be used to find the minimum or maximum of a function.

In machine learning, the Hessian matrix is used in a variety of applications, including:

Gradient descent: The Hessian matrix can be used to accelerate the convergence of gradient descent algorithms.
Newton's method: The Hessian matrix can be used to implement Newton's method, which is a more efficient way to find the minimum or maximum of a function than gradient descent.
Bayesian optimization: The Hessian matrix can be used to calculate the uncertainty of a Bayesian optimization model.
Neural network training: The Hessian matrix can be used to regularize neural networks, which can help to prevent overfitting.
The Hessian matrix can be a computationally expensive quantity to compute, but it can be very valuable in machine learning applications.

Here are some of the benefits of using Hessian computation in AI:

Can help to find the minimum or maximum of a function: The Hessian matrix can be used to calculate the local curvature of a function, which can be used to determine the stability of a solution to an optimization problem. It can also be used to find the minimum or maximum of a function.
Can accelerate the convergence of gradient descent algorithms: The Hessian matrix can be used to accelerate the convergence of gradient descent algorithms, which can save time and improve the accuracy of the results.
Can be used to regularize neural networks: The Hessian matrix can be used to regularize neural networks, which can help to prevent overfitting.
Here are some of the challenges of using Hessian computation in AI:

Can be computationally expensive: The Hessian matrix can be a computationally expensive quantity to compute, especially for large datasets.
May not be necessary in all cases: The Hessian matrix may not be necessary in all cases, and it may not always be worth the computational cost.
Can be difficult to interpret: The Hessian matrix can be difficult to interpret, especially for high-dimensional problems.
Overall, Hessian computation can be a valuable tool in AI, but it is important to weigh the benefits and challenges before using it.
'''

'''
We offer a convenience API to compute hessians: torch.func.hessiani. Hessians are the jacobian of the jacobian (or the partial derivative of the partial derivative, aka second order).

This suggests that one can just compose functorch jacobian transforms to compute the Hessian. Indeed, under the hood, hessian(f) is simply jacfwd(jacrev(f)).

Note: to boost performance: depending on your model, you may also want to use jacfwd(jacfwd(f)) or jacrev(jacrev(f)) instead to compute hessians leveraging the rule of thumb above regarding wider vs taller matrices.
'''

import torch
from torch.func import hessian

import torch.nn.functional as F
from functools import partial
_ = torch.manual_seed(0)

from torch.func import jacrev, jacfwd

def predict(weight, bias, x):
    return F.linear(x, weight, bias).tanh()

# lets reduce the size in order not to overwhelm Colab. Hessians require
# significant memory:
Din = 512
Dout = 32
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)

hess_api = hessian(predict, argnums=2)(weight, bias, x)
hess_fwdfwd = jacfwd(jacfwd(predict, argnums=2), argnums=2)(weight, bias, x)
hess_revrev = jacrev(jacrev(predict, argnums=2), argnums=2)(weight, bias, x)

torch.allclose(hess_api, hess_fwdfwd)

