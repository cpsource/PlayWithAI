# see also  https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/

'''
Sure. Here is an example of a Jacobian used in AI:

Backpropagation: In neural network training, the Jacobian matrix is used to calculate the gradients of the loss function with respect to the network's parameters. This is done by recursively applying the chain rule, which essentially breaks down the gradient calculation into a series of partial derivatives. The Jacobian matrix is then used to update the network's parameters using a gradient descent algorithm.
Here is a reference for the Jacobian used in AI:

A Gentle Introduction to the Jacobian: https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/
This article provides a detailed explanation of the Jacobian matrix and its applications in machine learning. It also includes a section on backpropagation, which is one of the most common applications of the Jacobian in AI.
'''

import torch
import torch.nn.functional as F
from functools import partial
_ = torch.manual_seed(0)

def predict(weight, bias, x):
    return F.linear(x, weight, bias).tanh()

D = 16
weight = torch.randn(D, D)
bias = torch.randn(D)
x = torch.randn(D)  # feature vector

def compute_jac(xp):
    jacobian_rows = [torch.autograd.grad(predict(weight, bias, xp), xp, vec)[0]
                     for vec in unit_vectors]
    return torch.stack(jacobian_rows)

xp = x.clone().requires_grad_()
unit_vectors = torch.eye(D)

jacobian = compute_jac(xp)

print(jacobian.shape)
print(jacobian[0])  # show first row

print("Second Try")

from torch.func import vmap, vjp

_, vjp_fn = vjp(partial(predict, weight, bias), x)

ft_jacobian, = vmap(vjp_fn)(unit_vectors)

# let's confirm both methods compute the same result
assert torch.allclose(ft_jacobian, jacobian)
print(ft_jacobian.shape)
print(ft_jacobian[0])

print("Third Try")

from torch.func import jacrev

ft_jacobian = jacrev(predict, argnums=2)(weight, bias, x)

# Confirm by running the following:
assert torch.allclose(ft_jacobian, jacobian)

print("Fourth Try")
def get_perf(first, first_descriptor, second, second_descriptor):
    """takes torch.benchmark objects and compares delta of second vs first."""
    faster = second.times[0]
    slower = first.times[0]
    gain = (slower-faster)/slower
    if gain < 0: gain *=-1
    final_gain = gain*100
    print(f" Performance delta: {final_gain:.4f} percent improvement with {second_descriptor} ")

from torch.utils.benchmark import Timer

without_vmap = Timer(stmt="compute_jac(xp)", globals=globals())
with_vmap = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())

no_vmap_timer = without_vmap.timeit(500)
with_vmap_timer = with_vmap.timeit(500)

print(no_vmap_timer)
print(with_vmap_timer)

print("Fifth Try")

get_perf(no_vmap_timer, "without vmap",  with_vmap_timer, "vmap")

print("Sixth Try")
# note the change in input via ``argnums`` parameters of 0,1 to map to weight and bias
ft_jac_weight, ft_jac_bias = jacrev(predict, argnums=(0, 1))(weight, bias, x)

from torch.func import jacrev, jacfwd

Din = 32
Dout = 2048
weight = torch.randn(Dout, Din)

bias = torch.randn(Dout)
x = torch.randn(Din)

# remember the general rule about taller vs wider... here we have a taller matrix:
print(weight.shape)

using_fwd = Timer(stmt="jacfwd(predict, argnums=2)(weight, bias, x)", globals=globals())
using_bwd = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())

jacfwd_timing = using_fwd.timeit(500)
jacrev_timing = using_bwd.timeit(500)

print(f'jacfwd time: {jacfwd_timing}')
print(f'jacrev time: {jacrev_timing}')

get_perf(jacfwd_timing, "jacfwd", jacrev_timing, "jacrev", );

Din = 2048
Dout = 32
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)

using_fwd = Timer(stmt="jacfwd(predict, argnums=2)(weight, bias, x)", globals=globals())
using_bwd = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())

jacfwd_timing = using_fwd.timeit(500)
jacrev_timing = using_bwd.timeit(500)

print(f'jacfwd time: {jacfwd_timing}')
print(f'jacrev time: {jacrev_timing}')

get_perf(jacrev_timing, "jacrev", jacfwd_timing, "jacfwd")

