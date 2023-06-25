import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need

def tanh(x):                 # Define a function
    y = np.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)       # Obtain its gradient function
print(grad_tanh(1.0))               # Evaluate the gradient at x = 1.0

from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
import matplotlib.pyplot as plt
x = np.linspace(-7, 7, 200)
plt.plot(x, tanh(x),
         x, egrad(tanh)(x),                                     # first  derivative
         x, egrad(egrad(tanh))(x),                              # second derivative
         x, egrad(egrad(egrad(tanh)))(x))                       # third  derivative
plt.show()
