# Author: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

#
# Note: Vas ist das? It would be nice to have/fine an explanation of
# what the heck is going on here
#

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets

print(datasets.load_diabetes())

X, y = datasets.load_diabetes(return_X_y=True)

print(X)
print(y)

exit(0)

print("Computing regularization path using the LARS ...")
_, _, coefs = linear_model.lars_path(X, y, method="lasso", verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle="dashed")
plt.xlabel("|coef| / max|coef|")
plt.ylabel("Coefficients")
plt.title("LASSO Path")
plt.axis("tight")
plt.show()
