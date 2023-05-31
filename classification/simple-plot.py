import matplotlib.pyplot as plt
import numpy as np

# Define the x and y values
x = np.linspace(0, 10, 100)
y = x ** 2

# Plot the curve
plt.plot(x, y)

# Add a title and labels
plt.title("A Quadratic Curve")
plt.xlabel("x")
plt.ylabel("y")

# Show the plot
plt.show()

