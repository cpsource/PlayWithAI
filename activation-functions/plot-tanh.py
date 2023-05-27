import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values
x = np.linspace(-10, 10, 100)

# Calculate the tanh values
y = np.tanh(x)

# Plot the tanh function
plt.plot(x, y)

# Set the title and labels
plt.title("Tanh Function")
plt.xlabel("x")
plt.ylabel("y")

# Show the plot
plt.show()
