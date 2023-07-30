import numpy as np
import matplotlib.pyplot as plt

# Create two arrays
x = np.random.randint(0, 100, 100)
y = np.random.randint(0, 100, 100)
print(x)

# Plot the scatter plot
plt.scatter(x, y)

# Add a title
plt.title("Scatter Plot")

# Add labels to the x-axis and y-axis
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")

# Show the plot
plt.show()
