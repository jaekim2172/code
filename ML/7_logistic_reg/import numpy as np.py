import numpy as np
import matplotlib.pyplot as plt

# Define x values for each function within their valid domains
x1 = np.linspace(0.01, 1, 100)  # For -log(x), avoid 0 to prevent infinity
x2 = np.linspace(0, 0.99, 100)  # For -log(1-x), avoid 1 to prevent infinity

# Calculate y values
y1 = -np.log(x1)
y2 = -np.log(1 - x2)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x1, y1, label=r'$-log(x)$', color='blue')
plt.plot(x2, y2, label=r'$-log(1-x)$', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Plots of $-log(x)$ and $-log(1 - x)$')
plt.legend()
plt.grid(True)
plt.show()
