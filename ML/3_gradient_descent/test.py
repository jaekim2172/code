import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

def gradient_descent(x, y):
    m_curr = b_curr = 0
    rate = 0.01
    n = len(x)
    plt.scatter(x, y, color='red', marker='+', linewidth=5)
    
    for i in range(10000):
        y_predicted = m_curr * x + b_curr
        plt.plot(x, y_predicted, color='green')  # Plot line for each iteration
        md = -(2/n) * sum(x * (y - y_predicted))  # Slope gradient
        yd = -(2/n) * sum(y - y_predicted)  # Intercept gradient
        m_curr = m_curr - rate * md  # Update slope
        b_curr = b_curr - rate * yd  # Update intercept

# Define x and y arrays
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([5, 7, 9, 11, 13], dtype=float)

# Call the gradient descent function
gradient_descent(x, y)

print(gradient_descent(x,y))