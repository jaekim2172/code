import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Read the data from logi_data.csv
data = pd.read_csv('logi_data.csv')

# Define the features and target variable
X = data[['x_data']]
y = data['y_data']

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Create a range of values for x to plot the logistic regression curve
x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob = model.predict_proba(x_range)[:, 1]

# Plot the data points and logistic regression curve
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(x_range, y_prob, color='red', label='Logistic Regression Curve')
plt.xlabel('x_data')
plt.ylabel('Probability of y_data = 1')
plt.title('Logistic Regression')
plt.legend()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Simulate reading the data (since we can't actually read files here)
data = pd.DataFrame({
    'x_data': [2.8, 3.2, 3.8, 4.3, 5.5, 5.9, 6.2, 6.9, 7.2, 7.8, 8.2, 8.3, 9.3, 9.5, 9.9],
    'y_data': [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1]
})

# Define the features and target variable
X = data[['x_data']]
y = data['y_data']

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Create a range of values for x with an interval of 0.1
x_range = np.arange(X.min()[0], X.max()[0] + 0.1, 0.1).reshape(-1, 1)
y_prob = model.predict_proba(x_range)[:, 1]

# Plot the data points and logistic regression curve
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(x_range, y_prob, color='red', label='Logistic Regression Curve')
plt.xlabel('x_data')
plt.ylabel('Probability of y_data = 1')
plt.title('Logistic Regression with x interval of 0.1')
plt.legend()
plt.show()
