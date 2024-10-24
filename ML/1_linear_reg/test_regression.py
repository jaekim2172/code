import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('homeprices.csv')

# Check the structure of the dataset
print(dataset.head())  # Preview the first few rows
print(dataset.shape)   # Check the number of rows and columns

# Assuming the last column is the target (adjust the index if needed)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values  # Use -1 to select the last column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Feature scaling for y (if needed, only if y is continuous and not categorical)
y_train = y_train.reshape(-1, 1)  # Reshape if y is 1D for scaling
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)