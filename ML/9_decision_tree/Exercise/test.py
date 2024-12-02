import pandas as pd

# Read the Titanic dataset
df = pd.read_csv('titanic.csv')

# Display the first few rows of the dataset
print(df.head())
# Get basic information about the dataset
print("\nDataset Info:")
print(df.info())
# Display information about columns
print("\nColumn Information:")
print(df.columns)
print("\nColumn Data Types:")
print(df.dtypes)
# Display detailed information about the first few rows
print("\nDetailed Head Information:")
print(df.head().to_string())

# Get summary statistics of numeric columns in the head
print("\nSummary Statistics of First Few Rows:")
print(df.head().describe())

# Display memory usage of the head
print("\nMemory Usage of First Few Rows:")
print(df.head().memory_usage(deep=True))

