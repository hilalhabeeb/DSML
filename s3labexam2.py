from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the diabetes dataset
diabetes = load_diabetes()

# Extract input features and target
X = diabetes.data
y = diabetes.target

# Print input features and target values
print("Input Features (X):")
print(X[:5])  # Printing first 5 rows as an example, you can change the index to view more rows

print("\nTarget (y):")
print(y[:5])  # Printing first 5 target values as an example

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Get parameters of the linear regression model
coefficients = model.coef_
intercept = model.intercept_

print("\nParameters of the Linear Regression model:")
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Input values for prediction
input_values = [0.02, -0.05, 0.1, -0.15, 0.2, -0.25, 0.3, -0.35, 0.4, -0.45]

# Reshape the input values into a 2D array (as the model expects a 2D array)
input_values_reshaped = np.array(input_values).reshape(1, -1)

# Use the model to predict target values for the input values provided
predicted_values = model.predict(input_values_reshaped)

print("\nEstimated target values for the input values:")
print(predicted_values)
