import numpy as np

# Function for piecewise linear activation
def piecewise_linear(x):
    return np.maximum(x, 0)

# Creating matrices
X = np.array([[1, 2],
              [3, 4],
              [5, -6],
              [7, 8]])

W1 = np.random.rand(3, 2)  # Adjusted shape to (3, 2)
b1 = np.random.rand(3, 1)
W2 = np.random.rand(5, 3)
b2 = np.random.rand(1, 2)

# Compute Z1 = W1X + b1
Z1 = np.dot(W1, X.T)

# Compute A1 = piecewise_linear(Z1)
A1 = piecewise_linear(Z1)

# Compute Z2 = W2A1 + b2
Z2 = np.dot(W2, A1)

# Compute A2 = piecewise_linear(Z2)
A2 = piecewise_linear(Z2)

# Displaying values
print("Z1:\n", Z1)
print("\nA1:\n", A1)
print("\nZ2:\n", Z2)
print("\nA2:\n", A2)
