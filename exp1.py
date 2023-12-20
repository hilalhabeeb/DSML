import numpy as np


def input_matrix():
    rows = int(input("enter the no of rows"))
    cols = int(input("enter the number of columns"))
    matrix = []
    print("enter the elements")
    for i in range(rows):
        r = []
        for j in range(cols):
            element = int(input(f"enter the element at row {i + 1},column {j + 1}   ->"))
            r.append(element)
        matrix.append(r)
    return np.array(matrix)


print("Matrix 1:")
matrix1 = input_matrix()

print("Matrix 2:")
matrix2 = input_matrix()

print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)

sum = np.add(matrix1, matrix2)
print("sum of 2 matrix= \n", sum)

sub = np.subtract(matrix1, matrix2)
print("subtraction = \n", sub)

mul = np.multiply(matrix1, matrix2)
print("product = \n", mul)

div = np.divide(matrix1, matrix2)
print("division = \n", div)
