import numpy as np
matrix= np.array ([[5,6,4],
                  [2,5,6],
                  [3,5,6]])
U, S, VT = np.linalg.svd(matrix)
print("printing matrix U\n",U)

print("\nS in diagonal format")
print(np.diag(S))

print("\nVT matrix(transpose of right singular vectors)")
print(VT)


print("\n original matrix using svd components")
reconstructed_matrix=np.dot(U, np.dot(np.diag(S), VT))
print(reconstructed_matrix)