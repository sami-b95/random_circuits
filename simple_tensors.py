import numpy as np
import tensornetwork as tn


def computational_basis_vector(d_value, i):
    vector = np.zeros(d_value)
    vector[i] = 1
    return tn.Node(vector)

def computational_basis_matrix(d_value, i, j):
    matrix = np.zeros((d_value, d_value))
    matrix[i, j] = 1
    return tn.Node(matrix)