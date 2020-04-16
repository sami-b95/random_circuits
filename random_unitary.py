import numpy as np


# Code from https://medium.com/@sohaib.alam/unpacking-the-quantum-supremacy-benchmark-with-python-67a46709d
def haar_unitary(dim: int) -> np.ndarray:
    # follows the algorithm in https://arxiv.org/pdf/math-ph/0609050.pdf
    # returns a unitary of size dim x dim
    Z = np.array([np.random.normal(0, 1) + np.random.normal(0, 1) * 1j for _ in range(dim ** 2)]).reshape(dim, dim)
    Q, R = np.linalg.qr(Z)
    diag = np.diagonal(R)
    lamb = np.diag(diag) / np.absolute(diag)
    unitary = np.matmul(Q, lamb)

    # this condition asserts that the matrix is unitary
    assert np.allclose(unitary.conj().T @ unitary, np.eye(dim))

    return unitary
