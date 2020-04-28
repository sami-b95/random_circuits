from itertools import product
import numpy as np
import random
from qiskit import QuantumCircuit, QuantumRegister


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

def pseudorandom_1d_local_circuit_helper(qubits, length):
    qc = QuantumCircuit(*set(qubit.register for qubit in qubits))
    for _ in range(length):
        # Pick a random pair of neighbour qubits
        #i, j = random.sample(range(len(qubits)), 2)
        i = np.random.randint(len(qubits) - 1)
        # Apply a random unitary to it
        qc.unitary(haar_unitary(4), [qubits[i], qubits[i + 1]])
    return qc

def pseudorandom_1d_parallel_circuit_helper(qubits, length):
    qc = QuantumCircuit(*set(qubit.register for qubit in qubits))
    for step in range(length):
        for qubit_index in range(step % 2, len(qubits) - 1, 2):
            qc.unitary(haar_unitary(4), [qubits[qubit_index], qubits[qubit_index + 1]])
    return qc

def pseudorandom_1d_local_circuit(n_qubits, length):
    qr = QuantumRegister(n_qubits)
    return pseudorandom_1d_local_circuit_helper(qr[:], length)

def pseudorandom_1d_parallel_circuit(n_qubits, length):
    qr = QuantumRegister(n_qubits)
    return pseudorandom_1d_parallel_circuit_helper(qr[:], length)

def pseudorandom_local_circuit_helper(qubits, c, s):
    if qubits.ndim == 1:
        return pseudorandom_1d_local_circuit_helper(qubits, s)
    qc = QuantumCircuit(*set(qubit.register for qubit in qubits.flatten()))
    for _ in range(c):
        for i in range(qubits.shape[0]):
            qc += pseudorandom_local_circuit_helper(qubits[i], c, s)
        for j in product(range(qubits.shape[0]), repeat=(qubits.ndim - 1)):
            qc += pseudorandom_1d_local_circuit_helper(qubits[j], s)
    for i in range(qubits.shape[0]):
        qc += pseudorandom_local_circuit_helper(qubits[i], c, s)
    return qc

def pseudorandom_local_circuit(D, n_qubits_per_dimension, c, s):
    qr = QuantumRegister(n_qubits_per_dimension ** D)
    qubits = np.reshape(qr[:], (n_qubits_per_dimension,) * D)
    return pseudorandom_local_circuit_helper(qubits, c, s)

def pseudorandom_parallel_circuit_helper(qubits, c, s):
    if qubits.ndim == 1:
        return pseudorandom_1d_parallel_circuit_helper(qubits, s)
    qc = QuantumCircuit(*set(qubit.register for qubit in qubits.flatten()))
    for _ in range(c):
        for i in range(qubits.shape[0]):
            qc += pseudorandom_parallel_circuit_helper(qubits[i], c, s)
        for j in product(range(qubits.shape[0]), repeat=(qubits.ndim - 1)):
            row = qubits
            for j_coordinate in j:
                row = row[:, j_coordinate]
            qc += pseudorandom_1d_parallel_circuit_helper(row, s)
    for i in range(qubits.shape[0]):
        qc += pseudorandom_parallel_circuit_helper(qubits[i], c, s)
    return qc

def pseudorandom_parallel_circuit(D, n_qubits_per_dimension, c, s):
    qr = QuantumRegister(n_qubits_per_dimension ** D)
    qubits = np.reshape(qr[:], (n_qubits_per_dimension,) * D)
    return pseudorandom_parallel_circuit_helper(qubits, c, s)