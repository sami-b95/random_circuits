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

def sycamore_18(n_cycles):
    edges = {
        "A": [(1, 4), (2, 5), (8, 11), (3, 6), (9, 12), (10, 13)],
        "B": [(4, 7), (5, 8), (11, 14), (6, 9), (12, 15), (13, 16)],
        "C": [(0, 4), (1, 5), (7, 11), (2, 6), (8, 12), (9, 13)],
        "D": [(4, 8), (5, 9), (11, 15), (6, 10), (12, 16), (13, 17)]
    }
    previous_one_qubit_gates = [None] * 18
    qr = QuantumRegister(18)
    qc = QuantumCircuit(qr)
    def apply_pattern(letter):
        applied_edges = edges[letter]
        for edge in applied_edges:
            qc.unitary(np.array([
                [1, 0, 0, 0],
                [0, 0, -1j, 0],
                [0, -1j, 0, 0],
                [0, 0, 0, np.exp(-1j * np.pi / 6)]
            ]), [qr[edge[0]], qr[edge[1]]])
    for cycle in range(n_cycles):
        # Apply single-qubit gates
        for qubit_index in range(18):
            one_qubit_gate_choices = ["X", "Y", "W"]
            if cycle > 0:
                one_qubit_gate_choices.remove(previous_one_qubit_gates[qubit_index])
            previous_one_qubit_gates[qubit_index] = choice = np.random.choice(one_qubit_gate_choices)
            if choice == "X":
                qc.unitary(1 / np.sqrt(2) * np.array([[1, -1j], [-1j, 1]]), [qr[qubit_index]])
            elif choice == "Y":
                qc.unitary(1 / np.sqrt(2) * np.array([[1, -1], [1, 1]]), [qr[qubit_index]])
            elif choice == "W":
                qc.unitary(1 / np.sqrt(2) * np.array([[1, -np.exp(1j * np.pi / 4)], [np.exp(-1j * np.pi / 4), 1]]), [qr[qubit_index]])
            else:
                raise ValueError("unexpected 1-qubit gate choice {}".format(choice))
        # Apply patterns
        apply_pattern("A")
        apply_pattern("B")
        apply_pattern("C")
        apply_pattern("D")
        apply_pattern("C")
        apply_pattern("D")
        apply_pattern("A")
        apply_pattern("B")
    return qc