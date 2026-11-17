"""Reference solution — qc_04_qft_qpe."""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def qft_matrix(n):
    """F[k, j] = exp(2πi jk / N) / sqrt(N), N = 2**n (integers in Qiskit little-endian indexing)."""
    N = 2 ** n
    j = np.arange(N)
    return np.exp(2j * np.pi * np.outer(j, j) / N) / np.sqrt(N)


def qft_circuit(n, do_swaps=True):
    """QFT from H + controlled-phase + SWAP; equals qiskit.circuit.library.QFTGate(n)."""
    qc = QuantumCircuit(n, name="QFT")
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            qc.cp(np.pi / 2 ** (j - k), k, j)
    if do_swaps:
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
    return qc


def inverse_qft_circuit(n):
    return qft_circuit(n).inverse()


def qpe_circuit(theta, t):
    """Estimate φ = θ/2π for U = P(θ) = diag(1, e^{iθ}) with eigenstate |1>.
    Counting qubits 0..t-1 (qubit k controls U^(2^k)), target qubit t; counting qubit k -> clbit k."""
    qc = QuantumCircuit(t + 1, t)
    qc.x(t)
    qc.h(range(t))
    for k in range(t):
        qc.cp(theta * 2 ** k, k, t)
    qc.compose(inverse_qft_circuit(t), qubits=range(t), inplace=True)
    qc.measure(range(t), range(t))
    return qc


def run_counts(qc, shots=2000, seed=0):
    sim = AerSimulator()
    return sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed).result().get_counts()


def estimate_phase(counts, t):
    """Most frequent outcome m -> φ ≈ m / 2^t."""
    m = max(counts, key=counts.get)
    return int(m.replace(" ", ""), 2) / 2 ** t


def qpe_distribution(phi, t):
    """Exact P(m) = |(1/2^t) Σ_k e^{2πi k(φ - m/2^t)}|^2 for m = 0..2^t-1."""
    T = 2 ** t
    k = np.arange(T)
    return np.array([abs(np.exp(2j * np.pi * k * (phi - m / T)).sum() / T) ** 2 for m in range(T)])


