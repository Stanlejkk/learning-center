"""Reference solution — qc_03_grover."""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def _mcz(qc, n):
    """Multi-controlled Z on all n qubits (phase -1 on |1...1>)."""
    if n == 1:
        qc.z(0)
        return
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)


def phase_oracle(n, marked):
    """Flip the sign of each marked basis state (integers, Qiskit little-endian indexing)."""
    qc = QuantumCircuit(n, name="oracle")
    for m in marked:
        zeros = [q for q in range(n) if not (m >> q) & 1]
        for q in zeros:
            qc.x(q)
        _mcz(qc, n)
        for q in zeros:
            qc.x(q)
    return qc


def diffuser(n):
    """2|s><s| - I (up to a global phase) = H^n X^n (MCZ) X^n H^n."""
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    _mcz(qc, n)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover_circuit(n, marked, k):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    oracle, diff = phase_oracle(n, marked), diffuser(n)
    for _ in range(k):
        qc.compose(oracle, inplace=True)
        qc.compose(diff, inplace=True)
    return qc


def success_probability(n, marked, k):
    p = Statevector(grover_circuit(n, marked, k)).probabilities()
    return float(sum(p[m] for m in marked))


def theory_probability(N, M, k):
    th = np.arcsin(np.sqrt(M / N))
    return float(np.sin((2 * k + 1) * th) ** 2)


def optimal_iterations(N, M):
    return int(np.floor(np.pi / 4 * np.sqrt(N / M)))


def run_grover_counts(n, marked, shots=2000, seed=0):
    qc = grover_circuit(n, marked, optimal_iterations(2 ** n, len(marked)))
    qc.measure_all()
    sim = AerSimulator()
    return sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed).result().get_counts()
