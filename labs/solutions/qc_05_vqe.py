"""Reference solution — qc_05_vqe."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

# H = Z⊗Z + 0.5 X⊗I. Qiskit Pauli labels are little-endian: "XI" = X on qubit 1, I on qubit 0.
HAMILTONIAN = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5)])
N_PARAMS = 4


def hamiltonian_matrix():
    Z = np.diag([1.0, -1.0])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    I = np.eye(2)
    return np.kron(Z, Z) + 0.5 * np.kron(X, I)   # kron(qubit1, qubit0) matches Qiskit's ordering


def exact_ground_energy():
    return float(np.linalg.eigvalsh(hamiltonian_matrix())[0])


def ansatz(params):
    """Hardware-efficient ansatz: Ry layer, CNOT, Ry layer (4 parameters)."""
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 0)
    qc.ry(params[3], 1)
    return qc


def energy(params):
    psi = Statevector(ansatz(params))
    return float(np.real(psi.expectation_value(HAMILTONIAN)))


def parameter_shift_grad(params):
    params = np.asarray(params, dtype=float)
    g = np.zeros_like(params)
    for i in range(len(params)):
        e = np.zeros_like(params)
        e[i] = np.pi / 2
        g[i] = 0.5 * (energy(params + e) - energy(params - e))
    return g


def single_qubit_expz_grad(theta):
    """d/dθ <Z> for Ry(θ)|0> via parameter shift (analytically -sin θ)."""
    f = lambda t: np.cos(t)  # noqa: E731  (<Z> = cos θ)
    return 0.5 * (f(theta + np.pi / 2) - f(theta - np.pi / 2))


def vqe(init_params, lr=0.2, steps=200):
    p = np.asarray(init_params, dtype=float).copy()
    history = [energy(p)]
    for _ in range(steps):
        p = p - lr * parameter_shift_grad(p)
        history.append(energy(p))
    return p, history
