"""Reference solution — qc_02_entanglement_teleportation."""
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def bell_circuit(which="phi+"):
    """2-qubit circuit preparing a Bell state from |00>. Textbook |b1 b0> labels; Qiskit amplitude order."""
    qc = QuantumCircuit(2)
    if which in ("psi+", "psi-"):
        qc.x(1)
    if which in ("phi-", "psi-"):
        qc.x(0)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def measurement_observable(angle):
    """cos(a) Z + sin(a) X — a ±1 observable in the xz-plane."""
    return np.cos(angle) * Z + np.sin(angle) * X


def correlation(state, a, b):
    """<A(a) ⊗ B(b)> on a 2-qubit Statevector; A acts on qubit 0 (Alice), B on qubit 1 (Bob)."""
    op = np.kron(measurement_observable(b), measurement_observable(a))   # Qiskit: kron(q1, q0)
    v = np.asarray(state)
    return float(np.real(np.conj(v) @ op @ v))


def chsh_value(a0, a1, b0, b1):
    psi = Statevector(bell_circuit("phi+")).data
    E = lambda a, b: correlation(psi, a, b)  # noqa: E731
    return E(a0, b0) + E(a0, b1) + E(a1, b0) - E(a1, b1)


OPTIMAL_ANGLES = (0.0, np.pi / 2, np.pi / 4, -np.pi / 4)


def _sampled_E(a, b, shots, sim, seed):
    qc = bell_circuit("phi+")
    # rotate measurement axis (cos a, 0, sin a) onto z: Ry(-a)
    qc.ry(-a, 0)
    qc.ry(-b, 1)
    qc.measure_all()
    counts = sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed).result().get_counts()
    tot = 0
    for k, c in counts.items():
        parity = (-1) ** (k.count("1"))
        tot += parity * c
    return tot / shots


def chsh_sampled(shots=4000, seed=7):
    sim = AerSimulator()
    a0, a1, b0, b1 = OPTIMAL_ANGLES
    E = lambda a, b, s: _sampled_E(a, b, shots, sim, seed + s)  # noqa: E731
    return E(a0, b0, 0) + E(a0, b1, 1) + E(a1, b0, 2) - E(a1, b1, 3)


def prepare_input(qc, q, theta, phi):
    qc.ry(theta, q)
    qc.rz(phi, q)


def teleport_circuit(theta, phi):
    """Dynamic teleportation: q0 = input, q1 = Alice's Bell half, q2 = Bob. Creg 'm' (2 bits)."""
    q = QuantumRegister(3, "q")
    m = ClassicalRegister(2, "m")
    qc = QuantumCircuit(q, m)
    prepare_input(qc, 0, theta, phi)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, m[0])
    qc.measure(1, m[1])
    with qc.if_test((m[1], 1)):
        qc.x(2)
    with qc.if_test((m[0], 1)):
        qc.z(2)
    return qc


def teleport_deferred(theta, phi):
    """Same protocol with the measurements deferred: classical control -> quantum control."""
    qc = QuantumCircuit(3)
    prepare_input(qc, 0, theta, phi)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(1, 2)
    qc.cz(0, 2)
    return qc
