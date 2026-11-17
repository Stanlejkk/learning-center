"""Reference solution — qc_01_statevector: a tiny statevector simulator (Qiskit little-endian convention)."""
import numpy as np

# ---- gates -------------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j]).astype(complex)
T = np.diag([1, np.exp(1j * np.pi / 4)])
# 4x4 in the textbook basis |a b>, a = FIRST qubit argument (control), b = second (target)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def rx(t):
    return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]])


def ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)


def rz(t):
    return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])


# ---- simulator ---------------------------------------------------------------------------
def zero_state(n):
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    return psi


def apply_1q(state, U, q, n):
    """Apply 2x2 U to qubit q (Qiskit order: qubit 0 = least-significant bit)."""
    psi = state.reshape([2] * n)            # axis 0 = qubit n-1, ..., axis n-1 = qubit 0
    ax = n - 1 - q
    psi = np.tensordot(U, psi, axes=([1], [ax]))   # new axis 0 is the acted-on qubit
    psi = np.moveaxis(psi, 0, ax)
    return psi.reshape(-1)


def apply_2q(state, U, qa, qb, n):
    """Apply 4x4 U (basis |a b>, a = qubit qa as the most-significant bit of U) to qubits qa, qb."""
    psi = state.reshape([2] * n)
    axa, axb = n - 1 - qa, n - 1 - qb
    U4 = U.reshape(2, 2, 2, 2)              # [a', b', a, b]
    psi = np.tensordot(U4, psi, axes=([2, 3], [axa, axb]))
    psi = np.moveaxis(psi, [0, 1], [axa, axb])
    return psi.reshape(-1)


_ONE = {"x": X, "y": Y, "z": Z, "h": H, "s": S, "t": T}
_TWO = {"cx": CNOT, "cz": CZ, "swap": SWAP}
_ROT = {"rx": rx, "ry": ry, "rz": rz}


def simulate(ops, n):
    """ops: list like [("h", 0), ("cx", 0, 1), ("ry", 0.3, 2)] -> final statevector (Qiskit order)."""
    psi = zero_state(n)
    for op in ops:
        name = op[0]
        if name in _ONE:
            psi = apply_1q(psi, _ONE[name], op[1], n)
        elif name in _ROT:
            psi = apply_1q(psi, _ROT[name](op[1]), op[2], n)
        elif name in _TWO:
            psi = apply_2q(psi, _TWO[name], op[1], op[2], n)
        else:
            raise ValueError(f"unknown gate {name}")
    return psi


def probabilities(state):
    return np.abs(state) ** 2


def measure_counts(state, shots, seed=0):
    """Sample computational-basis outcomes; keys are Qiskit-style bitstrings q_{n-1}...q_0."""
    n = int(np.log2(len(state)))
    p = probabilities(state)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(state), size=shots, p=p)
    vals, cnt = np.unique(idx, return_counts=True)
    return {format(int(v), f"0{n}b"): int(c) for v, c in zip(vals, cnt)}


def to_qiskit(ops, n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for op in ops:
        name = op[0]
        if name in _ONE:
            getattr(qc, name)(op[1])
        elif name in _ROT:
            getattr(qc, name)(op[1], op[2])
        else:
            getattr(qc, name)(op[1], op[2])
    return qc
