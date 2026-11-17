"""Checks for qc_01_statevector. Each check_* receives the notebook globals as `ns`."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def _rand_state(n, rng):
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


def check_apply_1q_matches_qiskit(ns):
    apply_1q = ns["apply_1q"]
    rng = np.random.default_rng(0)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    S = np.diag([1, 1j])
    for n in (1, 2, 3):
        for q in range(n):
            for U, name in ((H, "h"), (S, "s")):
                psi = _rand_state(n, rng)
                qc = QuantumCircuit(n)
                getattr(qc, name)(q)
                want = Statevector(psi).evolve(qc).data
                got = np.asarray(apply_1q(psi.copy(), U.astype(complex), q, n))
                assert got.shape == (2 ** n,), f"apply_1q must return a flat vector of length {2 ** n}"
                assert np.allclose(got, want, atol=1e-9), (
                    f"apply_1q({name}, qubit {q}, n={n}) disagrees with Qiskit — remember qubit 0 is the "
                    "least-significant bit (rightmost in |q_{n-1}...q_0>)")


def check_apply_2q_cnot_matches_qiskit(ns):
    apply_2q = ns["apply_2q"]
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    rng = np.random.default_rng(1)
    for n in (2, 3):
        for c in range(n):
            for t in range(n):
                if c == t:
                    continue
                psi = _rand_state(n, rng)
                qc = QuantumCircuit(n)
                qc.cx(c, t)
                want = Statevector(psi).evolve(qc).data
                got = np.asarray(apply_2q(psi.copy(), CNOT, c, t, n))
                assert np.allclose(got, want, atol=1e-9), f"apply_2q(CNOT, control={c}, target={t}, n={n}) is wrong"


def check_simulate_ghz_and_random(ns):
    simulate = ns["simulate"]
    ghz = simulate([("h", 0), ("cx", 0, 1), ("cx", 1, 2)], 3)
    want = np.zeros(8, dtype=complex)
    want[0] = want[7] = 1 / np.sqrt(2)
    assert np.allclose(ghz, want, atol=1e-9), "GHZ state wrong"
    rng = np.random.default_rng(2)
    n = 4
    ops = []
    qc = QuantumCircuit(n)
    for _ in range(25):
        kind = rng.integers(4)
        if kind == 0:
            g = ["h", "x", "s", "t", "z", "y"][rng.integers(6)]
            q = int(rng.integers(n))
            ops.append((g, q))
            getattr(qc, g)(q)
        elif kind == 1:
            g = ["rx", "ry", "rz"][rng.integers(3)]
            th, q = float(rng.uniform(0, 2 * np.pi)), int(rng.integers(n))
            ops.append((g, th, q))
            getattr(qc, g)(th, q)
        else:
            g = ["cx", "cz", "swap"][rng.integers(3)]
            a, b = (int(v) for v in rng.choice(n, 2, replace=False))
            ops.append((g, a, b))
            getattr(qc, g)(a, b)
    got = simulate(ops, n)
    assert np.allclose(got, Statevector(qc).data, atol=1e-8), "random 4-qubit circuit disagrees with Qiskit Statevector"


def check_measure_counts(ns):
    measure_counts, simulate = ns["measure_counts"], ns["simulate"]
    c = measure_counts(simulate([("x", 0)], 2), 100, seed=3)
    assert c == {"01": 100}, f"X on qubit 0 of 2 qubits must always give '01' (Qiskit order q1q0), got {c}"
    psi = simulate([("ry", 2 * np.arccos(np.sqrt(0.3)), 0)], 1)   # P(0)=0.3
    c = measure_counts(psi, 20000, seed=4)
    assert sum(c.values()) == 20000
    p0 = c.get("0", 0) / 20000
    assert abs(p0 - 0.3) < 0.02, f"sampled P(0)={p0:.3f}, expected ~0.3"
    assert measure_counts(psi, 500, seed=5) == measure_counts(psi, 500, seed=5), "sampling must be reproducible for a fixed seed"
