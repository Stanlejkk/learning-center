"""Checks for qc_03_grover."""
import numpy as np
from qiskit.quantum_info import Operator


def _equal_up_to_phase(A, B, tol=1e-8):
    k = np.argmax(np.abs(B))
    idx = np.unravel_index(k, B.shape)
    if abs(A[idx]) < 1e-12:
        return False
    ph = A[idx] / B[idx]
    return abs(abs(ph) - 1) < tol and np.allclose(A, ph * B, atol=tol)


def check_phase_oracle(ns):
    phase_oracle = ns["phase_oracle"]
    for n, marked in [(3, [5]), (3, [0, 6]), (4, [9]), (4, [1, 2, 15])]:
        U = Operator(phase_oracle(n, marked)).data
        want = np.diag([-1 if x in marked else 1 for x in range(2 ** n)]).astype(complex)
        assert _equal_up_to_phase(U, want), f"phase_oracle({n}, {marked}) is not diag(±1) with -1 on the marked states"


def check_diffuser(ns):
    diffuser = ns["diffuser"]
    for n in (2, 3, 4):
        N = 2 ** n
        s = np.ones(N) / np.sqrt(N)
        want = 2 * np.outer(s, s) - np.eye(N)
        U = Operator(diffuser(n)).data
        assert _equal_up_to_phase(U, want), f"diffuser({n}) != 2|s><s| - I (up to global phase)"


def check_success_matches_theory(ns):
    sp, tp = ns["success_probability"], ns["theory_probability"]
    for n, marked in [(3, [3]), (4, [10]), (4, [1, 7])]:
        for k in range(0, 6):
            got, want = sp(n, marked, k), tp(2 ** n, len(marked), k)
            assert abs(got - want) < 1e-6, f"n={n}, marked={marked}, k={k}: simulated {got:.4f} vs sin^2((2k+1)θ) = {want:.4f}"


def check_optimal_iterations_and_sampling(ns):
    oi = ns["optimal_iterations"]
    assert oi(8, 1) == 2 and oi(16, 1) == 3 and oi(1024, 1) == 25 and oi(16, 4) == 1, "optimal_iterations wrong"
    counts = ns["run_grover_counts"](4, [11], shots=2000, seed=3)
    top = max(counts, key=counts.get)
    assert int(top, 2) == 11, f"most frequent outcome should be the marked item 11 = 1011, got {top}"
    assert counts[top] / 2000 > 0.9, "after the optimal number of iterations the success rate should be ~96%"
