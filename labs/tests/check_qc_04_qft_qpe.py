"""Checks for qc_04_qft_qpe."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator


def check_qft_matrix(ns):
    F = np.asarray(ns["qft_matrix"](3))
    assert F.shape == (8, 8)
    assert np.allclose(F.conj().T @ F, np.eye(8), atol=1e-10), "QFT matrix must be unitary"
    x = np.random.default_rng(0).normal(size=8)
    # numpy's ifft uses e^{+2πi jk/N}/N  ->  F x = sqrt(N) * ifft(x)
    assert np.allclose(F @ x, np.sqrt(8) * np.fft.ifft(x), atol=1e-10), "sign/normalisation convention: e^{+2πi jk/N}/sqrt(N)"


def check_qft_circuit_matches_library(ns):
    qft_circuit, qft_matrix = ns["qft_circuit"], ns["qft_matrix"]
    for n in (1, 2, 3, 4):
        U = Operator(qft_circuit(n)).data
        ref = QuantumCircuit(n)
        ref.append(QFTGate(n), range(n))
        assert np.allclose(U, Operator(ref).data, atol=1e-9), f"qft_circuit({n}) differs from QFTGate({n})"
        assert np.allclose(U, qft_matrix(n), atol=1e-9), f"qft_circuit({n}) differs from qft_matrix({n})"
    ops = dict(qft_circuit(4).count_ops())
    assert ops.get("cp", 0) == 6 and ops.get("swap", 0) == 2, f"expected 6 controlled phases and 2 swaps for n=4, got {ops}"


def check_inverse_qft(ns):
    U = Operator(ns["qft_circuit"](3)).data
    V = Operator(ns["inverse_qft_circuit"](3)).data
    assert np.allclose(V @ U, np.eye(8), atol=1e-9), "inverse_qft_circuit(n) must undo qft_circuit(n)"


def check_qpe_exact_phases(ns):
    qpe_circuit, run_counts, estimate_phase = ns["qpe_circuit"], ns["run_counts"], ns["estimate_phase"]
    # T gate: θ = π/4 -> φ = 1/8 -> outcome 001 with certainty
    c = run_counts(qpe_circuit(np.pi / 4, 3), shots=500, seed=1)
    assert c == {"001": 500}, f"QPE of the T gate with t=3 should always give '001', got {c}"
    phi = 5 / 16
    est = estimate_phase(run_counts(qpe_circuit(2 * np.pi * phi, 4), shots=500, seed=2), 4)
    assert abs(est - phi) < 1e-12, f"φ = 5/16 should be read exactly with t=4, got {est}"


def check_qpe_inexact_phase(ns):
    qpe_circuit, run_counts, estimate_phase = ns["qpe_circuit"], ns["run_counts"], ns["estimate_phase"]
    t, phi = 5, 1 / 3
    counts = run_counts(qpe_circuit(2 * np.pi * phi, t), shots=3000, seed=3)
    est = estimate_phase(counts, t)
    assert abs(est - phi) <= 1 / 2 ** t, f"estimate {est:.4f} should be within 1/2^t of 1/3"
    dist = np.asarray(ns["qpe_distribution"](phi, t))
    assert abs(dist.sum() - 1) < 1e-9, "qpe_distribution must sum to 1"
    m = int(round(phi * 2 ** t))
    emp = counts.get(format(m, f"0{t}b"), 0) / 3000
    assert abs(emp - dist[m]) < 0.04, f"empirical P(m={m}) = {emp:.3f} vs theory {dist[m]:.3f}"
    assert dist[m] > 4 / np.pi ** 2, "the nearest estimate has probability ≥ 4/π²"
