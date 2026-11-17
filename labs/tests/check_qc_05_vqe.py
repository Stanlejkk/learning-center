"""Checks for qc_05_vqe."""
import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp


def check_hamiltonian_and_exact_energy(ns):
    Hm = np.asarray(ns["hamiltonian_matrix"]())
    ref = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5)]).to_matrix()
    assert np.allclose(Hm, ref), "hamiltonian_matrix must equal Z⊗Z + 0.5 X⊗I in Qiskit's ordering (X on qubit 1)"
    e0 = ns["exact_ground_energy"]()
    assert abs(e0 + np.sqrt(1.25)) < 1e-9, f"exact ground energy should be -sqrt(1.25) ≈ -1.1180, got {e0}"


def check_energy_function(ns):
    energy, ansatz = ns["energy"], ns["ansatz"]
    assert abs(energy(np.zeros(4)) - 1.0) < 1e-9, "with all angles 0 the state is |00> and <H> = <ZZ> = 1"
    rng = np.random.default_rng(0)
    Hm = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5)]).to_matrix()
    for _ in range(3):
        p = rng.uniform(-np.pi, np.pi, 4)
        U = Operator(ansatz(p)).data
        psi = U[:, 0]
        assert abs(energy(p) - np.real(psi.conj() @ Hm @ psi)) < 1e-9, "energy(params) != <psi(params)|H|psi(params)>"


def check_parameter_shift_matches_finite_difference(ns):
    grad, energy = ns["parameter_shift_grad"], ns["energy"]
    rng = np.random.default_rng(1)
    for _ in range(3):
        p = rng.uniform(-np.pi, np.pi, 4)
        g = np.asarray(grad(p))
        h = 1e-5
        fd = np.array([(energy(p + h * e) - energy(p - h * e)) / (2 * h) for e in np.eye(4)])
        assert np.allclose(g, fd, atol=1e-6), f"parameter-shift gradient {g} != finite difference {fd}"
    for th in (0.3, 1.2, -2.0):
        assert abs(ns["single_qubit_expz_grad"](th) + np.sin(th)) < 1e-12, "d<Z>/dθ for Ry(θ)|0> is -sin θ"


def check_vqe_converges(ns):
    p, hist = ns["vqe"](np.array([0.1, 0.2, -0.3, 0.4]), lr=0.2, steps=150)
    e0 = -np.sqrt(1.25)
    assert hist[-1] <= hist[0], "energy should decrease"
    assert abs(hist[-1] - e0) < 1e-3, f"VQE final energy {hist[-1]:.5f} should reach the exact {e0:.5f} within 1e-3"
    assert abs(ns["energy"](p) - hist[-1]) < 1e-9, "returned params must reproduce the last energy in history"
