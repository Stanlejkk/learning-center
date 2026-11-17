"""Checks for qc_02_entanglement_teleportation."""
import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
from qiskit_aer import AerSimulator

S2 = 1 / np.sqrt(2)
# amplitude vectors in Qiskit order (index = 2*q1 + q0); Bell states are defined up to a global phase
BELL = {"phi+": [S2, 0, 0, S2], "phi-": [S2, 0, 0, -S2], "psi+": [0, S2, S2, 0], "psi-": [0, S2, -S2, 0]}


def check_bell_circuits(ns):
    bell_circuit = ns["bell_circuit"]
    for name, vec in BELL.items():
        qc = bell_circuit(name)
        assert qc.num_qubits == 2, "bell_circuit must return a 2-qubit circuit"
        got = Statevector(qc).data
        ov = abs(np.vdot(vec, got))
        assert abs(ov - 1) < 1e-9, f"bell_circuit('{name}') prepares the wrong state (|overlap| = {ov:.3f})"


def check_chsh_exact(ns):
    chsh_value, correlation = ns["chsh_value"], ns["correlation"]
    s = chsh_value(0.0, np.pi / 2, np.pi / 4, -np.pi / 4)
    assert abs(s - 2 * np.sqrt(2)) < 1e-9, f"optimal CHSH should be 2*sqrt(2) ~ 2.828, got {s}"
    rng = np.random.default_rng(0)
    phi = np.array(BELL["phi+"], dtype=complex)
    for _ in range(5):
        a, b = rng.uniform(-np.pi, np.pi, 2)
        assert abs(correlation(phi, a, b) - np.cos(a - b)) < 1e-9, "on |Phi+>, E(a,b) must equal cos(a-b)"
    assert abs(chsh_value(0, 0, 0, 0) - 2.0) < 1e-9, "all-equal angles should give S = 2"


def check_chsh_sampled_violates_bell(ns):
    s = ns["chsh_sampled"](shots=4000, seed=11)
    assert 2.6 < s < 3.05, f"sampled CHSH value {s:.3f} should be close to 2.83 (> 2: Bell violation)"


def _target(theta, phi):
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


def check_teleport_deferred(ns):
    td = ns["teleport_deferred"]
    for theta, phi in [(0.0, 0.0), (np.pi, 0.0), (1.1, 0.4), (2.3, -1.7), (np.pi / 2, np.pi / 2)]:
        qc = td(theta, phi)
        assert qc.num_qubits == 3 and qc.num_clbits == 0, "deferred version: 3 qubits, no measurements"
        rho_bob = partial_trace(Statevector(qc), [0, 1])
        f = state_fidelity(rho_bob, Statevector(_target(theta, phi)))
        assert f > 1 - 1e-9, f"Bob's qubit (q2) fidelity {f:.4f} for theta={theta:.2f}, phi={phi:.2f}"


def check_teleport_dynamic_on_aer(ns):
    tc = ns["teleport_circuit"]
    sim = AerSimulator()
    for k, (theta, phi) in enumerate([(1.1, 0.4), (2.3, -1.7), (np.pi, 0.0)]):
        qc = tc(theta, phi).copy()
        assert qc.num_qubits == 3, "teleport_circuit: 3 qubits (q0 input, q1 Alice, q2 Bob)"
        assert qc.num_clbits >= 2, "teleport_circuit must measure Alice's two qubits"
        # undo the preparation on Bob's qubit; if teleportation worked, q2 is now |0> with certainty
        qc.rz(-phi, 2)
        qc.ry(-theta, 2)
        chk = ClassicalRegister(1, "chk")
        qc.add_register(chk)
        qc.measure(2, chk[0])
        counts = sim.run(transpile(qc, sim), shots=400, seed_simulator=100 + k).result().get_counts()
        bad = sum(c for key, c in counts.items() if key.split(" ")[0] == "1")
        assert bad == 0, f"Bob's qubit is not the teleported state ({bad}/400 failures) — check the X/Z corrections"
        outcomes = {key.split(" ")[-1] for key in counts}
        assert len(outcomes) >= 3, "Alice's measurement outcomes should be (close to) uniformly random"
