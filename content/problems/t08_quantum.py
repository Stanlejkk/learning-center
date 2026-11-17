"""Track 8 — Quantum computing (ids qc.*).

Grounding: Nielsen & Chuang, *Quantum Computation and Quantum Information* (N&C);
Shor 1994/95 (quant-ph/9508027); Grover 1996 (quant-ph/9605043); MIT 8.370x / 18.435J;
de Wolf's lecture notes (arXiv:1907.09415); the Qiskit documentation.

Conventions used in every prompt (stated explicitly where it matters):
  * Multi-qubit kets are written big-endian: in |q1 q2 ...> qubit 1 is the LEFT tensor factor, and
    vectors list amplitudes in the order |00>, |01>, |10>, |11> (N&C convention).
    Qiskit is little-endian (qubit 0 is the RIGHTMOST bit) — taught in qc.tensor level 3.
  * R_z(a) = diag(e^{-ia/2}, e^{ia/2}),  R_y(a) = [[cos a/2, -sin a/2],[sin a/2, cos a/2]].
  * QFT|j> = (1/sqrt N) sum_k e^{+2 pi i jk/N} |k>.
Answers are computed with sympy; the checker compares numerically so any equivalent exact form is accepted.
"""
from __future__ import annotations

import itertools
import math
import random

import sympy as sp

from engine.checker import fmt
from engine.content import Problem, gen

L = sp.latex
I = sp.I
PI = sp.pi
R2 = sp.sqrt(2)


# ============================================================================ helpers
def nz(rng: random.Random, lo: int, hi: int) -> int:
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def cs(e):
    """Canonical simplified complex scalar in a+bi form."""
    e = sp.simplify(sp.expand_complex(sp.simplify(e)))
    return sp.Integer(0) if e == 0 else e


def cmat(m: sp.Matrix) -> sp.Matrix:
    return m.applyfunc(cs)


def vec(*xs) -> sp.Matrix:
    return sp.Matrix(list(xs))


def kron(*ms: sp.Matrix) -> sp.Matrix:
    out = ms[0]
    for m in ms[1:]:
        r1, c1 = out.shape
        r2, c2 = m.shape
        out = sp.Matrix(r1 * r2, c1 * c2, lambda i, j: out[i // r2, j // c2] * m[i % r2, j % c2])
    return out


def dag(m: sp.Matrix) -> sp.Matrix:
    return m.H


def vans(v) -> list[str]:
    """Vector answer: list of formatted components."""
    return [fmt(cs(x)) for x in list(v)]


def mans(m: sp.Matrix) -> str:
    return fmt(cmat(m))


def ket(lbl) -> str:
    return r"\lvert " + str(lbl) + r"\rangle"


def bra(lbl) -> str:
    return r"\langle " + str(lbl) + r"\rvert"


def coef_tex(c) -> tuple[str, str]:
    """Return (sign, latex-of-|c|-ish) for writing c|x> inside a sum."""
    c = cs(c)
    sign = "+"
    if c.could_extract_minus_sign():
        sign, c = "-", cs(-c)
    if c == 1:
        return sign, ""
    t = L(c)
    if isinstance(c, sp.Add):
        t = r"\left(" + t + r"\right)"
    return sign, t


def state_tex(v, n: int | None = None) -> str:
    v = list(v)
    n = n or max(1, int(round(math.log2(len(v)))))
    parts = []
    for idx, c in enumerate(v):
        if cs(c) == 0:
            continue
        sign, t = coef_tex(c)
        parts.append((sign, t + ket(format(idx, f"0{n}b"))))
    if not parts:
        return "0"
    s = ("-" if parts[0][0] == "-" else "") + parts[0][1]
    for sign, t in parts[1:]:
        s += f" {sign} {t}"
    return s


def lin_tex(terms) -> str:
    """Render sum c_k * name_k with clean signs, e.g. [(2,'Z'),(-1,'X')] -> '2Z - X'."""
    out = ""
    for c, name in terms:
        if c == 0:
            continue
        mag = "" if abs(c) == 1 else str(abs(c))
        lead = ("-" if c < 0 else "") if not out else (" - " if c < 0 else " + ")
        out += lead + mag + (r"\," if mag else "") + name
    return out or "0"


def vec_tex(v) -> str:
    return r"\begin{pmatrix}" + r" \\ ".join(L(cs(x)) for x in list(v)) + r"\end{pmatrix}"


def mat_tex(m: sp.Matrix) -> str:
    return L(cmat(m))


def bloch_vector(v) -> list:
    a, b = v[0], v[1]
    ab = sp.conjugate(a) * b
    return [cs(2 * sp.re(ab)), cs(2 * sp.im(ab)), cs(sp.Abs(a) ** 2 - sp.Abs(b) ** 2)]


def expval(op: sp.Matrix, v: sp.Matrix):
    return cs((dag(v) * op * v)[0, 0])


def normalize(v: sp.Matrix) -> sp.Matrix:
    nrm = sp.sqrt(sum(sp.Abs(x) ** 2 for x in v))
    return (v / nrm).applyfunc(sp.simplify)


# gates & states
H = sp.Matrix([[1, 1], [1, -1]]) / R2
X = sp.Matrix([[0, 1], [1, 0]])
Y = sp.Matrix([[0, -I], [I, 0]])
Z = sp.Matrix([[1, 0], [0, -1]])
S = sp.Matrix([[1, 0], [0, I]])
T = sp.Matrix([[1, 0], [0, sp.exp(I * PI / 4)]])
ID = sp.eye(2)
GATES = {"I": ID, "X": X, "Y": Y, "Z": Z, "H": H, "S": S, "T": T}
GATE_TEX = {"I": "I", "X": "X", "Y": "Y", "Z": "Z", "H": "H", "S": "S", "T": "T"}

STATES = {
    "0": vec(1, 0), "1": vec(0, 1), "+": vec(1, 1) / R2, "-": vec(1, -1) / R2,
    "+i": vec(1, I) / R2, "-i": vec(1, -I) / R2,
}

CNOT12 = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CNOT21 = sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
CZ = sp.diag(1, 1, 1, -1)
SWAP = sp.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def Rz(a):
    return sp.diag(sp.exp(-I * a / 2), sp.exp(I * a / 2))


def Ry(a):
    return sp.Matrix([[sp.cos(a / 2), -sp.sin(a / 2)], [sp.sin(a / 2), sp.cos(a / 2)]])


def ctrl(U: sp.Matrix, control_first: bool = True) -> sp.Matrix:
    P0, P1 = sp.diag(1, 0), sp.diag(0, 1)
    return kron(P0, ID) + kron(P1, U) if control_first else kron(ID, P0) + kron(U, P1)


def qft_matrix(N: int) -> sp.Matrix:
    w = sp.exp(2 * PI * I / N)
    return sp.Matrix(N, N, lambda j, k: w ** (j * k)) / sp.sqrt(N)


def rand_amp(rng, allow_complex=True, lo=-3, hi=3):
    v = nz(rng, lo, hi)
    if allow_complex and rng.random() < 0.35:
        v = v * I
    return sp.Integer(v) if not isinstance(v, sp.Basic) else v


CONV = (r"Convention: qubit 1 is the left tensor factor; list amplitudes in the order "
        r"$\lvert 00\rangle, \lvert 01\rangle, \lvert 10\rangle, \lvert 11\rangle$.")
HINT_V1 = "amplitudes [c0, c1], each typed separately, e.g. [1/sqrt(2), i/sqrt(2)]  (i = sqrt(-1))"
HINT_V2 = ("amplitudes [c00, c01, c10, c11], each typed separately, e.g. [1/sqrt(2), 0, 0, -i/sqrt(2)] "
           "(do not factor a constant outside the brackets)")
HINT_M = "matrix as rows, e.g. [[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]]  (i = sqrt(-1))"
HINT_BITS = "just the bits, most significant first, e.g. 011"

ANGLES = [PI / 6, PI / 4, PI / 3, PI / 2, 2 * PI / 3, 3 * PI / 4, 5 * PI / 6, PI,
          7 * PI / 6, 5 * PI / 4, 4 * PI / 3, 3 * PI / 2, 5 * PI / 3, 7 * PI / 4, 11 * PI / 6]


# ============================================================================ qc.qubits
@gen("qc.qubits")
def qubit_states(rng, level):
    if level == 1:
        a, b = nz(rng, 1, 5), rng.choice([1, -1]) * nz(rng, 1, 5) * rng.choice([1, I])
        n2 = a ** 2 + sp.Abs(b) ** 2
        c = 1 / sp.sqrt(n2)
        return Problem(
            prompt=(f"Find the real constant $N>0$ such that $N\\left({state_tex([a, b])}\\right)$ is a valid "
                    "(normalised) qubit state."),
            answer=fmt(cs(c)), kind="num",
            rubric=["Normalisation means |α|² + |β|² = 1",
                    "Use the modulus squared |β|² = β β* (not β²) for complex amplitudes",
                    "Solve N² (|a|² + |b|²) = 1 and take the positive root"],
            solution=(f"$\\langle\\psi\\vert\\psi\\rangle = N^2(|{a}|^2 + |{L(b)}|^2) = {n2}N^2 = 1$, "
                      f"so $N = 1/\\sqrt{{{n2}}} = {L(cs(c))}$."),
            misconceptions=[(fmt(1 / (a + sp.Abs(b))), "forgot-amplitude-squared"),
                            (fmt(1 / sp.sqrt(a ** 2 + b ** 2)) if sp.re(b ** 2) + a ** 2 > 0 else "0",
                             "squared-without-conjugate")],
        )
    if level == 2:
        p = vec(rand_amp(rng), rand_amp(rng))
        q = vec(rand_amp(rng), rand_amp(rng))
        ip = cs((dag(p) * q)[0, 0])
        noconj = cs((p.T * q)[0, 0])
        return Problem(
            prompt=(f"Let $\\lvert\\phi\\rangle = {state_tex(p)}$ and $\\lvert\\psi\\rangle = {state_tex(q)}$ "
                    "(unnormalised). Compute the inner product $\\langle\\phi\\vert\\psi\\rangle$ in the form $a+bi$."),
            answer=fmt(ip), kind="expr", answer_hint="a complex number, e.g. 3 - 2*i",
            rubric=["Form the bra: conjugate-transpose the coefficients of |φ⟩",
                    "Use ⟨0|0⟩ = ⟨1|1⟩ = 1, ⟨0|1⟩ = 0",
                    "Sum the products and collect real/imaginary parts"],
            solution=(f"$\\langle\\phi\\vert = {L(cs(sp.conjugate(p[0])))}\\langle 0\\rvert + "
                      f"\\left({L(cs(sp.conjugate(p[1])))}\\right)\\langle 1\\rvert$, so "
                      f"$\\langle\\phi\\vert\\psi\\rangle = ({L(cs(sp.conjugate(p[0])))})({L(q[0])}) + "
                      f"({L(cs(sp.conjugate(p[1])))})({L(q[1])}) = {L(ip)}$."),
            misconceptions=[(fmt(noconj), "forgot-bra-conjugate"), (fmt(cs(sp.conjugate(ip))), "reversed-inner-product")],
        )
    phis = [k * PI / 4 for k in range(8)]
    f1, f2 = rng.sample(phis, 2)
    fid = cs(sp.cos((f1 - f2) / 2) ** 2)
    return Problem(
        prompt=(f"Let $\\lvert\\psi\\rangle = \\frac{{1}}{{\\sqrt2}}\\left(\\lvert 0\\rangle + e^{{i {L(f1)}}}\\lvert 1\\rangle\\right)$ "
                f"and $\\lvert\\phi\\rangle = \\frac{{1}}{{\\sqrt2}}\\left(\\lvert 0\\rangle + e^{{i {L(f2)}}}\\lvert 1\\rangle\\right)$. "
                "Derive a general formula for the overlap $|\\langle\\phi\\vert\\psi\\rangle|^2$ in terms of the "
                "relative phase difference, then evaluate it exactly."),
        answer=fmt(fid), kind="num",
        rubric=["Write ⟨φ|ψ⟩ = (1 + e^{i(φ_ψ − φ_φ)})/2",
                "Factor out e^{iΔ/2} to get e^{iΔ/2} cos(Δ/2)",
                "Overlap = cos²(Δ/2)", "Evaluate at the given angles"],
        solution=(f"$\\langle\\phi\\vert\\psi\\rangle = \\tfrac12(1 + e^{{i\\Delta}})$ with $\\Delta = {L(f1 - f2)}$, "
                  "$= e^{i\\Delta/2}\\cos(\\Delta/2)$, so $|\\langle\\phi\\vert\\psi\\rangle|^2 = \\cos^2(\\Delta/2) = "
                  f"{L(fid)}$. Only the *relative* phase matters (a global phase would cancel)."),
        misconceptions=[(fmt(cs(sp.cos(f1 - f2) ** 2)), "dropped-half-angle"),
                        (fmt(cs(sp.cos((f1 - f2) / 2))), "forgot-amplitude-squared")],
    )


# ============================================================================ qc.single_gates
_SIMPLE = ["0", "1", "+", "-", "+i", "-i"]


@gen("qc.single_gates")
def gate_sequences(rng, level):
    if level == 1:
        g = rng.choice(["X", "Y", "Z", "H", "S", "T"])
        s = rng.choice(_SIMPLE)
        out = GATES[g] * STATES[s]
        return Problem(
            prompt=f"Compute ${g}{ket(s)}$ as a column vector in the $\\{{\\lvert0\\rangle,\\lvert1\\rangle\\}}$ basis.",
            answer=vans(out), kind="vector", answer_hint=HINT_V1,
            rubric=[f"Write {ket(s)} as a column vector", f"Write the matrix of {g}", "Multiply matrix × vector"],
            solution=(f"${g} = {mat_tex(GATES[g])}$, ${ket(s)} = {vec_tex(STATES[s])}$, so "
                      f"${g}{ket(s)} = {vec_tex(out)} = {state_tex(out)}$."),
        )
    if level == 2:
        k = rng.choice([2, 3])
        seq = [rng.choice(["X", "Y", "Z", "H", "S", "T"]) for _ in range(k)]
        while len(set(seq)) == 1:
            seq = [rng.choice(["X", "Y", "Z", "H", "S", "T"]) for _ in range(k)]
        s = rng.choice(["0", "1", "+"])
        v = STATES[s]
        for g in seq:
            v = GATES[g] * v
        w = STATES[s]
        for g in reversed(seq):
            w = GATES[g] * w
        circ = " \\to ".join(seq)
        opstr = "".join(reversed(seq))
        return Problem(
            prompt=(f"A qubit starts in ${ket(s)}$. The circuit applies, in time order, ${circ}$. "
                    "Give the final state vector."),
            answer=vans(v), kind="vector", answer_hint=HINT_V1,
            rubric=["Time order left→right means the matrix product is written right→left",
                    f"Final state = {opstr}{ket(s)}", "Apply one gate at a time (or multiply matrices first)",
                    "Simplify phases like e^{iπ/4}"],
            solution=(f"State $= {opstr}{ket(s)} = {vec_tex(v)}$, i.e. ${state_tex(v)}$."),
            misconceptions=[(vans(w), "reversed-gate-order")],
        )
    # level 3: overall unitary of a 3-gate circuit
    pool = ["X", "Z", "H", "S", "T", "Y"]
    seq = rng.sample(pool, 3) if rng.random() < 0.6 else rng.choice([["H", "Z", "H"], ["H", "X", "H"], ["H", "S", "H"], ["S", "H", "S"], ["H", "T", "H"]])
    U = GATES[seq[2]] * GATES[seq[1]] * GATES[seq[0]]
    Uw = GATES[seq[0]] * GATES[seq[1]] * GATES[seq[2]]
    return Problem(
        prompt=(f"A circuit applies ${seq[0]}$, then ${seq[1]}$, then ${seq[2]}$ (time order). "
                "Compute the single $2\\times 2$ unitary the whole circuit implements."),
        answer=mans(U), kind="matrix", answer_hint=HINT_M,
        rubric=[f"Write U = {seq[2]}·{seq[1]}·{seq[0]} (last gate leftmost)",
                "Multiply the 2×2 matrices carefully", "Check U is unitary (columns orthonormal)",
                "Recognise identities such as HZH = X, HXH = Z if they appear"],
        solution=(f"$U = {seq[2]}{seq[1]}{seq[0]} = {mat_tex(U)}$. "
                  "Useful identities (N&C Ex. 4.13): $HXH = Z$, $HZH = X$, $HYH = -Y$, $T^2 = S$, $S^2 = Z$."),
        misconceptions=[(mans(Uw), "reversed-gate-order")],
    )


@gen("qc.single_gates")
def bloch_sphere(rng, level):
    th = rng.choice([PI / 3, PI / 2, 2 * PI / 3])
    ph = rng.choice([k * PI / 4 for k in range(8)] if level > 1 else [0, PI / 2, PI, 3 * PI / 2])
    v = vec(cs(sp.cos(th / 2)), cs(sp.exp(I * ph) * sp.sin(th / 2)))
    if level == 1:
        bv = bloch_vector(v)
        return Problem(
            prompt=(f"Give the Bloch vector $(x, y, z)$ of the state ${state_tex(v)}$. "
                    "(Recall $\\cos\\tfrac\\theta2\\lvert0\\rangle + e^{i\\varphi}\\sin\\tfrac\\theta2\\lvert1\\rangle "
                    "\\mapsto (\\sin\\theta\\cos\\varphi, \\sin\\theta\\sin\\varphi, \\cos\\theta)$.)"),
            answer=[fmt(c) for c in bv], kind="vector", answer_hint="x, y, z e.g. [sqrt(3)/2, 0, 1/2]",
            rubric=["Read off cos(θ/2) and e^{iφ} sin(θ/2)", "Find θ (note the half angle) and φ",
                    "Convert to (sin θ cos φ, sin θ sin φ, cos θ)"],
            solution=(f"$\\cos\\frac\\theta2 = {L(v[0])}$ gives $\\theta = {L(th)}$, and $\\varphi = {L(ph)}$. "
                      f"Bloch vector $= ({', '.join(L(c) for c in bv)})$. "
                      "Equivalently $x = 2\\,\\mathrm{Re}(\\alpha^*\\beta),\\ y = 2\\,\\mathrm{Im}(\\alpha^*\\beta),\\ z = |\\alpha|^2-|\\beta|^2$."),
            misconceptions=[([fmt(cs(sp.sin(th / 2) * sp.cos(ph))), fmt(cs(sp.sin(th / 2) * sp.sin(ph))), fmt(cs(sp.cos(th / 2)))],
                             "bloch-half-angle")],
        )
    if level == 2:
        return Problem(
            prompt=(f"Write ${state_tex(v)}$ as $\\cos\\tfrac\\theta2\\lvert0\\rangle + e^{{i\\varphi}}\\sin\\tfrac\\theta2\\lvert1\\rangle$ "
                    "with $\\theta\\in[0,\\pi]$, $\\varphi\\in[0, 2\\pi)$. Give $\\theta, \\varphi$."),
            answer=[fmt(th), fmt(ph)], kind="list", answer_hint="theta, phi  e.g. pi/3, 3*pi/2",
            rubric=["|α| = cos(θ/2) fixes θ (remember the factor 1/2)",
                    "φ = arg(β) − arg(α) (α is already real and positive)", "Put φ in [0, 2π)"],
            solution=(f"$\\cos\\frac\\theta2 = {L(v[0])} \\Rightarrow \\theta = {L(th)}$; "
                      f"$\\beta = {L(v[1])} = e^{{i {L(ph)}}}\\sin\\frac\\theta2 \\Rightarrow \\varphi = {L(ph)}$."),
            misconceptions=[([fmt(th / 2), fmt(ph)], "bloch-half-angle"),
                            ([fmt(th), fmt(sp.Mod(-ph, 2 * PI))], "phase-sign-flipped")],
        )
    a = rng.choice([PI / 3, PI / 2, 2 * PI / 3, PI / 4, 3 * PI / 4, PI / 6, 5 * PI / 6, PI])
    st = H * Rz(a) * H * vec(1, 0)
    bv = bloch_vector(st)
    return Problem(
        prompt=(f"Starting from $\\lvert 0\\rangle$, apply $H$, then $R_z({L(a)})$, then $H$, where "
                "$R_z(\\alpha) = \\mathrm{diag}(e^{-i\\alpha/2}, e^{i\\alpha/2})$. Give the final Bloch vector $(x,y,z)$. "
                "(On paper: show $H R_z(\\alpha) H = R_x(\\alpha)$ and interpret geometrically.)"),
        answer=[fmt(c) for c in bv], kind="vector", answer_hint="x, y, z",
        rubric=["H maps the z-axis to the x-axis, so H R_z(α) H = R_x(α)",
                "R_x(α) rotates the Bloch vector about x by angle α",
                "Start at (0,0,1); the result is (0, −sin α, cos α)",
                "(Or compute the state vector and use x = 2Re(α*β) etc.)"],
        solution=(f"$H R_z(\\alpha) H = R_x(\\alpha)$, a rotation by $\\alpha = {L(a)}$ about $x$: "
                  f"$(0,0,1) \\mapsto (0, -\\sin\\alpha, \\cos\\alpha) = ({', '.join(L(c) for c in bv)})$. "
                  "Global phases never affect the Bloch vector."),
        misconceptions=[([fmt(c) for c in bloch_vector(Rz(a) * H * vec(1, 0))], "ignored-final-hadamard"),
                        ([ "0", fmt(cs(sp.sin(a))), fmt(cs(sp.cos(a)))], "rotation-sense-flipped")],
    )


# ============================================================================ qc.measurement
@gen("qc.measurement")
def born_rule(rng, level):
    if level == 1:
        a, b = nz(rng, 1, 4), rng.choice([1, -1]) * nz(rng, 1, 4) * rng.choice([1, I])
        n2 = a ** 2 + sp.Abs(b) ** 2
        p1 = sp.Rational(int(sp.Abs(b) ** 2), int(n2))
        return Problem(
            prompt=(f"A qubit is in the state $\\dfrac{{1}}{{\\sqrt{{{n2}}}}}\\left({state_tex([a, b])}\\right)$. "
                    "It is measured in the computational basis. What is the probability of outcome 1?"),
            answer=fmt(p1), kind="num",
            rubric=["Born rule: P(1) = |⟨1|ψ⟩|²", "Use the modulus squared of the (normalised) amplitude"],
            solution=f"$P(1) = \\left|{L(b)}/\\sqrt{{{n2}}}\\right|^2 = {L(p1)}$.",
            misconceptions=[(fmt(cs(sp.Abs(b) / sp.sqrt(n2))), "forgot-amplitude-squared"),
                            (fmt(sp.Rational(a * a, int(n2))), "swapped-outcomes")],
        )
    if level == 2:
        while True:
            a, b = nz(rng, -3, 3), rand_amp(rng)
            n2 = a ** 2 + sp.Abs(b) ** 2
            pm = cs(sp.Abs(a - b) ** 2 / (2 * n2))
            if pm not in (0, 1):
                break
        outcome = rng.choice(["+", "-"])
        p = cs(sp.Abs(a + b) ** 2 / (2 * n2)) if outcome == "+" else pm
        return Problem(
            prompt=(f"The qubit $\\lvert\\psi\\rangle \\propto {state_tex([a, b])}$ (normalise it first) is measured in the "
                    f"$\\{{\\lvert+\\rangle, \\lvert-\\rangle\\}}$ basis. What is the probability of the outcome ${ket(outcome)}$?"),
            answer=fmt(p), kind="num",
            rubric=["Normalise the state", f"Compute ⟨{outcome}|ψ⟩ = (α {outcome} β)/√2",
                    "Take the modulus squared"],
            solution=(f"Norm$^2 = {n2}$. $\\langle {outcome}\\vert\\psi\\rangle = \\frac{{({a}) {outcome} ({L(b)})}}{{\\sqrt2\\sqrt{{{n2}}}}}$, "
                      f"so $P = {L(p)}$."),
            misconceptions=[(fmt(cs(sp.Abs(b) ** 2 / n2)), "measured-in-wrong-basis"),
                            (fmt(cs(sp.Abs(a + b if outcome == "-" else a - b) ** 2 / (2 * n2))), "swapped-plus-minus")],
        )
    while True:
        amps = [rand_amp(rng) if rng.random() < 0.8 else sp.Integer(0) for _ in range(4)]
        m = rng.choice([0, 1])
        c0, c1 = amps[2 * m], amps[2 * m + 1]
        if (c0 != 0 or c1 != 0) and sum(1 for x in amps if x != 0) >= 3:
            break
    n2 = sum(sp.Abs(x) ** 2 for x in amps)
    pm = cs((sp.Abs(c0) ** 2 + sp.Abs(c1) ** 2) / n2)
    post = normalize(vec(c0, c1))
    return Problem(
        prompt=(f"Two qubits are in the state $\\lvert\\psi\\rangle \\propto {state_tex(amps)}$. "
                f"Qubit 1 (the left one) is measured and gives outcome ${m}$. "
                f"Give the normalised post-measurement state of qubit 2. "
                f"(On paper also compute the probability of this outcome.)"),
        answer=vans(post), kind="vector", answer_hint=HINT_V1,
        rubric=["Normalise the 2-qubit state", f"Project onto |{m}⟩ for qubit 1: keep the |{m}0⟩, |{m}1⟩ terms",
                f"Probability = sum of their |amplitude|² (here {pm})", "Renormalise the surviving qubit-2 vector"],
        solution=(f"Surviving terms: ${state_tex([c0, c1] if m == 0 else [0, 0, c0, c1])}$. "
                  f"$P({m}) = {L(pm)}$. Renormalising gives qubit 2 in ${state_tex(post)}$."),
        misconceptions=[(vans(vec(c0, c1)), "forgot-to-renormalise"),
                        (vans(normalize(vec(amps[m], amps[m + 2])) if (amps[m] != 0 or amps[m + 2] != 0) else vec(0, 0)),
                         "measured-wrong-qubit")],
    )


# ============================================================================ qc.tensor
@gen("qc.tensor")
def tensor_states(rng, level):
    if level == 1:
        a, b = rng.choice(_SIMPLE), rng.choice(_SIMPLE)
        out = kron(STATES[a], STATES[b])
        return Problem(
            prompt=f"Write ${ket(a)}\\otimes{ket(b)}$ as a 4-component vector. " + CONV,
            answer=vans(out), kind="vector", answer_hint=HINT_V2,
            rubric=["Write each factor as a 2-vector", "Kronecker: (a0 b, a1 b) — each entry of the left vector multiplies the whole right vector"],
            solution=f"${vec_tex(STATES[a])}\\otimes{vec_tex(STATES[b])} = {vec_tex(out)}$.",
            misconceptions=[(vans(kron(STATES[b], STATES[a])), "wrong-tensor-order")],
        )
    if level == 2:
        p = vec(rand_amp(rng), rand_amp(rng))
        q = vec(rand_amp(rng), rand_amp(rng))
        np_, nq = sum(sp.Abs(x) ** 2 for x in p), sum(sp.Abs(x) ** 2 for x in q)
        out = kron(normalize(p), normalize(q))
        return Problem(
            prompt=(f"Let $\\lvert a\\rangle = \\frac{{1}}{{\\sqrt{{{np_}}}}}\\left({state_tex(p)}\\right)$ and "
                    f"$\\lvert b\\rangle = \\frac{{1}}{{\\sqrt{{{nq}}}}}\\left({state_tex(q)}\\right)$. "
                    "Compute $\\lvert a\\rangle\\otimes\\lvert b\\rangle$. " + CONV),
            answer=vans(out), kind="vector", answer_hint=HINT_V2,
            rubric=["Multiply the normalisation constants", "Coefficient of |xy⟩ is a_x · b_y",
                    "Keep the order: left factor index is the most significant bit"],
            solution=(f"Coefficient of $\\lvert xy\\rangle$ is $a_x b_y$: the result is ${vec_tex(out)}$ "
                      f"(norm 1 automatically, since $\\|a\\otimes b\\| = \\|a\\|\\|b\\|$)."),
            misconceptions=[(vans(kron(normalize(q), normalize(p))), "wrong-tensor-order"),
                            (vans(kron(p, q)), "forgot-normalisation")],
        )
    while True:
        p = vec(nz(rng, 1, 3), rand_amp(rng))
        q = vec(nz(rng, 1, 3), rand_amp(rng))
        if p[1] / p[0] != q[1] / q[0]:
            break
    full = kron(p, q)
    qn = normalize(q)
    return Problem(
        prompt=(f"The two-qubit state $\\lvert\\psi\\rangle \\propto {state_tex(full)}$ is a product state "
                "$\\lvert a\\rangle\\otimes\\lvert b\\rangle$. Find the normalised state $\\lvert b\\rangle$ of qubit 2, "
                "choosing the global phase so that its $\\lvert0\\rangle$ amplitude is real and positive."),
        answer=vans(qn), kind="vector", answer_hint=HINT_V1,
        rubric=["Group terms by qubit 1: c00|0⟩|0⟩ + c01|0⟩|1⟩ + …", "Factor: ratio c01/c00 = c11/c10 = b1/b0",
                "Build (b0, b1) ∝ (c00, c01) and normalise", "Fix the phase so b0 > 0"],
        solution=(f"Rows of the coefficient matrix are proportional: $(c_{{00}}, c_{{01}}) = ({L(full[0])}, {L(full[1])}) \\propto "
                  f"({L(q[0])}, {L(q[1])})$, so $\\lvert b\\rangle = {state_tex(qn)}$ and $\\lvert a\\rangle \\propto {state_tex(p)}$."),
        misconceptions=[(vans(normalize(p)), "wrong-tensor-order")],
    )


@gen("qc.tensor")
def tensor_gates(rng, level):
    if level == 1:
        A, B = rng.sample(["I", "X", "Z", "H", "S"], 2)
        M = kron(GATES[A], GATES[B])
        return Problem(
            prompt=f"Compute the $4\\times 4$ matrix ${A}\\otimes {B}$ (Kronecker product, ${A}$ acts on qubit 1). " + CONV,
            answer=mans(M), kind="matrix", answer_hint=HINT_M,
            rubric=["A⊗B is the block matrix [[a00 B, a01 B],[a10 B, a11 B]]", "Fill all 16 entries"],
            solution=f"${A}\\otimes {B} = {mat_tex(M)}$.",
            misconceptions=[(mans(kron(GATES[B], GATES[A])), "wrong-tensor-order")],
        )
    if level == 2:
        A, B = rng.sample(["X", "Y", "Z", "H", "S"], 2)
        bits = rng.choice(["00", "01", "10", "11"])
        out = kron(GATES[A] * STATES[bits[0]], GATES[B] * STATES[bits[1]])
        return Problem(
            prompt=f"Compute $({A}\\otimes {B}){ket(bits)}$ as a 4-vector. " + CONV,
            answer=vans(out), kind="vector", answer_hint=HINT_V2,
            rubric=["(A⊗B)(|x⟩⊗|y⟩) = (A|x⟩)⊗(B|y⟩)", "Apply each gate to its own qubit", "Take the Kronecker product"],
            solution=(f"$= ({A}{ket(bits[0])})\\otimes({B}{ket(bits[1])}) = {vec_tex(GATES[A] * STATES[bits[0]])}\\otimes"
                      f"{vec_tex(GATES[B] * STATES[bits[1]])} = {vec_tex(out)}$."),
            misconceptions=[(vans(kron(GATES[B] * STATES[bits[0]], GATES[A] * STATES[bits[1]])), "gates-on-wrong-qubits")],
        )
    g0, g1 = rng.sample(["X", "H", "Y", "SH"], 2)
    gm = {"X": X, "H": H, "Y": Y, "SH": S * H}
    gname = {"X": "X", "H": "H", "Y": "Y", "SH": "H then S"}
    s0, s1 = gm[g0] * vec(1, 0), gm[g1] * vec(1, 0)
    out = kron(s1, s0)          # Qiskit: index = q1*2 + q0
    wrong = kron(s0, s1)
    return Problem(
        prompt=(f"In Qiskit, a 2-qubit circuit starts in $\\lvert 00\\rangle$ and applies **{gname[g0]}** to qubit 0 and "
                f"**{gname[g1]}** to qubit 1. `Statevector(circuit)` lists amplitudes by index $= 2q_1 + q_0$, i.e. "
                "in the order of labels $\\lvert q_1 q_0\\rangle = \\lvert00\\rangle, \\lvert01\\rangle, \\lvert10\\rangle, \\lvert11\\rangle$ "
                "(little-endian). Give that statevector array."),
        answer=vans(out), kind="vector", answer_hint=HINT_V2,
        rubric=["Compute each single-qubit state", "Qiskit's statevector is |q1⟩⊗|q0⟩ (qubit 0 is the RIGHTMOST factor)",
                "Kronecker product in that order"],
        solution=(f"Qubit 0: ${state_tex(s0)}$; qubit 1: ${state_tex(s1)}$. Qiskit orders as $\\lvert q_1\\rangle\\otimes\\lvert q_0\\rangle$, "
                  f"so the array is ${vec_tex(out)}$. Writing $\\lvert q_0\\rangle\\otimes\\lvert q_1\\rangle$ (textbook order) "
                  "would give a permuted vector — a classic Qiskit bug."),
        misconceptions=[(vans(wrong), "little-endian-qiskit-ordering")],
    )


# ============================================================================ qc.multi_gates
@gen("qc.multi_gates")
def cnot_circuits(rng, level):
    if level == 1:
        bits = [rng.randint(0, 1), rng.randint(0, 1)]
        ops = [rng.choice(["CNOT12", "CNOT21", "X1", "X2"]) for _ in range(3)]
        b = list(bits)
        for o in ops:
            if o == "CNOT12":
                b[1] ^= b[0]
            elif o == "CNOT21":
                b[0] ^= b[1]
            elif o == "X1":
                b[0] ^= 1
            else:
                b[1] ^= 1
        names = {"CNOT12": "CNOT$_{1\\to2}$", "CNOT21": "CNOT$_{2\\to1}$", "X1": "$X$ on qubit 1", "X2": "$X$ on qubit 2"}
        res = f"{b[0]}{b[1]}"
        bw = list(bits)
        for o in ops:
            if o == "CNOT12":
                bw[0] ^= bw[1]
            elif o == "CNOT21":
                bw[1] ^= bw[0]
            elif o == "X1":
                bw[0] ^= 1
            else:
                bw[1] ^= 1
        return Problem(
            prompt=(f"Start in ${ket(f'{bits[0]}{bits[1]}')}$ and apply, in order: " + ", ".join(names[o] for o in ops) +
                    ". (CNOT$_{c\\to t}$ flips qubit $t$ iff qubit $c$ is 1.) Which basis state results? Type the two bits $q_1q_2$."),
            answer=res, kind="text", answer_hint="two bits, e.g. 10",
            rubric=["Track the classical bits through each gate", "CNOT: target ← target XOR control"],
            solution=f"Tracking the bits gives ${ket(res)}$ (CNOT acts as a reversible XOR on basis states).",
            misconceptions=[(f"{bw[0]}{bw[1]}", "control-target-swapped")],
        )
    if level == 2:
        a, b = rng.choice(_SIMPLE), rng.choice(_SIMPLE)
        while a in ("0", "1") and b in ("0", "1"):
            a, b = rng.choice(_SIMPLE), rng.choice(_SIMPLE)
        out = CNOT12 * kron(STATES[a], STATES[b])
        return Problem(
            prompt=(f"Compute CNOT$_{{1\\to2}}\\left({ket(a)}\\otimes{ket(b)}\\right)$ (control = qubit 1). " + CONV +
                    " On paper, decide whether the result is entangled."),
            answer=vans(out), kind="vector", answer_hint=HINT_V2,
            rubric=["Expand the product state in the computational basis",
                    "CNOT swaps the amplitudes of |10⟩ and |11⟩", "Collect terms"],
            solution=(f"${ket(a)}\\otimes{ket(b)} = {state_tex(kron(STATES[a], STATES[b]))}$; swapping the "
                      f"$\\lvert10\\rangle,\\lvert11\\rangle$ amplitudes gives ${state_tex(out)}$."),
            misconceptions=[(vans(CNOT21 * kron(STATES[a], STATES[b])), "control-target-swapped")],
        )
    choice = rng.randint(0, 3)
    if choice == 0:
        A, B = rng.choice(["I", "H", "X", "Z", "S"]), rng.choice(["I", "H", "X", "Z", "S"])
        U = CNOT12 * kron(GATES[A], GATES[B])
        wrong = kron(GATES[A], GATES[B]) * CNOT12
        desc = f"first ${A}\\otimes {B}$, then CNOT$_{{1\\to2}}$"
        expl = f"$U = \\mathrm{{CNOT}}_{{1\\to2}}({A}\\otimes {B})$"
    else:
        name, U, wrong, desc, expl = [
            ("cz", (kron(ID, H) * CNOT12 * kron(ID, H)), CNOT12, "$I\\otimes H$, then CNOT$_{1\\to2}$, then $I\\otimes H$",
             "$(I\\otimes H)\\mathrm{CNOT}(I\\otimes H) = \\mathrm{CZ} = \\mathrm{diag}(1,1,1,-1)$ because $HXH = Z$"),
            ("rev", (kron(H, H) * CNOT12 * kron(H, H)), CNOT12, "$H\\otimes H$, then CNOT$_{1\\to2}$, then $H\\otimes H$",
             "Hadamards on both sides reverse control and target: the result is CNOT$_{2\\to1}$"),
            ("swap", CNOT12 * CNOT21 * CNOT12, CNOT12, "CNOT$_{1\\to2}$, CNOT$_{2\\to1}$, CNOT$_{1\\to2}$",
             "Three alternating CNOTs implement SWAP (XOR-swap trick)"),
        ][choice - 1]
    return Problem(
        prompt=f"Find the $4\\times4$ unitary implemented by the circuit: {desc} (time order). " + CONV,
        answer=mans(U), kind="matrix", answer_hint=HINT_M,
        rubric=["Write each layer as a 4×4 matrix (Kronecker products for parallel gates)",
                "Multiply in reverse time order (last layer leftmost)", "Sanity check: unitary; identify a known gate"],
        solution=f"{expl}: $U = {mat_tex(U)}$.",
        misconceptions=[(mans(wrong), "reversed-gate-order")],
    )


@gen("qc.multi_gates")
def controlled_u(rng, level):
    if level == 1:
        g = rng.choice(["X", "Y", "Z", "H", "S", "T"])
        U = ctrl(GATES[g], True)
        return Problem(
            prompt=f"Write the $4\\times4$ matrix of controlled-${g}$ with **qubit 1 as control** and qubit 2 as target. " + CONV,
            answer=mans(U), kind="matrix", answer_hint=HINT_M,
            rubric=["C-U = |0⟩⟨0|⊗I + |1⟩⟨1|⊗U", "Block diagonal: diag(I, U)"],
            solution=f"$C\\text{{-}}{g} = \\lvert0\\rangle\\langle0\\rvert\\otimes I + \\lvert1\\rangle\\langle1\\rvert\\otimes {g} = {mat_tex(U)}$.",
            misconceptions=[(mans(ctrl(GATES[g], False)), "control-target-swapped")],
        )
    if level == 2:
        g, eig = rng.choice([("Z", "1"), ("S", "1"), ("T", "1"), ("X", "-"), ("Y", "-i"), ("Y", "+i"), ("Z", "0")])
        inp = kron(STATES["+"], STATES[eig])
        out = ctrl(GATES[g], True) * inp
        lam = cs((dag(STATES[eig]) * GATES[g] * STATES[eig])[0, 0])
        return Problem(
            prompt=(f"Apply controlled-${g}$ (control qubit 1) to ${ket('+')}\\otimes{ket(eig)}$. Give the output 4-vector. " + CONV +
                    " On paper: explain why the *control* qubit changes (phase kickback)."),
            answer=vans(out), kind="vector", answer_hint=HINT_V2,
            rubric=[f"{ket(eig)} is an eigenvector of {g}: find the eigenvalue λ",
                    "C-U(|0⟩+|1⟩)|u⟩/√2 = (|0⟩ + λ|1⟩)|u⟩/√2", "Expand the product into 4 amplitudes"],
            solution=(f"${g}{ket(eig)} = {L(lam)}{ket(eig)}$, so the output is "
                      f"$\\frac{{1}}{{\\sqrt2}}(\\lvert0\\rangle + ({L(lam)})\\lvert1\\rangle)\\otimes{ket(eig)} = {vec_tex(out)}$: "
                      "the eigenvalue is *kicked back* onto the control as a relative phase."),
            misconceptions=[(vans(inp), "ignored-phase-kickback")] if lam != 1 else [],
        )
    g = rng.choice(["X", "Y", "Z", "H", "S", "T"])
    U = ctrl(GATES[g], False)
    return Problem(
        prompt=(f"Derive the $4\\times4$ matrix of controlled-${g}$ with **qubit 2 as control** and **qubit 1 as target**. " + CONV),
        answer=mans(U), kind="matrix", answer_hint=HINT_M,
        rubric=["Write C-U = I⊗|0⟩⟨0| + U⊗|1⟩⟨1| (projector on the control's slot)",
                "Compute each Kronecker product", "Check: columns for |x0⟩ unchanged"],
        solution=(f"$I\\otimes\\lvert0\\rangle\\langle0\\rvert + {g}\\otimes\\lvert1\\rangle\\langle1\\rvert = {mat_tex(U)}$. "
                  "It is *not* block-diagonal: the target is the most significant qubit."),
        misconceptions=[(mans(ctrl(GATES[g], True)), "control-target-swapped")],
    )


# ============================================================================ qc.entanglement
@gen("qc.entanglement")
def product_test(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            p, q = vec(nz(rng, -3, 3), nz(rng, -3, 3)), vec(nz(rng, -3, 3), rand_amp(rng))
            amps = list(kron(p, q))
        else:
            while True:
                amps = [rand_amp(rng, False) if rng.random() < 0.85 else sp.Integer(0) for _ in range(4)]
                if amps[0] * amps[3] - amps[1] * amps[2] != 0:
                    break
        det = cs(amps[0] * amps[3] - amps[1] * amps[2])
        ans = "B" if det == 0 else "A"
        return Problem(
            prompt=(f"Is the two-qubit state $\\lvert\\psi\\rangle \\propto {state_tex(amps)}$ entangled?\n\n"
                    "A) entangled  B) product (not entangled)"),
            answer=ans, kind="choice",
            rubric=["Arrange amplitudes as a 2×2 matrix [[c00, c01],[c10, c11]]",
                    "Product state ⇔ the matrix has rank 1 ⇔ c00·c11 − c01·c10 = 0", "Conclude"],
            solution=f"$c_{{00}}c_{{11}} - c_{{01}}c_{{10}} = {L(det)}$" + (": product state (rank-1 coefficient matrix)." if det == 0 else " $\\neq 0$: entangled."),
            misconceptions=[("B" if ans == "A" else "A", "entanglement-det-test")],
        )
    if level == 2:
        while True:
            amps = [rand_amp(rng) for _ in range(4)]
            if amps[0] * amps[3] - amps[1] * amps[2] != 0:
                break
        n2 = sum(sp.Abs(x) ** 2 for x in amps)
        C = cs(2 * sp.Abs(amps[0] * amps[3] - amps[1] * amps[2]) / n2)
        return Problem(
            prompt=(f"For the pure state $\\lvert\\psi\\rangle = \\frac{{1}}{{\\sqrt{{{n2}}}}}\\left({state_tex(amps)}\\right)$ "
                    "compute the concurrence $C = 2\\,|c_{00}c_{11} - c_{01}c_{10}|$ (normalised amplitudes). "
                    "$C=0$ for product states, $C=1$ for Bell states."),
            answer=fmt(C), kind="num",
            rubric=["Use normalised amplitudes (divide by the norm — it enters squared)",
                    "Compute the determinant c00c11 − c01c10", "Take 2|·|"],
            solution=(f"$c_{{00}}c_{{11}} - c_{{01}}c_{{10}} = \\frac{{{L(cs(amps[0] * amps[3] - amps[1] * amps[2]))}}}{{{n2}}}$, "
                      f"so $C = {L(C)}$."),
            misconceptions=[(fmt(cs(C * n2)), "forgot-normalisation"), (fmt(cs(C / 2)), "dropped-factor-two")],
        )
    while True:
        a, b, c = nz(rng, 1, 4), rand_amp(rng, True, -4, 4), rand_amp(rng, True, -4, 4)
        t = cs(b * c / a)
        if t != cs(a * b / c):
            break
    return Problem(
        prompt=(f"For which value of $t$ is the (unnormalised) two-qubit state ${a}\\lvert00\\rangle + ({L(b)})\\lvert01\\rangle + "
                f"({L(c)})\\lvert10\\rangle + t\\lvert11\\rangle$, **not** entangled? Prove the product condition on paper."),
        answer=fmt(t), kind="expr", answer_hint="a number, e.g. 3/2 or 2*i",
        rubric=["Show (a0|0⟩+a1|1⟩)⊗(b0|0⟩+b1|1⟩) has c00c11 = c01c10",
                "Conversely, rank-1 coefficient matrix ⇒ product", f"Solve {a}·t = ({L(b)})({L(c)})"],
        solution=(f"Product $\\iff c_{{00}}c_{{11}} = c_{{01}}c_{{10}} \\iff {a}t = ({L(b)})({L(c)}) \\iff t = {L(t)}$."),
        misconceptions=[(fmt(cs(a * b / c)), "det-wrong-pairing"), (fmt(cs(-b * c / a)), "sign-error")],
    )


def _bell(name):
    return {"Phi+": vec(1, 0, 0, 1) / R2, "Phi-": vec(1, 0, 0, -1) / R2,
            "Psi+": vec(0, 1, 1, 0) / R2, "Psi-": vec(0, 1, -1, 0) / R2}[name]


BELL_TEX = {"Phi+": r"\lvert\Phi^+\rangle", "Phi-": r"\lvert\Phi^-\rangle", "Psi+": r"\lvert\Psi^+\rangle", "Psi-": r"\lvert\Psi^-\rangle"}


def _obs(a):
    return sp.cos(a) * Z + sp.sin(a) * X


@gen("qc.entanglement")
def bell_states(rng, level):
    if level == 1:
        bits = rng.choice(["00", "01", "10", "11"])
        out = CNOT12 * kron(H, ID) * kron(STATES[bits[0]], STATES[bits[1]])
        return Problem(
            prompt=(f"Apply $H$ to qubit 1, then CNOT$_{{1\\to2}}$, to ${ket(bits)}$. Give the output vector. " + CONV),
            answer=vans(out), kind="vector", answer_hint=HINT_V2,
            rubric=["H on qubit 1: |x⟩ → (|0⟩ + (−1)^x|1⟩)/√2", "CNOT copies qubit 1 into qubit 2 (XOR)",
                    "Identify which Bell state results"],
            solution=(f"$(H\\otimes I){ket(bits)} = {state_tex(kron(H, ID) * kron(STATES[bits[0]], STATES[bits[1]]))}$, then CNOT gives "
                      f"${state_tex(out)}$ — the Bell basis is $\\lvert\\beta_{{xy}}\\rangle = \\frac{{1}}{{\\sqrt2}}(\\lvert0y\\rangle + (-1)^x\\lvert1\\bar y\\rangle)$."),
            misconceptions=[(vans(CNOT21 * kron(H, ID) * kron(STATES[bits[0]], STATES[bits[1]])), "control-target-swapped")],
        )
    if level == 2:
        bname = rng.choice(["Phi+", "Psi-", "Phi-", "Psi+"])
        a, b = rng.choice(ANGLES[:8] + [0]), rng.choice(ANGLES[:8] + [0])
        E = expval(kron(_obs(a), _obs(b)), _bell(bname))
        return Problem(
            prompt=(f"Alice and Bob share ${BELL_TEX[bname]}$. Alice measures $A = \\cos\\alpha\\, Z + \\sin\\alpha\\, X$ and Bob "
                    f"$B = \\cos\\beta\\, Z + \\sin\\beta\\, X$ with $\\alpha = {L(a)}$, $\\beta = {L(b)}$ (each outcome $\\pm1$). "
                    "Compute the correlation $E = \\langle A\\otimes B\\rangle$."),
            answer=fmt(E), kind="num",
            rubric=["Expand A⊗B into ZZ, ZX, XZ, XX terms",
                    f"Evaluate ⟨ZZ⟩, ⟨XX⟩, ⟨ZX⟩, ⟨XZ⟩ on the Bell state", "Combine with the cos/sin weights (often a cos(α ± β) form)"],
            solution=(f"For ${BELL_TEX[bname]}$: $\\langle ZZ\\rangle = {expval(kron(Z, Z), _bell(bname))}$, "
                      f"$\\langle XX\\rangle = {expval(kron(X, X), _bell(bname))}$, $\\langle ZX\\rangle = \\langle XZ\\rangle = 0$. "
                      f"So $E = {L(E)}$."),
            misconceptions=[(fmt(cs(sp.cos(a - b))), "assumed-phi-plus-correlations")] if bname != "Phi+" else
            [(fmt(cs(sp.cos(a + b))), "sign-of-xx-correlation")],
        )
    a0 = rng.choice([0, PI / 4, PI / 2])
    a1 = a0 + rng.choice([PI / 2, PI / 4, -PI / 2])
    b0 = a0 + rng.choice([PI / 4, -PI / 4, PI / 6])
    b1 = b0 + rng.choice([-PI / 2, PI / 2, PI / 3])
    st = _bell("Phi+")
    Ef = lambda a, b: expval(kron(_obs(a), _obs(b)), st)  # noqa: E731
    Sv = cs(Ef(a0, b0) + Ef(a0, b1) + Ef(a1, b0) - Ef(a1, b1))
    return Problem(
        prompt=(f"On $\\lvert\\Phi^+\\rangle$, measurements in the $xz$-plane at angle $\\theta$ "
                "($\\cos\\theta Z + \\sin\\theta X$) give $E(\\alpha,\\beta) = \\cos(\\alpha-\\beta)$ — derive this on paper. "
                f"With $a = {L(a0)}$, $a' = {L(a1)}$, $b = {L(b0)}$, $b' = {L(b1)}$, compute the CHSH value "
                "$S = E(a,b) + E(a,b') + E(a',b) - E(a',b')$. (Local hidden variables: $|S|\\le 2$; Tsirelson: $|S|\\le 2\\sqrt2$.)"),
        answer=fmt(Sv), kind="num",
        rubric=["Derive E(α,β) = ⟨ZZ⟩cosα cosβ + ⟨XX⟩ sinα sinβ = cos(α−β)", "Evaluate the four correlators",
                "Combine with the sign pattern + + + −", "Compare with 2 and 2√2"],
        solution=(f"$S = \\cos({L(a0 - b0)}) + \\cos({L(a0 - b1)}) + \\cos({L(a1 - b0)}) - \\cos({L(a1 - b1)}) = {L(Sv)}$"
                  + (" — a Bell violation." if abs(float(Sv)) > 2 + 1e-9 else ".")),
        misconceptions=[(fmt(cs(Ef(a0, b0) + Ef(a0, b1) + Ef(a1, b0) + Ef(a1, b1))), "chsh-sign-pattern")],
    )


# ============================================================================ qc.teleportation
_CORR = {"00": "A", "01": "B", "10": "C", "11": "D"}


@gen("qc.teleportation")
def teleportation(rng, level):
    m1, m2 = rng.randint(0, 1), rng.randint(0, 1)
    if level == 1:
        key = f"{m1}{m2}"
        wrong = {"A": "B", "B": "C", "C": "B", "D": "A"}[_CORR[key]]
        return Problem(
            prompt=("Standard teleportation (N&C Fig. 1.13): Alice does CNOT (her qubit $\\to$ her Bell half), then $H$ on her qubit, "
                    f"and measures: $M_1 = {m1}$ (the qubit being teleported) and $M_2 = {m2}$ (her half of the Bell pair). "
                    "Which correction must Bob apply?\n\nA) nothing  B) $X$  C) $Z$  D) $X$ then $Z$ (i.e. $ZX$)"),
            answer=_CORR[key], kind="choice",
            rubric=["Rewrite |ψ⟩|Φ+⟩ in Alice's Bell basis (or track the circuit)",
                    "Bob's qubit is X^{M2} Z^{M1}|ψ⟩", "Correction: Z^{M1} X^{M2} (X first, then Z)"],
            solution=(f"Bob holds $X^{{{m2}}}Z^{{{m1}}}\\lvert\\psi\\rangle$, so he applies $Z^{{{m1}}}X^{{{m2}}}$: answer {_CORR[key]}."),
            misconceptions=[(wrong, "swapped-measurement-bits")],
        )
    if level == 2:
        while True:
            al, be = nz(rng, 1, 4), rand_amp(rng, True, -4, 4)
            if al != sp.Abs(be):
                break
        n2 = al ** 2 + sp.Abs(be) ** 2
        psi = vec(al, be) / sp.sqrt(n2)
        bob = (X ** m2) * (Z ** m1) * psi
        return Problem(
            prompt=(f"Alice teleports $\\lvert\\psi\\rangle = \\frac{{1}}{{\\sqrt{{{n2}}}}}\\left({state_tex([al, be])}\\right)$ "
                    f"using $\\lvert\\Phi^+\\rangle$ and the standard circuit (CNOT, then $H$ on her qubit). She obtains "
                    f"$M_1 = {m1}$, $M_2 = {m2}$. What is Bob's qubit **before** any correction?"),
            answer=vans(bob), kind="vector", answer_hint=HINT_V1,
            rubric=["Write |ψ⟩|Φ+⟩ after CNOT and H as ½ Σ |m1 m2⟩ ⊗ X^{m2}Z^{m1}|ψ⟩",
                    f"Pick the branch m1m2 = {m1}{m2}", "Apply Z^{m1} first, then X^{m2}, to (α, β)"],
            solution=(f"Branch ${ket(f'{m1}{m2}')}$: Bob holds $X^{{{m2}}}Z^{{{m1}}}\\lvert\\psi\\rangle = {state_tex(bob)}$. "
                      f"Each of the four outcomes has probability $1/4$, independent of $\\lvert\\psi\\rangle$ — "
                      "so no information travels without the 2 classical bits."),
            misconceptions=[(vans(psi), "assumed-already-corrected")],
        )
    b1, b2 = rng.randint(0, 1), rng.randint(0, 1)
    U = (Z ** b1) * (X ** b2)
    out = kron(U, ID) * _bell("Phi+")
    return Problem(
        prompt=(f"Superdense coding: Alice and Bob share $\\lvert\\Phi^+\\rangle$ (Alice holds qubit 1). To send the bits "
                f"$b_1b_2 = {b1}{b2}$ Alice applies $Z^{{{b1}}}X^{{{b2}}}$ ($X^{{{b2}}}$ first) to her qubit and sends it to Bob. "
                "Give the two-qubit state Bob now holds. " + CONV + " On paper, show that CNOT$_{1\\to2}$ then $H\\otimes I$ maps it to "
                f"${ket(f'{b1}{b2}')}$."),
        answer=vans(out), kind="vector", answer_hint=HINT_V2,
        rubric=["X on qubit 1 flips the first bit of each term", "Z on qubit 1 adds a − sign to terms with first bit 1",
                "Identify the resulting Bell state", "Decoding: CNOT then H returns |b1 b2⟩"],
        solution=(f"$(Z^{{{b1}}}X^{{{b2}}}\\otimes I)\\lvert\\Phi^+\\rangle = {state_tex(out)}$. The four encodings give the four "
                  "orthogonal Bell states, so 1 qubit + shared entanglement carries 2 bits."),
        misconceptions=[(vans(_bell("Phi+")), "forgot-encoding-gate")] if (b1 or b2) else [],
    )


# ============================================================================ qc.oracles
def _table_tex(n, f):
    rows = " & ".join(format(x, f"0{n}b") for x in range(2 ** n))
    vals = " & ".join(str(v) for v in f)
    return (r"$$\begin{array}{c|" + "c" * (2 ** n) + r"} x & " + rows + r" \\ \hline f(x) & " + vals + r"\end{array}$$")


@gen("qc.oracles")
def deutsch_jozsa(rng, level):
    if level == 1:
        n = rng.choice([2, 3])
        N = 2 ** n
        if rng.random() < 0.4:
            f = [rng.randint(0, 1)] * N
        else:
            ones = set(rng.sample(range(N), N // 2))
            f = [1 if x in ones else 0 for x in range(N)]
        amp = sp.Rational(sum((-1) ** v for v in f), N)
        p = amp ** 2
        return Problem(
            prompt=(f"Run Deutsch–Jozsa on $n={n}$ input qubits with the oracle $f$ below "
                    "($U_f\\lvert x\\rangle\\lvert y\\rangle = \\lvert x\\rangle\\lvert y\\oplus f(x)\\rangle$, ancilla in $\\lvert-\\rangle$). "
                    f"What is the probability that the input register is measured as all zeros?\n\n{_table_tex(n, f)}"),
            answer=fmt(p), kind="num",
            rubric=["Phase kickback: |x⟩|−⟩ → (−1)^{f(x)}|x⟩|−⟩",
                    "Amplitude of |0…0⟩ after the final H^{⊗n} is (1/2^n) Σ_x (−1)^{f(x)}",
                    "Constant ⇒ 1, balanced ⇒ 0"],
            solution=(f"Amplitude $= \\frac{{1}}{{{N}}}\\sum_x(-1)^{{f(x)}} = {L(amp)}$, so $P(0\\ldots0) = {L(p)}$ — "
                      + ("constant." if p == 1 else "balanced.")),
            misconceptions=[(fmt(1 - p), "constant-balanced-swapped")],
        )
    if level == 2:
        n = rng.choice([3, 4])
        s = [rng.randint(0, 1) for _ in range(n)]
        if not any(s):
            s[rng.randrange(n)] = 1
        f = [sum(int(b) * si for b, si in zip(format(x, f"0{n}b"), s)) % 2 for x in range(2 ** n)]
        sstr = "".join(map(str, s))
        return Problem(
            prompt=(f"Bernstein–Vazirani: $f(x) = s\\cdot x \\bmod 2$ for a hidden $s\\in\\{{0,1\\}}^{n}$. The truth table is below "
                    "(bit strings written $x_1x_2\\ldots$). One quantum query with $H^{\\otimes n}$, $U_f$ (ancilla $\\lvert-\\rangle$), "
                    f"$H^{{\\otimes n}}$ yields $s$ with certainty. What string is measured?\n\n{_table_tex(n, f)}"),
            answer=sstr, kind="text", answer_hint=HINT_BITS,
            rubric=["Phase oracle gives (1/√2^n) Σ_x (−1)^{s·x}|x⟩", "This equals H^{⊗n}|s⟩, so H^{⊗n} returns |s⟩",
                    "Read s off the table: s_i = f(e_i) (inputs with a single 1)"],
            solution=(f"$s_i = f(e_i)$: the single-1 inputs give $s = {sstr}$; check $f(x) = s\\cdot x$ on the other rows. "
                      "Classically you would need $n$ queries."),
            misconceptions=[(sstr[::-1], "little-endian-qiskit-ordering")],
        )
    n = rng.choice([2, 3])
    N = 2 ** n
    while True:
        k = rng.randint(1, N - 1)
        if k != N // 2:
            break
    ones = set(rng.sample(range(N), k))
    f = [1 if x in ones else 0 for x in range(N)]
    amp = sp.Rational(N - 2 * k, N)
    return Problem(
        prompt=(f"The Deutsch–Jozsa circuit is run on $n={n}$ qubits with an $f$ that is **neither** constant nor balanced "
                f"(table below). Derive the probability of measuring $0^{n}$ in terms of the number $k$ of inputs with $f(x)=1$, and "
                f"evaluate it.\n\n{_table_tex(n, f)}"),
        answer=fmt(amp ** 2), kind="num",
        rubric=["Amplitude of |0^n⟩ = (1/2^n) Σ_x (−1)^{f(x)}", "Σ_x (−1)^{f(x)} = (2^n − k) − k",
                "Probability = ((2^n − 2k)/2^n)²"],
        solution=(f"$k={k}$: amplitude $= ({N} - 2\\cdot{k})/{N} = {L(amp)}$, probability ${L(amp ** 2)}$."),
        misconceptions=[(fmt(abs(amp)), "forgot-amplitude-squared"), (fmt(sp.Rational(N - k, N)), "counted-zeros-only")],
    )


# ============================================================================ qc.grover
@gen("qc.grover")
def grover_counts(rng, level):
    if level == 1:
        n = rng.randint(3, 16)
        Mk = rng.choice([1, 1, 2, 4])
        N = 2 ** n
        k = math.floor(math.pi / 4 * math.sqrt(N / Mk))
        return Problem(
            prompt=(f"Grover search over $N = 2^{{{n}}} = {N}$ items with $M = {Mk}$ marked item(s). Using the standard estimate "
                    "$k^* = \\lfloor \\frac{\\pi}{4}\\sqrt{N/M}\\rfloor$, how many Grover iterations should you run?"),
            answer=str(k), kind="num",
            rubric=["Compute N/M", "Take the square root", "Multiply by π/4 ≈ 0.7854 and round down"],
            solution=(f"$\\frac{{\\pi}}{{4}}\\sqrt{{{N}/{Mk}}} \\approx {math.pi / 4 * math.sqrt(N / Mk):.3f}$, so $k^* = {k}$ — "
                      "versus $\\sim N/M$ classical queries (Grover 1996)."),
            misconceptions=[(str(N // Mk), "classical-query-count"), (str(math.floor(math.sqrt(N / Mk))), "dropped-pi-over-4")],
        )
    if level == 2:
        n = rng.randint(3, 7)
        N = 2 ** n
        Mk = rng.choice([1, 1, 2, 3])
        kopt = math.floor(math.pi / 4 * math.sqrt(N / Mk))
        k = rng.randint(1, kopt + 1)
        th = math.asin(math.sqrt(Mk / N))
        p = math.sin((2 * k + 1) * th) ** 2
        return Problem(
            prompt=(f"Grover search: $N = {N}$, $M = {Mk}$ marked, $k = {k}$ iterations. With $\\sin\\theta = \\sqrt{{M/N}}$, "
                    "the success probability is $\\sin^2((2k+1)\\theta)$. Compute it (3 decimals)."),
            answer=f"{p:.6f}", kind="num", tol=2e-3, answer_hint="a decimal, e.g. 0.945",
            rubric=["θ = arcsin(√(M/N)) (radians!)", "Each iteration rotates by 2θ in the (good, bad) plane",
                    "Evaluate sin²((2k+1)θ)"],
            solution=(f"$\\theta = \\arcsin\\sqrt{{{Mk}/{N}}} = {th:.5f}$, $(2k+1)\\theta = {(2 * k + 1) * th:.5f}$, "
                      f"$P = {p:.4f}$." + (" Note the overshoot past the optimum." if k > kopt else "")),
            misconceptions=[(f"{math.sin(k * th) ** 2:.6f}", "wrong-rotation-angle"),
                            (f"{math.sin((2 * k + 1) * math.sqrt(Mk / N)) ** 2:.6f}", "skipped-arcsin")],
        )
    n = rng.randint(2, 6)
    N = 2 ** n
    amp = sp.Rational(3 * N - 4, N) / sp.sqrt(N)
    p = cs(amp ** 2)
    return Problem(
        prompt=(f"Grover with $N = {N}$, one marked item $w$, starting from the uniform superposition. Using "
                "*inversion about the mean* (oracle flips $w$'s sign, diffuser maps $a_x \\mapsto 2\\bar a - a_x$), derive "
                "the probability of measuring $w$ after **one** iteration."),
        answer=fmt(p), kind="num",
        rubric=["Start: all amplitudes 1/√N", "Oracle: a_w = −1/√N; new mean = (N − 2)/(N√N)",
                "Diffuser: a_w → 2·mean + 1/√N = (3N − 4)/(N√N)", "Square it: (3 − 4/N)²/N"],
        solution=(f"Mean after oracle $\\bar a = \\frac{{N-2}}{{N\\sqrt N}}$; $a_w' = 2\\bar a + \\frac{{1}}{{\\sqrt N}} = "
                  f"\\frac{{3N-4}}{{N\\sqrt N}} = {L(cs(amp))}$, so $P = {L(p)}$ (≈ {float(p):.4f}; equals $\\sin^2 3\\theta$)."),
        misconceptions=[(fmt(cs(amp)), "forgot-amplitude-squared"), (fmt(sp.Rational(9, N)), "approximated-small-theta")],
    )


@gen("qc.grover")
def grover_diffuser(rng, level):
    if level == 1:
        a = [rng.randint(-3, 4) for _ in range(4)]
        mean = sp.Rational(sum(a), 4)
        out = [2 * mean - x for x in a]
        return Problem(
            prompt=(f"Apply the Grover diffuser $D = 2\\lvert s\\rangle\\langle s\\rvert - I$ on $N=4$ "
                    f"($\\lvert s\\rangle$ uniform) to the (unnormalised) amplitude vector $({', '.join(map(str, a))})$."),
            answer=[fmt(x) for x in out], kind="vector", answer_hint="4 numbers, e.g. [1/2, 3/2, -1, 0]",
            rubric=["⟨s|a⟩|s⟩ = (mean of the a_x) × (1,1,1,1)", "D a = 2·mean − a_x componentwise"],
            solution=f"Mean $= {L(mean)}$; $a_x \\mapsto 2\\cdot{L(mean)} - a_x$: $({', '.join(L(x) for x in out)})$.",
            misconceptions=[([fmt(x - mean) for x in a], "reflected-wrong-way"), ([fmt(mean - x) for x in a], "missing-factor-two")],
        )
    if level == 2:
        n = rng.choice([2, 3])
        N = 2 ** n
        w = rng.randrange(N)
        a = [sp.Integer(1) / sp.sqrt(N)] * N
        a[w] = -a[w]
        mean = sum(a) / N
        out = [cs(2 * mean - x) for x in a]
        wl = format(w, f"0{n}b")
        return Problem(
            prompt=(f"$n={n}$ qubits start in $H^{{\\otimes {n}}}\\lvert 0\\rangle^{{\\otimes {n}}}$. Apply one Grover iteration "
                    f"(oracle marking ${ket(wl)}$, then $D = 2\\lvert s\\rangle\\langle s\\rvert - I$). Give the full {N}-component state "
                    "vector in the order $\\lvert 0\\ldots0\\rangle, \\ldots, \\lvert 1\\ldots1\\rangle$."),
            answer=[fmt(x) for x in out], kind="vector", answer_hint=f"{N} amplitudes, e.g. [{', '.join(['0'] * (N - 1) + ['1'])}]",
            rubric=["Uniform amplitudes 1/√N", f"Oracle flips the sign of index {w}",
                    "Mean = (N−2)/(N√N)", "Each amplitude a → 2·mean − a"],
            solution=f"Result: $({', '.join(L(x) for x in out)})$" + (" — for $N=4$ one iteration finds $w$ with certainty." if N == 4 else "."),
            misconceptions=[([fmt(cs(-x)) for x in out], "global-sign-convention")],
        )
    n = rng.randint(3, 8)
    N = 2 ** n
    return Problem(
        prompt=(f"For $N = 2^{{{n}}} = {N}$ items, for which number $M$ of marked items does a **single** Grover iteration "
                "succeed with probability exactly 1? Derive it from $\\sin^2(3\\theta)$ with $\\sin^2\\theta = M/N$."),
        answer=str(N // 4), kind="num",
        rubric=["Success after k=1: sin²(3θ) = 1", "So 3θ = π/2, θ = π/6", "sin²(π/6) = 1/4 = M/N"],
        solution=f"$3\\theta = \\pi/2 \\Rightarrow \\sin^2\\theta = 1/4 \\Rightarrow M = N/4 = {N // 4}$.",
        misconceptions=[(str(N // 2), "inverted-angle"), ("1", "assumed-single-marked")],
    )


# ============================================================================ qc.qft
@gen("qc.qft")
def qft_states(rng, level):
    if level == 1:
        j = rng.randrange(4)
        out = qft_matrix(4) * sp.Matrix([1 if k == j else 0 for k in range(4)])
        wrong = dag(qft_matrix(4)) * sp.Matrix([1 if k == j else 0 for k in range(4)])
        jl = format(j, "02b")
        return Problem(
            prompt=(f"Compute the 2-qubit QFT of ${ket(jl)}$ (i.e. $j = {j}$, $N = 4$), with "
                    "$\\mathrm{QFT}\\lvert j\\rangle = \\frac{1}{\\sqrt N}\\sum_k e^{2\\pi i jk/N}\\lvert k\\rangle$. " + CONV),
            answer=vans(out), kind="vector", answer_hint=HINT_V2,
            rubric=["ω = e^{2πi/4} = i", "Amplitude of |k⟩ is ω^{jk}/2", "List k = 0..3"],
            solution=f"$\\omega = i$: amplitudes $i^{{{j}k}}/2$ give ${vec_tex(out)}$.",
            misconceptions=[(vans(wrong), "inverse-qft-sign")] if j % 2 == 1 else [],
        )
    if level == 2:
        j1, j2 = sorted(rng.sample(range(4), 2))
        sgn = rng.choice([1, -1])
        v = sp.Matrix([0] * 4)
        v[j1], v[j2] = 1 / R2, sgn / R2
        out = qft_matrix(4) * v
        return Problem(
            prompt=(f"Compute $\\mathrm{{QFT}}_4\\left({state_tex(v)}\\right)$. " + CONV +
                    " (Use linearity; on paper note which outcomes interfere destructively.)"),
            answer=vans(out), kind="vector", answer_hint=HINT_V2,
            rubric=["QFT is linear: transform each basis state", "Add the two images with the coefficients",
                    "Simplify; some amplitudes cancel"],
            solution=f"$= {vec_tex(out)}$, i.e. ${state_tex(out)}$.",
            misconceptions=[(vans(dag(qft_matrix(4)) * v), "inverse-qft-sign")],
        )
    N = rng.choice([8, 16])
    r = rng.choice([2, 4] if N == 8 else [2, 4, 8])
    x0 = rng.randrange(r)
    support = [x0 + m * r for m in range(N // r)]
    ks = [m * N // r for m in range(r)]
    return Problem(
        prompt=(f"A register of $\\log_2 {N}$ qubits holds the periodic state $\\frac{{1}}{{\\sqrt{{{N // r}}}}}\\sum_{{m=0}}^{{{N // r - 1}}} "
                f"\\lvert {x0} + {r}m\\rangle$ (period $r={r}$, offset ${x0}$). After applying $\\mathrm{{QFT}}_{{{N}}}$, which outcomes "
                "$k$ have nonzero probability? Derive it with a geometric sum."),
        answer=[str(k) for k in ks], kind="set", answer_hint="the set of k values, e.g. 0, 4",
        rubric=["Amplitude of |k⟩ ∝ e^{2πi x0 k/N} Σ_m e^{2πi r m k/N}",
                "Geometric sum is nonzero iff rk/N is an integer", "So k ∈ {0, N/r, 2N/r, …}; the offset only adds a phase"],
        solution=(f"$\\sum_{{m}} e^{{2\\pi i\\, {r} m k/{N}}} \\neq 0 \\iff {N // r} \\mid k$, giving $k\\in\\{{{', '.join(map(str, ks))}\\}}$, "
                  f"each with probability $1/{r}$. The offset ${x0}$ is only a phase — this is the heart of period finding."),
        misconceptions=[([str(s) for s in support], "confused-input-output") if support != ks else ([str(k + 1) for k in ks], "off-by-one")],
    )


@gen("qc.qft")
def qft_matrix_gen(rng, level):
    if level == 1:
        N = 8
        j, k = rng.randint(1, 7), rng.randint(1, 7)
        e = cs(sp.exp(2 * PI * I * j * k / N) / sp.sqrt(N))
        return Problem(
            prompt=(f"Give the matrix entry $F_{{{j}{k}}} = \\langle {k}\\vert\\mathrm{{QFT}}_{{8}}\\vert {j}\\rangle = "
                    f"\\frac{{1}}{{\\sqrt 8}}\\,\\omega^{{{j}\\cdot{k}}}$, $\\omega = e^{{2\\pi i/8}}$, in the form $a+bi$."),
            answer=fmt(e), kind="num", answer_hint="exact complex number, e.g. -1/4 + i/4",
            rubric=[f"Reduce jk = {j * k} mod 8", "ω^m = e^{iπm/4}: read off cos and sin", "Divide by √8 = 2√2"],
            solution=f"${j}\\cdot{k} \\equiv {j * k % 8} \\pmod 8$, so $F = e^{{i\\pi\\cdot{j * k % 8}/4}}/(2\\sqrt2) = {L(e)}$.",
            misconceptions=[(fmt(cs(sp.conjugate(e))), "inverse-qft-sign"), (fmt(cs(e * sp.sqrt(8))), "forgot-normalisation")],
        )
    if level == 2:
        which = rng.choice(["F", "Fdag", "F2"])
        F = qft_matrix(4)
        M = {"F": F, "Fdag": dag(F), "F2": F * F}[which]
        desc = {"F": "$\\mathrm{QFT}_4$", "Fdag": "$\\mathrm{QFT}_4^\\dagger$ (the inverse QFT)",
                "F2": "$\\mathrm{QFT}_4^2$ (apply the QFT twice)"}[which]
        wrong = {"F": dag(F), "Fdag": F, "F2": sp.eye(4)}[which]
        return Problem(
            prompt=f"Write the $4\\times4$ matrix of {desc}. (Basis order $\\lvert0\\rangle..\\lvert3\\rangle$ = $\\lvert00\\rangle..\\lvert11\\rangle$.)",
            answer=mans(M), kind="matrix", answer_hint=HINT_M,
            rubric=["F_{kj} = ω^{jk}/2 with ω = i", "Inverse: conjugate (ω → ω̄ = −i)",
                    "F² maps |j⟩ → |−j mod N⟩ (a permutation)"],
            solution=f"{desc} $= {mat_tex(M)}$.",
            misconceptions=[(mans(wrong), "inverse-qft-sign" if which != "F2" else "assumed-qft-involution")],
        )
    n = rng.randint(3, 12)
    return Problem(
        prompt=(f"The textbook QFT circuit on $n = {n}$ qubits (N&C Fig. 5.1) uses Hadamards, controlled-$R_k$ phase gates "
                "$R_k = \\mathrm{diag}(1, e^{2\\pi i/2^k})$ and final SWAPs to reverse qubit order. Give "
                "(number of controlled-$R_k$ gates, number of SWAPs)."),
        answer=[str(n * (n - 1) // 2), str(n // 2)], kind="list", answer_hint="two integers, e.g. 6, 2",
        rubric=["Qubit j (1-indexed) gets H then controlled-R_2..R_{n−j+1}: n − j gates",
                "Σ_j (n − j) = n(n−1)/2", "Reversing n qubits needs ⌊n/2⌋ swaps", "So the QFT is O(n²) gates vs O(n 2^n) for the FFT"],
        solution=f"$\\frac{{n(n-1)}}{{2}} = {n * (n - 1) // 2}$ controlled phases and $\\lfloor n/2\\rfloor = {n // 2}$ SWAPs.",
        misconceptions=[([str(n * n), str(n // 2)], "overcounted-phases"), ([str(n * (n - 1) // 2), str(n)], "overcounted-swaps")],
    )


# ============================================================================ qc.phase_estimation
@gen("qc.phase_estimation")
def phase_estimation(rng, level):
    if level == 1:
        t = rng.randint(2, 4)
        j = rng.randrange(1, 2 ** t)
        theta = sp.Rational(2 * j, 2 ** t) * PI
        bits = format(j, f"0{t}b")
        return Problem(
            prompt=(f"Phase estimation with $t = {t}$ counting qubits is applied to $U = P({L(theta)}) = \\mathrm{{diag}}(1, e^{{i{L(theta)}}})$ "
                    "and its eigenstate $\\lvert1\\rangle$. Writing the eigenvalue as $e^{2\\pi i\\varphi}$, the register yields "
                    "$2^t\\varphi$ exactly. Which bit string is measured (most significant bit first)?"),
            answer=bits, kind="text", answer_hint=HINT_BITS,
            rubric=["Eigenvalue e^{iθ} = e^{2πiφ} ⇒ φ = θ/2π", f"2^t φ = {j}", f"Write {j} in {t} binary digits"],
            solution=f"$\\varphi = {L(theta / (2 * PI))}$, $2^{t}\\varphi = {j}$, so the outcome is ${bits}$.",
            misconceptions=[(bits[::-1], "little-endian-qiskit-ordering")] if bits != bits[::-1] else [],
        )
    if level == 2:
        t = rng.randint(3, 5)
        m = rng.randrange(1, 2 ** t)
        bits = format(m, f"0{t}b")
        theta = cs(sp.Rational(2 * m, 2 ** t) * PI)
        return Problem(
            prompt=(f"A phase-estimation run with $t={t}$ counting qubits on an eigenstate of $U$ (eigenvalue $e^{{i\\theta}}$, "
                    f"$\\theta\\in[0,2\\pi)$) returns ${bits}$ with certainty (MSB first). What is $\\theta$?"),
            answer=fmt(theta), kind="num", answer_hint="an angle, e.g. 3*pi/4",
            rubric=[f"Convert the bits to an integer m = {m}", "φ = m / 2^t", "θ = 2πφ"],
            solution=f"$m = {m}$, $\\varphi = {m}/{2 ** t}$, $\\theta = 2\\pi\\varphi = {L(theta)}$.",
            misconceptions=[(fmt(sp.Rational(m, 2 ** t)), "forgot-2pi"),
                            (fmt(cs(sp.Rational(2 * int(bits[::-1], 2), 2 ** t) * PI)), "little-endian-qiskit-ordering")],
        )
    t = rng.choice([1, 2])
    phi = rng.choice([sp.Rational(1, 3), sp.Rational(1, 5), sp.Rational(1, 6), sp.Rational(2, 5), sp.Rational(3, 8), sp.Rational(2, 3)])
    T2 = 2 ** t
    m = int(round(float(phi) * T2)) % T2
    dlt = float(phi) - m / T2
    p = abs(sum(complex(math.cos(2 * math.pi * k * dlt), math.sin(2 * math.pi * k * dlt)) for k in range(T2)) / T2) ** 2
    return Problem(
        prompt=(f"Phase estimation with $t={t}$ counting qubit(s) on an eigenstate with eigenvalue $e^{{2\\pi i\\varphi}}$, "
                f"$\\varphi = {L(phi)}$ (not a multiple of $1/2^t$). Derive the probability of measuring $m = {m}$ "
                f"(the closest $t$-bit estimate), using $P(m) = \\left|\\frac{{1}}{{2^t}}\\sum_{{k=0}}^{{2^t-1}} e^{{2\\pi i k(\\varphi - m/2^t)}}\\right|^2$. "
                "Give it to 3 decimals."),
        answer=f"{p:.6f}", kind="num", tol=2e-3, answer_hint="a decimal, e.g. 0.684",
        rubric=["After controlled-U^{2^k} the register is (1/√2^t) Σ_k e^{2πiφk}|k⟩",
                "Inverse QFT amplitude of |m⟩ is a geometric sum", "Use |Σ| = |sin(π2^tδ)/sin(πδ)| with δ = φ − m/2^t",
                "Square and divide by 4^t"],
        solution=(f"$\\delta = \\varphi - {m}/{T2} = {L(phi - sp.Rational(m, T2))}$, "
                  f"$P = \\frac{{\\sin^2(\\pi 2^t\\delta)}}{{4^t\\sin^2(\\pi\\delta)}} \\approx {p:.4f}$ "
                  "(≥ 4/π² ≈ 0.405 is guaranteed for the nearest estimate)."),
        misconceptions=[(f"{math.sqrt(p):.6f}", "forgot-amplitude-squared"), ("1", "assumed-exact-phase")],
    )


# ============================================================================ qc.shor
_SEMIPRIMES = [15, 21, 33, 35, 39, 51, 55, 57, 65, 69, 77, 85, 91]


def _order(a, N):
    r, x = 1, a % N
    while x != 1:
        x = x * a % N
        r += 1
    return r


def _convergents(num, den):
    cf = []
    while den:
        q = num // den
        cf.append(q)
        num, den = den, num - q * den
    convs = []
    for k in range(len(cf)):
        frac = sp.Integer(cf[k])
        for c in reversed(cf[:k]):
            frac = c + 1 / frac
        convs.append(sp.Rational(frac))
    return cf, convs


@gen("qc.shor")
def shor_order(rng, level):
    if level == 1:
        while True:
            N = rng.choice(_SEMIPRIMES[:6])
            a = rng.randint(2, N - 2)
            if math.gcd(a, N) == 1 and _order(a, N) <= 12:
                break
        r = _order(a, N)
        pw = [pow(a, k, N) for k in range(1, r + 1)]
        return Problem(
            prompt=f"Find the order $r$ of $a = {a}$ modulo $N = {N}$: the smallest $r\\ge1$ with $a^r \\equiv 1 \\pmod{{N}}$.",
            answer=str(r), kind="num",
            rubric=["Compute successive powers a^k mod N (multiply the previous result by a, reduce)", "Stop at the first 1"],
            solution=f"Powers ${a}^k \\bmod {N}$: ${', '.join(map(str, pw))}$, so $r = {r}$.",
            misconceptions=[(str(r - 1), "off-by-one-order")] if r > 2 else [],
        )
    if level == 2:
        while True:
            N = rng.choice(_SEMIPRIMES)
            a = rng.randint(2, N - 2)
            if math.gcd(a, N) != 1:
                continue
            r = _order(a, N)
            if r % 2 == 0 and pow(a, r // 2, N) != N - 1 and r <= 12:
                break
        x = pow(a, r // 2, N)
        f1, f2 = math.gcd(x - 1, N), math.gcd(x + 1, N)
        return Problem(
            prompt=(f"Shor's classical post-processing: the order of $a={a}$ mod $N={N}$ is $r={r}$. Compute the two factors "
                    "$\\gcd(a^{r/2}-1, N)$ and $\\gcd(a^{r/2}+1, N)$."),
            answer=[str(f1), str(f2)], kind="set", answer_hint="two integers, e.g. 3, 5",
            rubric=[f"r is even and a^(r/2) ≢ −1 (mod N) — check it", f"a^(r/2) mod N = {x}",
                    "N | (a^{r/2}−1)(a^{r/2}+1) but divides neither factor", "Euclid's algorithm for each gcd"],
            solution=(f"${a}^{{{r // 2}}} \\equiv {x} \\pmod{{{N}}}$; $\\gcd({x - 1}, {N}) = {f1}$, $\\gcd({x + 1}, {N}) = {f2}$, "
                      f"and ${f1}\\cdot{f2} = {N}$ (Shor 1994, §5)."),
            misconceptions=[([str(x - 1), str(x + 1)], "skipped-gcd")],
        )
    while True:
        N = rng.choice(_SEMIPRIMES[:8])
        a = rng.randint(2, N - 2)
        if math.gcd(a, N) != 1:
            continue
        r = _order(a, N)
        if r < 3:
            continue
        s = rng.randint(1, r - 1)
        if math.gcd(s, r) != 1:
            continue
        t = (N * N - 1).bit_length()
        Q = 2 ** t
        y = round(s * Q / r)
        cf, convs = _convergents(y, Q)
        cands = [c for c in convs if c.q < N]
        if cands and cands[-1].q == r and y % Q != 0 and sp.Rational(y, Q) != sp.Rational(s, r):
            break
    conv_tex = ", ".join(L(c) for c in convs if c.q < N)
    return Problem(
        prompt=(f"Order finding for $a = {a}$, $N = {N}$ uses $t = {t}$ counting qubits ($Q = 2^t = {Q} \\ge N^2$). "
                f"The measurement gives $y = {y}$, so $y/Q \\approx s/r$. Expand $y/Q$ as a continued fraction, list its "
                f"convergents, and give the order $r$ (the largest convergent denominator below $N$; verify $a^r\\equiv1$)."),
        answer=str(r), kind="num",
        rubric=["Continued-fraction expansion via Euclid on (y, Q)", "Build convergents p_k/q_k",
                "Pick the convergent with denominator < N closest to y/Q", f"Verify {a}^r ≡ 1 (mod {N})"],
        solution=(f"${y}/{Q} = [{'; '.join(map(str, cf))}]$, convergents (denominator $<N$): ${conv_tex}$. "
                  f"The last is ${L(sp.Rational(s, r))}$, so $r = {r}$; indeed ${a}^{{{r}}} \\equiv 1 \\pmod {{{N}}}$. "
                  "Works because $|y/Q - s/r| \\le 1/(2Q) < 1/(2r^2)$ (Shor 1994, §5)."),
        misconceptions=[(str(s), "confused-numerator-denominator")],
    )


# ============================================================================ qc.density_noise
def _proj(v):
    return v * dag(v)


_FRACS = [sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(1, 4), sp.Rational(3, 4), sp.Rational(1, 5)]


@gen("qc.density_noise")
def density_matrices(rng, level):
    if level in (1, 2):
        s1, s2 = rng.sample(_SIMPLE, 2)
        p = rng.choice(_FRACS)
        rho = cmat(p * _proj(STATES[s1]) + (1 - p) * _proj(STATES[s2]))
        if level == 1:
            return Problem(
                prompt=(f"A source emits ${ket(s1)}$ with probability ${L(p)}$ and ${ket(s2)}$ with probability ${L(1 - p)}$. "
                        "Write the density matrix $\\rho$."),
                answer=mans(rho), kind="matrix", answer_hint=HINT_M,
                rubric=["ρ = Σ p_i |ψ_i⟩⟨ψ_i|", "Outer product |ψ⟩⟨ψ| = column × conjugated row", "Weighted sum; check Tr ρ = 1"],
                solution=f"$\\rho = {L(p)}{ket(s1)}{bra(s1)} + {L(1 - p)}{ket(s2)}{bra(s2)} = {mat_tex(rho)}$.",
                misconceptions=[(mans(cmat(p * STATES[s1] * STATES[s1].T + (1 - p) * STATES[s2] * STATES[s2].T)), "forgot-bra-conjugate")],
            )
        pur = cs((rho * rho).trace())
        return Problem(
            prompt=(f"For the ensemble ${L(p)}$ of ${ket(s1)}$ and ${L(1 - p)}$ of ${ket(s2)}$, compute the purity $\\mathrm{{Tr}}(\\rho^2)$."),
            answer=fmt(pur), kind="num",
            rubric=["Build ρ", "Tr ρ² = Σ_ij |ρ_ij|²", "Or: p² + (1−p)² + 2p(1−p)|⟨ψ1|ψ2⟩|²"],
            solution=(f"$\\rho = {mat_tex(rho)}$, $\\mathrm{{Tr}}\\rho^2 = {L(pur)}$ "
                      f"(overlap $|\\langle\\psi_1\\vert\\psi_2\\rangle|^2 = {L(cs(sp.Abs((dag(STATES[s1]) * STATES[s2])[0, 0]) ** 2))}$). "
                      "Purity 1 ⇔ pure; $1/2$ ⇔ maximally mixed."),
            misconceptions=[(fmt(p ** 2 + (1 - p) ** 2), "ignored-state-overlap"), ("1", "mixture-assumed-pure")],
        )
    while True:
        amps = [rand_amp(rng, True, -2, 2) if rng.random() < 0.85 else sp.Integer(0) for _ in range(4)]
        if amps[0] * amps[3] - amps[1] * amps[2] != 0:
            break
    v = normalize(sp.Matrix(amps))
    Mc = sp.Matrix([[v[0], v[1]], [v[2], v[3]]])
    which = rng.choice(["A", "B"])
    rhoA = cmat(Mc * dag(Mc))
    rhoB = cmat((dag(Mc) * Mc).T)
    ans, wrong = (rhoA, rhoB) if which == "A" else (rhoB, rhoA)
    keep = "1" if which == "A" else "2"
    n2 = sum(sp.Abs(x) ** 2 for x in amps)
    return Problem(
        prompt=(f"For $\\lvert\\psi\\rangle = \\frac{{1}}{{\\sqrt{{{n2}}}}}\\left({state_tex(amps)}\\right)$ compute the reduced density "
                f"matrix of qubit {keep}, $\\rho_{keep} = \\mathrm{{Tr}}_{{{'2' if keep == '1' else '1'}}}\\lvert\\psi\\rangle\\langle\\psi\\rvert$."),
        answer=mans(ans), kind="matrix", answer_hint=HINT_M,
        rubric=["Write |ψ⟩ = Σ c_ij |i⟩|j⟩ (C = coefficient matrix)",
                "Tr_2: (ρ_1)_{ii'} = Σ_j c_ij c*_{i'j}  (= C C†)", "Tr_1: (ρ_2)_{jj'} = Σ_i c_ij c*_{ij'}",
                "Check: Hermitian, trace 1; mixed because the state is entangled"],
        solution=f"$\\rho_{keep} = {mat_tex(ans)}$ (purity ${L(cs((ans * ans).trace()))} < 1$ since $\\lvert\\psi\\rangle$ is entangled).",
        misconceptions=[(mans(wrong), "traced-out-wrong-qubit")],
    )


@gen("qc.density_noise")
def noise_channels(rng, level):
    if level == 1:
        s = rng.choice(["0", "1", "+i", "-i"])
        p = rng.choice(_FRACS)
        rho = _proj(STATES[s])
        out = cmat((1 - p) * rho + p * X * rho * X)
        return Problem(
            prompt=f"A bit-flip channel $\\mathcal E(\\rho) = (1-p)\\rho + pX\\rho X$ with $p = {L(p)}$ acts on ${ket(s)}$. Give the output $\\rho$.",
            answer=mans(out), kind="matrix", answer_hint=HINT_M,
            rubric=["Write ρ = |ψ⟩⟨ψ|", "XρX swaps both rows and columns", "Mix with weights 1−p and p"],
            solution=f"$\\rho = {mat_tex(rho)}$, $X\\rho X = {mat_tex(X * rho * X)}$, output ${mat_tex(out)}$.",
            misconceptions=[(mans(cmat(p * rho + (1 - p) * X * rho * X)), "swapped-kraus-weights")],
        )
    if level == 2:
        s = rng.choice(["+", "-", "+i", "-i"])
        p, q = rng.choice(_FRACS), rng.choice(_FRACS)
        rho = _proj(STATES[s])
        r1 = (1 - p) * rho + p * X * rho * X
        out = cmat((1 - q) * r1 + q * Z * r1 * Z)
        return Problem(
            prompt=(f"${ket(s)}$ passes through a bit-flip channel ($p={L(p)}$) and then a phase-flip channel "
                    f"$\\rho\\mapsto(1-q)\\rho + qZ\\rho Z$ ($q={L(q)}$). Give the final density matrix."),
            answer=mans(out), kind="matrix", answer_hint=HINT_M,
            rubric=["Apply the bit flip: off-diagonal ρ01 ↔ ρ10 mix, diagonals mix",
                    "Phase flip multiplies off-diagonals by (1 − 2q)", "Diagonals unchanged by phase flip"],
            solution=f"After the bit flip: ${mat_tex(cmat(r1))}$; after the phase flip: ${mat_tex(out)}$.",
            misconceptions=[(mans(cmat(r1)), "skipped-second-channel")],
        )
    while True:
        a, b = nz(rng, 1, 3), nz(rng, -3, 3)
        if a != abs(b):
            break
    g = rng.choice(_FRACS)
    n2 = a * a + b * b
    v = vec(a, b) / sp.sqrt(n2)
    rho = _proj(v)
    E0 = sp.Matrix([[1, 0], [0, sp.sqrt(1 - g)]])
    E1 = sp.Matrix([[0, sp.sqrt(g)], [0, 0]])
    out = cmat(E0 * rho * dag(E0) + E1 * rho * dag(E1))
    return Problem(
        prompt=(f"Amplitude damping with $\\gamma = {L(g)}$ has Kraus operators $E_0 = \\begin{{pmatrix}}1&0\\\\0&\\sqrt{{1-\\gamma}}\\end{{pmatrix}}$, "
                f"$E_1 = \\begin{{pmatrix}}0&\\sqrt\\gamma\\\\0&0\\end{{pmatrix}}$. Apply it to $\\lvert\\psi\\rangle = \\frac{{1}}{{\\sqrt{{{n2}}}}}({state_tex([a, b])})$. "
                "Give the output density matrix (on paper: check $\\sum_k E_k^\\dagger E_k = I$)."),
        answer=mans(out), kind="matrix", answer_hint=HINT_M,
        rubric=["Verify completeness Σ E_k† E_k = I", "E0 ρ E0†: ρ11 → (1−γ)ρ11, off-diagonals × √(1−γ)",
                "E1 ρ E1† moves γρ11 into the |0⟩⟨0| entry", "Add; check trace 1"],
        solution=(f"$\\rho = {mat_tex(rho)}$ → ${mat_tex(out)}$: population flows to $\\lvert0\\rangle$ ($T_1$ decay) and "
                  "coherences shrink by $\\sqrt{1-\\gamma}$."),
        misconceptions=[(mans(cmat(E0 * rho * dag(E0))), "dropped-kraus-term")],
    )


# ============================================================================ qc.error_correction
_SYND = {0: "00", 1: "10", 2: "11", 3: "01"}


@gen("qc.error_correction")
def bitflip_code(rng, level):
    if level == 1:
        k = rng.randint(0, 3)
        err = "no error" if k == 0 else f"an $X$ error on qubit {k}"
        return Problem(
            prompt=(f"The 3-qubit bit-flip code encodes $\\alpha\\lvert000\\rangle + \\beta\\lvert111\\rangle$. With {err}, what syndrome "
                    "is measured? Give the bits $(s_1 s_2)$ for $Z_1Z_2$ and $Z_2Z_3$, with $s=1$ meaning eigenvalue $-1$."),
            answer=_SYND[k], kind="text", answer_hint="two bits, e.g. 10",
            rubric=["Z_iZ_j = +1 iff qubits i and j agree (parity check)",
                    "Both code words give the same syndrome — α, β are not revealed", "Combine the two parities"],
            solution=(f"Syndrome ${_SYND[k]}$. Lookup: 00 → no error, 10 → $X_1$, 11 → $X_2$, 01 → $X_3$ (N&C §10.1.1)."),
            misconceptions=[(_SYND[k][::-1], "syndrome-bit-order")] if _SYND[k] not in ("00", "11") else [],
        )
    if level == 2:
        p = rng.choice([sp.Rational(1, 10), sp.Rational(1, 5), sp.Rational(1, 4), sp.Rational(1, 3), sp.Rational(1, 20), sp.Rational(1, 100)])
        pl = 3 * p ** 2 - 2 * p ** 3
        return Problem(
            prompt=(f"Each of the 3 physical qubits of the bit-flip code suffers an independent $X$ error with probability $p = {L(p)}$. "
                    "Majority-vote correction fails iff 2 or more qubits flip. Compute the logical error probability exactly."),
            answer=fmt(pl), kind="num",
            rubric=["P(2 flips) = 3p²(1−p)", "P(3 flips) = p³", "Sum: 3p² − 2p³", "Compare with p: helps iff p < 1/2"],
            solution=f"$p_L = 3p^2(1-p) + p^3 = 3p^2 - 2p^3 = {L(pl)}$ (vs. unencoded $p = {L(p)}$).",
            misconceptions=[(fmt(3 * p ** 2), "forgot-triple-error"), (fmt(3 * p ** 2 * (1 - p)), "forgot-triple-error"),
                            (fmt(p ** 3), "only-counted-triple")],
        )
    if rng.random() < 0.5:
        k = rng.randint(1, 3)
        return Problem(
            prompt=(f"The 3-qubit **phase-flip** code encodes $\\alpha\\lvert{{+}}{{+}}{{+}}\\rangle + \\beta\\lvert{{-}}{{-}}{{-}}\\rangle$. "
                    f"A $Z$ error hits qubit {k}. Which stabilizers detect it, and what syndrome is measured for $(X_1X_2, X_2X_3)$ "
                    "($s=1$ means eigenvalue $-1$)? Derive it by conjugating the bit-flip code with $H^{\\otimes 3}$."),
            answer=_SYND[k], kind="text", answer_hint="two bits, e.g. 01",
            rubric=["HZH = X: the phase-flip code is the bit-flip code in the Hadamard basis",
                    "Stabilizers Z_iZ_j become X_iX_j", f"Z_{k} anticommutes with each X_iX_j containing qubit {k}"],
            solution=f"$Z_{k}$ anticommutes with the $X X$ checks that touch qubit {k}: syndrome ${_SYND[k]}$.",
            misconceptions=[("00", "z-error-invisible-to-x-checks")],
        )
    i, j = sorted(rng.sample([1, 2, 3], 2))
    return Problem(
        prompt=(f"In the 3-qubit bit-flip code, **two** $X$ errors occur, on qubits {i} and {j}, of "
                "$\\alpha\\lvert000\\rangle + \\beta\\lvert111\\rangle$. The decoder measures the syndrome and applies the "
                "single-qubit correction it indicates. What is the final state?\n\n"
                "A) $\\alpha\\lvert000\\rangle + \\beta\\lvert111\\rangle$ (corrected)  "
                "B) $\\alpha\\lvert111\\rangle + \\beta\\lvert000\\rangle$ (logical $X$)  "
                "C) $\\alpha\\lvert000\\rangle - \\beta\\lvert111\\rangle$ (logical $Z$)  "
                "D) the decoder reports a detected failure"),
        answer="B", kind="choice",
        rubric=[f"Syndrome of X_{i}X_{j} equals that of X_{6 - i - j}", f"Decoder applies X_{6 - i - j}",
                "Net operation X1X2X3 = logical X", "Distance-3 code: corrects 1 error, silently fails on 2"],
        solution=(f"$X_{i}X_{j}$ has the same syndrome as $X_{{{6 - i - j}}}$, so the decoder applies $X_{{{6 - i - j}}}$ and the net "
                  "error is $X_1X_2X_3 = \\bar X$: answer B. This is why $p_L \\approx 3p^2$."),
        misconceptions=[("A", "assumed-code-corrects-two"), ("D", "assumed-detection")],
    )


# ============================================================================ qc.variational
@gen("qc.variational")
def parameter_shift(rng, level):
    th = rng.choice(ANGLES)
    obs = rng.choice(["Z", "X"])
    f = (lambda t: sp.cos(t)) if obs == "Z" else (lambda t: sp.sin(t))
    if level == 1:
        val = cs(f(th))
        st = Ry(th) * vec(1, 0)
        return Problem(
            prompt=(f"Let $\\lvert\\psi(\\theta)\\rangle = R_y(\\theta)\\lvert0\\rangle$ with "
                    "$R_y(\\theta) = \\begin{pmatrix}\\cos\\frac\\theta2 & -\\sin\\frac\\theta2\\\\ \\sin\\frac\\theta2 & \\cos\\frac\\theta2\\end{pmatrix}$. "
                    f"Compute $\\langle {obs}\\rangle = \\langle\\psi\\vert {obs}\\vert\\psi\\rangle$ at $\\theta = {L(th)}$."),
            answer=fmt(val), kind="num",
            rubric=["|ψ⟩ = (cos θ/2, sin θ/2)", f"⟨{obs}⟩ = " + ("cos²(θ/2) − sin²(θ/2) = cos θ" if obs == "Z" else "2 sin(θ/2)cos(θ/2) = sin θ"),
                    "Evaluate"],
            solution=f"$\\lvert\\psi\\rangle = {vec_tex(st)}$, $\\langle {obs}\\rangle = {'\\cos' if obs == 'Z' else '\\sin'}\\theta = {L(val)}$.",
            misconceptions=[(fmt(cs(f(th / 2))), "bloch-half-angle")],
        )
    if level == 2:
        fp, fm = cs(f(th + PI / 2)), cs(f(th - PI / 2))
        g = cs((fp - fm) / 2)
        return Problem(
            prompt=(f"For $f(\\theta) = \\langle {obs}\\rangle$ in $R_y(\\theta)\\lvert0\\rangle$, use the parameter-shift rule "
                    "$f'(\\theta) = \\frac{1}{2}\\left[f(\\theta+\\frac\\pi2) - f(\\theta-\\frac\\pi2)\\right]$ to compute the gradient at "
                    f"$\\theta = {L(th)}$. (On paper: show why the rule is exact for gates $e^{{-i\\theta P/2}}$ with $P^2=I$.)"),
            answer=fmt(g), kind="num",
            rubric=["f(θ) = " + ("cos θ" if obs == "Z" else "sin θ"), "Evaluate at θ ± π/2 (two circuit runs)",
                    "Half the difference", "Matches the analytic derivative " + ("−sin θ" if obs == "Z" else "cos θ")],
            solution=(f"$f(\\theta\\pm\\frac\\pi2) = {L(fp)},\\ {L(fm)}$; $f'(\\theta) = \\frac12({L(fp)} - ({L(fm)})) = {L(g)}$. "
                      "Exact (not a finite difference) because $f$ is a sinusoid of period $2\\pi$ in $\\theta$."),
            misconceptions=[(fmt(cs(fp - fm)), "forgot-half-factor"), (fmt(cs(-g)), "gradient-sign")],
        )
    a, b = nz(rng, -3, 3), nz(rng, -3, 3)
    eta = rng.choice([sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 10), sp.Integer(1)])
    t0 = rng.choice([PI / 2, PI / 3, PI / 4, PI, PI / 6])
    E = lambda t: a * sp.cos(t) + b * sp.sin(t)  # noqa: E731
    grad = cs((E(t0 + PI / 2) - E(t0 - PI / 2)) / 2)
    t1 = cs(t0 - eta * grad)
    return Problem(
        prompt=(f"VQE toy: $H = {lin_tex([(a, 'Z\\otimes Z'), (b, 'X\\otimes I')])}$, ansatz $\\lvert\\psi(\\theta)\\rangle = "
                "(R_y(\\theta)\\otimes I)\\lvert00\\rangle$. Derive $E(\\theta) = \\langle H\\rangle$, compute "
                f"$E'(\\theta_0)$ with the parameter-shift rule at $\\theta_0 = {L(t0)}$, and take one gradient-descent step with "
                f"learning rate $\\eta = {L(eta)}$. Give $\\theta_1$."),
        answer=fmt(t1), kind="num", answer_hint="exact, e.g. pi/3 + 1/2",
        rubric=["⟨Z⊗Z⟩ = cos θ, ⟨X⊗I⟩ = sin θ on this ansatz", f"E(θ) = {a} cos θ + ({b}) sin θ",
                "E'(θ0) = [E(θ0+π/2) − E(θ0−π/2)]/2", "θ1 = θ0 − η E'(θ0)"],
        solution=(f"$E(\\theta) = {lin_tex([(a, '\\cos\\theta'), (b, '\\sin\\theta')])}$; $E'(\\theta_0) = {L(grad)}$; "
                  f"$\\theta_1 = {L(t0)} - {L(eta)}\\cdot({L(grad)}) = {L(t1)}$. Minimum energy $-\\sqrt{{a^2+b^2}} = {L(-sp.sqrt(a * a + b * b))}$."),
        misconceptions=[(fmt(cs(t0 + eta * grad)), "gradient-ascent-sign")],
    )


@gen("qc.variational")
def hamiltonian_expectation(rng, level):
    if level == 1:
        a, b, c = nz(rng, -4, 4), nz(rng, -4, 4), nz(rng, -4, 4)
        s = rng.choice(_SIMPLE)
        Hm = a * Z + b * X + c * Y
        val = expval(Hm, STATES[s])
        return Problem(
            prompt=f"Compute $\\langle\\psi\\vert H\\vert\\psi\\rangle$ for $H = {lin_tex([(a, 'Z'), (b, 'X'), (c, 'Y')])}$ and $\\lvert\\psi\\rangle = {ket(s)}$.",
            answer=fmt(val), kind="num",
            rubric=["Linearity: ⟨H⟩ = a⟨Z⟩ + b⟨X⟩ + c⟨Y⟩", f"Bloch components of {ket(s)} give ⟨Z⟩, ⟨X⟩, ⟨Y⟩", "Combine"],
            solution=(f"$\\langle Z\\rangle = {expval(Z, STATES[s])}$, $\\langle X\\rangle = {expval(X, STATES[s])}$, "
                      f"$\\langle Y\\rangle = {expval(Y, STATES[s])}$, so "
                      f"$\\langle H\\rangle = {L(val)}$."),
        )
    if level == 2:
        terms = rng.sample([("ZZ", kron(Z, Z)), ("XX", kron(X, X)), ("YY", kron(Y, Y)), ("ZI", kron(Z, ID)),
                            ("IX", kron(ID, X)), ("XI", kron(X, ID))], 3)
        cs_ = [nz(rng, -3, 3) for _ in terms]
        Hm = sum((c * m for c, (_, m) in zip(cs_, terms)), sp.zeros(4))
        if rng.random() < 0.5:
            bname = rng.choice(list(BELL_TEX))
            st, stex = _bell(bname), BELL_TEX[bname]
        else:
            a, b = rng.choice(_SIMPLE), rng.choice(_SIMPLE)
            st, stex = kron(STATES[a], STATES[b]), ket(a) + ket(b)
        val = expval(Hm, st)
        htex = lin_tex([(c, f"{n[0]}\\otimes {n[1]}") for c, (n, _) in zip(cs_, terms)])
        return Problem(
            prompt=f"Compute the energy $\\langle H\\rangle$ of ${stex}$ for $H = {htex}$.",
            answer=fmt(val), kind="num",
            rubric=["Expectation is linear in the Pauli terms", "Product states: ⟨A⊗B⟩ = ⟨A⟩⟨B⟩",
                    "Bell states: ⟨ZZ⟩, ⟨XX⟩, ⟨YY⟩ are ±1, single-qubit Paulis average to 0", "Sum with coefficients"],
            solution="Term by term: " + ", ".join(f"$\\langle {n}\\rangle = {expval(m, st)}$" for n, m in terms) + f"; total ${L(val)}$.",
        )
    a, b, c = nz(rng, -4, 4), nz(rng, -4, 4), rng.randint(-3, 3)
    Hm = a * Z + b * X + c * Y
    e0 = -sp.sqrt(a * a + b * b + c * c)
    return Problem(
        prompt=(f"Variational principle: for $H = {lin_tex([(a, 'Z'), (b, 'X'), (c, 'Y')])}$, the minimum of $\\langle\\psi\\vert H\\vert\\psi\\rangle$ "
                "over all single-qubit states is the ground energy. Derive it by writing $\\langle H\\rangle = \\vec h\\cdot\\vec r$ "
                "(Bloch vector $\\vec r$), and check against the eigenvalues of $H$."),
        answer=fmt(e0), kind="num",
        rubric=["⟨H⟩ = a z + b x + c y = h·r with |r| ≤ 1", "Minimised by r = −h/|h|",
                "E0 = −|h| = −√(a²+b²+c²)", "Check: eigenvalues of h·σ are ±|h|"],
        solution=(f"$E_0 = -\\|\\vec h\\| = -\\sqrt{{{a * a}+{b * b}+{c * c}}} = {L(e0)}$, the smaller eigenvalue of "
                  f"${mat_tex(Hm)}$."),
        misconceptions=[(fmt(-abs(a) - abs(b) - abs(c)), "added-pauli-norms"), (fmt(-sp.sqrt(a * a + b * b)), "dropped-y-term")],
    )


# ============================================================================ variety generators
# One extra generator per skill, each offering several genuinely different task types per level
# (dispatched with rng.randrange), so a learner does not see the same problem with new numbers.
def _veq(u, v) -> bool:
    """Exact equality of two sympy vectors/matrices (complex phases simplified)."""
    return all(cs(x) == 0 for x in list(u - v))


def _same_ray(u, v) -> bool:
    """Normalised u equals v up to a global phase."""
    return cs(sp.Abs((dag(v) * u)[0, 0]) ** 2) == 1


def _phase_fix(v: sp.Matrix) -> sp.Matrix:
    """Normalise v and fix the global phase so the first nonzero amplitude is real and positive."""
    v = normalize(v)
    k = next(i for i, x in enumerate(v) if cs(x) != 0)
    ph = cs(v[k] / sp.Abs(v[k]))
    return (v / ph).applyfunc(cs)


PAULI = {"I": ID, "X": X, "Y": Y, "Z": Z}
HINT_PAULI = "a Pauli string, qubit 1 first, with a minus sign only if negative, e.g. XZ or -YY (use I for identity)"
HINT_C = "the two coefficients [c0, c1], e.g. [1/sqrt(2), -i/sqrt(2)]  (i = sqrt(-1))"
HINT_S = "exact (log2 allowed) or a decimal to 4 places, e.g. 0.8113"
_LETTERS = "ABCDEF"
_BELL_ORDER = ["Phi+", "Phi-", "Psi+", "Psi-"]
_BELL_OPTS = "\n\nA) $\\lvert\\Phi^+\\rangle$  B) $\\lvert\\Phi^-\\rangle$  C) $\\lvert\\Psi^+\\rangle$  D) $\\lvert\\Psi^-\\rangle$"


def _pauli_string(M: sp.Matrix) -> str | None:
    """Write a Hermitian Clifford image M = ±P_1⊗...⊗P_n as a signed string like '-XZ'."""
    n = int(round(math.log2(M.shape[0])))
    for labels in itertools.product("IXYZ", repeat=n):
        P = kron(*[PAULI[x] for x in labels])
        for s in (1, -1):
            if cmat(M - s * P).is_zero_matrix:
                return ("-" if s < 0 else "") + "".join(labels)
    return None


def _ptex(ps: str) -> str:
    sign = "-" if ps.startswith("-") else ""
    return sign + "\\otimes ".join(ps.lstrip("-"))


def _phase_state(ph) -> str:
    """LaTeX of (|0> + e^{i ph}|1>)/sqrt2 with simple phases written as 1, i, -1, -i."""
    c = cs(sp.exp(I * ph))
    body = state_tex([1, c]) if c in (1, -1, I, -I) else r"\lvert0\rangle + e^{i" + L(ph) + r"}\lvert1\rangle"
    return r"\tfrac{1}{\sqrt2}\left(" + body + r"\right)"


def _norm_tex(amps) -> str:
    """'1/sqrt(n2) (a|..> + b|..>)' for integer/Gaussian-integer amplitudes."""
    n2 = sum(sp.Abs(x) ** 2 for x in amps)
    if n2 == 1:
        return state_tex(amps)
    if sp.sqrt(n2).is_integer:
        return f"\\frac{{1}}{{{sp.sqrt(n2)}}}\\left({state_tex(amps)}\\right)"
    return f"\\frac{{1}}{{\\sqrt{{{n2}}}}}\\left({state_tex(amps)}\\right)"


def _choice_line(opts) -> str:
    return "\n\n" + "  ".join(f"{_LETTERS[i]}) ${o}$" for i, o in enumerate(opts))


# ---------------------------------------------------------------- qc.qubits
_BASIS = {"Z": ("0", "1"), "X": ("+", "-"), "Y": ("+i", "-i")}
_BASIS_TEX = {b: r"\{" + ket(p) + ", " + ket(q) + r"\}" for b, (p, q) in _BASIS.items()}


@gen("qc.qubits")
def qubit_bases(rng, level):
    v = rng.randrange(3 if level == 1 else 2)
    if level == 1 and v == 0:
        s = rng.choice(_SIMPLE)
        b = rng.choice([k for k, pr in _BASIS.items() if s not in pr])
        p, q = _BASIS[b]
        c = [cs((dag(STATES[x]) * STATES[s])[0, 0]) for x in (p, q)]
        wrong = [cs((STATES[x].T * STATES[s])[0, 0]) for x in (p, q)]
        return Problem(
            prompt=(f"Rewrite ${ket(s)}$ in the basis ${_BASIS_TEX[b]}$: find $c_0, c_1$ with "
                    f"${ket(s)} = c_0{ket(p)} + c_1{ket(q)}$."),
            answer=vans(c), kind="vector", answer_hint=HINT_C,
            rubric=[f"c_0 = ⟨{p}|{s}⟩ and c_1 = ⟨{q}|{s}⟩ (project onto each new basis vector)",
                    "The bra conjugates the basis vector's amplitudes", "Check |c0|² + |c1|² = 1"],
            solution=(f"$c_0 = \\langle {p}\\vert {s}\\rangle = {L(c[0])}$, $c_1 = \\langle {q}\\vert {s}\\rangle = {L(c[1])}$. "
                      "Because the new basis is orthonormal, the coefficient of each basis vector is its overlap with the state."),
            misconceptions=[(vans(wrong), "forgot-bra-conjugate")],
        )
    if level == 1 and v == 1:
        ph = rng.randrange(8) * PI / 4
        beta = cs(-sp.exp(I * ph))
        return Problem(
            prompt=(f"Let $\\lvert\\psi\\rangle = {_phase_state(ph)}$. Find the complex number $\\beta$ for which "
                    "$\\lvert\\phi\\rangle = \\tfrac{1}{\\sqrt2}\\left(\\lvert0\\rangle + \\beta\\lvert1\\rangle\\right)$ is orthogonal to "
                    "$\\lvert\\psi\\rangle$."),
            answer=fmt(beta), kind="expr", answer_hint="a complex number, e.g. -1/sqrt(2) + i/sqrt(2)  or  -exp(i*pi/4)",
            rubric=["Orthogonal means ⟨ψ|φ⟩ = 0", "The bra ⟨ψ| conjugates the phase: e^{iφ} → e^{−iφ}",
                    "Solve ½(1 + e^{−iφ}β) = 0 for β"],
            solution=(f"$\\langle\\psi\\vert\\phi\\rangle = \\tfrac12\\left(1 + \\left({L(cs(sp.exp(-I * ph)))}\\right)\\beta\\right) = 0 "
                      f"\\Rightarrow \\beta = -\\left({L(cs(sp.exp(I * ph)))}\\right) = {L(beta)}$ (the bra conjugates the phase). "
                      "Orthogonal states sit at opposite points of the Bloch equator."),
            misconceptions=[(fmt(cs(-sp.exp(-I * ph))), "forgot-bra-conjugate"), (fmt(cs(sp.exp(I * ph))), "sign-error")],
        )
    if level == 1:
        b = rng.choice([1, I, -1, -I])
        glob = rng.choice([I, -1, -I])
        same = rng.random() < 0.5
        phi = [glob, glob * b] if same else [glob, glob * b * rng.choice([-1, I, -I])]
        ans = "A" if same else "B"
        ratio0, ratio1 = cs(phi[0]), cs(phi[1] / b)
        return Problem(
            prompt=(f"Do $\\lvert\\psi\\rangle = \\tfrac{{1}}{{\\sqrt2}}\\left({state_tex([1, b])}\\right)$ and "
                    f"$\\lvert\\phi\\rangle = \\tfrac{{1}}{{\\sqrt2}}\\left({state_tex(phi)}\\right)$ describe the same physical state "
                    "(identical measurement statistics in every basis)?\n\n"
                    "A) yes — they differ only by a global phase  B) no — they differ by a relative phase"),
            answer=ans, kind="choice",
            rubric=["Divide each amplitude of |φ⟩ by the matching amplitude of |ψ⟩",
                    "Same ratio for both ⇒ global phase (unobservable)",
                    "Different ratios ⇒ relative phase (observable, e.g. in the X basis)"],
            solution=(f"Ratios: $\\phi_0/\\psi_0 = {L(ratio0)}$, $\\phi_1/\\psi_1 = {L(ratio1)}$. "
                      + ("They agree, so $\\lvert\\phi\\rangle = " + ({-1: "-"}.get(ratio0, L(ratio0))) + "\\lvert\\psi\\rangle$: same physical state (A)."
                         if same else "They differ, so the *relative* phase changed: a different physical state (B).")),
            misconceptions=[("B" if same else "A", "global-vs-relative-phase")],
        )
    if level == 2 and v == 0:
        a, b = nz(rng, 1, 3), rand_amp(rng)
        perp = _phase_fix(vec(sp.conjugate(b), -a))
        wrong = _phase_fix(vec(b, -a))
        return Problem(
            prompt=(f"Find the normalised state $\\lvert\\psi^\\perp\\rangle$ orthogonal to $\\lvert\\psi\\rangle \\propto {state_tex([a, b])}$, "
                    "choosing the global phase so that its $\\lvert0\\rangle$ amplitude is real and positive."),
            answer=vans(perp), kind="vector", answer_hint=HINT_V1,
            rubric=["Write |ψ⊥⟩ ∝ c0|0⟩ + c1|1⟩ and impose ⟨ψ|ψ⊥⟩ = ā c0 + b̄ c1 = 0",
                    "One solution: (c0, c1) = (b̄, −ā) — conjugates!", "Normalise", "Multiply by a phase so c0 > 0"],
            solution=(f"$\\langle\\psi\\vert\\psi^\\perp\\rangle \\propto {a}c_0 + ({L(cs(sp.conjugate(b)))})c_1 = 0$ is solved by "
                      f"$(c_0, c_1) \\propto ({L(cs(sp.conjugate(b)))}, {-a})$. Normalising and fixing the phase: "
                      f"$\\lvert\\psi^\\perp\\rangle = {state_tex(perp)}$."),
            misconceptions=[(vans(wrong), "forgot-bra-conjugate")],
        )
    if level == 2:
        a, b = nz(rng, 1, 3), rand_amp(rng)
        n2 = a ** 2 + sp.Abs(b) ** 2
        psi = vec(a, b) / sp.sqrt(n2)
        basis = rng.choice(["X", "Y"])
        p, q = _BASIS[basis]
        c = [cs((dag(STATES[x]) * psi)[0, 0]) for x in (p, q)]
        wrong = [cs((STATES[x].T * psi)[0, 0]) for x in (p, q)]
        return Problem(
            prompt=(f"Express $\\lvert\\psi\\rangle = {_norm_tex([a, b])}$ in the basis ${_BASIS_TEX[basis]}$: give $c_0, c_1$ with "
                    f"$\\lvert\\psi\\rangle = c_0{ket(p)} + c_1{ket(q)}$."),
            answer=vans(c), kind="vector", answer_hint=HINT_C,
            rubric=[f"c_0 = ⟨{p}|ψ⟩, c_1 = ⟨{q}|ψ⟩", "Conjugate the basis amplitudes in the bra",
                    "Keep the 1/√(norm²) factor", "Check |c0|² + |c1|² = 1"],
            solution=(f"$c_0 = \\langle {p}\\vert\\psi\\rangle = {L(c[0])}$, $c_1 = \\langle {q}\\vert\\psi\\rangle = {L(c[1])}$ "
                      f"(and $|c_0|^2 + |c_1|^2 = {L(cs(sp.Abs(c[0]) ** 2 + sp.Abs(c[1]) ** 2))}$)."),
            misconceptions=[(vans(wrong), "forgot-bra-conjugate")],
        )
    if v == 0:
        pool = [STATES[k] for k in _SIMPLE] + [vec(sp.sqrt(3) / 2, sp.Rational(1, 2)), vec(sp.Rational(1, 2), sp.sqrt(3) / 2),
                                              vec(sp.sqrt(3) / 2, -sp.Rational(1, 2))]
        while True:
            u, w = rng.sample(pool, 2)
            ov = cs(sp.Abs((dag(u) * w)[0, 0]) ** 2)
            if 0 < ov < 1:
                break
        P = cs((1 + sp.sqrt(1 - ov)) / 2)
        return Problem(
            prompt=(f"Bob receives either $\\lvert\\psi_1\\rangle = {state_tex(u)}$ or $\\lvert\\psi_2\\rangle = {state_tex(w)}$, each with "
                    "probability $\\tfrac12$, and must guess which. The best possible measurement succeeds with the Helstrom probability "
                    "$P = \\tfrac12\\left(1 + \\sqrt{1 - |\\langle\\psi_1\\vert\\psi_2\\rangle|^2}\\right)$ (derive it on paper from the "
                    "trace distance of the two projectors). Compute $P$ exactly."),
            answer=fmt(P), kind="num",
            rubric=["Compute the overlap ⟨ψ1|ψ2⟩ (conjugate the bra)",
                    "Trace distance of two pure states = √(1 − |⟨ψ1|ψ2⟩|²)", "P = ½(1 + trace distance)"],
            solution=(f"$|\\langle\\psi_1\\vert\\psi_2\\rangle|^2 = {L(ov)}$, so $P = \\tfrac12\\left(1 + \\sqrt{{1 - {L(ov)}}}\\right) = {L(P)}$ "
                      f"(≈ {float(P):.4f}). Non-orthogonal states can never be distinguished perfectly."),
            misconceptions=[(fmt(cs((2 - ov) / 2)), "used-overlap-not-trace-distance"), (fmt(cs(1 - ov)), "confused-overlap-with-success")],
        )
    th = rng.choice([PI / 3, PI / 2, 2 * PI / 3])
    k = rng.randrange(8)
    ph, ph2 = k * PI / 4, ((k + 4) % 8) * PI / 4
    st = vec(cs(sp.cos(th / 2)), cs(sp.exp(I * ph) * sp.sin(th / 2)))
    return Problem(
        prompt=(f"Find the state orthogonal to $\\lvert\\psi\\rangle = {state_tex(st)}$ and write it as "
                "$\\cos\\tfrac{\\theta'}{2}\\lvert0\\rangle + e^{i\\varphi'}\\sin\\tfrac{\\theta'}{2}\\lvert1\\rangle$ with "
                "$\\theta'\\in[0,\\pi]$, $\\varphi'\\in[0,2\\pi)$. Give $\\theta', \\varphi'$. "
                "(On paper: show orthogonal states are antipodal on the Bloch sphere.)"),
        answer=[fmt(PI - th), fmt(ph2)], kind="list", answer_hint="theta', phi'  e.g. 2*pi/3, 5*pi/4",
        rubric=["Read off θ and φ of |ψ⟩ (remember the half angle)",
                "Orthogonal ⇔ Bloch vectors opposite: r' = −r", "−r has θ' = π − θ and φ' = φ + π (mod 2π)",
                "Check ⟨ψ|ψ'⟩ = 0"],
        solution=(f"$\\lvert\\psi\\rangle$ has $\\theta = {L(th)}$, $\\varphi = {L(ph)}$. The antipode has $\\theta' = \\pi - \\theta = {L(PI - th)}$, "
                  f"$\\varphi' = \\varphi + \\pi = {L(ph2)}$ (mod $2\\pi$). Check: "
                  "$\\langle\\psi\\vert\\psi'\\rangle = \\cos\\tfrac\\theta2\\sin\\tfrac\\theta2\\left(1 + e^{i(\\varphi'-\\varphi)}\\right) = 0$."),
        misconceptions=[([fmt(PI - th), fmt(ph)], "forgot-phase-shift"), ([fmt(th), fmt(ph2)], "kept-polar-angle")],
    )


# ---------------------------------------------------------------- qc.single_gates
_ROT_ANGLE = {"X": PI, "Y": PI, "Z": PI, "H": PI, "S": PI / 2, "T": PI / 4, "SH": 2 * PI / 3, "HS": 2 * PI / 3,
              "S^\\dagger": PI / 2}
_ROT_GATE = {"X": X, "Y": Y, "Z": Z, "H": H, "S": S, "T": T, "SH": S * H, "HS": H * S, "S^\\dagger": dag(S)}


@gen("qc.single_gates")
def gate_identities(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        pool = ["X", "Y", "Z", "H", "S"]
        while True:
            s, g = rng.choice(_SIMPLE), rng.choice(pool)
            out = GATES[g] * STATES[s]
            t = next((k for k in _SIMPLE if _veq(STATES[k], out)), None)
            if t is None or t == s:
                continue
            opts = [g] + rng.sample([p for p in pool if p != g], 3)
            rng.shuffle(opts)
            if sum(_veq(GATES[o] * STATES[s], STATES[t]) for o in opts) == 1:
                break
        ans = _LETTERS[opts.index(g)]
        mis = [(_LETTERS[i], "ignored-global-phase") for i, o in enumerate(opts)
               if o != g and _same_ray(GATES[o] * STATES[s], STATES[t])]
        return Problem(
            prompt=(f"Which gate maps ${ket(s)}$ to ${ket(t)}$ **exactly** (no extra global phase)?" + _choice_line(opts)),
            answer=ans, kind="choice",
            rubric=[f"Write {ket(s)} as a column vector", "Apply each candidate matrix",
                    "Compare with the target including the overall phase"],
            solution=(f"${g}{ket(s)} = {ket(t)}$ (multiply the matrix by the column vector), answer {ans}. Another gate may give ${ket(t)}$ only up to a "
                      "global phase such as $i$ — physically equivalent, but not equal as vectors."),
            misconceptions=mis,
        )
    if level == 1:
        G = rng.choice(["S", "T"])
        if rng.random() < 0.3:
            M, tex, wrong, tag = dag(GATES[G]), f"{G}^\\dagger", GATES[G], "dagger-forgot-conjugate"
        else:
            k = rng.choice([2, 3]) if G == "S" else rng.choice([2, 3, 4, 5, 6, 7])
            M, tex = GATES[G] ** k, f"{G}^{{{k}}}"
            wrong, tag = sp.diag(1, k * GATES[G][1, 1]), "multiplied-phase-instead-of-power"
        return Problem(
            prompt=(f"Write the $2\\times2$ matrix of ${tex}$ in simplest form (recall $S = \\mathrm{{diag}}(1, i)$, "
                    "$T = \\mathrm{diag}(1, e^{i\\pi/4})$)."),
            answer=mans(M), kind="matrix", answer_hint=HINT_M,
            rubric=["Diagonal matrices: powers/adjoints act entry by entry",
                    "(e^{iα})^k = e^{ikα}; the adjoint conjugates: e^{iα} → e^{−iα}",
                    "Simplify the phase (reduce mod 2π, write as a + bi)"],
            solution=f"${tex} = {mat_tex(M)}$ (useful: $T^2 = S$, $S^2 = Z$, $T^8 = I$, $S^\\dagger = S^3$).",
            misconceptions=[(mans(wrong), tag)],
        )
    if level == 2 and v == 0:
        name, M = rng.choice([("X", X), ("Y", Y), ("\\tfrac{X+Y}{\\sqrt2}", (X + Y) / R2), ("\\tfrac{X-Y}{\\sqrt2}", (X - Y) / R2)])
        lam = rng.choice([1, -1])
        ev = _phase_fix(next(vs[0] for val, _, vs in M.eigenvects() if cs(val - lam) == 0))
        other = _phase_fix(next(vs[0] for val, _, vs in M.eigenvects() if cs(val + lam) == 0))
        return Problem(
            prompt=(f"Find the normalised eigenvector of $M = {name} = {mat_tex(M)}$ with eigenvalue ${lam}$, fixing the global phase so "
                    "that its $\\lvert0\\rangle$ amplitude is real and positive."),
            answer=vans(ev), kind="vector", answer_hint=HINT_V1,
            rubric=[f"Solve (M − ({lam})I)v = 0: one row gives the ratio v1/v0", "Normalise",
                    "Fix the phase so v0 > 0", "Check Mv = λv"],
            solution=f"$(M - ({lam})I)v = 0$ gives, normalised and phase-fixed, $v = {state_tex(ev)}$.",
            misconceptions=[(vans(other), "wrong-eigenvalue")],
        )
    if level == 2:
        axis = rng.choice(["y", "z"])
        a = rng.choice([PI / 3, PI / 2, 2 * PI / 3, PI])
        s = rng.choice(["0", "1", "+"]) if axis == "y" else rng.choice(["+", "-", "+i"])
        R = Ry if axis == "y" else Rz
        out = R(a) * STATES[s]
        defn = ("$R_y(\\alpha) = \\begin{pmatrix}\\cos\\frac\\alpha2 & -\\sin\\frac\\alpha2\\\\ \\sin\\frac\\alpha2 & \\cos\\frac\\alpha2\\end{pmatrix}$"
                if axis == "y" else "$R_z(\\alpha) = \\mathrm{diag}(e^{-i\\alpha/2}, e^{i\\alpha/2})$")
        return Problem(
            prompt=(f"Apply $R_{axis}({L(a)})$ to ${ket(s)}$, where {defn}. Give the resulting state vector, keeping the global phase "
                    "exactly as the matrix produces it."),
            answer=vans(out), kind="vector", answer_hint=HINT_V1,
            rubric=["Evaluate the half angle α/2 first", "Write the 2×2 matrix with exact cos/sin/exponentials",
                    "Multiply by the input column vector", "Do not drop the global phase (it is part of the vector)"],
            solution=f"$R_{axis}({L(a)}) = {mat_tex(R(a))}$, so the output is ${vec_tex(out)}$.",
            misconceptions=[(vans(R(2 * a) * STATES[s]), "bloch-half-angle")],
        )
    if v == 0:
        name = rng.choice(list(_ROT_GATE))
        U, th = _ROT_GATE[name], _ROT_ANGLE[name]
        tr = cs(U.trace())
        return Problem(
            prompt=("Every single-qubit unitary can be written $U = e^{i\\alpha}R_{\\hat n}(\\theta) = e^{i\\alpha}\\left(\\cos\\tfrac\\theta2\\, I "
                    "- i\\sin\\tfrac\\theta2\\,\\hat n\\cdot\\vec\\sigma\\right)$. Derive $|\\mathrm{Tr}\\,U| = 2|\\cos\\tfrac\\theta2|$ and use it to "
                    f"find the Bloch-sphere rotation angle $\\theta\\in[0,\\pi]$ of $U = {name} = {mat_tex(U)}$."),
            answer=fmt(th), kind="num", answer_hint="an angle, e.g. 2*pi/3",
            rubric=["Paulis are traceless, so Tr U = 2e^{iα} cos(θ/2)", "Compute Tr U and its modulus",
                    "Solve |cos(θ/2)| = |Tr U|/2 with θ ∈ [0, π]"],
            solution=(f"$\\mathrm{{Tr}}\\,U = {L(cs(U[0, 0]))} + \\left({L(cs(U[1, 1]))}\\right)$, so $|\\mathrm{{Tr}}\\,U|/2 = "
                      f"{L(cs(sp.cos(th / 2)))} = \\cos\\tfrac\\theta2$ and $\\theta = {L(th)}$."),
            misconceptions=[(fmt(th / 2), "bloch-half-angle")],
        )
    us = {"H": H, "S": S, "SH": S * H, "HS": H * S, "S^\\dagger": dag(S), "HSH": H * S * H}
    name = rng.choice(list(us))
    U = us[name]
    P = rng.choice(["X", "Y", "Z"])
    res = _pauli_string(U * GATES[P] * dag(U))
    wres = _pauli_string(dag(U) * GATES[P] * U)
    opts = ["X", "-X", "Y", "-Y", "Z", "-Z"]
    return Problem(
        prompt=(f"Compute $U{P}U^\\dagger$ for $U = {name} = {mat_tex(U)}$ and identify it as a signed Pauli matrix "
                "(Clifford gates map Paulis to Paulis)." + _choice_line(opts)),
        answer=_LETTERS[opts.index(res)], kind="choice",
        rubric=["Conjugate one gate at a time, innermost (first-applied) gate first",
                "HXH = Z, HZH = X, HYH = −Y;  SXS† = Y, SYS† = −X, SZS† = Z", "Track the sign"],
        solution=f"$U{P}U^\\dagger = {_ptex(res)}$ (direct multiplication gives ${mat_tex(U * GATES[P] * dag(U))}$).",
        misconceptions=[(_LETTERS[opts.index(wres)], "conjugated-wrong-way")],
    )


# ---------------------------------------------------------------- qc.measurement
@gen("qc.measurement")
def measurement_stats(rng, level):
    v = rng.randrange(2 if level == 1 else 3)
    if level == 1 and v == 0:
        a, b = nz(rng, 1, 4), rng.choice([1, -1]) * nz(rng, 1, 4) * rng.choice([1, I])
        bb = int(sp.Abs(b) ** 2)
        n2 = a * a + bb
        ez = sp.Rational(a * a - bb, n2)
        return Problem(
            prompt=(f"A qubit is prepared in $\\lvert\\psi\\rangle = \\dfrac{{1}}{{\\sqrt{{{n2}}}}}\\left({state_tex([a, b])}\\right)$. "
                    "Compute the expectation value $\\langle Z\\rangle$ (outcome $+1$ for $\\lvert0\\rangle$, $-1$ for $\\lvert1\\rangle$)."),
            answer=fmt(ez), kind="num",
            rubric=["P(0) = |α|², P(1) = |β|² (normalised)", "⟨Z⟩ = (+1)P(0) + (−1)P(1)"],
            solution=f"$P(0) = {L(sp.Rational(a * a, n2))}$, $P(1) = {L(sp.Rational(bb, n2))}$, so $\\langle Z\\rangle = {L(ez)}$.",
            misconceptions=[(fmt(-ez), "swapped-outcomes"), (fmt(sp.Rational(a * a - bb)), "forgot-normalisation")],
        )
    if level == 1:
        while True:
            amps = [rand_amp(rng) if rng.random() < 0.8 else sp.Integer(0) for _ in range(4)]
            k = rng.randrange(4)
            if amps[k] != 0 and sum(1 for x in amps if x != 0) >= 2:
                break
        n2 = sum(sp.Abs(x) ** 2 for x in amps)
        ck = int(sp.Abs(amps[k]) ** 2)
        p = sp.Rational(ck, n2)
        bits = format(k, "02b")
        return Problem(
            prompt=(f"Both qubits of $\\lvert\\psi\\rangle \\propto {state_tex(amps)}$ are measured in the computational basis. "
                    f"What is the probability of the outcome $q_1q_2 = {bits}$?"),
            answer=fmt(p), kind="num",
            rubric=["Norm² = sum of |amplitude|²", f"P({bits}) = |c_{bits}|² / norm²"],
            solution=f"Norm$^2 = {n2}$, $|c_{{{bits}}}|^2 = {ck}$, so $P = {L(p)}$.",
            misconceptions=[(fmt(cs(sp.sqrt(ck) / sp.sqrt(n2))), "forgot-amplitude-squared"), (str(ck), "forgot-normalisation")],
        )
    if level == 2 and v == 0:
        while True:
            amps = [rand_amp(rng) if rng.random() < 0.85 else sp.Integer(0) for _ in range(4)]
            if sum(1 for x in amps if x != 0) >= 3:
                break
        w = [int(sp.Abs(x) ** 2) for x in amps]
        n2 = sum(w)
        zz = sp.Rational(w[0] + w[3] - w[1] - w[2], n2)
        return Problem(
            prompt=(f"Two qubits are in $\\lvert\\psi\\rangle \\propto {state_tex(amps)}$. Compute $\\langle Z\\otimes Z\\rangle$, the average "
                    "of the product of the two $\\pm1$ outcomes when both qubits are measured in the computational basis."),
            answer=fmt(zz), kind="num",
            rubric=["Z⊗Z has eigenvalue +1 on |00⟩, |11⟩ and −1 on |01⟩, |10⟩", "Probabilities = |c|²/norm²",
                    "⟨ZZ⟩ = P(equal) − P(different)"],
            solution=(f"$|c|^2 = ({', '.join(map(str, w))})$, norm$^2 = {n2}$: $\\langle ZZ\\rangle = "
                      f"\\frac{{{w[0]} + {w[3]} - {w[1]} - {w[2]}}}{{{n2}}} = {L(zz)}$."),
            misconceptions=[(fmt(sp.Rational(w[0] + w[3], n2)), "forgot-sign-weighting"), (fmt(zz * n2), "forgot-normalisation")],
        )
    if level == 2 and v == 1:
        while True:
            a, b = nz(rng, -3, 3), rand_amp(rng)
            obs = rng.choice(["X", "Y"])
            n2 = a * a + int(sp.Abs(b) ** 2)
            psi = vec(a, b) / sp.sqrt(n2)
            val = expval(GATES[obs], psi)
            if val != 0:
                break
        other = expval(GATES["Y" if obs == "X" else "X"], psi)
        return Problem(
            prompt=f"Compute $\\langle {obs}\\rangle$ for $\\lvert\\psi\\rangle = {_norm_tex([a, b])}$.",
            answer=fmt(val), kind="num",
            rubric=[f"⟨{obs}⟩ = ⟨ψ|{obs}|ψ⟩", f"{obs}|ψ⟩ = " + ("(β, α)" if obs == "X" else "(−iβ, iα)"),
                    "Equivalently ⟨X⟩ = 2 Re(ᾱβ), ⟨Y⟩ = 2 Im(ᾱβ)", "Divide by the norm²"],
            solution=(f"$\\bar\\alpha\\beta = {L(cs(a * b / n2))}$ (normalised), so $\\langle {obs}\\rangle = "
                      f"2\\,\\mathrm{{{'Re' if obs == 'X' else 'Im'}}}(\\bar\\alpha\\beta) = {L(val)}$."),
            misconceptions=[(fmt(other), "wrong-pauli-component")] if other != 0 else [],
        )
    if level == 2:
        while True:
            a, b = nz(rng, -3, 3), nz(rng, -3, 3)
            n2 = a * a + b * b
            o1, o2 = rng.choice(["+", "-"]), rng.choice(["0", "1"])
            p1 = sp.Rational((a + b) ** 2 if o1 == "+" else (a - b) ** 2, 2 * n2)
            if p1 != 0:
                break
        P = p1 / 2
        naive = p1 * sp.Rational(a * a if o2 == "0" else b * b, n2)
        return Problem(
            prompt=(f"The qubit $\\lvert\\psi\\rangle \\propto {state_tex([a, b])}$ is first measured in the "
                    "$\\{\\lvert+\\rangle,\\lvert-\\rangle\\}$ basis, and the post-measurement state is then measured in the computational "
                    f"basis. What is the probability of getting ${ket(o1)}$ and then ${ket(o2)}$?"),
            answer=fmt(P), kind="num",
            rubric=[f"P({o1}) = |⟨{o1}|ψ⟩|² (normalise first)", f"After the first measurement the state IS |{o1}⟩",
                    f"P({o2} | {o1}) = |⟨{o2}|{o1}⟩|² = 1/2", "Multiply"],
            solution=(f"$P({o1}) = {L(p1)}$; the state collapses to ${ket(o1)}$, which gives ${o2}$ with probability $\\tfrac12$: "
                      f"$P = {L(P)}$."),
            misconceptions=[(fmt(naive), "ignored-collapse")],
        )
    if v == 0:
        s = rng.choice(_SIMPLE)
        th = rng.choice([PI / 6, PI / 4, PI / 3, PI / 2, 2 * PI / 3, 3 * PI / 4, 5 * PI / 6])
        r = bloch_vector(STATES[s])
        rn = cs(r[0] * sp.sin(th) + r[2] * sp.cos(th))
        P = cs((1 + rn) / 2)
        return Problem(
            prompt=(f"A qubit in ${ket(s)}$ is measured along the axis $\\hat n = (\\sin\\theta, 0, \\cos\\theta)$, i.e. the observable "
                    f"$\\hat n\\cdot\\vec\\sigma = \\cos\\theta\\,Z + \\sin\\theta\\,X$, with $\\theta = {L(th)}$. Derive "
                    "$P(+1) = \\tfrac12(1 + \\vec r\\cdot\\hat n)$ ($\\vec r$ = Bloch vector) and evaluate it exactly."),
            answer=fmt(P), kind="num",
            rubric=["The +1 eigenstate of n·σ is the Bloch state pointing along n",
                    "|⟨n|ψ⟩|² = cos²(γ/2), γ = angle between r and n",
                    "cos²(γ/2) = (1 + cos γ)/2 = (1 + r·n)/2", f"Bloch vector of {ket(s)} is ({', '.join(L(c) for c in r)})"],
            solution=f"$\\vec r = ({', '.join(L(c) for c in r)})$, $\\vec r\\cdot\\hat n = {L(rn)}$, so $P(+1) = \\tfrac12\\left(1 + \\vec r\\cdot\\hat n\\right) = {L(P)}$.",
            misconceptions=[(fmt(rn), "dropped-constant-term"), (fmt(cs(rn ** 2)), "bloch-half-angle")],
        )
    if v == 1:
        while True:
            s1, s2 = rng.sample(_SIMPLE, 2)
            p = rng.choice(_FRACS)
            k = rng.choice([0, 1])
            l1 = cs(sp.Abs(STATES[s1][k]) ** 2)
            l2 = cs(sp.Abs(STATES[s2][k]) ** 2)
            den = cs(p * l1 + (1 - p) * l2)
            if den != 0 and l1 != 0:
                break
        post = cs(p * l1 / den)
        return Problem(
            prompt=(f"A device prepares ${ket(s1)}$ with probability ${L(p)}$ and ${ket(s2)}$ otherwise. You measure in the computational "
                    f"basis and get ${k}$. Using Bayes' rule with Born-rule likelihoods, what is the probability that the state was "
                    f"${ket(s1)}$?"),
            answer=fmt(post), kind="num",
            rubric=[f"Likelihoods: P({k}|{s1}) = |⟨{k}|{s1}⟩|², P({k}|{s2}) = |⟨{k}|{s2}⟩|²",
                    "Evidence P(k) = Σ prior × likelihood", "Posterior = prior × likelihood / evidence"],
            solution=(f"$P({k}\\mid {s1}) = {L(l1)}$, $P({k}\\mid {s2}) = {L(l2)}$, evidence $P({k}) = {L(den)}$, posterior "
                      f"$\\frac{{{L(p)}\\cdot {L(l1)}}}{{{L(den)}}} = {L(post)}$."),
            misconceptions=[(fmt(p), "ignored-measurement-update"), (fmt(l1), "confused-likelihood-posterior")],
        )
    a, b = nz(rng, 1, 4), nz(rng, -4, 4)
    n2 = a * a + b * b
    ez = sp.Rational(a * a - b * b, n2)
    var = 1 - ez ** 2
    eps = rng.choice([sp.Rational(1, 100), sp.Rational(1, 50), sp.Rational(1, 20), sp.Rational(1, 10)])
    N = sp.ceiling(var / eps ** 2)
    return Problem(
        prompt=(f"You estimate $\\langle Z\\rangle$ of $\\lvert\\psi\\rangle = {_norm_tex([a, b])}$ by averaging $N$ single-shot outcomes "
                "$\\pm1$. Derive the single-shot variance $\\mathrm{Var}(Z) = 1 - \\langle Z\\rangle^2$ and find the smallest $N$ for which "
                f"the standard error $\\sqrt{{\\mathrm{{Var}}/N}}$ is at most $\\varepsilon = {L(eps)}$."),
        answer=str(N), kind="num", answer_hint="an integer",
        rubric=["Z² = I so ⟨Z²⟩ = 1 and Var = 1 − ⟨Z⟩²", "⟨Z⟩ = (|α|² − |β|²)/norm²",
                "Require Var/N ≤ ε² ⇒ N ≥ Var/ε²", "Round up"],
        solution=(f"$\\langle Z\\rangle = {L(ez)}$, $\\mathrm{{Var}} = {L(var)}$, $N \\ge {L(var)}/{L(eps ** 2)} = {L(var / eps ** 2)}$, "
                  f"so $N = {N}$."),
        misconceptions=[(str(sp.ceiling(1 / eps ** 2)), "assumed-unit-variance"),
                        (str(sp.ceiling(sp.sqrt(var) / eps)), "forgot-to-square-error")],
    )


# ---------------------------------------------------------------- qc.tensor
@gen("qc.tensor")
def tensor_drills(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        while True:
            a, b, c = (rng.choice(_SIMPLE) for _ in range(3))
            bits = format(rng.randrange(8), "03b")
            amp = cs(STATES[a][int(bits[0])] * STATES[b][int(bits[1])] * STATES[c][int(bits[2])])
            rev = cs(STATES[a][int(bits[2])] * STATES[b][int(bits[1])] * STATES[c][int(bits[0])])
            if amp != 0:
                break
        return Problem(
            prompt=(f"What is the amplitude of ${ket(bits)}$ in the three-qubit product state "
                    f"${ket(a)}\\otimes{ket(b)}\\otimes{ket(c)}$? (Qubit 1 is the left factor and the left bit.)"),
            answer=fmt(amp), kind="num", answer_hint="an exact number, e.g. -i/(2*sqrt(2))",
            rubric=["Amplitude of |xyz⟩ in |a⟩⊗|b⟩⊗|c⟩ is a_x · b_y · c_z", "Read each factor's amplitude for its own bit", "Multiply"],
            solution=(f"Amplitude $= {L(cs(STATES[a][int(bits[0])]))}\\cdot\\left({L(cs(STATES[b][int(bits[1])]))}\\right)"
                      f"\\cdot\\left({L(cs(STATES[c][int(bits[2])]))}\\right) = {L(amp)}$."),
            misconceptions=[(fmt(rev), "wrong-tensor-order")],
        )
    if level == 1:
        n = rng.choice([3, 4])
        while True:
            bits = format(rng.randrange(2 ** n), f"0{n}b")
            if bits != bits[::-1]:
                break
        k = int(bits, 2)
        terms = " + ".join(str(2 ** (n - 1 - i)) for i, ch in enumerate(bits) if ch == "1")
        return Problem(
            prompt=(f"In a ${n}$-qubit register (qubit 1 = leftmost = most significant bit), the basis state ${ket(bits)}$ is the "
                    f"standard basis column vector $e_k$ of length ${2 ** n}$, with $k$ counted from $0$. What is $k$?"),
            answer=str(k), kind="num", answer_hint="an integer",
            rubric=["Read the bit string as a binary number, leftmost bit = 2^(n−1)", "That number is the (0-based) index of the single 1"],
            solution=f"${bits}_2 = {terms}" + (f" = {k}" if "+" in terms else "") + f"$, so $k = {k}$.",
            misconceptions=[(str(int(bits[::-1], 2)), "little-endian-qiskit-ordering"), (str(k + 1), "off-by-one")],
        )
    if level == 2 and v == 0:
        while True:
            a, b, c, d = (rng.choice(_SIMPLE) for _ in range(4))
            o1 = cs((dag(STATES[a]) * STATES[c])[0, 0])
            o2 = cs((dag(STATES[b]) * STATES[d])[0, 0])
            ov = cs(o1 * o2)
            if ov != 0 and ov != 1:
                break
        wrong = cs((STATES[a].T * STATES[c])[0, 0] * (STATES[b].T * STATES[d])[0, 0])
        return Problem(
            prompt=(f"Compute the inner product $\\left({bra(a)}\\otimes{bra(b)}\\right)\\left({ket(c)}\\otimes{ket(d)}\\right)$ of the "
                    "two product states."),
            answer=fmt(ov), kind="num", answer_hint="an exact (possibly complex) number, e.g. i/2",
            rubric=["Product rule: ⟨a b|c d⟩ = ⟨a|c⟩⟨b|d⟩", "Compute each single-qubit overlap (conjugate the bra)", "Multiply"],
            solution=f"$\\langle {a}\\vert {c}\\rangle = {L(o1)}$, $\\langle {b}\\vert {d}\\rangle = {L(o2)}$, product ${L(ov)}$.",
            misconceptions=[(fmt(wrong), "forgot-bra-conjugate")],
        )
    if level == 2:
        A, B, C, D = (rng.choice(["X", "Z", "H", "S", "Y"]) for _ in range(4))
        M = kron(GATES[A], GATES[B]) * kron(GATES[C], GATES[D])
        return Problem(
            prompt=(f"Compute the $4\\times4$ matrix $({A}\\otimes {B})({C}\\otimes {D})$, using the mixed-product rule rather than "
                    "multiplying $4\\times4$ matrices. " + CONV),
            answer=mans(M), kind="matrix", answer_hint=HINT_M,
            rubric=["(A⊗B)(C⊗D) = (AC)⊗(BD)", f"Compute {A}{C} and {B}{D} as 2×2 products", "Take one Kronecker product"],
            solution=(f"${A}{C} = {mat_tex(GATES[A] * GATES[C])}$, ${B}{D} = {mat_tex(GATES[B] * GATES[D])}$, so the product is "
                      f"${mat_tex(M)}$."),
            misconceptions=[(mans(kron(GATES[C] * GATES[A], GATES[D] * GATES[B])), "reversed-gate-order")],
        )
    A, B = rng.choice(["X", "Y", "Z", "H", "S", "T"]), rng.choice(["X", "Y", "Z", "H", "S", "T"])
    tA, tB = sp.expand(cs(GATES[A].trace())), sp.expand(cs(GATES[B].trace()))
    dA, dB = cs(GATES[A].det()), cs(GATES[B].det())
    tr, det = cs(tA * tB), cs(dA ** 2 * dB ** 2)
    return Problem(
        prompt=(f"Without writing out the $4\\times4$ matrix, find $\\mathrm{{Tr}}({A}\\otimes {B})$ and $\\det({A}\\otimes {B})$. On paper, "
                "prove $\\mathrm{Tr}(A\\otimes B) = \\mathrm{Tr}A\\,\\mathrm{Tr}B$ and $\\det(A\\otimes B) = (\\det A)^m(\\det B)^n$ "
                "for $A$ of size $n\\times n$ and $B$ of size $m\\times m$."),
        answer=[fmt(tr), fmt(det)], kind="list", answer_hint="trace, determinant  e.g. 0, 1",
        rubric=["Diagonal blocks of A⊗B are a_ii B ⇒ trace = Σ a_ii Tr B",
                "A⊗B = (A⊗I)(I⊗B); det(A⊗I_m) = (det A)^m, det(I_n⊗B) = (det B)^n",
                f"Tr {A} = {L(tA)}, Tr {B} = {L(tB)}, det {A} = {L(dA)}, det {B} = {L(dB)}"],
        solution=f"$\\mathrm{{Tr}} = ({L(tA)})({L(tB)}) = {L(tr)}$; $\\det = ({L(dA)})^2({L(dB)})^2 = {L(det)}$.",
        misconceptions=[([fmt(tr), fmt(cs(dA * dB))], "det-kron-exponent")],
    )


# ---------------------------------------------------------------- qc.multi_gates
_CLIFF_LAYERS = {
    "CNOT$_{1\\to2}$": CNOT12, "CNOT$_{2\\to1}$": CNOT21, "$H\\otimes I$": kron(H, ID), "$I\\otimes H$": kron(ID, H),
    "$S\\otimes I$": kron(S, ID), "$I\\otimes S$": kron(ID, S), "CZ": CZ,
}
_CNOT_RULES = "CNOT_{1→2}: X⊗I → X⊗X, I⊗X → I⊗X, Z⊗I → Z⊗I, I⊗Z → Z⊗Z (and Y = iXZ)"


def _rand_pauli2(rng) -> str:
    while True:
        p = rng.choice("IXYZ") + rng.choice("IXYZ")
        if p != "II":
            return p


@gen("qc.multi_gates")
def multi_gate_drills(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        b = [rng.randint(0, 1) for _ in range(3)]
        roles = rng.sample([0, 1, 2], 3)
        out, wrong = list(b), list(b)
        if rng.random() < 0.5:
            c1, c2, t = roles
            if b[c1] and b[c2]:
                out[t] ^= 1
            if b[c1] or b[c2]:
                wrong[t] ^= 1
            desc = (f"The Toffoli gate with controls $q_{c1 + 1}, q_{c2 + 1}$ and target $q_{t + 1}$ (flip the target iff **both** "
                    f"controls are 1) acts on ${ket(''.join(map(str, b)))}$.")
            rub = ["Check both control bits", "Flip the target only if both are 1"]
            tag = "toffoli-single-control"
        else:
            c, s1, s2 = roles
            if b[c]:
                out[s1], out[s2] = out[s2], out[s1]
            else:
                wrong[s1], wrong[s2] = wrong[s2], wrong[s1]
            desc = (f"The Fredkin (controlled-SWAP) gate with control $q_{c + 1}$ swaps $q_{s1 + 1}$ and $q_{s2 + 1}$ iff the control is 1. "
                    f"It acts on ${ket(''.join(map(str, b)))}$.")
            rub = ["Check the control bit", "If it is 1, exchange the two target bits"]
            tag = "control-condition-inverted"
        res, wres = "".join(map(str, out)), "".join(map(str, wrong))
        return Problem(
            prompt=desc + " Which basis state comes out? Type the three bits $q_1q_2q_3$.",
            answer=res, kind="text", answer_hint="three bits, e.g. 101",
            rubric=rub + ["Write the output bits in the order q1 q2 q3"],
            solution=f"Output ${ket(res)}$ — reversible classical gates simply permute basis states.",
            misconceptions=[(wres, tag)],
        )
    if level == 1:
        x, s = rng.choice(["0", "1"]), rng.choice(["+", "-", "+i", "-i"])
        inp = kron(STATES[x], STATES[s])
        out = CZ * inp
        return Problem(
            prompt=(f"Apply the controlled-$Z$ gate $\\mathrm{{CZ}} = \\mathrm{{diag}}(1,1,1,-1)$ to ${ket(x)}\\otimes{ket(s)}$. "
                    "Give the output 4-vector. " + CONV),
            answer=vans(out), kind="vector", answer_hint=HINT_V2,
            rubric=["Expand the input into the 4 amplitudes", "CZ flips the sign of the |11⟩ amplitude only",
                    "Equivalently: if qubit 1 is |1⟩, apply Z to qubit 2"],
            solution=f"Input ${vec_tex(inp)}$; flipping the $\\lvert11\\rangle$ sign gives ${vec_tex(out)}$.",
            misconceptions=[(vans(inp), "ignored-cz")] if x == "1" else [],
        )
    if level == 2 and v == 0:
        while True:
            a, b = rng.choice(_SIMPLE), rng.choice(_SIMPLE)
            if a not in ("0", "1") and b not in ("0", "1"):
                break
        inp = kron(STATES[a], STATES[b])
        out = CZ * inp
        det = cs(out[0] * out[3] - out[1] * out[2])
        return Problem(
            prompt=(f"Compute $\\mathrm{{CZ}}\\left({ket(a)}\\otimes{ket(b)}\\right)$ with $\\mathrm{{CZ}} = \\mathrm{{diag}}(1,1,1,-1)$. "
                    + CONV + " On paper, use the determinant test to decide whether the output is entangled."),
            answer=vans(out), kind="vector", answer_hint=HINT_V2,
            rubric=["Kronecker product of the two input states", "Negate the |11⟩ amplitude",
                    "Entangled ⇔ c00c11 − c01c10 ≠ 0"],
            solution=(f"${ket(a)}\\otimes{ket(b)} = {vec_tex(inp)}$, so CZ gives ${vec_tex(out)}$; "
                      f"$c_{{00}}c_{{11}} - c_{{01}}c_{{10}} = {L(det)}$" + (" ≠ 0: entangled." if det != 0 else ": product.")),
            misconceptions=[(vans(CNOT12 * inp), "confused-cz-with-cnot")],
        )
    if level == 2:
        P = _rand_pauli2(rng)
        c12 = rng.random() < 0.5
        U, W = (CNOT12, CNOT21) if c12 else (CNOT21, CNOT12)
        res = _pauli_string(U * kron(PAULI[P[0]], PAULI[P[1]]) * dag(U))
        wres = _pauli_string(W * kron(PAULI[P[0]], PAULI[P[1]]) * dag(W))
        name = "CNOT$_{1\\to2}$" if c12 else "CNOT$_{2\\to1}$"
        return Problem(
            prompt=(f"Heisenberg picture: compute $U({P[0]}\\otimes {P[1]})U^\\dagger$ for $U = $ {name} and write it as a signed Pauli "
                    "string."),
            answer=res, kind="text", answer_hint=HINT_PAULI,
            rubric=[_CNOT_RULES, "Split the input into single-qubit factors and map each", "Multiply the images (watch XZ = −iY)"],
            solution=f"$U({P[0]}\\otimes {P[1]})U^\\dagger = {_ptex(res)}$.",
            misconceptions=[(wres, "control-target-swapped")] if wres != res else [],
        )
    names = rng.sample(list(_CLIFF_LAYERS), 3)
    U = _CLIFF_LAYERS[names[2]] * _CLIFF_LAYERS[names[1]] * _CLIFF_LAYERS[names[0]]
    Uw = _CLIFF_LAYERS[names[0]] * _CLIFF_LAYERS[names[1]] * _CLIFF_LAYERS[names[2]]
    P = _rand_pauli2(rng)
    Pm = kron(PAULI[P[0]], PAULI[P[1]])
    res, wres = _pauli_string(U * Pm * dag(U)), _pauli_string(Uw * Pm * dag(Uw))
    steps, cur = [], Pm
    for nm in names:
        cur = _CLIFF_LAYERS[nm] * cur * dag(_CLIFF_LAYERS[nm])
        steps.append(f"after {nm}: ${_ptex(_pauli_string(cur))}$")
    return Problem(
        prompt=(f"A two-qubit circuit applies, in time order: {', '.join(names)}. Track the Pauli operator through it: give "
                f"$U({P[0]}\\otimes {P[1]})U^\\dagger$ as a signed Pauli string (this is how stabilizer simulators work)."),
        answer=res, kind="text", answer_hint=HINT_PAULI,
        rubric=["Conjugate by the layers in time order (first layer first)", _CNOT_RULES,
                "H: X ↔ Z, Y → −Y;  S: X → Y, Y → −X;  CZ: X⊗I → X⊗Z, I⊗X → Z⊗X", "Track the overall sign"],
        solution="Layer by layer: " + "; ".join(steps) + ".",
        misconceptions=[(wres, "reversed-gate-order")] if wres != res else [],
    )


# ---------------------------------------------------------------- qc.entanglement
_W3 = sp.Matrix([0, 1, 1, 0, 1, 0, 0, 0]) / sp.sqrt(3)
_GHZ3 = sp.Matrix([1, 0, 0, 0, 0, 0, 0, 1]) / R2


def _concurrence(c) -> sp.Expr:
    return cs(2 * sp.Abs(c[0] * c[3] - c[1] * c[2]))


@gen("qc.entanglement")
def entanglement_drills(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        b, P, q = rng.choice(_BELL_ORDER), rng.choice(["X", "Y", "Z"]), rng.choice([1, 2])
        out = (kron(PAULI[P], ID) if q == 1 else kron(ID, PAULI[P])) * _bell(b)
        tgt = next(n for n in _BELL_ORDER if _same_ray(out, _bell(n)))
        return Problem(
            prompt=f"Applying ${P}$ to qubit {q} of ${BELL_TEX[b]}$ gives which Bell state (up to a global phase)?" + _BELL_OPTS,
            answer=_LETTERS[_BELL_ORDER.index(tgt)], kind="choice",
            rubric=["Write the Bell state as a sum of two basis kets", f"Apply {P} to qubit {q} of each term",
                    "X flips the bit, Z adds (−1)^bit, Y = iXZ does both", "Match with Φ±, Ψ± ignoring an overall factor"],
            solution=f"$({P}\\text{{ on qubit {q}}}){BELL_TEX[b]} = {state_tex(out)} \\simeq {BELL_TEX[tgt]}$.",
            misconceptions=[(_LETTERS[_BELL_ORDER.index(b)], "assumed-state-unchanged")],
        )
    if level == 1:
        b, obs, a = rng.choice(_BELL_ORDER), rng.choice(["Z", "X"]), rng.choice([1, -1])
        corr = int(expval(kron(GATES[obs], GATES[obs]), _bell(b)))
        bob = a * corr
        return Problem(
            prompt=(f"Alice and Bob share ${BELL_TEX[b]}$. Alice measures ${obs}$ on her qubit (qubit 1) and gets ${a:+d}$. Bob then "
                    f"measures ${obs}$ on qubit 2. Which outcome ($+1$ or $-1$) does he get with certainty?"),
            answer=str(bob), kind="num", answer_hint="+1 or -1",
            rubric=[f"Bell states are eigenstates of {obs}⊗{obs}: find the eigenvalue", "Product of outcomes = that eigenvalue",
                    "Bob's outcome = eigenvalue × Alice's outcome"],
            solution=(f"$\\langle {obs}\\otimes {obs}\\rangle = {corr}$ on ${BELL_TEX[b]}$, so the outcomes multiply to ${corr}$: "
                      f"Bob gets ${bob:+d}$. (No signalling: Bob's own statistics are 50/50 until he compares notes.)"),
            misconceptions=[(str(-bob), "assumed-phi-plus-correlations")],
        )
    if level == 2 and v == 0:
        a, b = nz(rng, 1, 4), nz(rng, 1, 4)
        pair = rng.choice([(0, 3), (1, 2)])
        amps = [0, 0, 0, 0]
        amps[pair[0]], amps[pair[1]] = a, b * rng.choice([1, -1])
        n2 = a * a + b * b
        p = sp.Rational(a * a, n2)
        S = -(p * sp.log(p, 2) + (1 - p) * sp.log(1 - p, 2)) if p != sp.Rational(1, 2) else sp.Integer(1)
        return Problem(
            prompt=(f"Compute the entanglement entropy $S(\\rho_1) = -\\mathrm{{Tr}}\\,\\rho_1\\log_2\\rho_1$ (in bits) of "
                    f"$\\lvert\\psi\\rangle = {_norm_tex(amps)}$, where $\\rho_1$ is the reduced state of qubit 1."),
            answer=fmt(S), kind="num", tol=1e-3, answer_hint=HINT_S,
            rubric=["The state is already in Schmidt form: two orthogonal terms", "ρ1 = diag(p, 1 − p) with p = |a|²/norm²",
                    "S = −p log2 p − (1−p) log2(1−p)"],
            solution=(f"$\\rho_1 = \\mathrm{{diag}}({L(p)}, {L(1 - p)})$, so $S = -{L(p)}\\log_2{L(p)} - {L(1 - p)}\\log_2{L(1 - p)} "
                      f"\\approx {float(S):.4f}$ bits (1 bit = maximally entangled)."),
            misconceptions=[(fmt(sp.expand_log(S * sp.log(2), force=True)), "used-natural-log")] if p != sp.Rational(1, 2) else [],
        )
    if level == 2:
        a, b = rng.choice(_SIMPLE), rng.choice(_SIMPLE)
        psi = kron(STATES[a], STATES[b])
        c = [cs((dag(_bell(n)) * psi)[0, 0]) for n in _BELL_ORDER]
        return Problem(
            prompt=(f"Expand ${ket(a)}\\otimes{ket(b)}$ in the Bell basis: give the coefficients of $\\lvert\\Phi^+\\rangle, "
                    "\\lvert\\Phi^-\\rangle, \\lvert\\Psi^+\\rangle, \\lvert\\Psi^-\\rangle$ in that order "
                    "($\\lvert\\Phi^\\pm\\rangle = (\\lvert00\\rangle\\pm\\lvert11\\rangle)/\\sqrt2$, "
                    "$\\lvert\\Psi^\\pm\\rangle = (\\lvert01\\rangle\\pm\\lvert10\\rangle)/\\sqrt2$)."),
            answer=vans(c), kind="vector", answer_hint="four coefficients [cΦ+, cΦ−, cΨ+, cΨ−], e.g. [1/2, 1/2, -i/2, 0]",
            rubric=["Write the product state as 4 amplitudes c00, c01, c10, c11",
                    "Coefficient of a Bell state = its inner product with the state", "cΦ± = (c00 ± c11)/√2, cΨ± = (c01 ± c10)/√2"],
            solution=(f"Amplitudes ${vec_tex(psi)}$; the Bell coefficients are $({', '.join(L(x) for x in c)})$."),
        )
    if v == 0:
        for _ in range(400):
            amps = [rng.randint(-3, 3) for _ in range(4)]
            det = amps[0] * amps[3] - amps[1] * amps[2]
            n2 = sum(x * x for x in amps)
            if det == 0:
                continue
            disc = 1 - sp.Rational(4 * det * det, n2 * n2)
            if sp.sqrt(disc).is_rational:
                break
        lam1, lam2 = cs((1 + sp.sqrt(disc)) / 2), cs((1 - sp.sqrt(disc)) / 2)
        return Problem(
            prompt=(f"Find the Schmidt decomposition of $\\lvert\\psi\\rangle = {_norm_tex(amps)}$ and give the squared Schmidt "
                    "coefficients $\\lambda_1^2 \\ge \\lambda_2^2$ (the eigenvalues of the reduced density matrix $\\rho_1$)."),
            answer=[fmt(lam1), fmt(lam2)], kind="list", answer_hint="two numbers, larger first, e.g. 4/5, 1/5",
            rubric=["Coefficient matrix C = [[c00, c01],[c10, c11]] (normalised)", "ρ1 = C C†: trace 1, det = |det C|²",
                    "Eigenvalues solve λ² − λ + |det C|² = 0", "Schmidt coefficients are the singular values of C"],
            solution=(f"$\\det C = {L(sp.Rational(det, n2))}$, so $\\lambda^2 - \\lambda + {L(sp.Rational(det * det, n2 * n2))} = 0$ gives "
                      f"$\\lambda_1^2 = {L(lam1)}$, $\\lambda_2^2 = {L(lam2)}$."),
            misconceptions=[([fmt(cs(sp.sqrt(lam1))), fmt(cs(sp.sqrt(lam2)))], "gave-schmidt-coefficients-not-squares")],
        )
    name = rng.choice(["GHZ", "W"])
    psi = _GHZ3 if name == "GHZ" else _W3
    stex = ("\\tfrac{1}{\\sqrt2}(\\lvert000\\rangle + \\lvert111\\rangle)" if name == "GHZ"
            else "\\tfrac{1}{\\sqrt3}(\\lvert001\\rangle + \\lvert010\\rangle + \\lvert100\\rangle)")
    while True:
        basis = rng.choice(["Z", "X"])
        m = rng.choice(["0", "1"] if basis == "Z" else ["+", "-"])
        post = (kron(dag(STATES[m]), sp.eye(4)) * psi).applyfunc(cs)
        prob = cs(sum(sp.Abs(x) ** 2 for x in post))
        if prob != 0:
            break
    postn = normalize(post)
    C = _concurrence(postn)
    btex = "computational" if basis == "Z" else "$\\{\\lvert+\\rangle,\\lvert-\\rangle\\}$"
    return Problem(
        prompt=(f"Three qubits are in the {name} state ${stex}$. Qubit 1 is measured in the {btex} basis with outcome ${ket(m)}$. "
                "Compute the concurrence $C = 2|c_{00}c_{11} - c_{01}c_{10}|$ of the normalised post-measurement state of qubits 2 and 3."),
        answer=fmt(C), kind="num",
        rubric=[f"Project qubit 1 onto {ket(m)}: (⟨{m}|⊗I⊗I)|ψ⟩", f"Outcome probability = norm² (here {L(prob)})",
                "Renormalise the 2-qubit state", "Apply the concurrence formula"],
        solution=(f"Unnormalised post-state ${state_tex(post)}$ with probability ${L(prob)}$; normalised: ${state_tex(postn)}$, "
                  f"so $C = {L(C)}$. GHZ entanglement is fragile under $Z$ measurement but not under $X$; W keeps some entanglement."),
        misconceptions=[(fmt(_concurrence(post)), "forgot-normalisation")],
    )


# ---------------------------------------------------------------- qc.teleportation
_TEL_RES = {"Psi+": X, "Phi-": Z, "Psi-": X * Z}
_CORR_OPS = {"A": ID, "B": X, "C": Z, "D": Z * X}


def _prop_identity(M) -> bool:
    return cs(M[0, 1]) == 0 and cs(M[1, 0]) == 0 and cs(M[0, 0] - M[1, 1]) == 0


@gen("qc.teleportation")
def teleport_drills(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        k = rng.randint(1, 9)
        if rng.random() < 0.5:
            return Problem(
                prompt=(f"To teleport {k} unknown qubit state(s) from Alice to Bob, how many classical bits must Alice send, and how "
                        "many shared Bell pairs (ebits) are consumed? Give (bits, ebits)."),
                answer=[str(2 * k), str(k)], kind="list", answer_hint="two integers, e.g. 4, 2",
                rubric=["Each teleportation uses one Bell pair", "Alice's Bell measurement has 4 outcomes = 2 bits per qubit"],
                solution=f"$2\\cdot{k} = {2 * k}$ classical bits and ${k}$ ebits: 1 qubit = 2 cbits + 1 ebit.",
                misconceptions=[([str(k), str(k)], "one-bit-per-qubit")],
            )
        return Problem(
            prompt=(f"With superdense coding, Alice wants to send {2 * k} classical bits to Bob. How many qubits must she physically send, "
                    "and how many pre-shared Bell pairs are used? Give (qubits, ebits)."),
            answer=[str(k), str(k)], kind="list", answer_hint="two integers, e.g. 3, 3",
            rubric=["Each sent qubit (half of a Bell pair) carries 2 bits", "One Bell pair per sent qubit"],
            solution=f"${2 * k}/2 = {k}$ qubits and ${k}$ ebits: 1 qubit + 1 ebit = 2 cbits.",
            misconceptions=[([str(2 * k), str(k)], "one-bit-per-qubit")],
        )
    if level == 1:
        b = rng.choice(_BELL_ORDER)
        out = kron(H, ID) * CNOT12 * _bell(b)
        idx = next(i for i, x in enumerate(out) if cs(x) != 0)
        bits = format(idx, "02b")
        return Problem(
            prompt=(f"Superdense decoding: Bob holds both qubits in ${BELL_TEX[b]}$, applies CNOT$_{{1\\to2}}$ and then $H$ on qubit 1, "
                    "and measures both. Which bits $q_1q_2$ does he read?"),
            answer=bits, kind="text", answer_hint="two bits, e.g. 10",
            rubric=["CNOT maps Φ± → (|0⟩ ± |1⟩)|0⟩/√2 and Ψ± → (|0⟩ ± |1⟩)|1⟩/√2", "H turns |±⟩ into |0⟩/|1⟩",
                    "Read the resulting basis state (an overall sign is irrelevant)"],
            solution=f"$(H\\otimes I)\\,\\mathrm{{CNOT}}\\,{BELL_TEX[b]} = {state_tex(out)}$, so Bob reads ${bits}$.",
            misconceptions=[(bits[::-1], "swapped-measurement-bits")] if bits[0] != bits[1] else [],
        )
    if level == 2 and v == 0:
        while True:
            al, be = nz(rng, 1, 4), rand_amp(rng, True, -4, 4)
            n2 = al ** 2 + sp.Abs(be) ** 2
            psi = vec(al, be) / sp.sqrt(n2)
            br = {f"{m1}{m2}": (X ** m2) * (Z ** m1) * psi for m1 in (0, 1) for m2 in (0, 1)}
            if all(not _veq(br[x], br[y]) for x, y in itertools.combinations(br, 2)):
                break
        key = rng.choice(list(br))
        bob = br[key]
        return Problem(
            prompt=(f"Alice teleports $\\lvert\\psi\\rangle = {_norm_tex([al, be])}$ with the standard protocol (resource "
                    "$\\lvert\\Phi^+\\rangle$; CNOT then $H$ on her qubit; outcomes $M_1$, $M_2$). Before any correction, Bob's qubit is "
                    f"${_norm_tex(list(cs(x * sp.sqrt(n2)) for x in bob))}$. Which outcome $M_1M_2$ did Alice obtain?"),
            answer=key, kind="text", answer_hint="two bits M1M2, e.g. 01",
            rubric=["Bob's uncorrected qubit is X^{M2} Z^{M1}|ψ⟩", "Z^{M1}: sign of β flipped?", "X^{M2}: amplitudes swapped?",
                    "Read off M1 and M2"],
            solution=("The four branches $X^{M_2}Z^{M_1}\\lvert\\psi\\rangle$ are: "
                      + "; ".join(f"${x}$: ${_norm_tex([cs(c * sp.sqrt(n2)) for c in br[x]])}$" for x in sorted(br))
                      + f". Bob's qubit matches $M_1M_2 = {key}$."),
            misconceptions=[(key[::-1], "swapped-measurement-bits")] if key[0] != key[1] else [],
        )
    if level == 2:
        al, be = nz(rng, 1, 4), rand_amp(rng, True, -4, 4)
        n2 = al ** 2 + sp.Abs(be) ** 2
        psi = vec(al, be) / sp.sqrt(n2)
        start = kron(psi, _bell("Phi+"))
        after = kron(CNOT12, ID) * start
        return Problem(
            prompt=(f"Alice holds $\\lvert\\psi\\rangle = {_norm_tex([al, be])}$ (qubit 1) and shares $\\lvert\\Phi^+\\rangle$ with Bob "
                    "(qubits 2, 3). She applies CNOT from qubit 1 to qubit 2. Give the 3-qubit state as an 8-vector ordered "
                    "$\\lvert000\\rangle, \\lvert001\\rangle, \\ldots, \\lvert111\\rangle$ (labels $q_1q_2q_3$)."),
            answer=vans(after), kind="vector", answer_hint="8 amplitudes [c000, c001, ..., c111], each typed separately",
            rubric=["Expand |ψ⟩|Φ+⟩ = (α|0⟩ + β|1⟩)(|00⟩ + |11⟩)/√2", "CNOT flips qubit 2 in the terms where qubit 1 is 1",
                    "List the 8 amplitudes in binary order"],
            solution=(f"$\\lvert\\psi\\rangle\\lvert\\Phi^+\\rangle = {state_tex(start)}$; after the CNOT: ${state_tex(after)}$."),
            misconceptions=[(vans(start), "forgot-cnot")],
        )
    if v == 0:
        res = rng.choice(list(_TEL_RES))
        m1, m2 = rng.randint(0, 1), rng.randint(0, 1)
        pre = _TEL_RES[res] * (X ** m2) * (Z ** m1)
        ans = next(k for k, C in _CORR_OPS.items() if _prop_identity(C * pre))
        std = _CORR[f"{m1}{m2}"]
        return Problem(
            prompt=(f"Teleportation with a different resource: Alice and Bob share ${BELL_TEX[res]}$ instead of $\\lvert\\Phi^+\\rangle$ "
                    f"(Alice holds qubit 2, Bob qubit 3). Alice runs the standard protocol (CNOT, then $H$, measure) and gets "
                    f"$M_1 = {m1}$, $M_2 = {m2}$. Which correction makes Bob's qubit equal to $\\lvert\\psi\\rangle$ up to a global phase?"
                    "\n\nA) nothing  B) $X$  C) $Z$  D) $X$ then $Z$ (i.e. $ZX$)"),
            answer=ans, kind="choice",
            rubric=[f"Write {BELL_TEX[res]} = (I ⊗ P)|Φ+⟩ for a Pauli P on Bob's qubit",
                    "P commutes with everything Alice does, so Bob holds P X^{M2} Z^{M1}|ψ⟩",
                    "Find the correction C with C·P·X^{M2}Z^{M1} ∝ I (Paulis commute up to sign)"],
            solution=(f"${BELL_TEX[res]} = (I\\otimes P)\\lvert\\Phi^+\\rangle$ with $P = {'X' if res == 'Psi+' else 'Z' if res == 'Phi-' else 'XZ'}$; "
                      f"Bob holds $P X^{{{m2}}} Z^{{{m1}}}\\lvert\\psi\\rangle$, undone (up to a sign) by option {ans}."),
            misconceptions=[(std, "assumed-phi-plus-resource")] if std != ans else [],
        )
    b12, b34, bo = rng.choice(_BELL_ORDER), rng.choice(_BELL_ORDER), rng.choice(_BELL_ORDER)
    full = kron(_bell(b12), _bell(b34))
    bv = _bell(bo)
    r = sp.Matrix(4, 1, lambda idx, _: sum(sp.conjugate(bv[i2 * 2 + i3]) * full[(idx // 2) * 8 + i2 * 4 + i3 * 2 + idx % 2]
                                          for i2 in (0, 1) for i3 in (0, 1)))
    r = normalize(r)
    tgt = next(n for n in _BELL_ORDER if _same_ray(r, _bell(n)))
    return Problem(
        prompt=(f"Entanglement swapping: qubits 1, 2 are in ${BELL_TEX[b12]}$ and qubits 3, 4 in ${BELL_TEX[b34]}$. A Bell measurement on "
                f"qubits 2 and 3 yields ${BELL_TEX[bo]}$. Which Bell state (up to global phase) do qubits 1 and 4 now share, although "
                "they never interacted?" + _BELL_OPTS),
        answer=_LETTERS[_BELL_ORDER.index(tgt)], kind="choice",
        rubric=["Write each Bell state as (I⊗P)|Φ+⟩ with P ∈ {I, X, Z, XZ}",
                "Key identity: (⟨Φ+|_{23})|Φ+⟩_{12}|Φ+⟩_{34} = ½|Φ+⟩_{14}",
                "Move the Paulis through to qubits 1 and 4 (transpose when moving across a Φ+)",
                "Or: expand the 16 amplitudes and project directly"],
        solution=f"Projecting qubits 2, 3 onto ${BELL_TEX[bo]}$ leaves qubits 1, 4 in ${state_tex(r)} \\simeq {BELL_TEX[tgt]}$.",
        misconceptions=[(_LETTERS[_BELL_ORDER.index(bo)], "assumed-same-as-outcome")] if bo != tgt else [],
    )


# ---------------------------------------------------------------- qc.oracles
def _dot2(x: int, y: int) -> int:
    return bin(x & y).count("1") % 2


@gen("qc.oracles")
def oracle_drills(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        f = [rng.randint(0, 1) for _ in range(4)]
        x, y = rng.randrange(4), rng.randint(0, 1)
        xs = format(x, "02b")
        out = f"{xs}{y ^ f[x]}"
        return Problem(
            prompt=(f"The bit oracle $U_f\\lvert x\\rangle\\lvert y\\rangle = \\lvert x\\rangle\\lvert y\\oplus f(x)\\rangle$ for the $f$ below "
                    f"acts on $\\lvert {xs}\\rangle\\lvert {y}\\rangle$. Which basis state comes out? Type the three bits $x_1x_2y'$."
                    f"\n\n{_table_tex(2, f)}"),
            answer=out, kind="text", answer_hint="three bits, e.g. 101",
            rubric=[f"Look up f({xs}) in the table", "The input register x is unchanged", "y' = y XOR f(x)"],
            solution=f"$f({xs}) = {f[x]}$, so $y' = {y}\\oplus{f[x]} = {y ^ f[x]}$: output $\\lvert {xs}\\rangle\\lvert {y ^ f[x]}\\rangle$.",
            misconceptions=[(f"{xs}{f[x]}", "overwrote-instead-of-xor")],
        )
    if level == 1:
        while True:
            f = [rng.randint(0, 1) for _ in range(4)]
            if any(f):
                break
        out = [sp.Rational((-1) ** fx, 2) for fx in f]
        return Problem(
            prompt=("Apply the phase oracle $O_f\\lvert x\\rangle = (-1)^{f(x)}\\lvert x\\rangle$ to the uniform two-qubit superposition "
                    "$\\tfrac12\\sum_x\\lvert x\\rangle$, with $f$ given in the table. Give the 4 amplitudes in the order "
                    f"$\\lvert00\\rangle, \\lvert01\\rangle, \\lvert10\\rangle, \\lvert11\\rangle$.\n\n{_table_tex(2, f)}"),
            answer=[fmt(x) for x in out], kind="vector", answer_hint="4 amplitudes, e.g. [1/2, -1/2, 1/2, 1/2]",
            rubric=["Each amplitude starts at 1/2", "Multiply by −1 wherever f(x) = 1"],
            solution=f"Amplitudes $({', '.join(L(x) for x in out)})$ — the oracle writes $f$ into relative phases.",
            misconceptions=[(["1/2"] * 4, "ignored-phase-kickback")],
        )
    if level == 2 and v == 0:
        s = rng.randrange(1, 8)
        ys = [y for y in range(1, 8) if _dot2(y, s) == 0]
        y1, y2 = rng.sample(ys, 2)
        ss = format(s, "03b")
        return Problem(
            prompt=("Simon's problem on $n=3$ bits: $f(x) = f(x\\oplus s)$ for a hidden $s\\neq 000$. Each run returns some $y$ with "
                    f"$y\\cdot s = 0 \\pmod 2$. Two runs gave $y = {format(y1, '03b')}$ and $y = {format(y2, '03b')}$. Find $s$."),
            answer=ss, kind="text", answer_hint=HINT_BITS,
            rubric=["Each y gives one linear equation y·s = 0 (mod 2)", "Two independent equations in 3 unknowns: 2 solutions",
                    "Discard s = 000"],
            solution=(f"Solving $y_1\\cdot s = y_2\\cdot s = 0$ over $\\mathbb{{F}}_2$ leaves $\\{{000, {ss}\\}}$, so $s = {ss}$. "
                      "Classically one needs $\\sim 2^{n/2}$ queries; Simon needs $O(n)$."),
            misconceptions=[(format(y1 ^ y2, "03b"), "confused-y-with-s")] + ([(ss[::-1], "little-endian-qiskit-ordering")] if ss != ss[::-1] else []),
        )
    if level == 2:
        n = rng.choice([2, 3])
        N = 2 ** n
        ones = set(rng.sample(range(N), N // 2))
        f = [1 if x in ones else 0 for x in range(N)]
        z = rng.randrange(1, N)
        amp = sp.Rational(sum((-1) ** (f[x] + _dot2(x, z)) for x in range(N)), N)
        zs = format(z, f"0{n}b")
        return Problem(
            prompt=(f"The Deutsch–Jozsa circuit ($H^{{\\otimes n}}$, phase oracle via an ancilla in $\\lvert-\\rangle$, $H^{{\\otimes n}}$) "
                    f"runs on $n={n}$ qubits with the balanced $f$ below. What is the probability of measuring the specific string "
                    f"$z = {zs}$?\n\n{_table_tex(n, f)}"),
            answer=fmt(amp ** 2), kind="num",
            rubric=["After the oracle: (1/√N) Σ_x (−1)^{f(x)}|x⟩", "H^{⊗n}|x⟩ = (1/√N) Σ_z (−1)^{x·z}|z⟩",
                    "Amplitude of |z⟩ = (1/N) Σ_x (−1)^{f(x) + x·z}", "Square it"],
            solution=(f"$\\sum_x (-1)^{{f(x) + x\\cdot z}} = {amp * N}$, amplitude ${L(amp)}$, probability ${L(amp ** 2)}$."),
            misconceptions=[(fmt(amp), "forgot-amplitude-squared")] if amp not in (0, 1, -1) else [],
        )
    if v == 0:
        n = rng.randint(3, 6)
        P = sp.prod([1 - sp.Rational(2 ** i, 2 ** (n - 1)) for i in range(n - 1)])
        return Problem(
            prompt=(f"In Simon's algorithm on $n={n}$ bits, each run returns $y$ uniformly at random from the $2^{{n-1}}$ strings with "
                    f"$y\\cdot s = 0$ (including $y = 0$). Derive the probability that the first $n-1 = {n - 1}$ runs already give linearly "
                    "independent $y$'s over $\\mathbb{F}_2$ (so $s$ can be solved for)."),
            answer=fmt(P), kind="num",
            rubric=["The y's live in an (n−1)-dimensional subspace with 2^{n−1} elements",
                    "Run i+1 must avoid the span of the first i vectors (2^i elements)", "P = Π_{i=0}^{n−2} (1 − 2^i / 2^{n−1})"],
            solution=(f"$P = \\prod_{{i=0}}^{{{n - 2}}}\\left(1 - \\frac{{2^i}}{{2^{{{n - 1}}}}}\\right) = {L(P)}$ "
                      "(≈ 0.29 for large $n$; a few extra runs push it close to 1)."),
            misconceptions=[(fmt(sp.prod([1 - sp.Rational(2 ** i, 2 ** (n - 1)) for i in range(1, n - 1)])), "forgot-zero-vector")],
        )
    n = rng.choice([2, 3])
    N = 2 ** n
    k = rng.randint(0, N)
    ones = set(rng.sample(range(N), k))
    f = [1 if x in ones else 0 for x in range(N)]
    P = sp.Rational((N - k) ** 2 + k ** 2, N * N)
    return Problem(
        prompt=(f"A student runs the Deutsch–Jozsa circuit on $n={n}$ qubits but forgets both the $X$ and the $H$ on the ancilla, so it "
                "stays in $\\lvert0\\rangle$ and $U_f$ simply writes $f(x)$ into it. Derive the probability that the input register "
                f"reads $0^{n}$ (ancilla not measured) for the $f$ below.\n\n{_table_tex(n, f)}"),
        answer=fmt(P), kind="num",
        rubric=["After U_f: (1/√N) Σ_x |x⟩|f(x)⟩ — entangled, no phase kickback",
                "Amplitude of |0^n⟩|y⟩ after H^{⊗n}: (1/N)·#{x : f(x) = y}",
                "Sum the probabilities over y ∈ {0, 1}", f"Here #{{f=1}} = k = {k}"],
        solution=(f"$P(0^{n}) = \\left(\\tfrac{{N-k}}{{N}}\\right)^2 + \\left(\\tfrac{{k}}{{N}}\\right)^2 = "
                  f"\\frac{{{(N - k) ** 2} + {k ** 2}}}{{{N * N}}} = {L(P)}$. Without the $\\lvert-\\rangle$ ancilla the phases never "
                  "reach the register and the algorithm loses its power."),
        misconceptions=[(fmt(sp.Rational(N - 2 * k, N) ** 2), "assumed-phase-kickback")],
    )


# ---------------------------------------------------------------- qc.grover
def _grover_iter(N, M, k):
    """Exact amplitudes (marked, unmarked) after k Grover iterations from the uniform state."""
    a = b = 1 / sp.sqrt(N)
    hist = [(a, b)]
    for _ in range(k):
        a = -a
        mean = (M * a + (N - M) * b) / N
        a, b = sp.nsimplify(sp.simplify(2 * mean - a)), sp.simplify(2 * mean - b)
        hist.append((a, b))
    return hist


@gen("qc.grover")
def grover_drills(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        n = rng.randint(2, 8)
        N = 2 ** n
        frac = rng.choice([sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(3, 4)])
        M = int(N * frac)
        th = {sp.Rational(1, 4): PI / 6, sp.Rational(1, 2): PI / 4, sp.Rational(3, 4): PI / 3}[frac]
        return Problem(
            prompt=(f"Grover search over $N = {N}$ items with $M = {M}$ marked items. Each Grover iteration rotates the state by $2\\theta$ "
                    "in the plane of the (good, bad) states, where $\\sin\\theta = \\sqrt{M/N}$. What is the rotation angle $2\\theta$?"),
            answer=fmt(2 * th), kind="num", answer_hint="an angle in radians, e.g. pi/3",
            rubric=[f"M/N = {L(frac)}", "θ = arcsin(√(M/N))", "Double it"],
            solution=f"$\\sin\\theta = \\sqrt{{{L(frac)}}} \\Rightarrow \\theta = {L(th)}$, so $2\\theta = {L(2 * th)}$.",
            misconceptions=[(fmt(th), "forgot-factor-two")],
        )
    if level == 1:
        n = rng.randint(2, 5)
        N = 2 ** n
        M = rng.randint(1, max(1, N // 4))
        mean = sp.Rational(N - 2 * M, N) / sp.sqrt(N)
        return Problem(
            prompt=(f"Start from the uniform superposition over $N = {N}$ items (all amplitudes $1/\\sqrt{{{N}}}$) and apply the Grover "
                    f"oracle, which flips the sign of the $M = {M}$ marked item(s). What is the mean amplitude $\\bar a$ now?"),
            answer=fmt(cs(mean)), kind="num", answer_hint="exact, e.g. 3/(4*sqrt(8))",
            rubric=["N − M amplitudes are +1/√N, M are −1/√N", "Mean = (sum)/N"],
            solution=f"$\\bar a = \\frac{{({N} - {M}) - {M}}}{{{N}\\sqrt{{{N}}}}} = {L(cs(mean))}$.",
            misconceptions=[(fmt(1 / sp.sqrt(N)), "ignored-oracle"), (fmt(cs(sp.Rational(N - M, N) / sp.sqrt(N))), "dropped-marked-terms")],
        )
    if level == 2 and v == 0:
        N = rng.choice([8, 16, 32])
        hist = _grover_iter(N, 1, 2)
        P = cs(hist[2][0] ** 2)
        P1 = cs(hist[1][0] ** 2)
        return Problem(
            prompt=(f"Grover search, $N = {N}$, one marked item. Using inversion about the mean, track the marked amplitude and the common "
                    "unmarked amplitude through **two** iterations, starting from the uniform superposition, and give the success probability."),
            answer=fmt(P), kind="num",
            rubric=["Start a = b = 1/√N", "Oracle: a → −a", "Diffuser: mean m = (−a + (N−1)b)/N, then a → 2m + a, b → 2m − b",
                    "Repeat once more and square the marked amplitude"],
            solution=(f"After one iteration $a_1 = {L(cs(hist[1][0]))}$, $b_1 = {L(cs(hist[1][1]))}$; after two "
                      f"$a_2 = {L(cs(hist[2][0]))}$, so $P = {L(P)}$ (≈ {float(P):.4f}; equals $\\sin^2 5\\theta$)."),
            misconceptions=[(fmt(P1), "stopped-after-one-iteration")],
        )
    if level == 2:
        N = rng.choice([8, 16, 32, 64])
        M = rng.randint(2, max(2, N // 8))
        a1 = _grover_iter(N, M, 1)[1][0]
        P = cs(M * a1 ** 2)
        return Problem(
            prompt=(f"Grover search over $N = {N}$ items with $M = {M}$ marked items. After **one** iteration (oracle, then inversion about "
                    "the mean) from the uniform superposition, what is the probability of measuring *some* marked item?"),
            answer=fmt(P), kind="num",
            rubric=["All marked amplitudes stay equal (symmetry)", "Oracle flips the M marked amplitudes",
                    "Mean = (N − 2M)/(N√N); marked amplitude → 2·mean + 1/√N", "Multiply the squared amplitude by M"],
            solution=(f"Marked amplitude after one iteration ${L(cs(a1))}$; success probability $M a^2 = {L(P)}$ "
                      f"$= \\sin^2 3\\theta$ with $\\sin^2\\theta = {L(sp.Rational(M, N))}$."),
            misconceptions=[(fmt(cs(a1 ** 2)), "forgot-to-sum-over-marked")],
        )
    if v == 0:
        N = 2 ** rng.randint(3, 10)
        M = rng.randint(1, 4)
        tr = 2 - sp.Rational(4 * M, N)
        return Problem(
            prompt=("Restricted to the plane spanned by the good and bad states, the Grover iterate $G = D\\,O$ is a rotation by $2\\theta$ "
                    f"with $\\sin^2\\theta = M/N$. For $N = {N}$, $M = {M}$, derive the eigenvalues of $G$ on this plane and give "
                    "$\\mathrm{Tr}\\,G$ (restricted to the plane) exactly."),
            answer=fmt(tr), kind="num",
            rubric=["A 2D rotation by 2θ has eigenvalues e^{±2iθ}", "Trace = 2cos 2θ", "cos 2θ = 1 − 2 sin²θ = 1 − 2M/N"],
            solution=(f"$\\mathrm{{Tr}}\\,G = 2\\cos2\\theta = 2(1 - 2M/N) = {L(tr)}$. The eigenphases $\\pm2\\theta$ are what "
                      "quantum counting estimates with phase estimation."),
            misconceptions=[(fmt(cs(2 * sp.sqrt(1 - sp.Rational(M, N)))), "used-theta-not-2theta")],
        )
    r = rng.choice(["4", "2", "4/3"])
    j = rng.randint(0, 3)
    M, N = {"4": (2 ** j, 4 * 2 ** j), "2": (2 ** j, 2 * 2 ** j), "4/3": (3 * 2 ** j, 4 * 2 ** j)}[r]
    th = {"4": PI / 6, "2": PI / 4, "4/3": PI / 3}[r]
    k = rng.randint(1, 5)
    P = cs(sp.sin((2 * k + 1) * th) ** 2)
    return Problem(
        prompt=(f"Grover search with $N = {N}$ and $M = {M}$ marked items. Derive the success probability $\\sin^2((2k+1)\\theta)$ after "
                f"$k = {k}$ iterations exactly, and explain on paper why more iterations can make things worse."),
        answer=fmt(P), kind="num",
        rubric=["sin²θ = M/N gives an exact angle θ", "Each iteration adds 2θ to the angle from the bad axis",
                f"Evaluate sin²({2 * k + 1}θ)", "Note the periodicity: overshooting rotates past the good state"],
        solution=(f"$\\sin^2\\theta = {L(sp.Rational(M, N))} \\Rightarrow \\theta = {L(th)}$; $(2k+1)\\theta = {L((2 * k + 1) * th)}$, so "
                  f"$P = {L(P)}$."),
        misconceptions=[(fmt(cs(sp.sin((2 * k + 1) * th))), "forgot-amplitude-squared")] if P not in (0, 1) else [],
    )


# ---------------------------------------------------------------- qc.qft
@gen("qc.qft")
def qft_drills(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        j = rng.randrange(1, 8)
        l = rng.randint(1, 3)
        jb = format(j, "03b")
        ph = cs(2 * PI * sp.Rational(j % 2 ** l, 2 ** l))
        frac = "0." + jb[3 - l:]
        return Problem(
            prompt=("In the product form $\\mathrm{QFT}_8\\lvert j_1j_2j_3\\rangle = \\tfrac{1}{2\\sqrt2}"
                    "\\left(\\lvert0\\rangle + e^{2\\pi i\\,0.j_3}\\lvert1\\rangle\\right)"
                    "\\left(\\lvert0\\rangle + e^{2\\pi i\\,0.j_2j_3}\\lvert1\\rangle\\right)"
                    "\\left(\\lvert0\\rangle + e^{2\\pi i\\,0.j_1j_2j_3}\\lvert1\\rangle\\right)$ (binary fractions; output qubit 1 is the "
                    f"left factor), what relative phase $\\phi\\in[0,2\\pi)$ multiplies $\\lvert1\\rangle$ on output qubit {l} for the input "
                    f"${ket(jb)}$?"),
            answer=fmt(ph), kind="num", answer_hint="an angle in [0, 2π), e.g. 3*pi/2",
            rubric=[f"Output qubit {l} uses the binary fraction of the last {l} input bit(s)", f"0.{jb[3 - l:]} in binary = "
                    f"{sp.Rational(j % 2 ** l, 2 ** l)}", "φ = 2π × that fraction"],
            solution=f"${frac}_2 = {L(sp.Rational(j % 2 ** l, 2 ** l))}$, so $\\phi = 2\\pi\\cdot{L(sp.Rational(j % 2 ** l, 2 ** l))} = {L(ph)}$.",
            misconceptions=[(fmt(cs(2 * PI * sp.Rational(j, 8))), "used-full-fraction")],
        )
    if level == 1:
        j = rng.randrange(4)
        out = qft_matrix(4) * sp.Matrix([1 if k == j else 0 for k in range(4)])
        jb = format(j, "02b")
        wrong = format((-j) % 4, "02b")
        return Problem(
            prompt=(f"The 2-qubit state $\\tfrac12\\left({state_tex(2 * out)}\\right)$ equals $\\mathrm{{QFT}}_4\\lvert j\\rangle$ for one basis "
                    "state (with $\\mathrm{QFT}\\lvert j\\rangle = \\tfrac12\\sum_k i^{jk}\\lvert k\\rangle$). Which $\\lvert j\\rangle$ — i.e. what does "
                    "the inverse QFT output? Type the two bits."),
            answer=jb, kind="text", answer_hint="two bits, e.g. 01",
            rubric=["The amplitude of |k⟩ is i^{jk}/2", "Look at the |01⟩ amplitude: it equals i^j / 2", "Solve for j"],
            solution=f"The $\\lvert01\\rangle$ amplitude is $i^j/2 = {L(out[1])}$, so $j = {j}$, i.e. ${ket(jb)}$.",
            misconceptions=[(wrong, "inverse-qft-sign")] if wrong != jb else [],
        )
    if level == 2:
        a, b = rng.choice(_SIMPLE), rng.choice(_SIMPLE)
        psi = kron(STATES[a], STATES[b])
        out = qft_matrix(4) * psi
        return Problem(
            prompt=(f"Apply $\\mathrm{{QFT}}_4$ (with $\\omega = i$) to the product state ${ket(a)}\\otimes{ket(b)}$. " + CONV),
            answer=vans(out), kind="vector", answer_hint=HINT_V2,
            rubric=["Expand the input into its 4 amplitudes x_j", "Output amplitude y_k = ½ Σ_j i^{jk} x_j",
                    "Or: multiply by the 4×4 QFT matrix"],
            solution=f"Input ${vec_tex(psi)}$, output ${vec_tex(out)}$, i.e. ${state_tex(out)}$.",
            misconceptions=[(vans(dag(qft_matrix(4)) * psi), "inverse-qft-sign")],
        )
    if v == 0:
        while True:
            r = rng.choice([3, 5, 6])
            x0 = rng.randrange(min(r, 8 - r if r > 4 else r))
            support = list(range(x0, 8, r))
            Msz = len(support)
            k = rng.randrange(1, 8)
            amp = sum(sp.exp(2 * PI * I * x * k / 8) for x in support) / sp.sqrt(Msz * 8)
            P = cs(sp.nsimplify(sp.simplify(sp.expand_complex(amp * sp.conjugate(amp)))))
            if P != 0:
                break
        return Problem(
            prompt=(f"A 3-qubit register holds the uniform superposition of $\\lvert x\\rangle$ for $x\\in\\{{{', '.join(map(str, support))}\\}}$ "
                    f"(spacing $r = {r}$, which does **not** divide $N = 8$). After $\\mathrm{{QFT}}_8$, derive the probability of the outcome "
                    f"$k = {k}$ exactly."),
            answer=fmt(P), kind="num", tol=1e-3, answer_hint="exact, e.g. (2+sqrt(2))/16, or a decimal to 4 places",
            rubric=[f"Amplitude of |k⟩ = (1/√({Msz}·8)) Σ_x e^{{2πi xk/8}}", "Factor out the offset phase; sum the remaining roots of unity",
                    "Take |·|² (multiply by the complex conjugate)"],
            solution=(f"$\\left|\\sum_x e^{{2\\pi i\\,x\\cdot{k}/8}}\\right|^2 = {L(cs(P * Msz * 8))}$, so $P(k={k}) = {L(P)}$ "
                      f"(≈ {float(P):.4f}). When $r\\nmid N$ the peaks near multiples of $N/r$ are broadened — the reason Shor needs "
                      "continued fractions."),
            misconceptions=[(fmt(sp.Rational(1, r)), "assumed-r-divides-N")],
        )
    n = rng.randint(6, 20)
    m = rng.randint(2, min(n - 1, 8))
    cnt = sum(min(n - j, m - 1) for j in range(1, n + 1))
    return Problem(
        prompt=(f"Approximate QFT (Coppersmith): on $n = {n}$ qubits keep only the controlled-$R_k$ gates with $k \\le {m}$, dropping the tiny "
                f"rotations $R_k$, $k > {m}$. How many controlled-phase gates remain? (In the full circuit qubit $j$ receives "
                "$R_2, \\ldots, R_{n-j+1}$.)"),
        answer=str(cnt), kind="num", answer_hint="an integer",
        rubric=["Qubit j receives R_2..R_{n−j+1}: n − j gates", f"Keeping k ≤ {m} leaves min(n − j, {m - 1}) gates on qubit j",
                "Sum over j = 1..n"],
        solution=(f"$\\sum_{{j=1}}^{{{n}}}\\min(n-j, {m - 1}) = {cnt}$ (vs. $n(n-1)/2 = {n * (n - 1) // 2}$ for the exact QFT): "
                  "$O(nm)$ gates with error $O(n/2^m)$."),
        misconceptions=[(str(n * (n - 1) // 2), "forgot-to-drop-gates"), (str(n * (m - 1)), "overcounted-last-qubits")],
    )


# ---------------------------------------------------------------- qc.phase_estimation
_QPE_U = [("Z", Z, "1"), ("S", S, "1"), ("T", T, "1"), ("S^\\dagger", dag(S), "1"), ("T^\\dagger", dag(T), "1"),
          ("X", X, "-"), ("ST", S * T, "1"), ("ZT", Z * T, "1"), ("Y", Y, "+i")]


def _eigphase(U, e) -> sp.Rational:
    lam = cs((dag(STATES[e]) * U * STATES[e])[0, 0])
    for m in range(8):
        if cs(lam - sp.exp(2 * PI * I * m / 8)) == 0:
            return sp.Rational(m, 8)
    raise ValueError("phase not a multiple of 1/8")


@gen("qc.phase_estimation")
def qpe_drills(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        j = rng.randrange(1, 8)
        k = rng.randrange(3)
        phi = sp.Rational(j, 8)
        ang = cs(2 * PI * sp.Rational((2 ** k * j) % 8, 8))
        return Problem(
            prompt=(f"In phase estimation, $U$ has eigenvalue $e^{{2\\pi i\\varphi}}$ with $\\varphi = {L(phi)}$. The counting qubit that "
                    f"controls $U^{{2^{{{k}}}}}$ (prepared in $\\lvert+\\rangle$) picks up a relative phase $e^{{i\\phi}}$ on its "
                    "$\\lvert1\\rangle$ by phase kickback. Give $\\phi\\in[0,2\\pi)$."),
            answer=fmt(ang), kind="num", answer_hint="an angle in [0, 2π), e.g. pi/2",
            rubric=[f"U^{{2^{k}}} has eigenvalue e^{{2πi·2^{k}φ}}", "Kickback puts that eigenvalue on the control's |1⟩",
                    "Reduce the angle mod 2π"],
            solution=f"$\\phi = 2\\pi\\cdot 2^{{{k}}}\\cdot{L(phi)} = {L(2 * PI * 2 ** k * phi)} \\equiv {L(ang)} \\pmod{{2\\pi}}$.",
            misconceptions=[(fmt(cs(2 * PI * phi)), "forgot-power-of-two")] if k else [],
        )
    if level == 1:
        nb = rng.randint(3, 12)
        eps = rng.choice([sp.Rational(1, 4), sp.Rational(1, 10), sp.Rational(1, 20), sp.Rational(1, 100)])
        val = 2 + 1 / (2 * eps)
        c = next(c for c in range(20) if 2 ** c >= val)
        return Problem(
            prompt=(f"How many counting qubits $t$ does phase estimation need to give $\\varphi$ accurate to $n = {nb}$ bits with success "
                    f"probability at least $1-\\varepsilon$, $\\varepsilon = {L(eps)}$? Use N&C Eq. (5.35): "
                    "$t = n + \\left\\lceil \\log_2\\left(2 + \\tfrac{1}{2\\varepsilon}\\right)\\right\\rceil$."),
            answer=str(nb + c), kind="num", answer_hint="an integer",
            rubric=["Compute 2 + 1/(2ε)", "Smallest c with 2^c ≥ that value", "Add n"],
            solution=f"$2 + \\tfrac{{1}}{{2\\varepsilon}} = {L(val)}$, $\\lceil\\log_2 {L(val)}\\rceil = {c}$, so $t = {nb} + {c} = {nb + c}$.",
            misconceptions=[(str(nb), "ignored-error-padding")],
        )
    if level == 2 and v == 0:
        name, U, e = rng.choice(_QPE_U)
        phi = _eigphase(U, e)
        bits = format(int(phi * 8), "03b")
        return Problem(
            prompt=(f"Phase estimation with $t = 3$ counting qubits is run on $U = {name}$ with the eigenstate ${ket(e)}$ in the target "
                    "register. Find the eigenvalue, write it as $e^{2\\pi i\\varphi}$ with $\\varphi\\in[0,1)$, and give the measured "
                    "3-bit string (most significant bit first)."),
            answer=bits, kind="text", answer_hint=HINT_BITS,
            rubric=[f"Compute {name}{ket(e)} = λ{ket(e)}", "Write λ = e^{2πiφ} with φ ∈ [0, 1)", "8φ is an integer: write it in 3 bits"],
            solution=(f"$\\lambda = {L(cs((dag(STATES[e]) * U * STATES[e])[0, 0]))} = e^{{2\\pi i\\cdot {L(phi)}}}$, $8\\varphi = {int(phi * 8)}$, "
                      f"outcome ${bits}$."),
            misconceptions=[(bits[::-1], "little-endian-qiskit-ordering")] if bits != bits[::-1] else [],
        )
    if level == 2:
        name, U, _ = rng.choice([q for q in _QPE_U if q[2] == "1"])
        phi1 = _eigphase(U, "1")
        a, b = nz(rng, 1, 4), rand_amp(rng, True, -4, 4)
        n2 = a * a + int(sp.Abs(b) ** 2)
        which = rng.choice([0, 1])
        bits = "000" if which == 0 else format(int(phi1 * 8), "03b")
        P = sp.Rational(a * a if which == 0 else int(sp.Abs(b) ** 2), n2)
        Pw = 1 - P
        return Problem(
            prompt=(f"Phase estimation ($t = 3$ counting qubits) on $U = {name}$, but the target register holds the superposition "
                    f"${_norm_tex([a, b])}$ of eigenstates of $U$. What is the probability that the counting register reads ${bits}$?"),
            answer=fmt(P), kind="num",
            rubric=["|0⟩ and |1⟩ are eigenvectors of the diagonal U with phases φ0 = 0 and φ1", "By linearity QPE runs on each branch",
                    "Outcome for eigenstate |u⟩ occurs with probability |⟨u|ψ⟩|²", f"Match {bits} to its eigenstate"],
            solution=(f"$\\varphi_0 = 0 \\to 000$, $\\varphi_1 = {L(phi1)} \\to {format(int(phi1 * 8), '03b')}$. Reading ${bits}$ "
                      f"projects onto ${ket(which)}$: $P = {L(P)}$."),
            misconceptions=[(fmt(Pw), "swapped-eigenstates"), ("1", "assumed-deterministic")],
        )
    if v == 0:
        phi = rng.choice([sp.Rational(1, 3), sp.Rational(1, 6), sp.Rational(1, 8), sp.Rational(3, 8), sp.Rational(1, 12),
                          sp.Rational(2, 3), sp.Rational(5, 6), sp.Rational(5, 8)])
        P0 = cs(sp.cos(PI * phi) ** 2)
        return Problem(
            prompt=("Single-qubit phase estimation: one counting qubit in $\\lvert+\\rangle$ controls $U$ (eigenvalue "
                    "$e^{2\\pi i\\varphi}$ on the target eigenstate), then $H$ is applied and the counting qubit measured. Derive "
                    f"$P(0) = \\cos^2(\\pi\\varphi)$ and evaluate it exactly for $\\varphi = {L(phi)}$."),
            answer=fmt(P0), kind="num",
            rubric=["After kickback: (|0⟩ + e^{2πiφ}|1⟩)/√2", "H gives amplitude (1 + e^{2πiφ})/2 for |0⟩",
                    "|1 + e^{iα}|²/4 = cos²(α/2)", f"Evaluate at φ = {phi}"],
            solution=(f"$P(0) = \\left|\\tfrac{{1 + e^{{2\\pi i\\varphi}}}}{{2}}\\right|^2 = \\cos^2(\\pi\\varphi) = "
                      f"\\cos^2\\left({L(PI * phi)}\\right) = {L(P0)}$ — the basis of Kitaev's iterative phase estimation."),
            misconceptions=[(fmt(cs(sp.cos(2 * PI * phi) ** 2)), "dropped-half-angle"), (fmt(cs(sp.cos(PI * phi))), "forgot-amplitude-squared")],
        )
    while True:
        N = rng.choice(_SEMIPRIMES[:6])
        a = rng.randint(2, N - 2)
        if math.gcd(a, N) == 1 and 2 <= _order(a, N) <= 6:
            break
    r = _order(a, N)
    phases = [fmt(sp.Rational(s, r)) for s in range(r)]
    return Problem(
        prompt=(f"Order finding uses $U\\lvert y\\rangle = \\lvert {a}y \\bmod {N}\\rangle$. Show that "
                "$\\lvert u_s\\rangle = \\frac{1}{\\sqrt r}\\sum_{k=0}^{r-1} e^{-2\\pi i sk/r}\\lvert a^k \\bmod N\\rangle$ are eigenvectors "
                "with eigenvalue $e^{2\\pi i s/r}$, find the order $r$, and list all distinct eigenphases $\\varphi = s/r\\in[0,1)$."),
        answer=phases, kind="set", answer_hint="the set of phases, e.g. 0, 1/4, 1/2, 3/4",
        rubric=["U shifts a^k → a^{k+1}; reindex the sum to get the eigenvalue e^{2πis/r}",
                f"Find r: smallest r with {a}^r ≡ 1 (mod {N})", "Phases s/r for s = 0..r−1"],
        solution=(f"Powers of ${a}$ mod ${N}$: ${', '.join(str(pow(a, k, N)) for k in range(1, r + 1))}$, so $r = {r}$ and "
                  f"$\\varphi\\in\\{{{', '.join(L(sp.Rational(s, r)) for s in range(r))}\\}}$. Since $\\frac{{1}}{{\\sqrt r}}\\sum_s\\lvert u_s\\rangle = "
                  "\\lvert1\\rangle$, QPE on $\\lvert1\\rangle$ samples a random $s/r$."),
        misconceptions=[([fmt(sp.Rational(s, r)) for s in range(1, r + 1)], "off-by-one")],
    )


# ---------------------------------------------------------------- qc.shor
@gen("qc.shor")
def shor_drills(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        N = rng.choice(_SEMIPRIMES)
        while True:
            a = rng.randint(2, N - 2)
            if math.gcd(a, N) == 1:
                break
        k = rng.randint(10, 40)
        sq = [pow(a, 2 ** i, N) for i in range(k.bit_length())]
        used = [i for i in range(k.bit_length()) if k >> i & 1]
        return Problem(
            prompt=(f"Compute ${a}^{{{k}}} \\bmod {N}$ by repeated squaring (the classical core of the modular-exponentiation circuit "
                    "in Shor's algorithm)."),
            answer=str(pow(a, k, N)), kind="num", answer_hint="an integer in [0, N)",
            rubric=[f"Write {k} in binary: {format(k, 'b')}", f"Square repeatedly: a^1, a^2, a^4, … mod {N}",
                    "Multiply the powers for the 1-bits, reducing mod N each time"],
            solution=(f"${k} = {' + '.join(str(2 ** i) for i in used)}$; squares "
                      + ", ".join(f"${a}^{{{2 ** i}}} \\equiv {sq[i]}$" for i in range(len(sq)))
                      + f". Product of the used ones mod ${N}$: ${pow(a, k, N)}$."),
            misconceptions=[(str(a * k % N), "multiplied-instead-of-exponentiated")],
        )
    if level == 1:
        N = rng.choice(_SEMIPRIMES + [143, 187, 221, 247, 323, 391])
        n, t = N.bit_length(), (N * N - 1).bit_length()
        return Problem(
            prompt=(f"Register sizes for Shor's algorithm with $N = {N}$: the work register needs $n = \\lceil\\log_2 N\\rceil$ qubits and "
                    "the counting register the smallest $t$ with $2^t \\ge N^2$. Give $(t, n)$."),
            answer=[str(t), str(n)], kind="list", answer_hint="two integers, e.g. 8, 4",
            rubric=[f"2^{n - 1} < {N} ≤ 2^{n}", f"N² = {N * N}", f"2^{t - 1} < N² ≤ 2^{t}"],
            solution=f"$2^{{{n - 1}}} = {2 ** (n - 1)} < {N} \\le {2 ** n}$ so $n = {n}$; $N^2 = {N * N} \\le 2^{{{t}}} = {2 ** t}$ so $t = {t}$.",
            misconceptions=[([str(n), str(t)], "swapped-registers"), ([str(2 * n), str(n)], "assumed-t-equals-2n")],
        )
    if level == 2 and v == 0:
        target = rng.choice("ABC")
        while True:
            N = rng.choice(_SEMIPRIMES[:8])
            a = rng.randint(2, N - 2)
            if math.gcd(a, N) != 1:
                continue
            r = _order(a, N)
            if r > 12:
                continue
            cat = "B" if r % 2 else ("C" if pow(a, r // 2, N) == N - 1 else "A")
            if cat == target:
                break
        pw = [pow(a, k, N) for k in range(1, r + 1)]
        tail = {"A": f"$a^{{r/2}} \\equiv {pow(a, r // 2, N)}$, giving $\\gcd = {math.gcd(pow(a, r // 2, N) - 1, N)}, "
                     f"{math.gcd(pow(a, r // 2, N) + 1, N)}$: success (A).",
                "B": "$r$ is odd, so $a^{r/2}$ is not an integer power: failure (B), pick another $a$.",
                "C": f"$a^{{r/2}} \\equiv {N - 1} \\equiv -1$: the gcds are trivial, failure (C)."}[cat]
        return Problem(
            prompt=(f"In Shor's algorithm with $N = {N}$, the quantum part returns the order $r$ of $a = {a}$. Find $r$ and decide what the "
                    "classical post-processing does.\n\nA) success: $\\gcd(a^{r/2}\\pm1, N)$ gives nontrivial factors  "
                    "B) failure: $r$ is odd  C) failure: $a^{r/2} \\equiv -1 \\pmod N$"),
            answer=cat, kind="choice",
            rubric=[f"List powers of {a} mod {N} until 1", "Is r even?", "If so, is a^{r/2} ≡ −1 (mod N)?", "Otherwise the gcds factor N"],
            solution=f"Powers: ${', '.join(map(str, pw))}$, so $r = {r}$. " + tail,
            misconceptions=[("A", "skipped-order-checks")] if cat != "A" else [("C", "misread-minus-one")],
        )
    if level == 2:
        while True:
            Q = rng.choice([32, 64, 128, 256])
            y = rng.randrange(1, Q)
            cf, _ = _convergents(y, Q)
            if 3 <= len(cf) <= 6 and cf[-1] >= 2:
                break
        return Problem(
            prompt=(f"Expand $\\dfrac{{{y}}}{{{Q}}}$ as a continued fraction $[a_0; a_1, \\ldots, a_m]$ with Euclid's algorithm (last "
                    "coefficient $\\ge 2$) — the step that turns Shor's measurement $y/Q$ into a guess for $s/r$."),
            answer=[str(c) for c in cf], kind="list", answer_hint="the coefficients in order, e.g. 0, 3, 1, 2",
            rubric=["a0 = ⌊y/Q⌋ (0 here)", "Repeatedly invert the fractional part: Euclid on (Q, y)", "Stop when the remainder is 0"],
            solution=f"${y}/{Q} = [{cf[0]}; {', '.join(map(str, cf[1:]))}]$ — the quotients of Euclid's algorithm on $({y}, {Q})$.",
        )
    if v == 0:
        for _ in range(2000):
            N = rng.choice(_SEMIPRIMES[:10])
            a = rng.randint(2, N - 2)
            if math.gcd(a, N) != 1:
                continue
            r = _order(a, N)
            t = (N * N - 1).bit_length()
            Q = 2 ** t
            s1, s2 = rng.randrange(1, r), rng.randrange(1, r)
            d1, d2 = sp.Rational(s1, r).q, sp.Rational(s2, r).q
            if not (d1 < r and d2 < r and math.lcm(d1, d2) == r and d1 != d2):
                continue
            y1, y2 = round(s1 * Q / r), round(s2 * Q / r)
            c1 = [c for c in _convergents(y1, Q)[1] if c.q < N]
            c2 = [c for c in _convergents(y2, Q)[1] if c.q < N]
            if c1 and c2 and c1[-1] == sp.Rational(s1, r) and c2[-1] == sp.Rational(s2, r):
                break
        return Problem(
            prompt=(f"Order finding for $a = {a}$, $N = {N}$ with $Q = 2^{{{t}}} = {Q}$. Two runs return $y_1 = {y1}$ and $y_2 = {y2}$. "
                    "Use continued fractions on each $y_i/Q$ (largest convergent denominator below $N$). Neither denominator alone "
                    "satisfies $a^q\\equiv 1$; combine them to find the order $r$."),
            answer=str(r), kind="num", answer_hint="an integer",
            rubric=["CF expansion of each y_i/Q → reduced fraction s_i'/q_i", "Each q_i divides r (s_i and r may share factors)",
                    "r = lcm(q_1, q_2); verify a^r ≡ 1 (mod N)"],
            solution=(f"${y1}/{Q} \\approx {L(sp.Rational(s1, r))}$ and ${y2}/{Q} \\approx {L(sp.Rational(s2, r))}$: denominators ${d1}, {d2}$, "
                      f"$r = \\mathrm{{lcm}}({d1}, {d2}) = {r}$; check ${a}^{{{r}}} \\equiv {pow(a, r, N)} \\pmod{{{N}}}$."),
            misconceptions=[(str(max(d1, d2)), "used-single-run")] + ([(str(d1 * d2), "multiplied-denominators")] if d1 * d2 != r else []),
        )
    while True:
        p, q = rng.sample([3, 5, 7, 11, 13], 2)
        N = p * q
        a = rng.randint(2, N - 2)
        if math.gcd(a, N) == 1 and a % p != 1 and a % q != 1:
            break
    rp, rq = _order(a % p, p), _order(a % q, q)
    r = math.lcm(rp, rq)
    return Problem(
        prompt=(f"Find the order of $a = {a}$ modulo $N = {N} = {p}\\cdot{q}$ without listing powers mod ${N}$: compute the orders $r_p$, $r_q$ "
                f"of $a$ modulo ${p}$ and ${q}$, and prove on paper that $r = \\mathrm{{lcm}}(r_p, r_q)$ (Chinese remainder theorem)."),
        answer=str(r), kind="num", answer_hint="an integer",
        rubric=[f"Reduce: a ≡ {a % p} (mod {p}), a ≡ {a % q} (mod {q})", "Find each small order by listing powers",
                "a^r ≡ 1 mod N ⇔ a^r ≡ 1 mod p and mod q ⇔ r_p | r and r_q | r", "r = lcm(r_p, r_q)"],
        solution=f"$r_{{{p}}} = {rp}$, $r_{{{q}}} = {rq}$, so $r = \\mathrm{{lcm}}({rp}, {rq}) = {r}$ (check: ${a}^{{{r}}} \\bmod {N} = {pow(a, r, N)}$).",
        misconceptions=[(str(rp * rq), "multiplied-orders")] if rp * rq != r else [],
    )


# ---------------------------------------------------------------- qc.density_noise
_BLOCH_COMPS = [0, sp.Rational(1, 2), -sp.Rational(1, 2), sp.Rational(1, 3), -sp.Rational(1, 3), sp.Rational(1, 4), -sp.Rational(1, 4)]
_ENT_VECS = [(sp.Rational(1, 2), 0, 0), (0, 0, sp.Rational(1, 2)), (0, sp.Rational(1, 3), 0), (sp.Rational(3, 10), 0, sp.Rational(2, 5)),
             (0, sp.Rational(3, 5), 0), (sp.Rational(4, 5), 0, 0), (0, 0, sp.Rational(3, 5)), (sp.Rational(2, 5), sp.Rational(3, 10), 0),
             (0, 0, 0)]


def _rho_from_bloch(r) -> sp.Matrix:
    return cmat((ID + r[0] * X + r[1] * Y + r[2] * Z) / 2)


def _rand_bloch(rng):
    while True:
        r = [rng.choice(_BLOCH_COMPS) for _ in range(3)]
        if any(r) and sum(x * x for x in r) <= 1:
            return r


@gen("qc.density_noise")
def density_drills(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        r = _rand_bloch(rng)
        rho = _rho_from_bloch(r)
        return Problem(
            prompt=(f"Find the Bloch vector $\\vec r = (x, y, z)$ of the qubit density matrix $\\rho = {mat_tex(rho)}$, using "
                    "$\\rho = \\tfrac12(I + xX + yY + zZ)$."),
            answer=[fmt(x) for x in r], kind="vector", answer_hint="x, y, z  e.g. [1/2, 0, -1/3]",
            rubric=["ρ = ½[[1 + z, x − iy],[x + iy, 1 − z]]", "z = ρ00 − ρ11", "x = 2 Re ρ10, y = 2 Im ρ10"],
            solution=(f"$z = \\rho_{{00}} - \\rho_{{11}} = {L(r[2])}$, $\\rho_{{10}} = {L(rho[1, 0])}$ gives $x = {L(r[0])}$, $y = {L(r[1])}$."),
            misconceptions=[([fmt(r[0]), fmt(-r[1]), fmt(r[2])], "y-component-sign")],
        )
    if level == 1:
        r = _rand_bloch(rng)
        rho = _rho_from_bloch(r)
        t = rng.choice(["+", "-", "+i", "-i", "1"])
        p = cs((dag(STATES[t]) * rho * STATES[t])[0, 0])
        return Problem(
            prompt=(f"A qubit has density matrix $\\rho = {mat_tex(rho)}$. It is measured in a basis that contains ${ket(t)}$. What is the "
                    f"probability of the outcome ${ket(t)}$? (Use $P = \\langle t\\vert\\rho\\vert t\\rangle$.)"),
            answer=fmt(p), kind="num",
            rubric=[f"Write {ket(t)} as a column vector", "Compute ρ|t⟩, then the inner product with ⟨t| (conjugated)",
                    "Check 0 ≤ P ≤ 1"],
            solution=f"$\\langle t\\vert\\rho\\vert t\\rangle = {L(p)}$ (in Bloch terms: $\\tfrac12(1 + \\vec r\\cdot\\hat t)$).",
            misconceptions=[(fmt(rho[0, 0]), "read-diagonal-only")],
        )
    if level == 2 and v == 0:
        s, p = rng.choice(_SIMPLE), rng.choice(_FRACS)
        rho = _proj(STATES[s])
        out = cmat((1 - p) * rho + p * ID / 2)
        return Problem(
            prompt=(f"The depolarizing channel $\\mathcal E(\\rho) = (1-p)\\rho + p\\,\\tfrac{{I}}{{2}}$ with $p = {L(p)}$ acts on ${ket(s)}$. "
                    "Give the output density matrix."),
            answer=mans(out), kind="matrix", answer_hint=HINT_M,
            rubric=["ρ = |ψ⟩⟨ψ|", "Shrink ρ by (1 − p) and add p/2 on the diagonal", "Check trace 1; Bloch vector shrinks by (1 − p)"],
            solution=f"$\\rho = {mat_tex(rho)}$, so $\\mathcal E(\\rho) = {mat_tex(out)}$.",
            misconceptions=[(mans(cmat((1 - p) * rho + p * ID)), "forgot-half-identity")],
        )
    if level == 2:
        s = rng.choice(_SIMPLE)
        name, P = rng.choice([("bit-flip", "X"), ("phase-flip", "Z"), ("bit-phase-flip", "Y")])
        p = rng.choice(_FRACS)
        rho = _proj(STATES[s])
        out = (1 - p) * rho + p * GATES[P] * rho * GATES[P]
        F = cs((dag(STATES[s]) * out * STATES[s])[0, 0])
        return Problem(
            prompt=(f"Compute the fidelity $F = \\langle\\psi\\vert\\mathcal E(\\rho)\\vert\\psi\\rangle$ between $\\lvert\\psi\\rangle = {ket(s)}$ "
                    f"and its image under the {name} channel $\\rho\\mapsto(1-p)\\rho + p\\,{P}\\rho {P}$ with $p = {L(p)}$."),
            answer=fmt(F), kind="num",
            rubric=["F = (1 − p) + p |⟨ψ|P|ψ⟩|²", f"Compute ⟨ψ|{P}|ψ⟩ (is {ket(s)} an eigenstate of {P}?)", "Combine"],
            solution=(f"$\\langle\\psi\\vert {P}\\vert\\psi\\rangle = {L(expval(GATES[P], STATES[s]))}$, so "
                      f"$F = (1-p) + p\\cdot{L(cs(sp.Abs(expval(GATES[P], STATES[s])) ** 2))} = {L(F)}$. Eigenstates of the error are immune."),
            misconceptions=[(fmt(1 - p), "assumed-every-error-harmful")],
        )
    if v == 0:
        r = list(rng.choice(_ENT_VECS))
        r = [x * rng.choice([1, -1]) for x in r]
        rl = sp.sqrt(sum(x * x for x in r))
        rho = _rho_from_bloch(r)
        l1, l2 = (1 + rl) / 2, (1 - rl) / 2
        S = -(l1 * sp.log(l1, 2) + l2 * sp.log(l2, 2)) if rl != 0 else sp.Integer(1)
        dg = [rho[0, 0], rho[1, 1]]
        Sd = cs(-sum(d * sp.log(d, 2) for d in dg if d != 0))
        return Problem(
            prompt=(f"Compute the von Neumann entropy $S(\\rho) = -\\mathrm{{Tr}}\\,\\rho\\log_2\\rho$ (in bits) of $\\rho = {mat_tex(rho)}$. "
                    "(Derive that the eigenvalues are $\\tfrac12(1\\pm|\\vec r|)$ for Bloch vector $\\vec r$.)"),
            answer=fmt(S), kind="num", tol=1e-3, answer_hint=HINT_S,
            rubric=["Read off the Bloch vector and |r|", "Eigenvalues of ½(I + r·σ) are ½(1 ± |r|) because (r·σ)² = |r|² I",
                    "S = −Σ λ log2 λ (use the eigenvalues, not the diagonal entries)"],
            solution=(f"$|\\vec r| = {L(rl)}$, eigenvalues ${L(l1)}, {L(l2)}$, so $S = -{L(l1)}\\log_2{L(l1)} - {L(l2)}\\log_2{L(l2)} "
                      f"\\approx {float(S):.4f}$ bits." if rl != 0 else "$\\rho = I/2$ is maximally mixed: $S = 1$ bit."),
            misconceptions=[(fmt(Sd), "used-diagonal-not-eigenvalues")] if cs(Sd - S) != 0 else [],
        )
    s = rng.choice(["+", "+i"])
    obs = "X" if s == "+" else "Y"
    q = rng.choice([sp.Rational(1, 3), sp.Rational(1, 4), sp.Rational(1, 5), sp.Rational(1, 10), sp.Rational(3, 4), sp.Rational(2, 3)])
    n = rng.randint(2, 5)
    val = (1 - 2 * q) ** n
    return Problem(
        prompt=(f"The state ${ket(s)}$ passes through ${n}$ consecutive phase-flip channels $\\rho\\mapsto(1-q)\\rho + qZ\\rho Z$ with "
                f"$q = {L(q)}$. Derive how the off-diagonal element of $\\rho$ evolves and give the final $\\langle {obs}\\rangle$."),
        answer=fmt(val), kind="num",
        rubric=["ZρZ keeps the diagonal and negates the off-diagonals", "One channel multiplies ρ01 by (1 − q) − q = 1 − 2q",
                f"After n channels: (1 − 2q)^n", f"⟨{obs}⟩ = 2·(Re or Im of ρ10) starts at 1"],
        solution=(f"$\\rho_{{01}} \\to (1-2q)^n\\rho_{{01}}$, so $\\langle {obs}\\rangle = (1 - 2\\cdot{L(q)})^{{{n}}} = {L(val)}$ — "
                  "exponential dephasing ($T_2$ decay)."),
        misconceptions=[(fmt((1 - q) ** n), "forgot-factor-two"), (fmt(1 - 2 * q * n), "linearised-decay")],
    )


# ---------------------------------------------------------------- qc.error_correction
def _hamming_H():
    return [[(i >> (2 - row)) & 1 for i in range(1, 8)] for row in range(3)]


@gen("qc.error_correction")
def qec_drills(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        a, b = nz(rng, 1, 4), rand_amp(rng, True, -4, 4)
        n2 = a * a + int(sp.Abs(b) ** 2)
        al, be = sp.Integer(a) / sp.sqrt(n2), b / sp.sqrt(n2)
        enc = [al] + [0] * 6 + [be]
        clone = kron(vec(al, be), vec(al, be), vec(al, be))
        return Problem(
            prompt=(f"Encode $\\lvert\\psi\\rangle = {_norm_tex([a, b])}$ into the 3-qubit bit-flip code "
                    "($\\lvert0\\rangle\\mapsto\\lvert000\\rangle$, $\\lvert1\\rangle\\mapsto\\lvert111\\rangle$, extended linearly). "
                    "Give the encoded state as an 8-vector ordered $\\lvert000\\rangle, \\lvert001\\rangle, \\ldots, \\lvert111\\rangle$."),
            answer=vans(enc), kind="vector", answer_hint="8 amplitudes [c000, ..., c111], each typed separately",
            rubric=["Encoding is linear: α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩", "Only the first and last amplitudes are nonzero",
                    "It is NOT |ψ⟩⊗|ψ⟩⊗|ψ⟩ (no-cloning)"],
            solution=(f"$\\alpha\\lvert000\\rangle + \\beta\\lvert111\\rangle$ with $\\alpha = {L(cs(al))}$, $\\beta = {L(cs(be))}$: "
                      f"${vec_tex(sp.Matrix(enc))}$ (prepared with two CNOTs from qubit 1)."),
            misconceptions=[(vans(clone), "cloned-instead-of-encoded")],
        )
    if level == 1:
        d = rng.randint(2, 11)
        return Problem(
            prompt=(f"A quantum code has distance $d = {d}$. How many arbitrary single-qubit errors can it **correct**, and how many can it "
                    "**detect**? Give (correct, detect)."),
            answer=[str((d - 1) // 2), str(d - 1)], kind="list", answer_hint="two integers, e.g. 1, 2",
            rubric=["Detect up to d − 1 errors (they cannot map a codeword to another)", "Correct up to t = ⌊(d − 1)/2⌋"],
            solution=f"Correct $\\lfloor({d}-1)/2\\rfloor = {(d - 1) // 2}$, detect ${d} - 1 = {d - 1}$.",
            misconceptions=[([str(d // 2), str(d)], "off-by-one")],
        )
    if level == 2 and v == 0:
        while True:
            err = "".join(rng.choice("IXYZ") for _ in range(3))
            if err != "III":
                break
        flip = [1 if c in "XY" else 0 for c in err]
        syn = f"{(flip[0] + flip[1]) % 2}{(flip[1] + flip[2]) % 2}"
        xonly = [1 if c == "X" else 0 for c in err]
        wsyn = f"{(xonly[0] + xonly[1]) % 2}{(xonly[1] + xonly[2]) % 2}"
        return Problem(
            prompt=(f"In the 3-qubit bit-flip code (stabilizers $Z_1Z_2$, $Z_2Z_3$) the Pauli error ${_ptex(err)}$ occurs. What syndrome "
                    "$(s_1s_2)$ is measured? ($s = 1$ iff the error anticommutes with that stabilizer.)"),
            answer=syn, kind="text", answer_hint="two bits, e.g. 10",
            rubric=["X and Y anticommute with Z; I and Z commute with Z", "Z_iZ_j anticommutes with the error iff an odd number of qubits i, j get X/Y",
                    "Compute both parities"],
            solution=(f"Bit-flipping components (X or Y) on qubits: {', '.join(str(i + 1) for i in range(3) if flip[i]) or 'none'}. "
                      f"Syndrome ${syn}$. Note $Z$ components are invisible to this code."),
            misconceptions=[(wsyn, "y-error-ignored")] if wsyn != syn else [],
        )
    if level == 2:
        Hm = _hamming_H()
        codewords = [w for w in range(128) if all(sum(Hm[r][i] * ((w >> (6 - i)) & 1) for i in range(7)) % 2 == 0 for r in range(3))]
        c = rng.choice(codewords)
        k = rng.randint(1, 7)
        rw = c ^ (1 << (7 - k))
        rbits = format(rw, "07b")
        syn = format(k, "03b")
        htex = "\\begin{pmatrix}" + " \\\\ ".join(" & ".join(map(str, row)) for row in Hm) + "\\end{pmatrix}"
        return Problem(
            prompt=("The classical $[7,4]$ Hamming code (the basis of the Steane code) has parity-check matrix $H$ below; column $i$ is "
                    f"$i$ in binary. The word $r = {rbits}$ arrives with exactly one bit flipped. Compute the syndrome $Hr \\bmod 2$ and give "
                    f"the position (1–7) of the flipped bit.\n\n$$H = {htex}$$"),
            answer=str(k), kind="num", answer_hint="an integer from 1 to 7",
            rubric=["Each syndrome bit is the parity of r over the 1s in that row of H", "Read the 3 syndrome bits top to bottom as a binary number",
                    "Columns of H are 1..7 in binary, so the syndrome IS the error position"],
            solution=f"$Hr = ({', '.join(syn)})^T = {syn}_2 = {k}$: bit {k} flipped (codeword ${format(c, '07b')}$).",
            misconceptions=[(str(int(syn[::-1], 2)), "syndrome-bit-order")] if syn != syn[::-1] else [],
        )
    if v == 0:
        n = rng.choice([3, 5, 7, 9])
        t = (n + 1) // 2
        C = math.comb(n, t)
        return Problem(
            prompt=(f"An $n = {n}$-qubit repetition code with majority-vote decoding faces independent bit flips with small probability $p$. "
                    f"Derive the leading-order logical error rate $p_L \\approx C\\,p^{{t}}$ and give $(C, t)$."),
            answer=[str(C), str(t)], kind="list", answer_hint="two integers C, t, e.g. 3, 2",
            rubric=["Majority vote fails iff more than n/2 qubits flip", f"Smallest failing number of flips t = (n+1)/2 = {t}",
                    f"There are C(n, t) ways to choose them; higher orders are O(p^(t+1))"],
            solution=f"$p_L = \\binom{{{n}}}{{{t}}}p^{{{t}}}(1-p)^{{{n - t}}} + \\ldots \\approx {C}\\,p^{{{t}}}$.",
            misconceptions=[([str(C), str(t - 1)], "off-by-one"), ([str(n), str(t)], "counted-single-patterns")],
        )
    codes = [("five-qubit", 5, 1, 3), ("Steane", 7, 1, 3), ("Shor", 9, 1, 3), ("four-qubit error-detecting", 4, 2, 2),
             ("distance-3 surface", 9, 1, 3), ("distance-5 surface", 25, 1, 5)]
    name, n, k, d = rng.choice(codes)
    g = n - k
    return Problem(
        prompt=(f"The {name} code is an $[[{n},{k},{d}]]$ stabilizer code. How many independent stabilizer generators does it have, and how "
                "many distinct syndromes can they produce? Give (generators, syndromes). On paper, compare with the $1 + 3n$ "
                "single-qubit errors (quantum Hamming bound)."),
        answer=[str(g), str(2 ** g)], kind="list", answer_hint="two integers, e.g. 4, 16",
        rubric=["Each independent generator halves the Hilbert space: 2^n / 2^{n−k} = 2^k", f"So n − k = {g} generators",
                f"Each generator gives one ±1 bit: 2^{g} syndromes", f"Compare with 1 + 3n = {1 + 3 * n}"],
        solution=(f"$n - k = {g}$ generators and $2^{{{g}}} = {2 ** g}$ syndromes"
                  + (f" (≥ $1 + 3n = {1 + 3 * n}$" + (", with equality: a perfect code)." if 2 ** g == 1 + 3 * n else ").")
                     if 2 ** g >= 1 + 3 * n else f" < $1 + 3n = {1 + 3 * n}$: it cannot correct every single-qubit error (here d = {d}).")),
        misconceptions=[([str(n), str(2 ** n)], "counted-qubits-not-generators")],
    )


# ---------------------------------------------------------------- qc.variational
_VQA_TERMS = [("Z\\otimes Z", "ZZ"), ("X\\otimes I", "XI"), ("I\\otimes X", "IX"), ("Y\\otimes Y", "YY"), ("X\\otimes X", "XX"),
              ("Z\\otimes I", "ZI")]


def _rand_counts(rng, total):
    cuts = sorted(rng.sample(range(1, total), 3))
    return [cuts[0], cuts[1] - cuts[0], cuts[2] - cuts[1], total - cuts[2]]


def _Rx(a):
    return sp.Matrix([[sp.cos(a / 2), -I * sp.sin(a / 2)], [-I * sp.sin(a / 2), sp.cos(a / 2)]])


@gen("qc.variational")
def vqa_drills(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        total = rng.choice([100, 200, 500, 1000])
        n = _rand_counts(rng, total)
        obs = rng.choice(["Z\\otimes Z", "Z\\otimes I", "I\\otimes Z"])
        sgn = {"Z\\otimes Z": [1, -1, -1, 1], "Z\\otimes I": [1, 1, -1, -1], "I\\otimes Z": [1, -1, 1, -1]}[obs]
        est = sp.Rational(sum(s * c for s, c in zip(sgn, n)), total)
        numer = " ".join(f"{'+' if s > 0 else '-'} {c}" for s, c in zip(sgn, n)).lstrip("+ ")
        return Problem(
            prompt=(f"Measuring both qubits of a VQE ansatz in the computational basis {total} times gave the counts "
                    f"$00{{:}}\\ {n[0]}$, $01{{:}}\\ {n[1]}$, $10{{:}}\\ {n[2]}$, $11{{:}}\\ {n[3]}$. Estimate $\\langle {obs}\\rangle$."),
            answer=fmt(est), kind="num", answer_hint="a fraction or decimal, e.g. 0.12",
            rubric=[f"Eigenvalue of {obs} on each outcome: ({', '.join(f'{s:+d}' for s in sgn)}) for 00, 01, 10, 11",
                    "Weight each count by its eigenvalue", "Divide by the number of shots"],
            solution=f"$\\langle {obs}\\rangle \\approx \\frac{{{numer}}}{{{total}}} = {L(est)}$.",
            misconceptions=[(fmt(-est), "parity-sign-error")],
        )
    if level == 1:
        n, Lr, g = rng.randint(2, 8), rng.randint(1, 6), rng.choice([1, 2, 3])
        P = g * n * Lr
        return Problem(
            prompt=(f"A hardware-efficient ansatz on $n = {n}$ qubits has $L = {Lr}$ layers, each applying {g} parametrised single-qubit "
                    "rotation(s) to every qubit (plus fixed CNOTs). How many circuit evaluations (distinct expectation values) does one "
                    "full parameter-shift gradient need?"),
            answer=str(2 * P), kind="num", answer_hint="an integer",
            rubric=["Count parameters: rotations per qubit per layer × n × L", "Parameter shift needs 2 evaluations (θ ± π/2) per parameter"],
            solution=f"$P = {g}\\cdot{n}\\cdot{Lr} = {P}$ parameters, so $2P = {2 * P}$ evaluations (each estimated with many shots).",
            misconceptions=[(str(P), "forgot-two-shifts")],
        )
    if level == 2 and v == 0:
        T_ = rng.choice([100, 200, 400])
        nz_, nx_ = _rand_counts(rng, T_), _rand_counts(rng, T_)
        a, b, c = nz(rng, -3, 3), nz(rng, -3, 3), nz(rng, -3, 3)
        zz = sp.Rational(nz_[0] - nz_[1] - nz_[2] + nz_[3], T_)
        zi = sp.Rational(nz_[0] + nz_[1] - nz_[2] - nz_[3], T_)
        xx = sp.Rational(nx_[0] - nx_[1] - nx_[2] + nx_[3], T_)
        E = a * zz + b * xx + c * zi
        htex = lin_tex([(a, "Z\\otimes Z"), (b, "X\\otimes X"), (c, "Z\\otimes I")])
        return Problem(
            prompt=(f"Estimate $\\langle H\\rangle$ for $H = {htex}$ from these counts ({T_} shots per setting). "
                    f"$Z$-basis: $00{{:}}\\ {nz_[0]}$, $01{{:}}\\ {nz_[1]}$, $10{{:}}\\ {nz_[2]}$, $11{{:}}\\ {nz_[3]}$. "
                    f"$X$-basis (outcome 0 means $\\lvert+\\rangle$): $00{{:}}\\ {nx_[0]}$, $01{{:}}\\ {nx_[1]}$, $10{{:}}\\ {nx_[2]}$, $11{{:}}\\ {nx_[3]}$."),
            answer=fmt(E), kind="num", answer_hint="a fraction or decimal",
            rubric=["⟨ZZ⟩ and ⟨ZI⟩ both come from the Z-basis counts (parity / first bit)", "⟨XX⟩ is the parity in the X-basis counts",
                    "Combine with the coefficients"],
            solution=(f"$\\langle ZZ\\rangle = {L(zz)}$, $\\langle ZI\\rangle = {L(zi)}$, $\\langle XX\\rangle = {L(xx)}$, so "
                      f"$\\langle H\\rangle = {L(E)}$ (≈ {float(E):.3f})."),
            misconceptions=[(fmt(a * zz + b * xx), "dropped-local-term")],
        )
    if level == 2:
        g = rng.choice([PI / 6, PI / 4, PI / 3, PI / 2, 2 * PI / 3])
        b = rng.choice([PI / 16, PI / 12, PI / 8, PI / 6, 3 * PI / 16])
        Cm = sp.diag(0, 1, 1, 0)
        UC = sp.diag(1, sp.exp(-I * g), sp.exp(-I * g), 1)
        UB = kron(_Rx(2 * b), _Rx(2 * b))
        st = UB * UC * kron(STATES["+"], STATES["+"])
        val = cs(sp.simplify((dag(st) * Cm * st)[0, 0]))
        return Problem(
            prompt=("QAOA ($p = 1$) for MaxCut on a single edge: cost $C = \\tfrac12(1 - Z_1Z_2)$, state "
                    "$\\lvert\\gamma,\\beta\\rangle = e^{-i\\beta(X_1+X_2)}e^{-i\\gamma C}\\lvert{+}{+}\\rangle$. With "
                    f"$\\gamma = {L(g)}$ and $\\beta = {L(b)}$, compute $\\langle C\\rangle$ exactly."),
            answer=fmt(val), kind="num",
            hint="Show ⟨C⟩ = ½ + ½ sin(4β) sin(γ): e^{−iγC} only puts the phase e^{−iγ} on |01⟩, |10⟩; then rotate Z1Z2 back through the mixer.",
            rubric=["e^{−iγC}|++⟩ = ½(|00⟩ + e^{−iγ}|01⟩ + e^{−iγ}|10⟩ + |11⟩)",
                    "Heisenberg: the mixer maps Z1Z2 → (cos2β Z1 + sin2β Y1)(cos2β Z2 + sin2β Y2)",
                    "Evaluate ⟨Z1Z2⟩ = −sin 4β sin γ on the phased state", "⟨C⟩ = ½(1 − ⟨Z1Z2⟩)"],
            solution=(f"$\\langle C\\rangle = \\tfrac12 + \\tfrac12\\sin 4\\beta\\,\\sin\\gamma = \\tfrac12 + \\tfrac12\\sin({L(4 * b)})\\sin({L(g)}) "
                      f"= {L(val)}$; the optimum $\\gamma = \\pi/2$, $\\beta = \\pi/8$ cuts the edge with certainty."),
            misconceptions=[(fmt(sp.Rational(1, 2)), "ignored-variational-parameters")],
        )
    if v == 0:
        m = rng.randint(2, 4)
        terms = rng.sample(_VQA_TERMS, m)
        cs_ = [nz(rng, -4, 4) for _ in range(m)]
        eps = rng.choice([sp.Rational(1, 10), sp.Rational(1, 20), sp.Rational(1, 50)])
        Ssq = sum(c * c for c in cs_)
        Nper = sp.ceiling(Ssq / eps ** 2)
        htex = lin_tex([(c, t[0]) for c, t in zip(cs_, terms)])
        return Problem(
            prompt=(f"$H = {htex}$ is estimated by measuring each of its {m} Pauli terms separately with the same number $N$ of shots per "
                    "term. Using $\\mathrm{Var}(P)\\le 1$ for any Pauli $P$, derive the $N$ that guarantees a standard error at most "
                    f"$\\varepsilon = {L(eps)}$ for the estimate of $\\langle H\\rangle$, and give the **total** shot count $mN$ (worst case)."),
            answer=str(m * Nper), kind="num", answer_hint="an integer",
            rubric=["Var(Σ c_i P̂_i) = Σ c_i² Var(P_i)/N ≤ Σ c_i²/N (independent settings)",
                    "Require Σ c_i²/N ≤ ε² ⇒ N = ⌈Σ c_i² / ε²⌉", "Total = m·N"],
            solution=(f"$\\sum c_i^2 = {Ssq}$, $N = \\lceil {Ssq}/{L(eps ** 2)}\\rceil = {Nper}$, total ${m}\\cdot{Nper} = {m * Nper}$ shots — "
                      "why grouping commuting terms matters."),
            misconceptions=[(str(Nper), "forgot-terms-multiply"), (str(m * sp.ceiling(sp.sqrt(Ssq) / eps)), "forgot-to-square-error")],
        )
    a, b = nz(rng, -3, 3), nz(rng, -3, 3)
    c = rng.randint(-2, 2)
    t0 = rng.choice([PI / 2, PI / 3, PI / 4, PI / 6, PI])
    f2 = cs(-a * sp.cos(t0) - b * sp.sin(t0))
    f1 = cs(-a * sp.sin(t0) + b * sp.cos(t0))
    etex = lin_tex([(a, "\\cos\\theta"), (b, "\\sin\\theta")]) + (f" {'+' if c > 0 else '-'} {abs(c)}" if c else "")
    return Problem(
        prompt=(f"For $E(\\theta) = {etex}$ (every expectation value of a single $R_y(\\theta)$-parametrised circuit has this form), derive "
                "the second-order parameter-shift rule $E''(\\theta) = \\tfrac14\\left[E(\\theta+\\pi) - 2E(\\theta) + E(\\theta-\\pi)\\right]$ "
                f"and use it to compute $E''(\\theta_0)$ at $\\theta_0 = {L(t0)}$."),
        answer=fmt(f2), kind="num",
        rubric=["Apply the first-order rule twice: shifts ±π/2 twice give θ ± π and θ (twice)",
                "E(θ ± π) = −(E(θ) − c) + c for the sinusoidal part", "Evaluate the combination at θ0",
                "Check against the analytic E'' = −a cos θ − b sin θ"],
        solution=(f"$E(\\theta_0\\pm\\pi) = {L(cs(2 * c - (a * sp.cos(t0) + b * sp.sin(t0) + c)))}$, $E(\\theta_0) = "
                  f"{L(cs(a * sp.cos(t0) + b * sp.sin(t0) + c))}$, so $E''(\\theta_0) = {L(f2)}$ — exact, no finite-difference error."),
        misconceptions=[(fmt(f1), "computed-first-derivative")],
    )
