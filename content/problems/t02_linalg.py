"""Track 2 — Linear algebra (Strang-style), through SVD/PCA, matrix calculus and complex inner products.

Conventions (see engine/content.py and t00_foundations.py):
  * Each generator: (rng, level) -> Problem, deterministic given rng.
  * Levels: 1 = mechanical, 2 = multi-step, 3 = derivation / reasoning with a checkable final result.
  * Matrices are built from chosen factors (unimodular P, integer eigenvalues, Pythagorean
    directions, ...) so that everything stays paper-friendly. All answers are computed with sympy.
"""
from __future__ import annotations

import random

import sympy as sp

from engine.checker import fmt
from engine.content import Problem, gen

L = sp.latex
I = sp.I

QUADS = [((1, 2, 2), 3), ((2, 3, 6), 7), ((1, 4, 8), 9), ((4, 4, 7), 9), ((2, 6, 9), 11), ((0, 3, 4), 5)]


def nz(rng: random.Random, lo: int, hi: int) -> int:
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def vec(vals) -> list[str]:
    return [fmt(sp.nsimplify(sp.simplify(v))) for v in vals]


def tup(vals) -> str:
    return "(" + ", ".join(L(sp.simplify(v)) for v in vals) + ")"


def rand_mat(rng, r, c, lo=-3, hi=3) -> sp.Matrix:
    return sp.Matrix(r, c, lambda *_: rng.randint(lo, hi))


def unimod2(rng) -> sp.Matrix:
    """Random 2x2 integer matrix with det ±1 and small entries (not diagonal)."""
    while True:
        a, b = rng.randint(-2, 2), rng.randint(-2, 2)
        P = sp.Matrix([[1, a], [0, 1]]) * sp.Matrix([[1, 0], [b, 1]])
        if rng.random() < 0.3:
            P = P * sp.Matrix([[0, 1], [1, 0]])
        if max(abs(v) for v in P) <= 3 and not P.is_diagonal():
            return P


def unimod3(rng) -> sp.Matrix:
    while True:
        Lm = sp.Matrix([[1, 0, 0], [rng.randint(-1, 1), 1, 0], [rng.randint(-1, 1), rng.randint(-1, 1), 1]])
        Um = sp.Matrix([[1, rng.randint(-1, 1), rng.randint(-1, 1)], [0, 1, rng.randint(-1, 1)], [0, 0, 1]])
        P = Lm * Um
        if max(abs(v) for v in P) <= 3 and not P.is_diagonal():
            return P


def quad_vec(rng):
    """Random signed/permuted integer 3-vector with integer norm."""
    v, n = rng.choice(QUADS)
    v = list(v)
    rng.shuffle(v)
    return [c * rng.choice([-1, 1]) for c in v], n


# =========================================================================== vectors
@gen("la.vectors")
def vectors(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            u = [rng.randint(-5, 5) for _ in range(3)]
            v = [rng.randint(-5, 5) for _ in range(3)]
            d = sum(a * b for a, b in zip(u, v))
            return Problem(
                prompt=f"Compute the dot product $\\mathbf u\\cdot\\mathbf v$ for $\\mathbf u = {tup(u)}$, $\\mathbf v = {tup(v)}$.",
                answer=str(d), kind="num",
                rubric=["Multiply matching components", "Add the three products"],
                solution="$= " + " + ".join(f"({a})({b})" for a, b in zip(u, v)) + f" = {d}$.",
                misconceptions=[(str(sum(u) * sum(v)), "summed-then-multiplied")],
            )
        q, n = quad_vec(rng)
        k = rng.randint(1, 2)
        u = [k * c for c in q]
        return Problem(
            prompt=f"Compute the Euclidean norm $\\|\\mathbf u\\|$ of $\\mathbf u = {tup(u)}$.",
            answer=str(k * n), kind="num",
            rubric=["Square each component", "Sum", "Take the square root"],
            solution=f"$\\|\\mathbf u\\| = \\sqrt{{{sum(c*c for c in u)}}} = {k*n}$.",
            misconceptions=[(str(sum(c * c for c in u)), "forgot-square-root"), (str(sum(abs(c) for c in u)), "used-l1-norm")],
        )
    if level == 2:
        while True:
            u, nu = quad_vec(rng)
            v, nv = quad_vec(rng)
            d = sum(a * b for a, b in zip(u, v))
            if d != 0:
                break
        cosv = sp.Rational(d, nu * nv)
        return Problem(
            prompt=(f"Find $\\cos\\theta$ where $\\theta$ is the angle between $\\mathbf u = {tup(u)}$ and $\\mathbf v = {tup(v)}$. "
                    "Is the angle acute or obtuse? (Type $\\cos\\theta$.)"),
            answer=fmt(cosv), kind="num",
            rubric=["u·v", f"‖u‖ = {nu}, ‖v‖ = {nv}", "cos θ = u·v / (‖u‖‖v‖)", "Sign of cos θ decides acute/obtuse"],
            solution=f"$\\cos\\theta = \\frac{{{d}}}{{{nu}\\cdot{nv}}} = {L(cosv)}$ ({'acute' if d > 0 else 'obtuse'}).",
            misconceptions=[(str(d), "forgot-normalisation")],
        )
    while True:
        u = [rng.randint(-4, 4) for _ in range(3)]
        v, nv = quad_vec(rng)
        U, V = sp.Matrix(u), sp.Matrix(v)
        perp = U - (U.dot(V) / V.dot(V)) * V
        if perp.norm() != 0 and U.dot(V) != 0:
            break
    dist = sp.simplify(perp.norm())
    return Problem(
        prompt=(f"Find the distance from the point $\\mathbf p = {tup(u)}$ to the line through the origin spanned by "
                f"$\\mathbf v = {tup(v)}$. On paper: write $\\mathbf p = \\hat{{\\mathbf p}} + \\mathbf r$ with "
                "$\\hat{\\mathbf p}$ on the line and $\\mathbf r \\perp \\mathbf v$."),
        answer=fmt(dist), kind="num", answer_hint="exact, e.g. sqrt(17)/3",
        rubric=["Projection coefficient c = p·v / v·v", "p̂ = c v", "r = p − p̂ and check r·v = 0", "Distance = ‖r‖"],
        solution=(f"$c = \\frac{{{U.dot(V)}}}{{{nv**2}}}$, $\\mathbf r = {tup(perp)}$, $\\|\\mathbf r\\| = {L(dist)}$. "
                  f"(Check: $\\|\\mathbf r\\|^2 = \\|\\mathbf p\\|^2 - (\\mathbf p\\cdot\\mathbf v)^2/\\|\\mathbf v\\|^2$.)"),
        misconceptions=[(fmt(sp.simplify((U.dot(V) / V.dot(V)) * V.norm())), "gave-projection-length")],
    )


# =========================================================================== matrix multiplication
@gen("la.matmul")
def matmul(rng, level):
    if level == 1:
        A, B = rand_mat(rng, 2, 3), rand_mat(rng, 3, 2)
        return Problem(
            prompt=f"Compute $AB$ for $A = {L(A)}$ and $B = {L(B)}$.",
            answer=fmt(A * B), kind="matrix",
            rubric=["Shape check: (2×3)(3×2) = 2×2", "Entry (i,j) = row i of A · column j of B", "Four dot products"],
            solution=f"$AB = {L(A * B)}$.",
            misconceptions=[(fmt(B * A), "multiplication-order")],
        )
    if level == 2:
        while True:
            A, B = rand_mat(rng, 2, 2), rand_mat(rng, 2, 2)
            Cm = A * B - B * A
            if Cm != sp.zeros(2, 2):
                break
        return Problem(
            prompt=f"Let $A = {L(A)}$ and $B = {L(B)}$. Compute the commutator $AB - BA$.",
            answer=fmt(Cm), kind="matrix",
            rubric=["Compute AB", "Compute BA (different in general!)", "Subtract entrywise",
                    "Sanity check: the trace of a commutator is 0"],
            solution=f"$AB = {L(A*B)}$, $BA = {L(B*A)}$, $AB - BA = {L(Cm)}$.",
            misconceptions=[("[[0, 0], [0, 0]]", "assumed-matrices-commute")],
        )
    V = unimod2(rng)
    W = rand_mat(rng, 2, 2, -4, 4)
    while W.det() == 0:
        W = rand_mat(rng, 2, 2, -4, 4)
    A = W * V.inv()
    v1, v2 = list(V.col(0)), list(V.col(1))
    w1, w2 = list(W.col(0)), list(W.col(1))
    return Problem(
        prompt=(f"A linear map $T:\\mathbb R^2\\to\\mathbb R^2$ satisfies $T{tup(v1)} = {tup(w1)}$ and "
                f"$T{tup(v2)} = {tup(w2)}$. Find the standard matrix $A$ of $T$ (so $T(\\mathbf x) = A\\mathbf x$)."),
        answer=fmt(A), kind="matrix",
        rubric=["Write the data as AV = W with V = [v₁ v₂], W = [w₁ w₂]", "Invert V (det V = ±1 here)",
                "A = W V⁻¹ (order matters)", "Check A v₁ = w₁"],
        solution=f"$V = {L(V)}$, $V^{{-1}} = {L(V.inv())}$, $A = WV^{{-1}} = {L(A)}$.",
        misconceptions=[(fmt(V.inv() * W), "inverse-on-wrong-side"), (fmt(W * V), "forgot-inverse")],
    )


@gen("la.matmul", name="matvec_and_layers")
def matvec(rng, level):
    if level == 1:
        A = rand_mat(rng, 3, 3)
        xv = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
        r = A * xv
        return Problem(
            prompt=f"Compute $A\\mathbf x$ for $A = {L(A)}$ and $\\mathbf x = {tup(xv)}$.",
            answer=vec(r), kind="vector",
            rubric=["Row picture: each entry is a row of A dotted with x",
                    "Or column picture: x₁·col₁ + x₂·col₂ + x₃·col₃"],
            solution=f"$A\\mathbf x = {tup(r)}$.",
            misconceptions=[(fmt(list(A.T * xv)), "used-transpose")],
        )
    if level == 2:
        A = rand_mat(rng, 3, 3)
        uu = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
        vv = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
        val = (uu.T * A * vv)[0]
        return Problem(
            prompt=f"Compute the scalar $\\mathbf u^\\top A\\mathbf v$ for $A = {L(A)}$, $\\mathbf u = {tup(uu)}$, $\\mathbf v = {tup(vv)}$.",
            answer=str(val), kind="num",
            rubric=["First compute A v (a vector)", "Then dot with u", "Equivalently Σᵢⱼ uᵢ Aᵢⱼ vⱼ"],
            solution=f"$A\\mathbf v = {tup(A*vv)}$, $\\mathbf u^\\top(A\\mathbf v) = {val}$.",
            misconceptions=[(str((vv.T * A * uu)[0]), "swapped-u-and-v")],
        )
    while True:
        W1 = rand_mat(rng, 3, 2, -2, 2)
        b1 = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
        xv = sp.Matrix([nz(rng, -2, 2) for _ in range(2)])
        z = W1 * xv + b1
        if any(v > 0 for v in z) and any(v < 0 for v in z):
            break
    W2 = rand_mat(rng, 2, 3, -2, 2)
    b2 = sp.Matrix([rng.randint(-2, 2) for _ in range(2)])
    hv = z.applyfunc(lambda v: max(v, 0))
    out = W2 * hv + b2
    return Problem(
        prompt=(f"Forward pass of a 2-layer network: $\\mathbf y = W_2\\,\\mathrm{{ReLU}}(W_1\\mathbf x + \\mathbf b_1) + \\mathbf b_2$ with "
                f"$W_1 = {L(W1)}$, $\\mathbf b_1 = {tup(b1)}$, $W_2 = {L(W2)}$, $\\mathbf b_2 = {tup(b2)}$, "
                f"$\\mathbf x = {tup(xv)}$. Compute $\\mathbf y$. (State the shapes of every intermediate on paper.)"),
        answer=vec(out), kind="vector",
        rubric=["Shapes: W₁ is 3×2, so z = W₁x + b₁ ∈ ℝ³", "Apply ReLU entrywise (negatives → 0)",
                "W₂ is 2×3, y ∈ ℝ²", "Add b₂"],
        solution=f"$\\mathbf z = {tup(z)}$, $\\mathbf h = {tup(hv)}$, $\\mathbf y = W_2\\mathbf h + \\mathbf b_2 = {tup(out)}$.",
        misconceptions=[(fmt(list(W2 * z + b2)), "forgot-activation")],
    )


# =========================================================================== linear systems / LU
def _lu_pair(rng):
    Lm = sp.Matrix([[1, 0, 0], [rng.randint(-2, 2), 1, 0], [rng.randint(-2, 2), rng.randint(-2, 2), 1]])
    Um = sp.Matrix([[rng.choice([1, 2, -1, 3]), rng.randint(-3, 3), rng.randint(-3, 3)],
                    [0, rng.choice([1, 2, -1, -2]), rng.randint(-3, 3)],
                    [0, 0, rng.choice([1, -1, 2])]])
    return Lm, Um


@gen("la.linear_systems")
def linear_systems(rng, level):
    if level == 1:
        _, U = _lu_pair(rng)
        xs = sp.Matrix([rng.randint(-4, 4) for _ in range(3)])
        b = U * xs
        return Problem(
            prompt=f"Solve $U\\mathbf x = \\mathbf b$ by back substitution, where $U = {L(U)}$ and $\\mathbf b = {tup(b)}$.",
            answer=vec(xs), kind="vector",
            rubric=["Solve the last equation for x₃", "Substitute into the 2nd row for x₂", "Then the 1st row for x₁"],
            solution=f"$\\mathbf x = {tup(xs)}$.",
            misconceptions=[(fmt(list(reversed(list(xs)))), "reversed-order")],
        )
    if level == 2:
        while True:
            Lm, U = _lu_pair(rng)
            A = Lm * U
            if max(abs(v) for v in A) <= 9:
                break
        xs = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
        b = A * xs
        return Problem(
            prompt=f"Solve $A\\mathbf x = \\mathbf b$ by Gaussian elimination: $A = {L(A)}$, $\\mathbf b = {tup(b)}$.",
            answer=vec(xs), kind="vector",
            rubric=["Augment [A | b]", "Eliminate below pivot 1, then below pivot 2 (record multipliers)",
                    "Back substitution", "Check A x = b"],
            solution=f"Elimination gives $U = {L(U)}$, $\\mathbf c = {tup(Lm.inv() * b)}$; back-substitution: $\\mathbf x = {tup(xs)}$.",
            misconceptions=[(fmt(list(U.inv() * b)), "forgot-to-transform-rhs")],
        )
    while True:
        Lm, U = _lu_pair(rng)
        A = Lm * U
        if Lm != sp.eye(3) and max(abs(v) for v in A) <= 12:
            break
    Lneg = sp.Matrix(3, 3, lambda r, c: -Lm[r, c] if r > c else Lm[r, c])
    return Problem(
        prompt=(f"Find the LU factorisation $A = LU$ (no row exchanges) of $A = {L(A)}$, with $L$ unit lower triangular. "
                "Type $L$."),
        answer=fmt(Lm), kind="matrix",
        rubric=["ℓ₂₁ = a₂₁/a₁₁, ℓ₃₁ = a₃₁/a₁₁ (multipliers for column 1)", "Update rows 2, 3",
                "ℓ₃₂ from the new pivot in column 2", "L holds the multipliers with a + sign; check LU = A"],
        solution=f"$L = {L(Lm)}$, $U = {L(U)}$.",
        misconceptions=[(fmt(Lneg), "lu-multiplier-sign")],
    )


# =========================================================================== inverses & determinants
@gen("la.inverse_det")
def inverse_det(rng, level):
    if level == 1:
        while True:
            A = rand_mat(rng, 2, 2, -4, 4)
            if A.det() in (1, -1, 2, -2):
                break
        a, b, c, d = A
        adj = sp.Matrix([[d, -b], [-c, a]])
        mis = [(fmt(sp.Matrix([[d, b], [c, a]]) / A.det()), "inverse-sign-pattern")]
        if A.det() != 1:
            mis.append((fmt(adj), "forgot-divide-by-det"))
        return Problem(
            prompt=f"Find $A^{{-1}}$ for $A = {L(A)}$.",
            answer=fmt(A.inv()), kind="matrix",
            rubric=[f"det A = ad − bc = {A.det()}", "Swap a and d, negate b and c", "Divide by det A", "Check A A⁻¹ = I"],
            solution=f"$A^{{-1}} = \\frac{{1}}{{{A.det()}}}{L(adj)} = {L(A.inv())}$.",
            misconceptions=mis,
        )
    if level == 2:
        while True:
            A = rand_mat(rng, 3, 3)
            if A.det() != 0:
                break
        return Problem(
            prompt=f"Compute $\\det A$ for $A = {L(A)}$.",
            answer=str(A.det()), kind="num",
            rubric=["Cofactor expansion along a row/column with zeros (or eliminate to triangular form)",
                    "Track the checkerboard signs (+ − +)", "Sum the terms"],
            solution=f"$\\det A = {A.det()}$.",
            misconceptions=[(str(A[0, 0] * A[1, 1] * A[2, 2]), "determinant-diagonal-only"), (str(-A.det()), "cofactor-sign")],
        )
    if rng.random() < 0.5:
        dA, dB = nz(rng, -4, 4), nz(rng, -5, 5)
        c = rng.choice([2, 3, -1, -2])
        val = sp.Rational(c ** 3 * dB, dA)
        return Problem(
            prompt=(f"$A$ and $B$ are $3\\times 3$ with $\\det A = {dA}$ and $\\det B = {dB}$. "
                    f"Compute $\\det\\left({c}\\,A^{{-1}}B^\\top\\right)$."),
            answer=fmt(val), kind="num",
            rubric=["det(cM) = cⁿ det M for n×n M (here n = 3)", "det(A⁻¹) = 1/det A", "det(Bᵀ) = det B",
                    "det of a product = product of dets"],
            solution=f"$= ({c})^3\\cdot\\frac{{1}}{{{dA}}}\\cdot({dB}) = {L(val)}$.",
            misconceptions=[(fmt(sp.Rational(c * dB, dA)), "determinant-scalar-not-cubed"),
                            (fmt(sp.Rational(c ** 3 * dA, dB)), "inverted-wrong-matrix")],
        )
    kk = sp.Symbol("k")
    while True:
        A = rand_mat(rng, 3, 3)
        r, c = rng.randint(0, 2), rng.randint(0, 2)
        A[r, c] = kk
        dpoly = sp.expand(A.det())
        alpha = dpoly.coeff(kk, 1)
        if alpha != 0:
            break
    sol = sp.solve(dpoly, kk)[0]
    return Problem(
        prompt=f"For which value of $k$ is $A = {L(A)}$ singular (not invertible)?",
        answer=fmt(sol), kind="num",
        rubric=["Singular ⇔ det A = 0", "Expand det A along the row or column containing k (it is linear in k)",
                f"det A = {dpoly}", "Solve for k"],
        solution=f"$\\det A = {L(dpoly)} = 0 \\Rightarrow k = {L(sol)}$.",
        misconceptions=[(fmt(-sol), "sign-error")],
    )


# =========================================================================== subspaces
@gen("la.subspaces")
def subspaces(rng, level):
    if level == 1:
        while True:
            r = rng.randint(1, 3)
            A = rand_mat(rng, 3, r, -2, 2) * rand_mat(rng, r, 4, -2, 2)
            if A.rank() == r and max(abs(v) for v in A) <= 8:
                break
        return Problem(
            prompt=f"What is the rank of $A = {L(A)}$?",
            answer=str(r), kind="num",
            rubric=["Row reduce to echelon form", "Count the pivots", "rank = number of pivots"],
            solution=f"RREF: ${L(A.rref()[0])}$, which has ${r}$ pivot(s).",
            misconceptions=[("3", "rank-counted-rows")],
        )
    if level == 2:
        while True:
            a, b, c, d = (rng.randint(-3, 3) for _ in range(4))
            R0 = sp.Matrix([[1, 0, a, b], [0, 1, c, d]])
            M = rand_mat(rng, 3, 2, -2, 2)
            A = M * R0
            if M.rank() == 2 and max(abs(v) for v in A) <= 9:
                break
        s1, s2 = [-a, -c, 1, 0], [-b, -d, 0, 1]
        ans = sp.Matrix([s1, s2])
        return Problem(
            prompt=(f"Find the special solutions of $A\\mathbf x = \\mathbf 0$ for $A = {L(A)}$. "
                    "(Columns 1 and 2 are the pivot columns, $x_3, x_4$ are free.) "
                    "Type the matrix whose **rows** are $\\mathbf s_1$ (with $x_3=1, x_4=0$) and $\\mathbf s_2$ (with $x_3=0, x_4=1$)."),
            answer=fmt(ans), kind="matrix", answer_hint="[[s1 entries], [s2 entries]]  (two rows of length 4)",
            rubric=["Row reduce to R = rref(A)", "Identify pivot vs free columns", "Set one free variable to 1, the other 0",
                    "Solve for pivot variables (they are minus the free-column entries of R)"],
            solution=f"$R = {L(A.rref()[0])}$, $\\mathbf s_1 = {tup(s1)}$, $\\mathbf s_2 = {tup(s2)}$.",
            misconceptions=[(fmt(sp.Matrix([[a, c, 1, 0], [b, d, 0, 1]])), "special-solutions-forgot-negate")],
        )
    while True:
        A = rand_mat(rng, 3, 2, -2, 2) * rand_mat(rng, 2, 3, -2, 2)
        if A.rank() != 2:
            continue
        yv = A.T.nullspace()[0]
        if yv[2] != 0 and max(abs(v) for v in A) <= 8:
            break
    while True:
        x0 = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
        bvec = A * x0
        cval = bvec[2]
        if cval != 0 and (bvec[0], bvec[1]) != (0, 0):
            break
    yv = yv / sp.gcd(list(yv)) if all(v.is_integer for v in yv) else yv
    return Problem(
        prompt=(f"$A = {L(A)}$ has rank 2. For which value of $c$ is $\\mathbf b = ({bvec[0]}, {bvec[1]}, c)$ in the "
                "column space $C(A)$? (Use the left null space.)"),
        answer=str(cval), kind="num",
        rubric=["Eliminate on [A | b] — or find y with Aᵀy = 0", f"Left null space spanned by y = {list(yv)}",
                "b ∈ C(A) ⇔ yᵀb = 0", "Solve the linear equation for c"],
        solution=f"$\\mathbf y = {tup(yv)}$ spans $N(A^\\top)$; $\\mathbf y^\\top\\mathbf b = 0 \\Rightarrow c = {cval}$.",
        misconceptions=[(str(-cval), "sign-error")],
    )


# =========================================================================== orthogonality
@gen("la.orthogonality")
def orthogonality(rng, level):
    if level == 1:
        while True:
            a = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
            b = sp.Matrix([rng.randint(-4, 4) for _ in range(3)])
            if a.dot(a) != 0 and a.dot(b) != 0:
                break
        p = (a.dot(b) / a.dot(a)) * a
        return Problem(
            prompt=f"Project $\\mathbf b = {tup(b)}$ onto the line through $\\mathbf a = {tup(a)}$. Give the projection vector $\\mathbf p$.",
            answer=vec(p), kind="vector",
            rubric=["p = (aᵀb / aᵀa) a", f"aᵀb = {a.dot(b)}, aᵀa = {a.dot(a)}", "Scale a", "Check (b − p) ⟂ a"],
            solution=f"$\\mathbf p = \\frac{{{a.dot(b)}}}{{{a.dot(a)}}}{tup(a)} = {tup(p)}$.",
            misconceptions=[(fmt(list(a.dot(b) * a)), "projection-forgot-divide")],
        )
    if level == 2:
        while True:
            ts = list(rng.choice([(-2, -1, 1, 2), (-3, -1, 1, 3), (0, 1, 2, 3), (-1, 0, 1, 2)]))
            ys = [rng.randint(-3, 6) for _ in ts]
            A = sp.Matrix([[1, tv] for tv in ts])
            bb = sp.Matrix(ys)
            xh = (A.T * A).inv() * A.T * bb
            if xh[1] != 0:
                break
        pts = ", ".join(f"({tv}, {yv})" for tv, yv in zip(ts, ys))
        return Problem(
            prompt=(f"Fit the best least-squares line $y = c + d\\,t$ to the points $(t, y)$ = {pts}. "
                    "Give $(c, d)$."),
            answer=vec(xh), kind="vector", answer_hint="[c, d]",
            rubric=["Write A x ≈ b with rows [1, tᵢ]", f"AᵀA = {(A.T*A).tolist()}, Aᵀb = {list(A.T*bb)}",
                    "Solve the normal equations AᵀA x̂ = Aᵀb", "Check: the residual sums to 0"],
            solution=f"$A^\\top A = {L(A.T*A)}$, $A^\\top\\mathbf b = {tup(A.T*bb)}$, so $(c, d) = {tup(xh)}$.",
            misconceptions=[(fmt(list(reversed(list(xh)))), "intercept-slope-swapped")],
        )
    best = None
    for _ in range(300):
        a, na = quad_vec(rng)
        b = [rng.randint(-4, 4) for _ in range(3)]
        A_, B_ = sp.Matrix(a), sp.Matrix(b)
        r12 = A_.dot(B_) / na
        Bp = B_ - (A_.dot(B_) / na ** 2) * A_
        if Bp.norm() == 0 or r12 == 0:
            continue
        best = (a, b, na, r12, Bp)
        if sp.sqrt(Bp.dot(Bp)).is_rational:
            break
    a, b, na, r12, Bp = best
    r22 = sp.sqrt(Bp.dot(Bp))
    R = sp.Matrix([[na, r12], [0, r22]])
    return Problem(
        prompt=(f"Apply Gram–Schmidt to the columns of $A = \\big[\\,\\mathbf a\\ \\ \\mathbf b\\,\\big]$ with "
                f"$\\mathbf a = {tup(a)}$, $\\mathbf b = {tup(b)}$, and write $A = QR$ ($Q$ with orthonormal columns, "
                "$R$ upper triangular with positive diagonal). Type $R$."),
        answer=fmt(R), kind="matrix", answer_hint="[[r11, r12], [0, r22]] — exact, sqrt allowed",
        rubric=[f"q₁ = a/‖a‖ with ‖a‖ = {na}", "r₁₂ = q₁ᵀb", "B = b − r₁₂ q₁ (orthogonal to a)",
                "r₂₂ = ‖B‖, q₂ = B/r₂₂"],
        solution=(f"$r_{{11}} = {na}$, $r_{{12}} = \\mathbf q_1^\\top\\mathbf b = {L(r12)}$, "
                  f"$\\mathbf B = {tup(Bp)}$, $r_{{22}} = \\|\\mathbf B\\| = {L(r22)}$. $R = {L(R)}$."),
        misconceptions=[(fmt(sp.Matrix([[na, sp.Matrix(a).dot(sp.Matrix(b))], [0, r22]])), "r12-used-unnormalised-a")],
    )


# =========================================================================== eigenvalues
def _eig2(rng, lo=-4, hi=5, cap=12):
    while True:
        P = unimod2(rng)
        d1, d2 = rng.sample(range(lo, hi + 1), 2)
        A = P * sp.diag(d1, d2) * P.inv()
        if max(abs(v) for v in A) <= cap and A[0, 1] != 0 and A[1, 0] != 0:
            return A, P, d1, d2


@gen("la.eigen")
def eigen(rng, level):
    if level == 1:
        A, _, d1, d2 = _eig2(rng)
        return Problem(
            prompt=f"Find the eigenvalues of $A = {L(A)}$.",
            answer=[str(d1), str(d2)], kind="set",
            rubric=["Characteristic polynomial det(A − λI) = λ² − (tr A)λ + det A",
                    f"tr A = {A.trace()}, det A = {A.det()}", "Solve the quadratic", "Check: sum = trace, product = det"],
            solution=f"$\\lambda^2 - ({A.trace()})\\lambda + ({A.det()}) = 0 \\Rightarrow \\lambda = {d1}, {d2}$.",
            misconceptions=[(f"{A[0, 0]}, {A[1, 1]}", "diagonal-entries-not-eigenvalues"),
                            (f"{-d1}, {-d2}", "characteristic-poly-sign")],
        )
    if level == 2:
        A, P, d1, d2 = _eig2(rng)
        if P[0, 0] == 0:
            P = P[:, [1, 0]]
            d1, d2 = d2, d1
        ev = [1, sp.Rational(P[1, 0], P[0, 0])]
        mis = []
        if P[0, 1] != 0:
            mis.append((fmt([1, sp.Rational(P[1, 1], P[0, 1])]), "eigenvector-of-other-eigenvalue"))
        return Problem(
            prompt=(f"$\\lambda = {d1}$ is an eigenvalue of $A = {L(A)}$. Find a corresponding eigenvector, "
                    "scaled so that its **first** component is 1."),
            answer=vec(ev), kind="vector", answer_hint="[1, something]",
            rubric=[f"Form A − ({d1})I", "Its rows are dependent; solve one row = 0", "Scale so x₁ = 1", "Check A v = λ v"],
            solution=f"$A - ({d1})I = {L(A - d1 * sp.eye(2))}$, null vector $\\mathbf v = {tup(ev)}$.",
            misconceptions=mis,
        )
    if rng.random() < 0.5:
        fr = [sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 4), sp.Rational(1, 5), sp.Rational(2, 5),
              sp.Rational(3, 10), sp.Rational(1, 10)]
        p, q = rng.sample(fr, 2)
        A = sp.Matrix([[1 - p, q], [p, 1 - q]])
        ss = [q / (p + q), p / (p + q)]
        return Problem(
            prompt=(f"A two-state Markov chain evolves as $\\mathbf x_{{k+1}} = A\\mathbf x_k$ with column-stochastic "
                    f"$A = {L(A)}$. Find the steady state $\\mathbf x_\\infty$ (the eigenvector for $\\lambda = 1$ whose "
                    "entries sum to 1). On paper also find the other eigenvalue and explain why $\\mathbf x_k \\to \\mathbf x_\\infty$."),
            answer=vec(ss), kind="vector", answer_hint="[x1, x2] with x1 + x2 = 1",
            rubric=["Columns sum to 1 ⇒ λ = 1 is an eigenvalue", "Solve (A − I)x = 0",
                    "Normalise so the entries sum to 1", f"Other eigenvalue = tr A − 1 = {L(1 - p - q)}, |λ₂| < 1"],
            solution=f"$(A - I)\\mathbf x = 0$ gives $-{L(p)}x_1 + {L(q)}x_2 = 0$, so $\\mathbf x_\\infty = {tup(ss)}$; $\\lambda_2 = {L(1 - p - q)}$.",
            misconceptions=[(fmt(list(reversed(ss))), "steady-state-swapped")],
        )
    P = unimod2(rng)
    d1, d2 = rng.choice([(-1, 2), (1, 2), (0, 2), (1, -1), (2, -2), (0, -1), (1, 3)])
    k = rng.randint(4, 6) if max(abs(d1), abs(d2)) <= 2 else rng.randint(3, 4)
    cvec = sp.Matrix([nz(rng, -2, 2), nz(rng, -2, 2)])
    A = P * sp.diag(d1, d2) * P.inv()
    v = P * cvec
    res = A ** k * v
    return Problem(
        prompt=(f"Let $A = {L(A)}$, whose eigenpairs are $\\lambda_1 = {d1}$, $\\mathbf v_1 = {tup(P.col(0))}$ and "
                f"$\\lambda_2 = {d2}$, $\\mathbf v_2 = {tup(P.col(1))}$. Compute $A^{{{k}}}\\mathbf u$ for $\\mathbf u = {tup(v)}$ "
                "without multiplying matrices repeatedly."),
        answer=vec(res), kind="vector",
        rubric=["Write u = c₁v₁ + c₂v₂ (solve a 2×2 system)", f"c = ({cvec[0]}, {cvec[1]})",
                f"A^{k} u = c₁λ₁^{k} v₁ + c₂λ₂^{k} v₂", "Add the vectors"],
        solution=(f"$\\mathbf u = {cvec[0]}\\mathbf v_1 + ({cvec[1]})\\mathbf v_2$, so "
                  f"$A^{{{k}}}\\mathbf u = {cvec[0]}\\cdot({d1})^{{{k}}}\\mathbf v_1 + ({cvec[1]})({d2})^{{{k}}}\\mathbf v_2 = {tup(res)}$."),
        misconceptions=[(fmt(list(k * A * v)), "power-as-multiple")],
    )


@gen("la.eigen", name="eigen_properties")
def eigen_properties(rng, level):
    if level == 1:
        A, _, d1, d2 = _eig2(rng)
        return Problem(
            prompt=(f"$A = {L(A)}$ has eigenvalue $\\lambda_1 = {d1}$. Without solving the characteristic polynomial, "
                    "find the other eigenvalue."),
            answer=str(d2), kind="num",
            rubric=["Sum of eigenvalues = trace", f"tr A = {A.trace()}", "λ₂ = tr A − λ₁", "Check: λ₁λ₂ = det A"],
            solution=f"$\\lambda_2 = \\operatorname{{tr}}A - \\lambda_1 = {A.trace()} - ({d1}) = {d2}$ (and ${d1}\\cdot({d2}) = {A.det()} = \\det A$ ✓).",
            misconceptions=[(str(d1 - A.trace()), "trace-sign"), (str(A.det()), "gave-determinant")],
        )
    if level == 2:
        while True:
            lams = [rng.randint(-3, 3) for _ in range(3)]
            c = nz(rng, -3, 3)
            vals = [l * l + c * l for l in lams]
            if len(set(vals)) == 3:
                break
        A = sp.Matrix([[lams[0], rng.randint(-3, 3), rng.randint(-3, 3)],
                       [0, lams[1], rng.randint(-3, 3)],
                       [0, 0, lams[2]]])
        Bm = A ** 2 + c * A
        wrong = [l * l + c for l in lams]
        return Problem(
            prompt=f"Let $A = {L(A)}$. Find the eigenvalues of $B = A^2 {'+' if c > 0 else '-'} {abs(c)}A$ (no need to compute $B$).",
            answer=[str(v) for v in vals], kind="set",
            rubric=["A is triangular → eigenvalues on the diagonal", "If Av = λv then p(A)v = p(λ)v",
                    f"Apply p(λ) = λ² {c:+d}λ to each eigenvalue"],
            solution=f"$\\lambda(A) = {lams}$; $\\lambda(B) = \\lambda^2 {c:+d}\\lambda = {vals}$. (Indeed $B = {L(Bm)}$.)",
            misconceptions=[(", ".join(map(str, wrong)), "polynomial-misapplied")],
        )
    a, d = rng.randint(-3, 4), rng.randint(-3, 4)
    while a == d:
        d = rng.randint(-3, 4)
    c = nz(rng, -3, 3)
    kk = sp.Symbol("k")
    A = sp.Matrix([[a, kk], [c, d]])
    ksol = sp.solve(sp.discriminant(A.charpoly().as_expr(), sp.Symbol("lambda")), kk)[0]
    lam = sp.Rational(a + d, 2)
    return Problem(
        prompt=(f"For which value of $k$ does $A = {L(A)}$ have a **repeated** eigenvalue? Give $k$ and that eigenvalue. "
                "On paper: is $A$ diagonalizable then?"),
        answer=[fmt(ksol), fmt(lam)], kind="list", answer_hint="k, lambda   e.g. -1/4, 3/2",
        rubric=["Characteristic polynomial λ² − (a+d)λ + (ad − kc)", "Repeated root ⇔ discriminant (a+d)² − 4(ad − kc) = 0",
                "Simplify to (a − d)² + 4kc = 0, solve for k", "λ = (a + d)/2; A ≠ λI so it is NOT diagonalizable"],
        solution=f"$(a-d)^2 + 4kc = {(a-d)**2} {4*c:+d}k = 0 \\Rightarrow k = {L(ksol)}$, $\\lambda = {L(lam)}$. A is a defective (Jordan) matrix.",
        misconceptions=[(f"{fmt(-ksol)}, {fmt(lam)}", "discriminant-sign")],
    )


# =========================================================================== diagonalization / spectral theorem
_DIRS = [(1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1), (1, 3), (3, 1)]


@gen("la.diagonalization")
def diagonalization(rng, level):
    if level == 1:
        P = unimod2(rng)
        d1, d2 = rng.sample(range(-3, 5), 2)
        D = sp.diag(d1, d2)
        A = P * D * P.inv()
        return Problem(
            prompt=f"Given $P = {L(P)}$ and $D = {L(D)}$, compute $A = PDP^{{-1}}$.",
            answer=fmt(A), kind="matrix",
            rubric=["P⁻¹ (det P = ±1)", "Compute PD (scales the columns of P)", "Multiply by P⁻¹ on the right",
                    "Check: tr A = tr D"],
            solution=f"$P^{{-1}} = {L(P.inv())}$, $A = {L(A)}$.",
            misconceptions=[(fmt(P.inv() * D * P), "diagonalization-order")],
        )
    v = sp.Matrix(rng.choice(_DIRS))
    nv2 = v.dot(v)
    P1 = v * v.T / nv2
    if level == 2:
        c = nz(rng, -2, 2)
        l2 = rng.randint(-3, 3)
        l1 = l2 + c * nv2
        A = l2 * sp.eye(2) + c * v * v.T
        return Problem(
            prompt=(f"The symmetric matrix $A = {L(A)}$ has eigenvalues ${l1}$ and ${l2}$. Write the spectral decomposition "
                    f"$A = \\lambda_1 P_1 + \\lambda_2 P_2$ with orthogonal projectors $P_k = \\mathbf q_k\\mathbf q_k^\\top$. "
                    f"Type $P_1$, the projector onto the eigenspace of $\\lambda_1 = {l1}$."),
            answer=fmt(P1), kind="matrix", answer_hint="exact fractions, e.g. [[1/5, 2/5], [2/5, 4/5]]",
            rubric=[f"Solve (A − {l1}I)v = 0", "Normalise: q₁ = v/‖v‖", "P₁ = q₁q₁ᵀ = vvᵀ/(vᵀv)",
                    "Check P₁² = P₁ and λ₁P₁ + λ₂(I − P₁) = A"],
            solution=f"Eigenvector $\\mathbf v = {tup(v)}$, $P_1 = \\frac{{1}}{{{nv2}}}{L(v * v.T)} = {L(P1)}$; $P_2 = I - P_1$.",
            misconceptions=[(fmt(v * v.T), "forgot-normalise-eigenvector")],
        )
    while True:
        c = rng.choice([-1, 1])
        l2 = rng.randint(-1, 2)
        l1 = l2 + c * nv2
        if abs(l1) <= 3 and l1 != 0 and l2 != l1:
            break
        v = sp.Matrix(rng.choice(_DIRS))
        nv2 = v.dot(v)
        P1 = v * v.T / nv2
    A = l2 * sp.eye(2) + c * v * v.T
    k = rng.randint(3, 5) if max(abs(l1), abs(l2)) <= 2 else rng.randint(3, 4)
    Ak = A ** k
    return Problem(
        prompt=(f"Use the spectral theorem to compute $A^{{{k}}}$ for the symmetric matrix $A = {L(A)}$. "
                "(Find the eigenvalues and orthonormal eigenvectors first; avoid repeated multiplication.)"),
        answer=fmt(Ak), kind="matrix",
        rubric=["Eigenvalues from trace and determinant", "Orthonormal eigenvectors q₁ ⟂ q₂",
                f"A^{k} = λ₁^{k} q₁q₁ᵀ + λ₂^{k} q₂q₂ᵀ", "Add the two rank-1 matrices"],
        solution=(f"$\\lambda_1 = {l1}$ with $\\mathbf v_1 = {tup(v)}$, $\\lambda_2 = {l2}$. $P_1 = {L(P1)}$, $P_2 = I - P_1$. "
                  f"$A^{{{k}}} = ({l1})^{{{k}}}P_1 + ({l2})^{{{k}}}P_2 = {L(Ak)}$."),
        misconceptions=[(fmt(A.applyfunc(lambda e: e ** k)), "entrywise-power")],
    )


# =========================================================================== SVD
def _svd2(rng, allow_zero=True):
    """A = U Σ Vᵀ with U, V both 45°-type orthogonal matrices, so A is an integer matrix."""
    H = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    R = sp.Matrix([[1, -1], [1, 1]]) / sp.sqrt(2)
    while True:
        U = rng.choice([H, R]) * sp.diag(rng.choice([-1, 1]), rng.choice([-1, 1]))
        V = rng.choice([H, R]) * sp.diag(rng.choice([-1, 1]), rng.choice([-1, 1]))
        s1 = rng.randint(2, 8)
        s2 = rng.choice([v for v in range(0 if allow_zero else 1, s1) if (s1 - v) % 2 == 0] or [None])
        if s2 is None:
            continue
        A = sp.simplify(U * sp.diag(s1, s2) * V.T)
        if all(e.is_integer for e in A):
            return A, U, V, s1, s2


@gen("la.svd")
def svd(rng, level):
    if level == 1:
        A, U, V, s1, s2 = _svd2(rng)
        mis = [(f"{s1**2}, {s2**2}", "forgot-square-root")]
        evs = list(A.eigenvals(multiple=True))
        if all(e.is_real for e in evs):
            mis.append((", ".join(fmt(e) for e in evs), "eigenvalues-not-singular-values"))
        return Problem(
            prompt=f"Find the singular values of $A = {L(A)}$.",
            answer=[str(s1), str(s2)], kind="set",
            rubric=["Form AᵀA (symmetric, PSD)", f"AᵀA = {(A.T*A).tolist()}", "Eigenvalues of AᵀA",
                    "Singular values = square roots"],
            solution=f"$A^\\top A = {L(A.T*A)}$ has eigenvalues ${s1**2}, {s2**2}$, so $\\sigma = {s1}, {s2}$.",
            misconceptions=mis,
        )
    if level == 2:
        A, U, V, s1, s2 = _svd2(rng, allow_zero=False)
        A1 = sp.simplify(s1 * U.col(0) * V.col(0).T)
        A2 = sp.simplify(s2 * U.col(1) * V.col(1).T)
        return Problem(
            prompt=(f"Compute the best rank-1 approximation $A_1$ (in the Frobenius/spectral norm) of $A = {L(A)}$. "
                    "On paper: state the approximation error $\\|A - A_1\\|_2$."),
            answer=fmt(A1), kind="matrix", answer_hint="exact fractions, e.g. [[5/2, 5/2], [5/2, 5/2]]",
            rubric=["Eigen-decompose AᵀA → σ₁ and v₁", "u₁ = A v₁ / σ₁", "A₁ = σ₁ u₁ v₁ᵀ (Eckart–Young)",
                    f"Error ‖A − A₁‖₂ = σ₂ = {s2}"],
            solution=(f"$\\sigma_1 = {s1}$, $\\mathbf v_1 = {tup(V.col(0))}$, $\\mathbf u_1 = {tup(U.col(0))}$; "
                      f"$A_1 = {L(A1)}$, error $\\sigma_2 = {s2}$."),
            misconceptions=[(fmt(A2), "used-smallest-singular-value")],
        )
    while True:
        uu = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
        vv = sp.Matrix([nz(rng, -2, 2) for _ in range(2)])
        bb = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
        if uu.dot(uu) != 0 and uu.dot(bb) != 0:
            break
    A = uu * vv.T
    xplus = A.pinv() * bb
    return Problem(
        prompt=(f"$A = {L(A)}$ has rank 1, so $A\\mathbf x = \\mathbf b$ with $\\mathbf b = {tup(bb)}$ has no unique "
                "least-squares solution. Write $A = \\sigma_1\\mathbf u_1\\mathbf v_1^\\top$, form the pseudoinverse "
                "$A^+ = \\sigma_1^{-1}\\mathbf v_1\\mathbf u_1^\\top$, and compute the minimum-norm least-squares solution "
                "$\\mathbf x^+ = A^+\\mathbf b$."),
        answer=vec(xplus), kind="vector",
        rubric=[f"A = u vᵀ with u = {list(uu)}, v = {list(vv)}", "σ₁ = ‖u‖‖v‖, u₁ = u/‖u‖, v₁ = v/‖v‖",
                "A⁺ = v uᵀ / (‖u‖²‖v‖²)", "x⁺ = v (uᵀb) / (‖u‖²‖v‖²)"],
        solution=(f"$\\|\\mathbf u\\|^2 = {uu.dot(uu)}$, $\\|\\mathbf v\\|^2 = {vv.dot(vv)}$, $\\mathbf u^\\top\\mathbf b = {uu.dot(bb)}$. "
                  f"$\\mathbf x^+ = {tup(xplus)}$ (it lies in the row space, so it has minimum norm)."),
        misconceptions=[(fmt(list(A.T * bb)), "pinv-forgot-scaling")],
    )


# =========================================================================== PCA
def _pca_data(rng):
    v = sp.Matrix(rng.choice(_DIRS))
    wv = sp.Matrix([-v[1], v[0]])
    s_, t_ = rng.sample([1, 2, 3], 2)
    mu = sp.Matrix([rng.randint(-3, 3), rng.randint(-3, 3)])
    pts = [mu + s_ * v, mu - s_ * v, mu + t_ * wv, mu - t_ * wv]
    rng.shuffle(pts)
    return pts, mu, v, wv, s_, t_


@gen("la.pca")
def pca(rng, level):
    if level == 1:
        mu = sp.Matrix([rng.randint(-3, 3), rng.randint(-3, 3)])
        while True:
            ds = [sp.Matrix([rng.randint(-3, 3), rng.randint(-3, 3)]) for _ in range(3)]
            ds.append(-(ds[0] + ds[1] + ds[2]))
            if max(abs(v) for d in ds for v in d) <= 4 and all(d != sp.zeros(2, 1) for d in ds):
                break
        pts = [mu + d for d in ds]
        S = sum((d * d.T for d in ds), sp.zeros(2, 2)) / 4
        S_n1 = sum((d * d.T for d in ds), sp.zeros(2, 2)) / 3
        S_raw = sum((p * p.T for p in pts), sp.zeros(2, 2)) / 4
        mis = [(fmt(S_n1), "covariance-used-n-minus-1")]
        if mu != sp.zeros(2, 1):
            mis.append((fmt(S_raw), "covariance-forgot-to-center"))
        return Problem(
            prompt=("Compute the covariance matrix $S = \\frac1n\\sum_k (\\mathbf x_k - \\bar{\\mathbf x})(\\mathbf x_k - \\bar{\\mathbf x})^\\top$ "
                    "(divide by $n$) of the four points " + ", ".join(f"${tup(p)}$" for p in pts) + "."),
            answer=fmt(S), kind="matrix",
            rubric=[f"Mean x̄ = {tup(mu)}", "Centre the data", "Sum of outer products (or XᵀX of centred X)", "Divide by n = 4"],
            solution=f"$\\bar{{\\mathbf x}} = {tup(mu)}$; centred points " + ", ".join(f"${tup(d)}$" for d in ds) + f"; $S = {L(S)}$.",
            misconceptions=mis,
        )
    pts, mu, v, wv, s_, t_ = _pca_data(rng)
    pts_s = ", ".join(f"${tup(p)}$" for p in pts)
    S = sum(((p - mu) * (p - mu).T for p in pts), sp.zeros(2, 2)) / 4
    big, bigdir = (s_, v) if s_ > t_ else (t_, wv)
    if level == 2:
        ratio = sp.Rational(big ** 2, s_ ** 2 + t_ ** 2)
        return Problem(
            prompt=(f"Run PCA (covariance with $1/n$) on the four points {pts_s}. What fraction of the total variance "
                    "is explained by the first principal component?"),
            answer=fmt(ratio), kind="num",
            rubric=[f"Mean = {tup(mu)}; centre the data", f"Covariance S = {S.tolist()}",
                    "Eigenvalues of S (trace/det)", "Ratio λ₁ / (λ₁ + λ₂)"],
            solution=(f"$S = {L(S)}$ with eigenvalues ${L(S.eigenvals(multiple=True)[0])}, {L(S.eigenvals(multiple=True)[1])}$; "
                      f"PC1 along ${tup(bigdir)}$. Explained fraction $= {L(ratio)}$."),
            misconceptions=[(fmt(sp.Rational(big, s_ + t_)), "used-std-not-variance")],
        )
    off = sp.Matrix([nz(rng, -3, 3), rng.randint(-3, 3)])
    xn = mu + off
    q = bigdir / bigdir.norm()
    rec = mu + (off.dot(q)) * q
    return Problem(
        prompt=(f"Run PCA on the four points {pts_s}, keep only the first principal component, and compute the "
                f"rank-1 reconstruction of the new point $\\mathbf x = {tup(xn)}$: "
                "$\\hat{\\mathbf x} = \\bar{\\mathbf x} + \\big(\\mathbf q_1^\\top(\\mathbf x - \\bar{\\mathbf x})\\big)\\mathbf q_1$."),
        answer=vec(rec), kind="vector",
        rubric=[f"Mean x̄ = {tup(mu)}", f"Covariance S = {S.tolist()}", "Top eigenvector q₁ (unit length; sign does not matter)",
                "Project the centred point onto q₁", "Add the mean back"],
        solution=(f"$\\mathbf q_1 = {tup(q)}$, $\\mathbf x - \\bar{{\\mathbf x}} = {tup(off)}$, "
                  f"coefficient ${L(off.dot(q))}$, $\\hat{{\\mathbf x}} = {tup(rec)}$."),
        misconceptions=[(fmt(list(rec - mu)), "pca-forgot-to-add-mean"),
                        (fmt(list(xn.dot(q) * q)), "pca-forgot-to-center")],
    )


# =========================================================================== matrix calculus
@gen("la.matrix_calculus")
def matrix_calculus(rng, level):
    if level == 1:
        while True:
            A = rand_mat(rng, 2, 2)
            if A != A.T:
                break
        b = sp.Matrix([rng.randint(-3, 3) for _ in range(2)])
        x0 = sp.Matrix([rng.randint(-2, 2) for _ in range(2)])
        while x0 == sp.zeros(2, 1):
            x0 = sp.Matrix([rng.randint(-2, 2) for _ in range(2)])
        g = (A + A.T) * x0 + b
        return Problem(
            prompt=(f"For $f(\\mathbf x) = \\mathbf x^\\top A\\mathbf x + \\mathbf b^\\top\\mathbf x$ with $A = {L(A)}$ "
                    f"(not symmetric) and $\\mathbf b = {tup(b)}$, compute $\\nabla f$ at $\\mathbf x = {tup(x0)}$."),
            answer=vec(g), kind="vector",
            rubric=["∇(xᵀAx) = (A + Aᵀ)x", "∇(bᵀx) = b", f"A + Aᵀ = {(A + A.T).tolist()}", "Evaluate"],
            solution=f"$\\nabla f = (A + A^\\top)\\mathbf x + \\mathbf b = {L(A + A.T)}{tup(x0)} + {tup(b)} = {tup(g)}$.",
            misconceptions=[(fmt(list(2 * A * x0 + b)), "assumed-symmetric")],
        )
    if level == 2:
        A = rand_mat(rng, 3, 2, -2, 2)
        b = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
        x0 = sp.Matrix([rng.randint(-2, 2) for _ in range(2)])
        r = A * x0 - b
        while r == sp.zeros(3, 1) or A.T * r == sp.zeros(2, 1):
            b = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
            r = A * x0 - b
        g = 2 * A.T * r
        return Problem(
            prompt=(f"Let $f(\\mathbf x) = \\|A\\mathbf x - \\mathbf b\\|^2$ with $A = {L(A)}$ and $\\mathbf b = {tup(b)}$. "
                    f"Derive $\\nabla f$ and evaluate it at $\\mathbf x = {tup(x0)}$."),
            answer=vec(g), kind="vector",
            rubric=["f = (Ax − b)ᵀ(Ax − b)", "Differential df = 2(Ax − b)ᵀA dx", "∇f = 2Aᵀ(Ax − b)",
                    f"Residual r = {list(r)}", "Multiply by 2Aᵀ"],
            solution=f"$\\mathbf r = A\\mathbf x - \\mathbf b = {tup(r)}$, $\\nabla f = 2A^\\top\\mathbf r = {tup(g)}$.",
            misconceptions=[(fmt(list(A.T * r)), "forgot-factor-2"), (fmt(list(2 * A.T * (b - A * x0))), "residual-sign")],
        )
    while True:
        A = rand_mat(rng, 3, 2, -2, 2)
        if A.rank() == 2:
            break
    b = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
    lam = rng.choice([1, 2, 3])
    M = A.T * A + lam * sp.eye(2)
    xr = M.inv() * A.T * b
    xols = (A.T * A).inv() * A.T * b
    return Problem(
        prompt=(f"Ridge regression: minimise $f(\\mathbf x) = \\|A\\mathbf x - \\mathbf b\\|^2 + {lam}\\|\\mathbf x\\|^2$ with "
                f"$A = {L(A)}$ and $\\mathbf b = {tup(b)}$. Derive the closed-form minimiser by setting $\\nabla f = 0$, "
                "then compute it."),
        answer=vec(xr), kind="vector",
        rubric=["∇f = 2Aᵀ(Ax − b) + 2λx", "Set to 0: (AᵀA + λI)x = Aᵀb",
                f"AᵀA + λI = {M.tolist()}, Aᵀb = {list(A.T*b)}", "Solve the 2×2 system",
                "(Hessian 2(AᵀA + λI) ≻ 0 ⇒ unique minimum)"],
        solution=f"$(A^\\top A + {lam}I) = {L(M)}$, $A^\\top\\mathbf b = {tup(A.T*b)}$, $\\mathbf x^* = {tup(xr)}$.",
        misconceptions=[(fmt(list(xols)), "ridge-dropped-regulariser")],
    )


@gen("la.matrix_calculus", name="hessians_and_softmax")
def softmax_calc(rng, level):
    if level == 1:
        while True:
            A = rand_mat(rng, 2, 2)
            if A != A.T:
                break
        H = A + A.T
        return Problem(
            prompt=f"Find the Hessian of $f(\\mathbf x) = \\mathbf x^\\top A\\mathbf x$ for $A = {L(A)}$.",
            answer=fmt(H), kind="matrix",
            rubric=["∇f = (A + Aᵀ)x", "Hessian = Jacobian of the gradient = A + Aᵀ", "Result is symmetric"],
            solution=f"$\\nabla^2 f = A + A^\\top = {L(H)}$.",
            misconceptions=[(fmt(2 * A), "assumed-symmetric"), (fmt(A), "forgot-transpose-term")],
        )
    while True:
        a = [rng.randint(1, 4) for _ in range(3)]
        if len(set(a)) > 1:
            break
    zs = ", ".join("0" if v == 1 else f"\\ln {v}" for v in a)
    p = sp.Matrix([sp.Rational(v, sum(a)) for v in a])
    if level == 2:
        cls = rng.randint(0, 2)
        yv = sp.Matrix([1 if k == cls else 0 for k in range(3)])
        g = p - yv
        return Problem(
            prompt=(f"A 3-class classifier outputs logits $\\mathbf z = ({zs})$; $\\mathbf p = \\mathrm{{softmax}}(\\mathbf z)$ and the "
                    f"loss is $\\mathcal L = -\\ln p_{{{cls + 1}}}$ (true class {cls + 1}). Compute $\\nabla_{{\\mathbf z}}\\mathcal L$."),
            answer=vec(g), kind="vector",
            rubric=[f"e^z = {a}, so p = e^z / Σ e^z", "L = −z_y + ln Σ_k e^(z_k)", "∂L/∂z_k = p_k − [k = y]",
                    "Subtract the one-hot vector"],
            solution=f"$\\mathbf p = {tup(p)}$, $\\nabla_{{\\mathbf z}}\\mathcal L = \\mathbf p - \\mathbf y = {tup(g)}$.",
            misconceptions=[(fmt(list(yv - p)), "loss-gradient-sign"), (fmt(list(p)), "forgot-one-hot")],
        )
    J = sp.diag(*p) - p * p.T
    return Problem(
        prompt=(f"For $\\mathbf p = \\mathrm{{softmax}}(\\mathbf z)$, derive $\\frac{{\\partial p_i}}{{\\partial z_j}}$ and then compute the "
                f"full Jacobian $J_{{ij}} = \\frac{{\\partial p_i}}{{\\partial z_j}}$ at $\\mathbf z = ({zs})$."),
        answer=fmt(J), kind="matrix", answer_hint="3x3 with exact fractions",
        rubric=["Quotient rule on p_i = e^(z_i) / Σ e^(z_k)", "∂p_i/∂z_j = p_i(δ_ij − p_j)",
                "J = diag(p) − ppᵀ", f"p = {list(p)}", "Check: rows sum to 0"],
        solution=f"$\\mathbf p = {tup(p)}$, $J = \\operatorname{{diag}}(\\mathbf p) - \\mathbf p\\mathbf p^\\top = {L(J)}$.",
        misconceptions=[(fmt(sp.diag(*p)), "softmax-jacobian-forgot-outer-product"),
                        (fmt(-p * p.T), "softmax-jacobian-forgot-diagonal")],
    )


# =========================================================================== complex vectors / Hermitian / unitary
_GAUSS = [1 + I, 1 - I, 2 + I, 1 + 2 * I, 2 - I, 2 * I, 2, 2 + 2 * I, -1 + I, 3 * I]


@gen("la.complex_inner")
def complex_inner(rng, level):
    if level == 1:
        def cz():
            return rng.randint(-3, 3) + rng.randint(-3, 3) * I
        while True:
            uu, vv = [cz(), cz()], [cz(), cz()]
            ip = sp.expand(sum(sp.conjugate(a) * b for a, b in zip(uu, vv)))
            if ip != 0 and any(sp.im(a) != 0 for a in uu):
                break
        noconj = sp.expand(sum(a * b for a, b in zip(uu, vv)))
        return Problem(
            prompt=(f"Compute the complex inner product $\\langle\\mathbf u,\\mathbf v\\rangle = \\mathbf u^H\\mathbf v = "
                    f"\\sum_k \\overline{{u_k}}\\,v_k$ for $\\mathbf u = {tup(uu)}$, $\\mathbf v = {tup(vv)}$."),
            answer=fmt(ip), kind="num", answer_hint="a complex number like 3 - 2*i",
            rubric=["Conjugate the entries of u (first argument)", "Multiply componentwise with v", "Add, using i² = −1"],
            solution=f"$\\overline{{\\mathbf u}} = {tup([sp.conjugate(a) for a in uu])}$, $\\langle\\mathbf u,\\mathbf v\\rangle = {L(ip)}$.",
            misconceptions=[(fmt(noconj), "forgot-conjugate"), (fmt(sp.conjugate(ip)), "conjugated-wrong-argument")],
        )
    if level == 2:
        while True:
            wz = rng.choice(_GAUSS) * rng.choice([1, -1])
            m2 = sp.expand(wz * sp.conjugate(wz))
            opts = [dd for dd in range(-8, 9) if sp.sqrt(dd * dd + 4 * m2).is_integer]
            if opts:
                break
        dd = rng.choice(opts)
        dval = rng.randint(-3, 3)
        A = sp.Matrix([[dval + dd, wz], [sp.conjugate(wz), dval]])
        evs = list(A.eigenvals(multiple=True))
        return Problem(
            prompt=f"Verify that $A = {L(A)}$ is Hermitian and find its eigenvalues.",
            answer=[fmt(e) for e in evs], kind="set",
            rubric=["A^H = Ā^T equals A ⇒ Hermitian ⇒ real eigenvalues", "det(A − λI) uses w·w̄ = |w|²",
                    f"λ² − ({A.trace()})λ + ({sp.expand(A.det())}) = 0", "Solve"],
            solution=f"$|w|^2 = {m2}$, characteristic polynomial $\\lambda^2 - ({A.trace()})\\lambda + ({sp.expand(A.det())})$, $\\lambda = {', '.join(L(e) for e in evs)}$.",
            misconceptions=[(f"{fmt(A[0, 0])}, {fmt(A[1, 1])}", "diagonal-entries-not-eigenvalues")],
        )
    psis = [(sp.Rational(3, 5), sp.Rational(4, 5) * I), (sp.Rational(4, 5), -sp.Rational(3, 5) * I),
            (1 / sp.sqrt(2), I / sp.sqrt(2)), ((1 + I) / 2, 1 / sp.sqrt(2)), (1 / sp.sqrt(2), (1 - I) / 2),
            (sp.Rational(3, 5), sp.Rational(-4, 5))]
    a0, a1 = rng.choice(psis)
    psi = sp.Matrix([a0, a1])
    while True:
        wz = rng.choice(_GAUSS)
        H = sp.Matrix([[rng.randint(-3, 3), wz], [sp.conjugate(wz), rng.randint(-3, 3)]])
        val = sp.nsimplify(sp.simplify((psi.H * H * psi)[0]))
        wrong = sp.nsimplify(sp.simplify((psi.T * H * psi)[0]))
        if val != 0:
            break
    return Problem(
        prompt=(f"A qubit is in state $|\\psi\\rangle = {L(a0)}|0\\rangle + \\left({L(a1)}\\right)|1\\rangle$. "
                f"Compute the expectation value $\\langle\\psi|H|\\psi\\rangle$ of the observable $H = {L(H)}$. "
                "On paper: check that $|\\psi\\rangle$ is normalised and explain why the result must be real."),
        answer=fmt(val), kind="num", answer_hint="exact real number, e.g. 7/25 or 1 + sqrt(2)/2",
        rubric=["‖ψ‖² = |a|² + |b|² = 1", "Compute H|ψ⟩", "Take the inner product with ⟨ψ| (conjugate the bra!)",
                "H Hermitian ⇒ ⟨ψ|H|ψ⟩ is real"],
        solution=f"$H|\\psi\\rangle = {tup(sp.expand(H * psi))}$, $\\langle\\psi|H|\\psi\\rangle = {L(val)}$.",
        misconceptions=[(fmt(wrong), "forgot-conjugate")],
    )


# =========================================================================== extra problem types (variety)
# One extra generator per skill, each level choosing between genuinely different tasks, so that a
# learner does not see the same sentence with new numbers over and over.
def comb(terms) -> str:
    """LaTeX for a sum of coef * name with clean signs/ones: comb([(2, 'A'), (-1, 'B')]) -> '2A - B'."""
    out = ""
    for c, name in terms:
        if isinstance(c, sp.Symbol):  # symbolic coefficient such as k
            out += (" + " if out else "") + f"{L(c)}{name}"
            continue
        if c == 0:
            continue
        mag = "" if abs(c) == 1 else str(abs(c))
        if not out:
            out = ("-" if c < 0 else "") + mag + name
        else:
            out += f" {'-' if c < 0 else '+'} " + mag + name
    return out


def sym2(rng, lo=-3, hi=3, cap=9):
    """Symmetric 2x2 integer matrix l2*I + c*v v^T with integer eigenvalues (l1 = l2 + c*|v|^2, l2)."""
    while True:
        v = sp.Matrix(rng.choice(_DIRS))
        c = nz(rng, -2, 2)
        l2 = rng.randint(lo, hi)
        A = l2 * sp.eye(2) + c * v * v.T
        if max(abs(e) for e in A) <= cap:
            return A, v, l2 + c * v.dot(v), l2


# --------------------------------------------------------------------------- vectors
@gen("la.vectors", name="vector_ops")
def vector_ops(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            while True:
                u = sp.Matrix([rng.randint(-4, 4) for _ in range(3)])
                v = sp.Matrix([rng.randint(-4, 4) for _ in range(3)])
                cr = u.cross(v)
                if all(c != 0 for c in cr):
                    break
            comp = [f"({u[1]})({v[2]}) - ({u[2]})({v[1]})", f"({u[2]})({v[0]}) - ({u[0]})({v[2]})",
                    f"({u[0]})({v[1]}) - ({u[1]})({v[0]})"]
            return Problem(
                prompt=f"Compute the cross product $\\mathbf u\\times\\mathbf v$ for $\\mathbf u = {tup(u)}$ and $\\mathbf v = {tup(v)}$.",
                answer=vec(cr), kind="vector",
                rubric=["Use (u₂v₃ − u₃v₂, u₃v₁ − u₁v₃, u₁v₂ − u₂v₁)", "Watch the order in the middle component",
                        "Check: the result is orthogonal to both u and v"],
                solution=f"$\\mathbf u\\times\\mathbf v = \\big({comp[0]},\\ {comp[1]},\\ {comp[2]}\\big) = {tup(cr)}$.",
                misconceptions=[(fmt(list(-cr)), "cross-product-order"),
                                (fmt([cr[0], -cr[1], cr[2]]), "cross-product-middle-sign")],
            )
        u = sp.Matrix([rng.randint(-4, 4) for _ in range(3)])
        v = sp.Matrix([rng.randint(-4, 4) for _ in range(3)])
        a, b = rng.choice([2, 3, -2, -3]), rng.choice([2, 3, -1, -2, -3])
        r = a * u + b * v
        return Problem(
            prompt=(f"Let $\\mathbf u = {tup(u)}$ and $\\mathbf v = {tup(v)}$. Form the linear combination "
                    f"${comb([(a, '\\mathbf u'), (b, '\\mathbf v')])}$."),
            answer=vec(r), kind="vector",
            rubric=[f"Scale u by {a}", f"Scale v by {b}", "Add componentwise"],
            solution=f"${a}{tup(u)} + ({b}){tup(v)} = {tup(a * u)} + {tup(b * v)} = {tup(r)}$.",
            misconceptions=[(fmt(list(a * u - b * v)), "sign-error")],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                v = [rng.randint(-4, 4) for _ in range(3)]
                a, b = rng.randint(-4, 4), rng.randint(-4, 4)
                if v[1] != 0 and (a * v[0] + b * v[2]) != 0:
                    break
            kk = sp.Symbol("k")
            ksol = sp.solve(a * v[0] + kk * v[1] + b * v[2], kk)[0]
            return Problem(
                prompt=(f"For which value of $k$ are $\\mathbf u = ({a}, k, {b})$ and $\\mathbf v = {tup(v)}$ orthogonal?"),
                answer=fmt(ksol), kind="num",
                rubric=["Orthogonal ⇔ u·v = 0", f"u·v = {a}·{v[0]} + {v[1]}k + {b}·{v[2]}", "Solve the linear equation for k"],
                solution=f"$\\mathbf u\\cdot\\mathbf v = {L(a * v[0] + kk * v[1] + b * v[2])} = 0 \\Rightarrow k = {L(ksol)}$.",
                misconceptions=[(fmt(-ksol), "sign-error")],
            )
        q, n = quad_vec(rng)
        m = rng.randint(1, 2)
        u = [m * c for c in q]
        unit = [sp.Rational(c, n) for c in q]
        return Problem(
            prompt=f"Find the unit vector pointing in the same direction as $\\mathbf u = {tup(u)}$.",
            answer=vec(unit), kind="vector", answer_hint="exact fractions, e.g. [1/3, -2/3, 2/3]",
            rubric=[f"‖u‖ = √{sum(c * c for c in u)} = {m * n}", "Divide every component by ‖u‖", "Check: the result has length 1"],
            solution=f"$\\|\\mathbf u\\| = {m * n}$, so $\\hat{{\\mathbf u}} = \\mathbf u/{m * n} = {tup(unit)}$.",
            misconceptions=[(fmt([sp.Rational(c, m * m * n * n) for c in u]), "divided-by-norm-squared")],
        )
    if rng.random() < 0.5:
        while True:
            P = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
            d1 = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
            d2 = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
            cr = d1.cross(d2)
            if cr != sp.zeros(3, 1):
                break
        Q, R = P + d1, P + d2
        area = sp.sqrt(cr.dot(cr)) / 2
        return Problem(
            prompt=(f"Find the area of the triangle with vertices $P = {tup(P)}$, $Q = {tup(Q)}$, $R = {tup(R)}$. "
                    "On paper: explain why $\\tfrac12\\|\\overrightarrow{PQ}\\times\\overrightarrow{PR}\\|$ is the area."),
            answer=fmt(area), kind="num", answer_hint="exact, e.g. sqrt(35)/2",
            rubric=[f"Edge vectors PQ = {list(d1)}, PR = {list(d2)}", "Cross product PQ × PR",
                    "‖PQ × PR‖ = area of the parallelogram (= base · height)", "Halve it for the triangle"],
            solution=(f"$\\overrightarrow{{PQ}}\\times\\overrightarrow{{PR}} = {tup(d1)}\\times{tup(d2)} = {tup(cr)}$, "
                      f"norm $\\sqrt{{{cr.dot(cr)}}}$, area $= {L(area)}$."),
            misconceptions=[(fmt(2 * area), "triangle-forgot-half")],
        )
    while True:
        a, b = rng.randint(1, 4), rng.randint(1, 4)
        c = rng.randint(-(a * b - 1), a * b - 1)
        al, be = rng.choice([1, 2, 3, -1, -2]), rng.choice([1, 2, 3, -1, -2, -3])
        sq = al * al * a * a + 2 * al * be * c + be * be * b * b
        if c != 0 and sq > 0:
            break
    ans = sp.sqrt(sq)
    return Problem(
        prompt=(f"Vectors $\\mathbf u, \\mathbf v$ satisfy $\\|\\mathbf u\\| = {a}$, $\\|\\mathbf v\\| = {b}$ and "
                f"$\\mathbf u\\cdot\\mathbf v = {c}$. Without knowing their components, compute "
                f"$\\|{comb([(al, '\\mathbf u'), (be, '\\mathbf v')])}\\|$. Derive the expansion you use."),
        answer=fmt(ans), kind="num", answer_hint="exact, e.g. sqrt(41)",
        rubric=["‖w‖² = w·w", "Expand (αu + βv)·(αu + βv) = α²‖u‖² + 2αβ u·v + β²‖v‖² (bilinearity, symmetry)",
                f"= {al * al * a * a} + ({2 * al * be * c}) + {be * be * b * b} = {sq}", "Take the square root"],
        solution=(f"$\\|{comb([(al, '\\mathbf u'), (be, '\\mathbf v')])}\\|^2 = ({al})^2\\cdot{a}^2 + 2({al})({be})({c}) + ({be})^2\\cdot{b}^2"
                  f" = {sq}$, so the norm is ${L(ans)}$."),
        misconceptions=[(fmt(sp.sqrt(al * al * a * a + be * be * b * b)), "dropped-cross-term"), (str(sq), "forgot-square-root")],
    )


# --------------------------------------------------------------------------- matmul
_MAPS = [
    ("rotate by $90^\\circ$ counterclockwise", sp.Matrix([[0, -1], [1, 0]])),
    ("reflect across the $x$-axis", sp.Matrix([[1, 0], [0, -1]])),
    ("reflect across the line $y = x$", sp.Matrix([[0, 1], [1, 0]])),
    ("stretch horizontally by a factor of $2$", sp.Matrix([[2, 0], [0, 1]])),
    ("stretch vertically by a factor of $3$", sp.Matrix([[1, 0], [0, 3]])),
    ("shear $(x, y)\\mapsto(x + y,\\ y)$", sp.Matrix([[1, 1], [0, 1]])),
    ("shear $(x, y)\\mapsto(x,\\ y - 2x)$", sp.Matrix([[1, 0], [-2, 1]])),
]


@gen("la.matmul", name="matrix_algebra")
def matrix_algebra(rng, level):
    if level == 1:
        A, B = rand_mat(rng, 2, 3), rand_mat(rng, 3, 2)
        if rng.random() < 0.5:
            AB = A * B
            tr = AB.trace()
            return Problem(
                prompt=(f"Find the trace of the product $AB$, where $A = {L(A)}$ and $B = {L(B)}$. "
                        "(Only the diagonal entries of $AB$ are needed.)"),
                answer=str(tr), kind="num",
                rubric=["AB is 2×2; trace = (AB)₁₁ + (AB)₂₂", "(AB)₁₁ = row 1 of A · column 1 of B",
                        "(AB)₂₂ = row 2 of A · column 2 of B", "Add (check: equals tr(BA))"],
                solution=f"$AB = {L(AB)}$, so $\\operatorname{{tr}}(AB) = {AB[0, 0]} + ({AB[1, 1]}) = {tr}$.",
                misconceptions=[(str(sum(AB)), "summed-all-entries")],
            )
        a, b = rng.choice([2, 3, -2]), rng.choice([1, -1, 2, -3])
        R = a * A + b * B.T
        return Problem(
            prompt=(f"Evaluate the matrix expression ${comb([(a, 'A'), (b, 'B^\\top')])}$, given $A = {L(A)}$ and $B = {L(B)}$."),
            answer=fmt(R), kind="matrix",
            rubric=["Transpose B (3×2 → 2×3: rows become columns)", f"Scale: {a}A and {b}Bᵀ", "Add entrywise"],
            solution=f"$B^\\top = {L(B.T)}$, so ${comb([(a, 'A'), (b, 'B^\\top')])} = {L(R)}$.",
            misconceptions=[(fmt(a * A - b * B.T), "sign-error")],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                A = rand_mat(rng, 2, 2, -2, 2)
                if not A.is_diagonal() and max(abs(v) for v in A ** 3) <= 30:
                    break
            return Problem(
                prompt=f"Use repeated multiplication to find $A^3$ for $A = {L(A)}$.",
                answer=fmt(A ** 3), kind="matrix",
                rubric=["Compute A² = A·A", "Then A³ = A²·A", "Do NOT cube the entries"],
                solution=f"$A^2 = {L(A ** 2)}$, $A^3 = A^2A = {L(A ** 3)}$.",
                misconceptions=[(fmt(A.applyfunc(lambda v: v ** 3)), "entrywise-power")],
            )
        A, B, Cm = rand_mat(rng, 3, 3, -2, 2), rand_mat(rng, 3, 3, -2, 2), rand_mat(rng, 3, 3, -2, 2)
        i, j = rng.randint(0, 2), rng.randint(0, 2)
        val = (A * B * Cm)[i, j]
        rowAB = A.row(i) * B
        return Problem(
            prompt=(f"Without computing the whole product, find the $({i + 1}, {j + 1})$ entry of $ABC$ for "
                    f"$A = {L(A)}$, $B = {L(B)}$, $C = {L(Cm)}$."),
            answer=str(val), kind="num",
            rubric=[f"(ABC)ᵢⱼ = (row {i + 1} of A) · B · (column {j + 1} of C)", f"Row vector r = (row {i + 1} of A) B",
                    f"Dot r with column {j + 1} of C"],
            solution=f"$\\mathbf r = {L(A.row(i))}B = {L(rowAB)}$; $\\mathbf r\\cdot C_{{:,{j + 1}}} = {val}$.",
            misconceptions=[(str((Cm * B * A)[i, j]), "multiplication-order")],
        )
    if rng.random() < 0.5:
        while True:
            (n1, M1), (n2, M2) = rng.sample(_MAPS, 2)
            if M2 * M1 != M1 * M2:
                break
        T = M2 * M1
        return Problem(
            prompt=(f"A transformation of the plane first does: {n1}; and then: {n2}. "
                    "Find the single $2\\times 2$ matrix of the combined transformation (acting on column vectors). "
                    "On paper: explain why the order of the factors matters here."),
            answer=fmt(T), kind="matrix",
            rubric=["Matrix of each map: its columns are the images of e₁ and e₂", f"First map M₁ = {M1.tolist()}",
                    f"Second map M₂ = {M2.tolist()}", "Composite = M₂M₁ (the first map acts first, so it is on the right)"],
            solution=f"$M_1 = {L(M1)}$, $M_2 = {L(M2)}$, so $M_2M_1 = {L(T)}$ (whereas $M_1M_2 = {L(M1 * M2)}$).",
            misconceptions=[(fmt(M1 * M2), "composition-order")],
        )
    while True:
        u = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
        v = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
        s_ = v.dot(u)
        if s_ in (-2, -1, 1, 2) and sum(1 for c in u if c) >= 2 and sum(1 for c in v if c) >= 2:
            break
    k = rng.randint(3, 5) if abs(s_) == 1 else rng.randint(3, 4)
    M = u * v.T
    Mk = M ** k
    return Problem(
        prompt=(f"Let $\\mathbf u = {tup(u)}$, $\\mathbf v = {tup(v)}$ and $M = \\mathbf u\\mathbf v^\\top$. Compute $M^{{{k}}}$ "
                "efficiently: regroup the product using associativity before multiplying anything out."),
        answer=fmt(Mk), kind="matrix",
        rubric=["M^k = u (vᵀu)(vᵀu)⋯(vᵀu) vᵀ by associativity", f"vᵀu = {s_} is a scalar",
                f"M^{k} = (vᵀu)^{k - 1} u vᵀ", "Form the outer product u vᵀ and scale it"],
        solution=(f"$\\mathbf v^\\top\\mathbf u = {s_}$, so $M^{{{k}}} = ({s_})^{{{k - 1}}}\\,\\mathbf u\\mathbf v^\\top = "
                  f"{'' if s_ ** (k - 1) == 1 else s_ ** (k - 1)}{L(M)} = {L(Mk)}$."),
        misconceptions=[(fmt(s_ ** k * M), "power-off-by-one"), (fmt(M.applyfunc(lambda e: e ** k)), "entrywise-power")],
    )


# --------------------------------------------------------------------------- linear systems
@gen("la.linear_systems", name="systems_more")
def systems_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            Lm, _ = _lu_pair(rng)
            while Lm == sp.eye(3):
                Lm, _ = _lu_pair(rng)
            cv = sp.Matrix([rng.randint(-4, 4) for _ in range(3)])
            b = Lm * cv
            return Problem(
                prompt=(f"Solve $L\\mathbf c = \\mathbf b$ by forward substitution, where $L = {L(Lm)}$ is unit lower "
                        f"triangular and $\\mathbf b = {tup(b)}$."),
                answer=vec(cv), kind="vector",
                rubric=["Row 1 gives c₁ directly", "Substitute c₁ into row 2 for c₂", "Substitute c₁, c₂ into row 3 for c₃"],
                solution=f"$\\mathbf c = {tup(cv)}$ (check: $L\\mathbf c = {tup(b)}$).",
                misconceptions=[(fmt(list(Lm.T.inv() * b)), "used-transpose")],
            )
        while True:
            A = rand_mat(rng, 2, 2, -5, 5)
            if A.det() != 0 and all(v != 0 for v in A):
                break
        xs = sp.Matrix([rng.randint(-4, 4) for _ in range(2)])
        b = A * xs
        D, Dx, Dy = A.det(), sp.Matrix([[b[0], A[0, 1]], [b[1], A[1, 1]]]).det(), sp.Matrix([[A[0, 0], b[0]], [A[1, 0], b[1]]]).det()
        return Problem(
            prompt=(f"Use Cramer's rule to solve ${comb([(A[0, 0], 'x'), (A[0, 1], 'y')])} = {b[0]}$, "
                    f"${comb([(A[1, 0], 'x'), (A[1, 1], 'y')])} = {b[1]}$."),
            answer=vec(xs), kind="vector", answer_hint="[x, y]",
            rubric=[f"D = det A = {D}", "Dₓ: replace column 1 by b; D_y: replace column 2 by b", "x = Dₓ/D, y = D_y/D"],
            solution=f"$D = {D}$, $D_x = {Dx}$, $D_y = {Dy}$, so $x = {L(sp.Rational(Dx, D))}$, $y = {L(sp.Rational(Dy, D))}$.",
            misconceptions=[(fmt([xs[1], xs[0]]), "swapped-x-y")],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                r1 = sp.Matrix([[rng.randint(-3, 3) for _ in range(3)]])
                r2 = sp.Matrix([[rng.randint(-3, 3) for _ in range(3)]])
                al, be = nz(rng, -2, 2), nz(rng, -2, 2)
                r3 = al * r1 + be * r2
                jj = rng.randint(0, 2)
                cols = [c for c in range(3) if c != jj]
                minor = sp.Matrix([[r1[cols[0]], r1[cols[1]]], [r2[cols[0]], r2[cols[1]]]]).det()
                if minor != 0 and max(abs(v) for v in r3) <= 9:
                    break
            p1, p2 = rng.randint(-5, 5), rng.randint(-5, 5)
            p3 = al * p1 + be * p2
            kk = sp.Symbol("k")
            row3 = [r3[c] if c != jj else kk for c in range(3)]
            eqs = [comb([(r[c], f"x_{c + 1}") for c in range(3)]) for r in (r1, r2, row3)]
            ksol = r3[jj]
            Am = sp.Matrix([list(r1), list(r2), row3])
            return Problem(
                prompt=("For which value of $k$ does the system\n\n"
                        f"$${eqs[0]} = {p1},\\quad {eqs[1]} = {p2},\\quad {eqs[2]} = {p3}$$\n\n"
                        "have infinitely many solutions?"),
                answer=fmt(ksol), kind="num",
                rubric=["A unique solution exists iff det A ≠ 0, so we need det A = 0",
                        f"Expand det A (linear in k): {sp.expand(Am.det())}", "Solve for k",
                        f"Check consistency: equation 3 = ({al})·eq1 + ({be})·eq2, right-hand side included"],
                solution=(f"$\\det A = {L(sp.expand(Am.det()))} = 0 \\Rightarrow k = {ksol}$. Then row 3 equals "
                          f"${al}\\cdot$row 1 $+ ({be})\\cdot$row 2 and ${p3} = {al}({p1}) + ({be})({p2})$, so the system is "
                          "consistent with one free variable: infinitely many solutions."),
                misconceptions=[(fmt(-ksol), "sign-error")],
            )
        while True:
            Lm, U = _lu_pair(rng)
            if Lm != sp.eye(3):
                break
        xs = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
        b = Lm * U * xs
        cv = U * xs
        return Problem(
            prompt=(f"Given the factorisation $A = LU$ with $L = {L(Lm)}$ and $U = {L(U)}$, solve $A\\mathbf x = \\mathbf b$ "
                    f"for $\\mathbf b = {tup(b)}$ using two triangular solves (do not form $A$)."),
            answer=vec(xs), kind="vector",
            rubric=["Forward substitution: L c = b", f"c = {list(cv)}", "Back substitution: U x = c", "Check A x = b"],
            solution=f"$L\\mathbf c = \\mathbf b \\Rightarrow \\mathbf c = {tup(cv)}$; $U\\mathbf x = \\mathbf c \\Rightarrow \\mathbf x = {tup(xs)}$.",
            misconceptions=[(fmt(list(U.inv() * b)), "skipped-forward-substitution")],
        )
    if rng.random() < 0.5:
        while True:
            c = [rng.randint(-3, 3), rng.randint(-3, 3), nz(rng, -2, 2)]
            ts = sorted(rng.sample(range(-2, 4), 3))
            tt = sp.Symbol("t")
            pol = c[0] + c[1] * tt + c[2] * tt ** 2
            ys = [pol.subs(tt, tv) for tv in ts]
            if max(abs(v) for v in ys) <= 25:
                break
        V = sp.Matrix([[1, tv, tv ** 2] for tv in ts])
        pts = ", ".join(f"$({tv}, {yv})$" for tv, yv in zip(ts, ys))
        return Problem(
            prompt=(f"Find the quadratic $p(t) = c_0 + c_1 t + c_2 t^2$ whose graph passes through the points {pts}. "
                    "Set up the $3\\times3$ (Vandermonde) linear system for the coefficients and solve it. "
                    "Type $(c_0, c_1, c_2)$."),
            answer=vec(c), kind="vector", answer_hint="[c0, c1, c2]",
            rubric=["One equation per point: c₀ + c₁tₖ + c₂tₖ² = yₖ", f"Matrix with rows [1, tₖ, tₖ²]: {V.tolist()}",
                    "Eliminate (subtract row 1 from rows 2, 3, ...)", "Back substitute; check all three points"],
            solution=f"$V = {L(V)}$, $V\\mathbf c = {tup(ys)}$ gives $\\mathbf c = {tup(c)}$, i.e. $p(t) = {L(pol)}$.",
            misconceptions=[(fmt(list(reversed(c))), "coefficient-order-reversed")],
        )
    while True:
        a, cc = rng.randint(-3, 3), rng.randint(-3, 3)
        R0 = sp.Matrix([[1, 0, a], [0, 1, cc]])
        M = rand_mat(rng, 3, 2, -2, 2)
        A = M * R0
        if M.rank() == 2 and max(abs(v) for v in A) <= 9 and (a, cc) != (0, 0):
            break
    xp = sp.Matrix([rng.randint(-3, 3), rng.randint(-3, 3), 0])
    b = A * xp
    sv = sp.Matrix([-a, -cc, 1])
    ans = sp.Matrix([list(xp), list(sv)])
    return Problem(
        prompt=(f"The system $A\\mathbf x = \\mathbf b$ with $A = {L(A)}$ and $\\mathbf b = {tup(b)}$ has infinitely many "
                "solutions. Write the complete solution as $\\mathbf x = \\mathbf x_p + t\\,\\mathbf s$, where the particular "
                "solution $\\mathbf x_p$ has $x_3 = 0$ and the null-space vector $\\mathbf s$ has $x_3 = 1$. "
                "Type the matrix whose rows are $\\mathbf x_p$ and $\\mathbf s$."),
        answer=fmt(ans), kind="matrix", answer_hint="[[xp entries], [s entries]]  (two rows of length 3)",
        rubric=["Row reduce [A | b] to [R | d]", "Pivot columns 1, 2; x₃ is free",
                "x_p: set x₃ = 0 and read x₁, x₂ from d", "s: solve A s = 0 with x₃ = 1"],
        solution=(f"$\\operatorname{{rref}}[A\\mid\\mathbf b] = {L(A.row_join(b).rref()[0])}$, so "
                  f"$\\mathbf x_p = {tup(xp)}$ and $\\mathbf s = {tup(sv)}$."),
        misconceptions=[(fmt(sp.Matrix([list(xp), [a, cc, 1]])), "special-solution-forgot-negate")],
    )


# --------------------------------------------------------------------------- inverses & determinants
@gen("la.inverse_det", name="det_inverse_more")
def det_inverse_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            n = rng.choice([3, 4])
            upper = rng.random() < 0.5
            A = sp.zeros(n, n)
            for r in range(n):
                for c in range(n):
                    if r == c:
                        A[r, c] = nz(rng, -3, 3)
                    elif (c > r) == upper:
                        A[r, c] = rng.randint(-5, 5)
            d = A.det()
            return Problem(
                prompt=f"Compute the determinant of the triangular matrix $A = {L(A)}$.",
                answer=str(d), kind="num",
                rubric=["Triangular ⇒ det = product of the diagonal entries", "Off-diagonal entries do not matter",
                        "Multiply carefully with signs"],
                solution="$\\det A = " + "\\cdot".join(f"({A[k, k]})" for k in range(n)) + f" = {d}$.",
                misconceptions=[(str(sum(A[k, k] for k in range(n))), "summed-diagonal")],
            )
        i, j = rng.sample(range(3), 2)
        cval = nz(rng, -4, 4)
        E = sp.eye(3)
        E[i, j] = cval
        return Problem(
            prompt=(f"The elementary matrix $E = {L(E)}$ adds {'' if cval == 1 else f'${cval}$ times '}row {j + 1} to row {i + 1}. "
                    "Find $E^{-1}$ by asking which row operation undoes it."),
            answer=fmt(E.inv()), kind="matrix",
            rubric=[f"E performs: row {i + 1} += ({cval}) · row {j + 1}", f"Undo: row {i + 1} −= ({cval}) · row {j + 1}",
                    "E⁻¹ is the identity with the off-diagonal entry negated", "Check E E⁻¹ = I"],
            solution=f"$E^{{-1}} = {L(E.inv())}$.",
            misconceptions=[(fmt(E.T), "inverse-is-transpose"), (fmt(E), "inverse-kept-sign")],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                A = unimod3(rng)
                if max(abs(v) for v in A.inv()) <= 6:
                    break
            if rng.random() < 0.3:
                A = -A
            Ainv = A.inv()
            return Problem(
                prompt=f"Use Gauss–Jordan elimination on $[\\,A \\mid I\\,]$ to find $A^{{-1}}$ for $A = {L(A)}$.",
                answer=fmt(Ainv), kind="matrix",
                rubric=["Write the augmented 3×6 matrix [A | I]", "Eliminate below the pivots",
                        "Eliminate above the pivots (and scale pivots to 1)", "Right half is A⁻¹; check A A⁻¹ = I"],
                solution=f"$[A\\mid I] \\to [I \\mid A^{{-1}}]$ with $A^{{-1}} = {L(Ainv)}$ ($\\det A = {A.det()}$).",
                misconceptions=[(fmt(Ainv.T), "cofactors-not-transposed")],
            )
        dA = nz(rng, -5, 5)
        cval = rng.choice([2, 3, -2, -1, 4])
        kval = nz(rng, -3, 3)
        ops = [("swap rows 1 and 3", -1), (f"multiply row 2 by ${cval}$", cval), (f"add ${kval}$ times row 1 to row 3", 1)]
        if rng.random() < 0.5:
            ops.append(("swap rows 1 and 2", -1))
        rng.shuffle(ops)
        factor = 1
        for _, f in ops:
            factor *= f
        ans = factor * dA
        no_swap = dA * sp.prod([f for _, f in ops if f != -1])
        steps = "; ".join(f"({k + 1}) {t}" for k, (t, _) in enumerate(ops))
        return Problem(
            prompt=(f"$A$ is a $3\\times 3$ matrix with $\\det A = {dA}$. A matrix $B$ is obtained from $A$ by the row "
                    f"operations {steps}. Find $\\det B$."),
            answer=str(ans), kind="num",
            rubric=["A row swap multiplies det by −1", "Scaling a row by c multiplies det by c",
                    "Adding a multiple of one row to another leaves det unchanged", "Multiply all factors by det A"],
            solution=("Factors: " + ", ".join(f"${f}$" for _, f in ops) + f"; $\\det B = {factor}\\cdot({dA}) = {ans}$."),
            misconceptions=[(str(no_swap), "row-swap-sign"), (str(ans * (1 + kval)), "row-replacement-changes-det")],
        )
    if rng.random() < 0.5:
        a, b, c = rng.sample(range(-3, 5), 3)
        V = sp.Matrix([[1, a, a ** 2], [1, b, b ** 2], [1, c, c ** 2]])
        d = V.det()
        return Problem(
            prompt=(f"Evaluate $\\det {L(V)}$ by recognising its Vandermonde structure. On paper: derive the general "
                    "formula $\\det\\begin{bmatrix}1&a&a^2\\\\1&b&b^2\\\\1&c&c^2\\end{bmatrix} = (b-a)(c-a)(c-b)$ by row "
                    "reduction, then substitute."),
            answer=str(d), kind="num",
            rubric=["Subtract row 1 from rows 2 and 3", "Factor (b − a) and (c − a) out of rows 2 and 3",
                    "The remaining 2×2 determinant is (c − b)", f"Substitute a = {a}, b = {b}, c = {c}"],
            solution=f"$(b-a)(c-a)(c-b) = ({b - a})({c - a})({c - b}) = {d}$.",
            misconceptions=[(str(-d), "vandermonde-sign")],
        )
    while True:
        A = rand_mat(rng, 3, 3)
        if A.det() in (-4, -3, -2, -1, 1, 2, 3, 4, 5, 6):
            break
    i, j = rng.randint(0, 2), rng.randint(0, 2)
    val = A.inv()[i, j]
    cof_ji = A.cofactor(j, i)
    wrong = sp.Rational(A.cofactor(i, j), A.det())
    return Problem(
        prompt=(f"Find only the $({i + 1}, {j + 1})$ entry of $A^{{-1}}$ for $A = {L(A)}$, using the cofactor formula "
                "$A^{-1} = \\frac{1}{\\det A}\\,C^\\top$ (do not compute the whole inverse)."),
        answer=fmt(val), kind="num",
        rubric=["det A (cofactor expansion)", f"(A⁻¹)_{i + 1}{j + 1} = C_{j + 1}{i + 1} / det A — note the transpose",
                f"C_{j + 1}{i + 1} = (−1)^({i + j + 2}) × minor (delete row {j + 1}, column {i + 1})", "Divide"],
        solution=f"$\\det A = {A.det()}$, $C_{{{j + 1}{i + 1}}} = {cof_ji}$, so $(A^{{-1}})_{{{i + 1}{j + 1}}} = {L(val)}$.",
        misconceptions=[(fmt(wrong), "cofactor-forgot-transpose"), (fmt(-val), "cofactor-sign")],
    )


# --------------------------------------------------------------------------- subspaces
@gen("la.subspaces", name="subspace_more")
def subspace_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            m, n = rng.randint(3, 7), rng.randint(3, 8)
            r = rng.randint(1, min(m, n) - 1) if min(m, n) > 1 else 1
            ans = [r, n - r, r, m - r]
            return Problem(
                prompt=(f"$A$ is a ${m}\\times {n}$ matrix of rank ${r}$. Give the dimensions of its four fundamental "
                        "subspaces in the order column space $C(A)$, null space $N(A)$, row space $C(A^\\top)$, "
                        "left null space $N(A^\\top)$."),
                answer=[str(v) for v in ans], kind="list", answer_hint="four numbers in that order, e.g. 2, 3, 2, 1",
                rubric=["dim C(A) = dim C(Aᵀ) = r", f"N(A) ⊂ ℝ^{n}: dim = n − r", f"N(Aᵀ) ⊂ ℝ^{m}: dim = m − r"],
                solution=(f"$\\dim C(A) = \\dim C(A^\\top) = {r}$, $\\dim N(A) = {n} - {r} = {n - r}$, "
                          f"$\\dim N(A^\\top) = {m} - {r} = {m - r}$."),
                misconceptions=[(", ".join(map(str, [r, m - r, r, n - r])), "nullspace-dims-swapped")],
            )
        while True:
            u = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
            v = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
            if u.row_join(v).rank() == 2:
                break
        c1, c2 = nz(rng, -3, 3), nz(rng, -3, 3)
        b = c1 * u + c2 * v
        return Problem(
            prompt=(f"The vector $\\mathbf b = {tup(b)}$ lies in the span of $\\mathbf u = {tup(u)}$ and "
                    f"$\\mathbf v = {tup(v)}$. Express $\\mathbf b = c_1\\mathbf u + c_2\\mathbf v$ and type $(c_1, c_2)$."),
            answer=vec([c1, c2]), kind="vector", answer_hint="[c1, c2]",
            rubric=["Write three equations c₁uₖ + c₂vₖ = bₖ", "Solve two of them for c₁, c₂", "Check the third equation"],
            solution=f"${c1}{tup(u)} + ({c2}){tup(v)} = {tup(b)}$, so $(c_1, c_2) = ({c1}, {c2})$.",
            misconceptions=[(fmt([c2, c1]), "coefficients-swapped")],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                n = rng.choice([4, 5])
                r = rng.choice([2, 3])
                piv = sorted(rng.sample(range(n), r))
                R = sp.zeros(3, n)
                for row, pc in enumerate(piv):
                    R[row, pc] = 1
                    for c in range(pc + 1, n):
                        if c not in piv:
                            R[row, c] = rng.randint(-2, 2)
                P = unimod3(rng)
                A = P * R
                if max(abs(v) for v in A) <= 9 and A.rank() == r:
                    break
            return Problem(
                prompt=(f"Which columns of $A = {L(A)}$ are pivot columns (and hence form a basis of $C(A)$)? "
                        "Type their column numbers."),
                answer=[str(p + 1) for p in piv], kind="set", answer_hint="column numbers, e.g. 1, 3",
                rubric=["Row reduce A to echelon form", "Pivot = first nonzero entry in each nonzero row",
                        "The pivot columns of the ORIGINAL A form a basis of C(A)"],
                solution=f"$\\operatorname{{rref}}(A) = {L(A.rref()[0])}$: pivots in columns ${', '.join(str(p + 1) for p in piv)}$.",
                misconceptions=[(", ".join(str(k + 1) for k in range(r)), "assumed-first-columns-are-pivots")],
            )
        while True:
            w1 = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
            w2 = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
            al, be = nz(rng, -2, 2), nz(rng, -2, 2)
            w3 = al * w1 + be * w2
            jj = rng.randint(0, 2)
            kk = sp.Symbol("k")
            w3k = sp.Matrix([w3[c] if c != jj else kk for c in range(3)])
            dpoly = sp.expand(w1.row_join(w2).row_join(w3k).det())
            if dpoly.coeff(kk, 1) != 0 and max(abs(v) for v in w3) <= 8:
                break
        ksol = sp.solve(dpoly, kk)[0]
        return Problem(
            prompt=(f"For which value of $k$ are the vectors $\\mathbf w_1 = {tup(w1)}$, $\\mathbf w_2 = {tup(w2)}$ and "
                    f"$\\mathbf w_3 = {tup(w3k)}$ linearly dependent?"),
            answer=fmt(ksol), kind="num",
            rubric=["Three vectors in ℝ³ are dependent ⇔ the matrix with them as columns is singular",
                    f"det = {dpoly} (linear in k)", "Set det = 0 and solve", "Optional: find the dependency c₁w₁ + c₂w₂ = w₃"],
            solution=(f"$\\det[\\mathbf w_1\\ \\mathbf w_2\\ \\mathbf w_3] = {L(dpoly)} = 0 \\Rightarrow k = {L(ksol)}$; "
                      f"then $\\mathbf w_3 = {comb([(al, '\\mathbf w_1'), (be, '\\mathbf w_2')])}$."),
            misconceptions=[(fmt(-ksol), "sign-error")],
        )
    if rng.random() < 0.5:
        while True:
            cv, av, bv = (sp.Matrix([rng.randint(-2, 2) for _ in range(4)]) for _ in range(3))
            shared = rng.random() < 0.6
            if shared:
                U = [cv + av, av]
                W = [bv + cv, 2 * bv - cv]
            else:
                dv = sp.Matrix([rng.randint(-2, 2) for _ in range(4)])
                U = [cv + av, av]
                W = [bv, dv]
            MU, MW = U[0].row_join(U[1]), W[0].row_join(W[1])
            if MU.rank() == 2 and MW.rank() == 2 and max(abs(v) for w in U + W for v in w) <= 6:
                rk = MU.row_join(MW).rank()
                if (shared and rk == 3) or (not shared and rk == 4):
                    break
        ans = 4 - rk
        return Problem(
            prompt=(f"In $\\mathbb R^4$ let $U = \\operatorname{{span}}\\{{{tup(U[0])}, {tup(U[1])}\\}}$ and "
                    f"$W = \\operatorname{{span}}\\{{{tup(W[0])}, {tup(W[1])}\\}}$ (each is a plane). "
                    "What is $\\dim(U\\cap W)$? Justify with the dimension formula."),
            answer=str(ans), kind="num",
            rubric=["dim(U ∩ W) = dim U + dim W − dim(U + W)", "dim U = dim W = 2 (the spanning pairs are independent)",
                    "dim(U + W) = rank of the 4×4 matrix of all four vectors", f"rank = {rk}"],
            solution=f"$\\dim(U+W) = {rk}$, so $\\dim(U\\cap W) = 2 + 2 - {rk} = {ans}$.",
            misconceptions=[(str(rk), "gave-dim-of-sum")],
        )
    while True:
        r1 = sp.Matrix([[rng.randint(-2, 2) for _ in range(5)]])
        r2 = sp.Matrix([[rng.randint(-2, 2) for _ in range(5)]])
        if rng.random() < 0.5:
            r3 = nz(rng, -2, 2) * r1 + nz(rng, -2, 2) * r2
        else:
            r3 = sp.Matrix([[rng.randint(-2, 2) for _ in range(5)]])
        M = r1.col_join(r2).col_join(r3)
        if r1.rank() and r1.col_join(r2).rank() == 2 and max(abs(v) for v in r3) <= 6 and all(any(r) for r in (r1, r2, r3)):
            break
    rk = M.rank()
    eqs = ",\\quad ".join(f"{comb([(r[c], f'x_{c + 1}') for c in range(5)])} = 0" for r in (r1, r2, r3))
    return Problem(
        prompt=(f"The subspace $V\\subset\\mathbb R^5$ consists of all $\\mathbf x$ satisfying $${eqs}.$$ What is $\\dim V$? "
                "(Check carefully whether the three equations are independent.)"),
        answer=str(5 - rk), kind="num",
        rubric=["V is the null space of the 3×5 coefficient matrix", "Row reduce to find its rank r",
                "Is equation 3 a combination of equations 1 and 2?", "dim V = 5 − r (rank–nullity)"],
        solution=f"Coefficient matrix ${L(M)}$ has rank ${rk}$, so $\\dim V = 5 - {rk} = {5 - rk}$.",
        misconceptions=[("2", "assumed-equations-independent")] + ([(str(rk), "gave-rank-not-nullity")] if rk != 2 else []),
    )


# --------------------------------------------------------------------------- orthogonality
_ORTHO3 = sp.Matrix([[1, 2, 2], [2, 1, -2], [2, -2, 1]]) / 3


def _rand_orth(rng):
    if rng.random() < 0.5:
        p, q, r = rng.choice([(3, 4, 5), (5, 12, 13)])
        Q = sp.Matrix([[p, -q], [q, p]]) / r
        if rng.random() < 0.5:
            Q = Q * sp.diag(1, -1)
        return Q
    perm = rng.sample(range(3), 3)
    Q = _ORTHO3[:, perm]
    return Q * sp.diag(*[rng.choice([-1, 1]) for _ in range(3)])


@gen("la.orthogonality", name="orthogonality_more")
def orthogonality_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            a = sp.Matrix([nz(rng, -3, 3), nz(rng, -3, 3)])
            P = a * a.T / a.dot(a)
            return Problem(
                prompt=f"Find the projection matrix $P$ that projects $\\mathbb R^2$ onto the line through $\\mathbf a = {tup(a)}$.",
                answer=fmt(P), kind="matrix", answer_hint="exact fractions, e.g. [[1/5, 2/5], [2/5, 4/5]]",
                rubric=["P = a aᵀ / (aᵀa)", f"aᵀa = {a.dot(a)}", "Outer product a aᵀ", "Check P² = P and P = Pᵀ"],
                solution=f"$P = \\frac{{1}}{{{a.dot(a)}}}{L(a * a.T)} = {L(P)}$.",
                misconceptions=[(fmt(a * a.T), "projection-forgot-divide")],
            )
        Q = _rand_orth(rng)
        n = Q.shape[0]
        b = sp.Matrix([rng.randint(-4, 4) for _ in range(n)])
        while b == sp.zeros(n, 1):
            b = sp.Matrix([rng.randint(-4, 4) for _ in range(n)])
        xs = Q.T * b
        return Problem(
            prompt=(f"The square matrix $Q = {L(Q)}$ has orthonormal columns. Solve $Q\\mathbf x = \\mathbf b$ for "
                    f"$\\mathbf b = {tup(b)}$ without any elimination."),
            answer=vec(xs), kind="vector", answer_hint="exact fractions allowed",
            rubric=["QᵀQ = I, so Q⁻¹ = Qᵀ", "x = Qᵀ b", "Each xₖ = (column k of Q) · b"],
            solution=f"$\\mathbf x = Q^\\top\\mathbf b = {tup(xs)}$.",
            misconceptions=[(fmt(list(Q * b)), "used-q-not-transpose")],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                A = rand_mat(rng, 3, 2, -2, 2)
                b = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
                if A.rank() == 2 and A.col(0).dot(A.col(1)) != 0:
                    p = A * (A.T * A).inv() * A.T * b
                    if p != b and p != sp.zeros(3, 1) and (A.T * A).det() <= 40:
                        break
            naive = sum(((A.col(k).dot(b) / A.col(k).dot(A.col(k))) * A.col(k) for k in range(2)), sp.zeros(3, 1))
            return Problem(
                prompt=(f"Project $\\mathbf b = {tup(b)}$ onto the plane spanned by the columns of $A = {L(A)}$. "
                        "Give the projection $\\mathbf p$."),
                answer=vec(p), kind="vector", answer_hint="exact fractions, e.g. [1/3, 2, -5/3]",
                rubric=["Normal equations AᵀA x̂ = Aᵀb", f"AᵀA = {(A.T * A).tolist()}, Aᵀb = {list(A.T * b)}",
                        "Solve for x̂", "p = A x̂; check (b − p) ⟂ both columns"],
                solution=(f"$A^\\top A = {L(A.T * A)}$, $\\hat{{\\mathbf x}} = {tup((A.T * A).inv() * A.T * b)}$, "
                          f"$\\mathbf p = A\\hat{{\\mathbf x}} = {tup(p)}$."),
                misconceptions=[(fmt(list(naive)), "summed-projections-nonorthogonal-columns")],
            )
        while True:
            u = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
            v = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
            nvec = u.cross(v)
            if nvec[2] != 0 and nvec[1] != 0:
                break
        ans = nvec / nvec[2]
        return Problem(
            prompt=(f"Find a vector spanning the orthogonal complement of the plane spanned by $\\mathbf u = {tup(u)}$ and "
                    f"$\\mathbf v = {tup(v)}$ in $\\mathbb R^3$, scaled so that its third component is $1$."),
            answer=vec(ans), kind="vector", answer_hint="[a, b, 1]",
            rubric=["The complement is the null space of the 2×3 matrix with rows uᵀ, vᵀ",
                    "Solve the two equations (or use n = u × v)", "Scale so the third component is 1",
                    "Check n·u = n·v = 0"],
            solution=f"$\\mathbf u\\times\\mathbf v = {tup(nvec)}$, scaled: ${tup(ans)}$.",
            misconceptions=[(fmt([ans[0], -ans[1], 1]), "cross-product-middle-sign")],
        )
    if rng.random() < 0.5:
        while True:
            A = rand_mat(rng, 3, 2, -2, 2)
            b = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
            if A.rank() == 2 and (A.T * A).det() <= 40:
                p = A * (A.T * A).inv() * A.T * b
                if p != b and p != sp.zeros(3, 1):
                    break
        r = b - p
        dist = sp.sqrt(r.dot(r))
        return Problem(
            prompt=(f"Find the distance from $\\mathbf b = {tup(b)}$ to the plane $C(A)$ spanned by the columns of "
                    f"$A = {L(A)}$. On paper: explain why the error $\\mathbf e = \\mathbf b - \\mathbf p$ is perpendicular "
                    "to the plane, which makes $\\|\\mathbf e\\|$ the shortest distance."),
            answer=fmt(sp.nsimplify(dist)), kind="num", answer_hint="exact, e.g. 2*sqrt(6)/3",
            rubric=["Normal equations AᵀA x̂ = Aᵀb", "p = A x̂", "e = b − p (check Aᵀe = 0)", "Distance = ‖e‖"],
            solution=(f"$\\mathbf p = {tup(p)}$, $\\mathbf e = {tup(r)}$, $\\|\\mathbf e\\| = {L(sp.nsimplify(dist))}$. "
                      "(Alternatively: $\\mathbf e$ is along the normal $\\mathbf a_1\\times\\mathbf a_2$.)"),
            misconceptions=[(fmt(sp.nsimplify(sp.sqrt(p.dot(p)))), "gave-projection-length"),
                            (fmt(sp.nsimplify(r.dot(r))), "forgot-square-root")],
        )
    xq, n = quad_vec(rng)
    while xq[0] == n:
        xq, n = quad_vec(rng)
    xv = sp.Matrix(xq)
    v = xv - n * sp.Matrix([1, 0, 0])
    H = sp.eye(3) - 2 * v * v.T / v.dot(v)
    return Problem(
        prompt=(f"A Householder reflection $H = I - \\dfrac{{2\\,\\mathbf v\\mathbf v^\\top}}{{\\mathbf v^\\top\\mathbf v}}$ with "
                f"$\\mathbf v = \\mathbf x - \\|\\mathbf x\\|\\mathbf e_1$ maps $\\mathbf x$ onto the $x_1$-axis (this is the "
                f"building block of QR). For $\\mathbf x = {tup(xv)}$, compute $H$. On paper: show $H$ is symmetric and "
                "orthogonal, and verify $H\\mathbf x = \\|\\mathbf x\\|\\mathbf e_1$."),
        answer=fmt(H), kind="matrix", answer_hint="3x3 with exact fractions",
        rubric=[f"‖x‖ = {n}", f"v = x − {n}e₁ = {list(v)}", f"vᵀv = {v.dot(v)}", "H = I − 2vvᵀ/(vᵀv)",
                f"Check: Hx = ({n}, 0, 0)"],
        solution=f"$\\mathbf v = {tup(v)}$, $\\mathbf v^\\top\\mathbf v = {v.dot(v)}$, $H = {L(H)}$, and $H\\mathbf x = {tup(H * xv)}$.",
        misconceptions=[(fmt(sp.eye(3) - v * v.T / v.dot(v)), "householder-forgot-factor-2")],
    )


# --------------------------------------------------------------------------- eigen
@gen("la.eigen", name="eigen_more")
def eigen_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            lams = rng.sample(range(-4, 6), 3)
            upper = rng.random() < 0.5
            A = sp.diag(*lams)
            for r in range(3):
                for c in range(3):
                    if r != c and (c > r) == upper:
                        A[r, c] = rng.randint(-4, 4)
            mis = []
            row = list(A.row(0)) if upper else list(A.row(2))
            if len(set(row)) == 3:
                mis.append((", ".join(map(str, row)), "used-a-row-not-diagonal"))
            return Problem(
                prompt=f"State the eigenvalues of the triangular matrix $A = {L(A)}$ and explain on paper why no computation is needed.",
                answer=[str(v) for v in lams], kind="set",
                rubric=["A − λI is triangular too", "det of a triangular matrix = product of its diagonal",
                        "det(A − λI) = (a₁₁ − λ)(a₂₂ − λ)(a₃₃ − λ)", "Eigenvalues = diagonal entries"],
                solution=f"$\\det(A - \\lambda I) = \\prod_k (a_{{kk}} - \\lambda)$, so $\\lambda \\in \\{{{', '.join(map(str, lams))}\\}}$.",
                misconceptions=mis,
            )
        jcol = rng.randint(0, 1)
        A, P, d1, d2 = _eig2(rng)
        while (d1, d2)[jcol] == 0:
            A, P, d1, d2 = _eig2(rng)
        sc = rng.choice([1, 2, -1])
        v = sc * P.col(jcol)
        lam = (d1, d2)[jcol]
        return Problem(
            prompt=(f"Check that $\\mathbf v = {tup(v)}$ is an eigenvector of $A = {L(A)}$ and give its eigenvalue."),
            answer=str(lam), kind="num",
            rubric=["Compute A v", "Compare A v with v: is it a multiple?", "The multiple is the eigenvalue"],
            solution=f"$A\\mathbf v = {tup(A * v)} = {lam}\\,{tup(v)}$, so $\\lambda = {lam}$.",
            misconceptions=[(str((d1, d2)[1 - jcol]), "eigenvalue-of-other-eigenvector")],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                B, _, d1, d2 = _eig2(rng, cap=8)
                f = rng.randint(-4, 5)
                if f not in (d1, d2):
                    break
            # embed the 2x2 block in positions (1,3) and the scalar in position 2
            A = sp.Matrix([[B[0, 0], 0, B[0, 1]], [0, f, 0], [B[1, 0], 0, B[1, 1]]])
            return Problem(
                prompt=(f"Find all eigenvalues of $A = {L(A)}$. (Hint: expand $\\det(A - \\lambda I)$ along the row with "
                        "two zeros.)"),
                answer=[str(d1), str(d2), str(f)], kind="set",
                rubric=["Expand det(A − λI) along row 2", f"Factor ({f} − λ) times a 2×2 determinant",
                        "The 2×2 part: λ² − (trace)λ + det", "Solve and collect all three eigenvalues"],
                solution=(f"$\\det(A-\\lambda I) = ({f}-\\lambda)\\big[\\lambda^2 - ({B.trace()})\\lambda + ({B.det()})\\big]$, "
                          f"so $\\lambda = {f}, {d1}, {d2}$."),
                misconceptions=[(f"{A[0, 0]}, {A[1, 1]}, {A[2, 2]}", "diagonal-entries-not-eigenvalues")],
            )
        while True:
            al, be = rng.randint(-3, 3), nz(rng, 1, 3)
            P = unimod2(rng)
            A = P * sp.Matrix([[al, -be], [be, al]]) * P.inv()
            if max(abs(v) for v in A) <= 9:
                break
        ev = [al + be * I, al - be * I]
        return Problem(
            prompt=(f"The matrix $A = {L(A)}$ has no real eigenvalues. Find its two complex eigenvalues."),
            answer=[fmt(v) for v in ev], kind="set", answer_hint="complex numbers, e.g. 1 + 2*i, 1 - 2*i",
            rubric=[f"tr A = {A.trace()}, det A = {A.det()}", "λ² − (tr A)λ + det A = 0",
                    "Discriminant is negative: √(negative) = i·√(…)", "Eigenvalues come in a conjugate pair"],
            solution=(f"$\\lambda^2 - ({A.trace()})\\lambda + {A.det()} = 0$, discriminant ${A.trace() ** 2 - 4 * A.det()}$, "
                      f"$\\lambda = {L(ev[0])},\\ {L(ev[1])}$."),
            misconceptions=[(f"{al + be}, {al - be}", "dropped-imaginary-unit")],
        )
    if rng.random() < 0.5:
        while True:
            r1 = rng.randint(2, 4)
            r2 = rng.choice([-2, -1, 1, 2, 3])
            if abs(r2) < r1:
                break
        c1, c2 = nz(rng, 1, 3), nz(rng, -3, 3)
        a, b = r1 + r2, -r1 * r2
        x0, x1 = c1 + c2, c1 * r1 + c2 * r2
        sgn_b = "+" if b > 0 else "-"
        return Problem(
            prompt=(f"A sequence satisfies $x_{{k+1}} = {a if a != 1 else ''}x_k {sgn_b} {abs(b) if abs(b) != 1 else ''}x_{{k-1}}$ with "
                    f"$x_0 = {x0}$, $x_1 = {x1}$. Write the recurrence as $\\mathbf u_{{k+1}} = A\\mathbf u_k$ with "
                    "$\\mathbf u_k = (x_k, x_{k-1})$ and use the eigenvalues of $A$ to find "
                    "$\\displaystyle\\lim_{k\\to\\infty} x_{k+1}/x_k$."),
            answer=str(r1), kind="num",
            rubric=[f"A = [[{a}, {b}], [1, 0]]", f"Characteristic polynomial λ² − {a}λ + ({-b}) = 0",
                    f"Eigenvalues {r1} and {r2}", "x_k = c₁λ₁^k + c₂λ₂^k is dominated by the largest |λ| (c₁ ≠ 0 here)"],
            solution=(f"$A = {L(sp.Matrix([[a, b], [1, 0]]))}$, $\\lambda^2 - {a}\\lambda {'+' if -b >= 0 else '-'} {abs(b)} = 0 \\Rightarrow "
                      f"\\lambda = {r1}, {r2}$. Here $x_k = {c1}\\cdot{r1}^k + ({c2})({r2})^k$, so the ratio tends to ${r1}$."),
            misconceptions=[(str(r2), "picked-smaller-eigenvalue")],
        )
    A, _, d1, d2 = _eig2(rng, lo=-3, hi=3, cap=8)
    while 0 in (d1, d2):
        A, _, d1, d2 = _eig2(rng, lo=-3, hi=3, cap=8)
    k = rng.choice([3, 4])
    lam = sp.Symbol("lambda")
    rem = sp.Poly(lam ** k, lam).rem(sp.Poly(A.charpoly(lam).as_expr(), lam))
    al, be = rem.coeff_monomial(lam), rem.coeff_monomial(1)
    assert A ** k == al * A + be * sp.eye(2)
    return Problem(
        prompt=(f"Use the Cayley–Hamilton theorem for $A = {L(A)}$ to write $A^{{{k}}} = \\alpha A + \\beta I$. "
                "Type $\\alpha, \\beta$."),
        answer=[fmt(al), fmt(be)], kind="list", answer_hint="alpha, beta   e.g. 7, -12",
        rubric=[f"A² = (tr A)A − (det A)I = {A.trace()}A − ({A.det()})I", "Multiply by A and substitute A² again",
                f"Repeat until A^{k} is linear in A", "Check with the eigenvalues: λ^k = αλ + β for both λ"],
        solution=(f"$A^2 = {A.trace()}A - ({A.det()})I$; reducing gives $A^{{{k}}} = {L(al)}A + ({L(be)})I$. "
                  f"Check with $\\lambda = {d1}$: $({d1})^{{{k}}} = {d1 ** k} = {al}\\cdot({d1}) + ({be})$."),
        misconceptions=[(f"{fmt(be)}, {fmt(al)}", "alpha-beta-swapped")],
    )


# --------------------------------------------------------------------------- diagonalization
@gen("la.diagonalization", name="diagonalization_more")
def diagonalization_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            a, d = rng.sample(range(-3, 5), 2)
            m = nz(rng, -2, 2)
            b = m * (d - a)
            A = sp.Matrix([[a, b], [0, d]])
            P = sp.Matrix([[1, m], [0, 1]])
            return Problem(
                prompt=(f"Diagonalise the triangular matrix $A = {L(A)}$: find $P$ with $A = P\\,\\mathrm{{diag}}({a}, {d})\\,P^{{-1}}$, "
                        f"where column 1 of $P$ is the eigenvector for $\\lambda = {a}$ scaled to $(1, 0)$ and column 2 is the "
                        f"eigenvector for $\\lambda = {d}$ scaled to have second component $1$."),
                answer=fmt(P), kind="matrix", answer_hint="[[1, p12], [0, 1]]",
                rubric=["Eigenvalues are the diagonal entries", f"λ = {a}: (A − {a}I)v = 0 gives v = (1, 0)",
                        f"λ = {d}: solve (A − {d}I)v = 0 with v₂ = 1", "Put the eigenvectors in the columns of P"],
                solution=f"$A - ({d})I = {L(A - d * sp.eye(2))}$: first row gives ${a - d}v_1 + ({b})v_2 = 0$, so $\\mathbf v = ({m}, 1)$; $P = {L(P)}$.",
                misconceptions=[(fmt(sp.Matrix([[1, -m], [0, 1]])), "eigenvector-sign")],
            )
        P = unimod2(rng)
        d1, d2 = rng.sample([-2, -1, 1, 2], 2)
        k = rng.choice([3, 4])
        D = sp.diag(d1, d2)
        Ak = P * D ** k * P.inv()
        return Problem(
            prompt=(f"Using the factorisation $A = PDP^{{-1}}$ with $P = {L(P)}$ and $D = {L(D)}$, compute $A^{{{k}}}$ "
                    "without forming $A$ first."),
            answer=fmt(Ak), kind="matrix",
            rubric=["A^k = P D^k P⁻¹ (the inner P⁻¹P cancel)", f"D^{k} = diag({d1}^{k}, {d2}^{k})", "P⁻¹ (det P = ±1)",
                    "Multiply P D^k P⁻¹"],
            solution=f"$D^{{{k}}} = {L(D ** k)}$, $P^{{-1}} = {L(P.inv())}$, $A^{{{k}}} = {L(Ak)}$.",
            misconceptions=[(fmt(P.inv() * D ** k * P), "diagonalization-order"), (fmt(P * D * P.inv() * k), "power-as-multiple")],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                A, P, d1, d2 = _eig2(rng)
                if P[0, 0] != 0 and P[0, 1] != 0:
                    break
            cols = [(d1, P.col(0) / P[0, 0]), (d2, P.col(1) / P[0, 1])]
            cols.sort(key=lambda t: -t[0])
            Pn = cols[0][1].row_join(cols[1][1])
            return Problem(
                prompt=(f"Diagonalise $A = {L(A)}$ as $A = PDP^{{-1}}$ with the eigenvalues in $D$ in **decreasing** order. "
                        "Scale each eigenvector (column of $P$) so that its first component is $1$, and type $P$."),
                answer=fmt(Pn), kind="matrix", answer_hint="[[1, 1], [a, b]]",
                rubric=["Eigenvalues from tr A and det A", f"Order: λ₁ = {cols[0][0]} > λ₂ = {cols[1][0]}",
                        "Null vector of A − λI for each eigenvalue", "Scale so the first component is 1; check A P = P D"],
                solution=(f"$\\lambda_1 = {cols[0][0]}$: $\\mathbf v_1 = {tup(cols[0][1])}$; $\\lambda_2 = {cols[1][0]}$: "
                          f"$\\mathbf v_2 = {tup(cols[1][1])}$. $P = {L(Pn)}$."),
                misconceptions=[(fmt(cols[1][1].row_join(cols[0][1])), "eigenvector-columns-swapped")],
            )
        a = rng.randint(-3, 3)
        bb = rng.randint(-3, 4)
        while bb == a:
            bb = rng.randint(-3, 4)
        p = 0 if rng.random() < 0.35 else nz(rng, -3, 3)
        A = sp.Matrix([[a, p, rng.randint(-3, 3)], [0, a, rng.randint(-3, 3)], [0, 0, bb]])
        gm = 3 - (A - a * sp.eye(3)).rank()
        total = gm + 1
        return Problem(
            prompt=(f"How many linearly independent eigenvectors does $A = {L(A)}$ have? On paper: decide whether $A$ is "
                    "diagonalizable."),
            answer=str(total), kind="num",
            rubric=[f"Eigenvalues {a} (twice) and {bb}", f"Geometric multiplicity of {a} = 3 − rank(A − {a}I)",
                    f"rank(A − {a}I) = {(A - a * sp.eye(3)).rank()}", f"λ = {bb} contributes 1",
                    "Diagonalizable ⇔ 3 independent eigenvectors"],
            solution=(f"$A - ({a})I = {L(A - a * sp.eye(3))}$ has rank ${(A - a * sp.eye(3)).rank()}$, so $\\lambda = {a}$ has "
                      f"${gm}$ independent eigenvector(s); with $\\lambda = {bb}$ the total is ${total}$ "
                      f"({'diagonalizable' if total == 3 else 'not diagonalizable'})."),
            misconceptions=[("3", "assumed-diagonalizable"), ("2", "counted-distinct-eigenvalues")],
        )
    if rng.random() < 0.5:
        mu = rng.choice([sp.Rational(1, 2), sp.Rational(-1, 2), sp.Rational(1, 3), sp.Rational(-1, 3)])
        while True:
            P = unimod2(rng)
            A = P * sp.diag(1, mu) * P.inv()
            if max(abs(v) for v in A) <= 4:
                break
        Lim = P * sp.diag(1, 0) * P.inv()
        return Problem(
            prompt=(f"Let $A = {L(A)}$. Find $\\displaystyle\\lim_{{k\\to\\infty}} A^k$ by diagonalising $A$. "
                    "On paper: explain what happens to each eigen-direction."),
            answer=fmt(Lim), kind="matrix", answer_hint="exact entries, e.g. [[2, -2], [1, -1]]",
            rubric=[f"Eigenvalues 1 and {L(mu)} (trace/det)", "Eigenvectors → P", "A^k = P diag(1, μ^k) P⁻¹",
                    "μ^k → 0 since |μ| < 1, so the limit is P diag(1, 0) P⁻¹ (projection onto the λ = 1 eigenvector)"],
            solution=(f"$\\lambda = 1$, $\\mathbf v_1 = {tup(P.col(0))}$; $\\lambda = {L(mu)}$, $\\mathbf v_2 = {tup(P.col(1))}$. "
                      f"$\\lim A^k = P\\,\\mathrm{{diag}}(1, 0)P^{{-1}} = {L(Lim)}$."),
            misconceptions=[("[[0, 0], [0, 0]]", "assumed-powers-vanish"), (fmt(P * sp.diag(0, 1) * P.inv()), "kept-wrong-eigenvalue")],
        )
    tt = sp.Symbol("t")
    A, P, d1, d2 = _eig2(rng, lo=-3, hi=2, cap=8)
    cvec = sp.Matrix([nz(rng, -2, 2), nz(rng, -2, 2)])
    x0 = P * cvec
    sol = cvec[0] * sp.exp(d1 * tt) * P.col(0) + cvec[1] * sp.exp(d2 * tt) * P.col(1)
    wrong = cvec[0] * sp.exp(d2 * tt) * P.col(0) + cvec[1] * sp.exp(d1 * tt) * P.col(1)
    return Problem(
        prompt=(f"Solve the linear ODE system $\\dfrac{{d\\mathbf x}}{{dt}} = A\\mathbf x$ with $A = {L(A)}$ and "
                f"$\\mathbf x(0) = {tup(x0)}$ by diagonalising $A$. Type $\\mathbf x(t)$."),
        answer=vec(list(sol)), kind="vector", answer_hint="[x1(t), x2(t)], e.g. [2*exp(3*t) - exp(-t), exp(3*t)]",
        rubric=["Eigenvalues and eigenvectors of A", "General solution x(t) = c₁e^(λ₁t)v₁ + c₂e^(λ₂t)v₂",
                f"Match x(0): c₁v₁ + c₂v₂ = {list(x0)}", "Write both components"],
        solution=(f"$\\lambda_1 = {d1}$, $\\mathbf v_1 = {tup(P.col(0))}$; $\\lambda_2 = {d2}$, $\\mathbf v_2 = {tup(P.col(1))}$; "
                  f"$c = ({cvec[0]}, {cvec[1]})$. $\\mathbf x(t) = \\big({L(sol[0])},\\ {L(sol[1])}\\big)$."),
        misconceptions=[(fmt(list(wrong)), "eigenvalue-eigenvector-mismatch")],
    )


# --------------------------------------------------------------------------- SVD
@gen("la.svd", name="svd_more")
def svd_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            A, U, V, s1, s2 = _svd2(rng)
            mis = [(str(s1 ** 2), "forgot-square-root")]
            evs = list(A.eigenvals(multiple=True))
            if all(e.is_real for e in evs):
                mis.append((fmt(max(abs(e) for e in evs)), "eigenvalue-not-singular-value"))
            return Problem(
                prompt=(f"What is the spectral norm $\\|A\\|_2 = \\max_{{\\|\\mathbf x\\|=1}}\\|A\\mathbf x\\|$ of $A = {L(A)}$? "
                        "(It equals the largest singular value.)"),
                answer=str(s1), kind="num",
                rubric=["‖A‖₂ = σ_max", "Form AᵀA", "Largest eigenvalue of AᵀA", "Take its square root"],
                solution=f"$A^\\top A = {L(A.T * A)}$ has largest eigenvalue ${s1 ** 2}$, so $\\|A\\|_2 = {s1}$.",
                misconceptions=mis,
            )
        A = rand_mat(rng, 2, 3, -3, 3)
        ss = sum(v * v for v in A)
        fro = sp.sqrt(ss)
        return Problem(
            prompt=(f"Compute the Frobenius norm $\\|A\\|_F$ of $A = {L(A)}$. On paper: note that $\\|A\\|_F^2 = "
                    "\\sum_k \\sigma_k^2 = \\operatorname{tr}(A^\\top A)$."),
            answer=fmt(fro), kind="num", answer_hint="exact, e.g. sqrt(30)",
            rubric=["Square every entry", "Sum the squares", "Take the square root"],
            solution=("$\\|A\\|_F^2 = " + " + ".join(f"{v * v}" for v in A) + f" = {ss}$, so $\\|A\\|_F = {L(fro)}$."),
            misconceptions=[(str(ss), "forgot-square-root"), (str(sum(abs(v) for v in A)), "used-entrywise-l1")],
        )
    if level == 2:
        A, U, V, s1, s2 = _svd2(rng, allow_zero=False)
        if rng.random() < 0.5:
            kap = sp.Rational(s1, s2)
            return Problem(
                prompt=(f"Find the condition number $\\kappa(A) = \\sigma_{{\\max}}/\\sigma_{{\\min}}$ of $A = {L(A)}$. "
                        "On paper: what does it say about solving $A\\mathbf x = \\mathbf b$ with noisy $\\mathbf b$?"),
                answer=fmt(kap), kind="num",
                rubric=["AᵀA and its eigenvalues", f"σ₁ = {s1}, σ₂ = {s2}", "κ = σ₁/σ₂",
                        "Relative error in x can be up to κ × relative error in b"],
                solution=f"$A^\\top A = {L(A.T * A)}$, eigenvalues ${s1 ** 2}, {s2 ** 2}$, so $\\kappa = {s1}/{s2} = {L(kap)}$.",
                misconceptions=[(fmt(kap ** 2), "used-eigenvalues-of-AtA")],
            )
        v1 = V.col(0) * (1 if V[0, 0] > 0 else -1)
        v2 = V.col(1) * (1 if V[0, 1] > 0 else -1)
        return Problem(
            prompt=(f"Find the first right singular vector $\\mathbf v_1$ of $A = {L(A)}$ (the unit eigenvector of $A^\\top A$ "
                    "for its largest eigenvalue), with sign chosen so that its first component is positive."),
            answer=vec(list(v1)), kind="vector", answer_hint="unit vector, e.g. [3, 4]/5 or [1, 2]/sqrt(5)",
            rubric=["Form AᵀA", f"Largest eigenvalue σ₁² = {s1 ** 2}", "Solve (AᵀA − σ₁²I)v = 0", "Normalise; make v₁ > 0"],
            solution=f"$A^\\top A = {L(A.T * A)}$; for $\\lambda = {s1 ** 2}$: $\\mathbf v_1 = {tup(v1)}$.",
            misconceptions=[(fmt(list(v2)), "used-smallest-singular-value")],
        )
    A, U, V, s1, s2 = _svd2(rng, allow_zero=False)
    if rng.random() < 0.5:
        frac = sp.Rational(s2 ** 2, s1 ** 2 + s2 ** 2)
        return Problem(
            prompt=(f"Compress $A = {L(A)}$ by its best rank-1 approximation $A_1$ (Eckart–Young). What fraction of "
                    "$\\|A\\|_F^2$ is lost, i.e. $\\|A - A_1\\|_F^2 / \\|A\\|_F^2$? Derive it in terms of singular values."),
            answer=fmt(frac), kind="num",
            rubric=["‖A‖_F² = σ₁² + σ₂²", "A − A₁ = σ₂u₂v₂ᵀ, so ‖A − A₁‖_F² = σ₂²", f"σ₁ = {s1}, σ₂ = {s2} (from AᵀA)",
                    "Ratio σ₂²/(σ₁² + σ₂²)"],
            solution=(f"$\\|A\\|_F^2 = {sum(v * v for v in A)} = {s1 ** 2} + {s2 ** 2}$; lost fraction "
                      f"$= {s2 ** 2}/{s1 ** 2 + s2 ** 2} = {L(frac)}$."),
            misconceptions=[(fmt(sp.Rational(s2, s1 + s2)), "used-sigma-not-sigma-squared"),
                            (fmt(1 - frac), "gave-retained-fraction")],
        )
    Qm = sp.simplify(U * V.T)
    return Problem(
        prompt=(f"Orthogonal Procrustes: the orthogonal matrix closest to $A = {L(A)}$ in Frobenius norm is "
                "$Q = UV^\\top$, where $A = U\\Sigma V^\\top$ is an SVD. Compute the SVD and type $Q$. "
                "On paper: explain why $Q$ does not depend on the sign choices of the singular vectors."),
        answer=fmt(Qm), kind="matrix",
        rubric=["Eigen-decompose AᵀA → V and σ₁, σ₂", "uₖ = A vₖ / σₖ", "Q = U Vᵀ = Σₖ uₖvₖᵀ",
                "Flipping vₖ also flips uₖ, so each uₖvₖᵀ is unchanged"],
        solution=(f"$\\sigma = {s1}, {s2}$; $V = {L(V)}$, $U = {L(U)}$; $Q = UV^\\top = {L(Qm)}$ "
                  f"(also $Q = A(A^\\top A)^{{-1/2}}$)."),
        misconceptions=[(fmt(A / s1), "normalised-by-largest-singular-value")],
    )


# --------------------------------------------------------------------------- PCA
def _four_points(rng):
    mu = sp.Matrix([rng.randint(-3, 3), rng.randint(-3, 3)])
    while True:
        ds = [sp.Matrix([rng.randint(-3, 3), rng.randint(-3, 3)]) for _ in range(3)]
        ds.append(-(ds[0] + ds[1] + ds[2]))
        if max(abs(v) for d in ds for v in d) <= 4 and all(d != sp.zeros(2, 1) for d in ds):
            return mu, ds, [mu + d for d in ds]


_UNITS2 = [(1, 0), (0, 1), (sp.Rational(3, 5), sp.Rational(4, 5)), (sp.Rational(4, 5), sp.Rational(-3, 5)),
           (sp.Rational(-3, 5), sp.Rational(4, 5)), (sp.Rational(5, 13), sp.Rational(12, 13))]


@gen("la.pca", name="pca_more")
def pca_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            mu, ds, pts = _four_points(rng)
            tv = sp.Rational(sum(d.dot(d) for d in ds), 4)
            raw = sp.Rational(sum(p.dot(p) for p in pts), 4)
            mis = [(fmt(sp.Rational(sum(d.dot(d) for d in ds), 3)), "covariance-used-n-minus-1")]
            if mu != sp.zeros(2, 1):
                mis.append((fmt(raw), "covariance-forgot-to-center"))
            return Problem(
                prompt=("Compute the total variance $\\operatorname{tr}(S) = \\frac1n\\sum_k\\|\\mathbf x_k - \\bar{\\mathbf x}\\|^2$ "
                        "(divide by $n$) of the four points " + ", ".join(f"${tup(p)}$" for p in pts) + "."),
                answer=fmt(tv), kind="num",
                rubric=[f"Mean x̄ = {tup(mu)}", "Centre each point", "Sum the squared lengths", "Divide by n = 4"],
                solution=(f"$\\bar{{\\mathbf x}} = {tup(mu)}$; centred points " + ", ".join(f"${tup(d)}$" for d in ds)
                          + f"; squared lengths sum to ${sum(d.dot(d) for d in ds)}$, so $\\operatorname{{tr}}S = {L(tv)}$."),
                misconceptions=mis,
            )
        mu = sp.Matrix([rng.randint(-4, 4), rng.randint(-4, 4)])
        q = sp.Matrix(rng.choice(_UNITS2[2:]))
        xv = mu + sp.Matrix([rng.randint(-4, 4), rng.randint(-4, 4)])
        while xv == mu:
            xv = mu + sp.Matrix([rng.randint(-4, 4), rng.randint(-4, 4)])
        score = q.dot(xv - mu)
        return Problem(
            prompt=(f"A PCA was fitted with data mean $\\bar{{\\mathbf x}} = {tup(mu)}$ and first principal direction "
                    f"$\\mathbf q_1 = {tup(q)}$. Compute the PC1 score (coordinate) $z_1 = \\mathbf q_1^\\top(\\mathbf x - \\bar{{\\mathbf x}})$ "
                    f"of the point $\\mathbf x = {tup(xv)}$."),
            answer=fmt(score), kind="num",
            rubric=["Centre: x − x̄", "Dot with the unit direction q₁"],
            solution=f"$\\mathbf x - \\bar{{\\mathbf x}} = {tup(xv - mu)}$, $z_1 = {L(score)}$.",
            misconceptions=[(fmt(q.dot(xv)), "pca-forgot-to-center")],
        )
    if level == 2:
        if rng.random() < 0.5:
            pts, mu, v, wv, s_, t_ = _pca_data(rng)
            big, small = (v, wv) if s_ > t_ else (wv, v)
            q = big / big.norm()
            q = q * (1 if q[0] > 0 else -1)
            q2 = small / small.norm()
            q2 = q2 * (1 if q2[0] > 0 else -1)
            S = sum(((p - mu) * (p - mu).T for p in pts), sp.zeros(2, 2)) / 4
            return Problem(
                prompt=("Find the first principal direction (unit vector, first component positive) of the four points "
                        + ", ".join(f"${tup(p)}$" for p in pts) + "."),
                answer=vec(list(q)), kind="vector", answer_hint="unit vector, e.g. [1, 2]/sqrt(5)",
                rubric=[f"Mean = {tup(mu)}; centre the data", f"Covariance S = {S.tolist()}",
                        "Largest eigenvalue of S and its eigenvector", "Normalise; fix the sign"],
                solution=f"$S = {L(S)}$; top eigenvector $\\mathbf q_1 = {tup(q)}$.",
                misconceptions=[(fmt(list(q2)), "picked-smallest-eigenvector")],
            )
        mu, ds, pts = _four_points(rng)
        S = sum((d * d.T for d in ds), sp.zeros(2, 2)) / 4
        uvec = sp.Matrix(rng.choice(_UNITS2))
        var = (uvec.T * S * uvec)[0]
        return Problem(
            prompt=("How much variance do the four points " + ", ".join(f"${tup(p)}$" for p in pts)
                    + f" have along the unit direction $\\mathbf u = {tup(uvec)}$? (Use $1/n$.) "
                    "Compute it as $\\mathbf u^\\top S\\mathbf u$ and also directly from the projected coordinates."),
            answer=fmt(var), kind="num",
            rubric=[f"Mean = {tup(mu)}", f"Covariance S = {S.tolist()}", "uᵀSu",
                    "Check: average of (uᵀ(xₖ − x̄))²"],
            solution=(f"$S = {L(S)}$, $\\mathbf u^\\top S\\mathbf u = {L(var)}$; projected coordinates "
                      + ", ".join(f"${L(uvec.dot(d))}$" for d in ds) + "."),
            misconceptions=[(fmt(sp.sqrt(var)), "gave-standard-deviation")],
        )
    if rng.random() < 0.5:
        while True:
            B, v, l1, l2 = sym2(rng, lo=1, hi=4, cap=8)
            f = rng.randint(1, 9)
            lams = [l1, l2, f]
            if min(lams) > 0 and len(set(lams)) == 3:
                break
        srt = sorted(lams, reverse=True)
        tot = sum(srt)
        cum = [sum(srt[:k + 1]) for k in range(3)]
        ths = [sp.Rational(p, 100) for p in (60, 70, 75, 80, 90, 95)]
        th = rng.choice([t_ for t_ in ths if all(sp.Rational(c, tot) != t_ for c in cum)])
        kmin = next(k + 1 for k in range(3) if sp.Rational(cum[k], tot) >= th)
        S = sp.Matrix([[B[0, 0], 0, B[0, 1]], [0, f, 0], [B[1, 0], 0, B[1, 1]]])
        pct = int(th * 100)
        return Problem(
            prompt=(f"A centred 3-feature dataset has covariance matrix $S = {L(S)}$. What is the smallest number $k$ of "
                    f"principal components that retains at least ${pct}\\%$ of the total variance?"),
            answer=str(kmin), kind="num",
            rubric=["Eigenvalues of S: feature 2 is uncorrelated → λ = S₂₂", "The remaining 2×2 block: trace/det",
                    "Sort eigenvalues decreasingly; total = tr S", f"Smallest k with cumulative fraction ≥ {pct}%"],
            solution=(f"Eigenvalues ${', '.join(map(str, srt))}$, total ${tot}$; cumulative fractions "
                      + ", ".join(f"${L(sp.Rational(c, tot))}$" for c in cum) + f", so $k = {kmin}$."),
            misconceptions=[(str(next(k + 1 for k in range(3) if sp.Rational(sum(sorted(lams)[:k + 1]), tot) >= th)),
                             "counted-from-smallest-eigenvalue")],
        )
    pts, mu, v, wv, s_, t_ = _pca_data(rng)
    big = v if s_ > t_ else wv
    q = big / big.norm()
    err = sum(((p - mu) - ((p - mu).dot(q)) * q).dot((p - mu) - ((p - mu).dot(q)) * q) for p in pts)
    err = sp.nsimplify(sp.simplify(err))
    return Problem(
        prompt=("Reconstruction error of PCA: project the four points " + ", ".join(f"${tup(p)}$" for p in pts)
                + " onto their first principal component (around the mean) and compute the **total squared reconstruction error** "
                "$\\sum_k \\|\\mathbf x_k - \\hat{\\mathbf x}_k\\|^2$. On paper: relate it to the discarded eigenvalue of $S$ (with $1/n$)."),
        answer=fmt(err), kind="num",
        rubric=[f"Mean = {tup(mu)}", "Covariance S and its eigenvalues λ₁ ≥ λ₂", "Each error is the component along q₂",
                "Total error = n · λ₂"],
        solution=(f"$S = {L(sum(((p - mu) * (p - mu).T for p in pts), sp.zeros(2, 2)) / 4)}$, "
                  f"$\\lambda_2 = {L(err / 4)}$, total error $= 4\\lambda_2 = {L(err)}$."),
        misconceptions=[(fmt(err / 4), "gave-mean-not-sum")],
    )


# --------------------------------------------------------------------------- matrix calculus
@gen("la.matrix_calculus", name="matrix_calc_more")
def matrix_calc_more(rng, level):
    xx, yy = sp.symbols("x y")
    if level == 1:
        if rng.random() < 0.5:
            cv = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
            x0 = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
            while x0 == cv:
                x0 = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
            g = 2 * (x0 - cv)
            return Problem(
                prompt=(f"Let $f(\\mathbf x) = \\|\\mathbf x - \\mathbf c\\|^2$ with $\\mathbf c = {tup(cv)}$. "
                        f"Compute the gradient $\\nabla f$ at $\\mathbf x = {tup(x0)}$."),
                answer=vec(list(g)), kind="vector",
                rubric=["f = (x − c)ᵀ(x − c)", "∇f = 2(x − c)", "Evaluate"],
                solution=f"$\\nabla f = 2(\\mathbf x - \\mathbf c) = 2{tup(x0 - cv)} = {tup(g)}$.",
                misconceptions=[(fmt(list(x0 - cv)), "forgot-factor-2")],
            )
        a, b, c = nz(rng, -3, 3), nz(rng, -3, 3), nz(rng, -3, 3)
        x0, y0 = nz(rng, -2, 2), nz(rng, -2, 2)
        F = sp.Matrix([a * xx ** 2 * yy, b * xx + c * yy ** 2, xx * yy])
        J = F.jacobian([xx, yy]).subs({xx: x0, yy: y0})
        return Problem(
            prompt=(f"Find the Jacobian matrix of $\\mathbf f(x, y) = \\big({L(F[0])},\\ {L(F[1])},\\ {L(F[2])}\\big)$ "
                    f"at the point $({x0}, {y0})$."),
            answer=fmt(J), kind="matrix", answer_hint="3x2: row k = gradient of component k, e.g. [[1, 2], [3, 4], [5, 6]]",
            rubric=["J has one row per output and one column per input (3×2)", "Row k = (∂fₖ/∂x, ∂fₖ/∂y)",
                    f"Evaluate at ({x0}, {y0})"],
            solution=f"$J = {L(F.jacobian([xx, yy]))}$, at the point: ${L(J)}$.",
            misconceptions=[(fmt(J.T), "jacobian-transposed")],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                a = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
                x0 = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
                s_ = a.dot(x0)
                if s_ not in (0, 1) and a != sp.zeros(3, 1):
                    break
            g = 2 * s_ * a
            return Problem(
                prompt=(f"Let $f(\\mathbf x) = (\\mathbf a^\\top\\mathbf x)^2$ with $\\mathbf a = {tup(a)}$. Derive $\\nabla f$ "
                        f"with the chain rule and evaluate it at $\\mathbf x = {tup(x0)}$."),
                answer=vec(list(g)), kind="vector",
                rubric=["Inner s = aᵀx has ∇s = a", "Outer g(s) = s², g'(s) = 2s", "∇f = 2(aᵀx) a",
                        f"aᵀx = {s_} at the point"],
                solution=f"$\\nabla f = 2(\\mathbf a^\\top\\mathbf x)\\mathbf a = 2\\cdot({s_}){tup(a)} = {tup(g)}$.",
                misconceptions=[(fmt(list(2 * a)), "chain-forgot-inner-value"), (fmt(list(s_ * a)), "forgot-factor-2")],
            )
        u = sp.Matrix([nz(rng, -3, 3) for _ in range(2)])
        v = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
        while v == sp.zeros(3, 1):
            v = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
        G = u * v.T
        return Problem(
            prompt=(f"For a $2\\times 3$ matrix $W$, let $f(W) = \\mathbf u^\\top W\\mathbf v$ with $\\mathbf u = {tup(u)}$ and "
                    f"$\\mathbf v = {tup(v)}$. Find the gradient $\\nabla_W f$ (the matrix of $\\partial f/\\partial W_{{ij}}$)."),
            answer=fmt(G), kind="matrix", answer_hint="2x3 matrix, same shape as W",
            rubric=["f = Σᵢⱼ uᵢ Wᵢⱼ vⱼ", "∂f/∂Wᵢⱼ = uᵢ vⱼ", "So ∇_W f = u vᵀ (same shape as W)"],
            solution=f"$\\nabla_W f = \\mathbf u\\mathbf v^\\top = {L(G)}$.",
            misconceptions=[(fmt(v * u.T), "gradient-transposed")],
        )
    if rng.random() < 0.5:
        while True:
            a, c = rng.randint(1, 4), rng.randint(1, 4)
            b = rng.randint(-2, 2)
            if a * c - b * b > 0:
                break
        Q = sp.Matrix([[a, b], [b, c]])
        bv = sp.Matrix([rng.randint(-4, 4) for _ in range(2)])
        x0 = sp.Matrix([rng.randint(-2, 2) for _ in range(2)])
        g0 = Q * x0 - bv
        while g0 == sp.zeros(2, 1):
            x0 = sp.Matrix([rng.randint(-2, 2) for _ in range(2)])
            g0 = Q * x0 - bv
        x1 = x0 - Q.inv() * g0
        return Problem(
            prompt=(f"Take one Newton step $\\mathbf x_1 = \\mathbf x_0 - [\\nabla^2 f]^{{-1}}\\nabla f(\\mathbf x_0)$ for "
                    f"$f(\\mathbf x) = \\tfrac12\\mathbf x^\\top Q\\mathbf x - \\mathbf b^\\top\\mathbf x$ with $Q = {L(Q)}$, "
                    f"$\\mathbf b = {tup(bv)}$, starting from $\\mathbf x_0 = {tup(x0)}$. Type $\\mathbf x_1$. "
                    "On paper: explain why one step lands exactly on the minimiser."),
            answer=vec(list(x1)), kind="vector", answer_hint="exact fractions allowed",
            rubric=["∇f = Qx − b, ∇²f = Q", f"∇f(x₀) = {list(g0)}", "Solve Q d = ∇f(x₀) and set x₁ = x₀ − d",
                    "x₁ = Q⁻¹b: f is quadratic, so its 2nd-order model is exact"],
            solution=(f"$\\nabla f(\\mathbf x_0) = {tup(g0)}$, $Q^{{-1}} = {L(Q.inv())}$, $\\mathbf x_1 = {tup(x1)}$ "
                      "$= Q^{-1}\\mathbf b$, where $\\nabla f = 0$."),
            misconceptions=[(fmt(list(x0 - g0)), "used-gradient-step")],
        )
    while True:
        W = rand_mat(rng, 2, 3, -2, 2)
        xv = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
        yv = sp.Matrix([rng.randint(-3, 3) for _ in range(2)])
        r = W * xv - yv
        if r != sp.zeros(2, 1) and xv != sp.zeros(3, 1):
            break
    G = r * xv.T
    return Problem(
        prompt=(f"A linear layer $\\hat{{\\mathbf y}} = W\\mathbf x$ is trained with loss $\\mathcal L = \\tfrac12\\|W\\mathbf x - \\mathbf y\\|^2$. "
                f"Derive $\\nabla_W\\mathcal L$ and evaluate it for $W = {L(W)}$, $\\mathbf x = {tup(xv)}$, "
                f"$\\mathbf y = {tup(yv)}$."),
        answer=fmt(G), kind="matrix", answer_hint="2x3 matrix, same shape as W",
        rubric=["Residual r = Wx − y", "∂L/∂Wᵢⱼ = rᵢ xⱼ (chain rule through ŷᵢ = Σⱼ Wᵢⱼxⱼ)", "∇_W L = r xᵀ",
                f"r = {list(r)}"],
        solution=f"$\\mathbf r = W\\mathbf x - \\mathbf y = {tup(r)}$, $\\nabla_W\\mathcal L = \\mathbf r\\mathbf x^\\top = {L(G)}$.",
        misconceptions=[(fmt(-G), "residual-sign"), (fmt(2 * G), "forgot-half-cancels")],
    )


# --------------------------------------------------------------------------- complex inner products
def ket(a0, a1) -> str:
    """LaTeX for a0|0> + a1|1>, dropping zero terms and bracketing negative or complex coefficients."""
    parts = []
    for c, lbl in ((a0, "|0\\rangle"), (a1, "|1\\rangle")):
        if c == 0:
            continue
        if c == 1:
            parts.append(lbl)
        elif c.is_real and c > 0:
            parts.append(L(c) + lbl)
        else:
            parts.append(f"\\left({L(c)}\\right){lbl}")
    return " + ".join(parts)


@gen("la.complex_inner", name="complex_more")
def complex_more(rng, level):
    def cz():
        return rng.randint(-3, 3) + rng.randint(-3, 3) * I

    if level == 1:
        if rng.random() < 0.5:
            while True:
                uu = [cz() for _ in range(rng.choice([2, 3]))]
                if any(sp.im(a) != 0 for a in uu):
                    break
            sq = sum(sp.expand(a * sp.conjugate(a)) for a in uu)
            nrm = sp.sqrt(sq)
            noconj = sp.expand(sum(a * a for a in uu))
            return Problem(
                prompt=f"Compute the norm $\\|\\mathbf u\\| = \\sqrt{{\\mathbf u^H\\mathbf u}}$ of the complex vector $\\mathbf u = {tup(uu)}$.",
                answer=fmt(nrm), kind="num", answer_hint="exact, e.g. sqrt(14)",
                rubric=["|a + bi|² = a² + b² (multiply by the conjugate)", "Sum the squared moduli", "Square root"],
                solution=("$\\|\\mathbf u\\|^2 = " + " + ".join(str(sp.expand(a * sp.conjugate(a))) for a in uu)
                          + f" = {sq}$, so $\\|\\mathbf u\\| = {L(nrm)}$."),
                misconceptions=[(fmt(sp.sqrt(noconj)), "forgot-conjugate"), (str(sq), "forgot-square-root")],
            )
        while True:
            A = sp.Matrix(2, 2, lambda *_: cz())
            if sum(1 for v in A if sp.im(v) != 0) >= 2 and A.H != A and A.H != A.T:
                break
        return Problem(
            prompt=f"Write down the conjugate transpose $A^H = \\overline{{A}}^\\top$ of $A = {L(A)}$.",
            answer=fmt(A.H), kind="matrix", answer_hint="e.g. [[1 - i, 2], [3*i, 4]]",
            rubric=["Transpose: entry (i, j) moves to (j, i)", "Conjugate every entry: a + bi → a − bi"],
            solution=f"$A^H = {L(A.H)}$.",
            misconceptions=[(fmt(A.T), "forgot-conjugate"), (fmt(A.conjugate()), "forgot-transpose")],
        )
    if level == 2:
        while True:
            al, be = cz(), cz()
            if al != 0 and be != 0 and (sp.im(al) != 0 or sp.im(be) != 0):
                break
        m0, m1 = sp.expand(al * sp.conjugate(al)), sp.expand(be * sp.conjugate(be))
        if rng.random() < 0.5:
            prob = sp.Rational(m1, m0 + m1)
            return Problem(
                prompt=(f"A qubit is prepared in the (unnormalised) state $|\\psi\\rangle = ({L(al)})|0\\rangle + ({L(be)})|1\\rangle$. "
                        "What is the probability of measuring $|1\\rangle$?"),
                answer=fmt(prob), kind="num",
                rubric=["Born rule: P(1) = |β|² / (|α|² + |β|²) for an unnormalised state", f"|α|² = {m0}",
                        f"|β|² = {m1}", "Divide"],
                solution=f"$|\\alpha|^2 = {m0}$, $|\\beta|^2 = {m1}$, $P(1) = {m1}/{m0 + m1} = {L(prob)}$.",
                misconceptions=[(str(m1), "forgot-normalise"), (fmt(sp.Rational(m0, m0 + m1)), "measured-wrong-basis-state")],
            )
        nrm = sp.sqrt(m0 + m1)
        ans = [sp.simplify(al / nrm), sp.simplify(be / nrm)]
        return Problem(
            prompt=(f"Normalise the complex vector $\\mathbf v = ({L(al)},\\ {L(be)})$: divide it by its norm "
                    "(a positive real number; do not change the global phase)."),
            answer=vec(ans), kind="vector", answer_hint="e.g. [1, i]/sqrt(2)  or  [(1+i)/sqrt(3), 1/sqrt(3)]",
            rubric=[f"|v₁|² = {m0}, |v₂|² = {m1}", f"‖v‖ = √{m0 + m1}", "Divide each component by ‖v‖",
                    "Check: |v₁|² + |v₂|² = 1"],
            solution=(f"$\\|\\mathbf v\\|^2 = {m0} + {m1} = {m0 + m1}$, $\\|\\mathbf v\\| = {L(nrm)}$, so "
                      f"$\\hat{{\\mathbf v}} = \\frac{{1}}{{{L(nrm)}}}({L(al)},\\ {L(be)}) = {tup(ans)}$."),
            misconceptions=[(fmt([sp.Rational(1, m0 + m1) * al, sp.Rational(1, m0 + m1) * be]), "divided-by-norm-squared")],
        )
    if rng.random() < 0.5:
        psis = [(sp.Rational(3, 5), sp.Rational(4, 5) * I), (sp.Rational(4, 5), -sp.Rational(3, 5) * I),
                (1 / sp.sqrt(2), I / sp.sqrt(2)), ((1 + I) / 2, 1 / sp.sqrt(2)), (1 / sp.sqrt(2), (1 - I) / 2),
                (sp.Rational(3, 5), sp.Rational(-4, 5)), (1 / sp.sqrt(2), 1 / sp.sqrt(2)), (sp.Integer(1), sp.Integer(0))]
        while True:
            (a0, a1), (b0, b1) = rng.sample(psis, 2)
            ip = sp.conjugate(a0) * b0 + sp.conjugate(a1) * b1
            fid = sp.nsimplify(sp.simplify(sp.expand(ip * sp.conjugate(ip))))
            wrong_ip = a0 * b0 + a1 * b1
            wrong = sp.nsimplify(sp.simplify(sp.expand(wrong_ip * sp.conjugate(wrong_ip))))
            if fid != 0 and fid != wrong:
                break
        return Problem(
            prompt=(f"Two qubit states are $|\\phi\\rangle = {ket(a0, a1)}$ and "
                    f"$|\\psi\\rangle = {ket(b0, b1)}$. Compute the fidelity "
                    "$|\\langle\\phi|\\psi\\rangle|^2$ (the probability that $|\\psi\\rangle$ passes a test for $|\\phi\\rangle$)."),
            answer=fmt(fid), kind="num", answer_hint="exact, e.g. 9/25 or 1/2",
            rubric=["Bra ⟨φ| = conjugate of the coefficients of |φ⟩", "⟨φ|ψ⟩ = φ̄₀ψ₀ + φ̄₁ψ₁",
                    "|z|² = z z̄", "Result lies in [0, 1]"],
            solution=f"$\\langle\\phi|\\psi\\rangle = {L(sp.simplify(ip))}$, so $|\\langle\\phi|\\psi\\rangle|^2 = {L(fid)}$.",
            misconceptions=[(fmt(wrong), "forgot-conjugate")],
        )
    (pa, pb, pc), n = quad_vec(rng)
    d = rng.randint(-3, 3)
    H = sp.Matrix([[d + pc, pa - pb * I], [pa + pb * I, d - pc]])
    shown = comb([(d, "I"), (pa, "\\sigma_x"), (pb, "\\sigma_y"), (pc, "\\sigma_z")])
    return Problem(
        prompt=(f"Find the eigenvalues of the qubit Hamiltonian $H = {shown}$, where "
                "$\\sigma_x, \\sigma_y, \\sigma_z$ are the Pauli matrices. On paper: show that "
                "$(\\mathbf n\\cdot\\boldsymbol\\sigma)^2 = \\|\\mathbf n\\|^2 I$ and use it."),
        answer=[str(d + n), str(d - n)], kind="set",
        rubric=[f"Write H as a 2×2 matrix: {fmt(H)}", "Pauli matrices anticommute and square to I",
                f"(n·σ)² = ‖n‖² I with ‖n‖ = {n}, so n·σ has eigenvalues ±{n}", f"Shift by {d}: λ = {d} ± {n}"],
        solution=f"$H = {L(H)}$, $\\|\\mathbf n\\| = \\sqrt{{({pa})^2 + ({pb})^2 + ({pc})^2}} = {n}$, eigenvalues ${d + n}, {d - n}$.",
        misconceptions=[(f"{d + pc}, {d - pc}", "diagonal-entries-not-eigenvalues")],
    )
