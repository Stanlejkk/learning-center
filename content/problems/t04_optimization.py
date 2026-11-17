"""Track 4 — Optimization (MIT 18.065 / Boyd & Vandenberghe / Adam paper territory, original problems).

All iterations run on small quadratics or simple polynomials so a few steps can be done by hand
with exact fractions. Conventions follow content/problems/t00_foundations.py.
"""
from __future__ import annotations

import random

import sympy as sp

from engine.checker import fmt
from engine.content import Problem, gen

L = sp.latex
R = sp.Rational
x, y = sp.symbols("x y")
FRAC_HINT = "an exact fraction, e.g. 7/4"
VEC_HINT = "a vector with exact entries, e.g. [1/2, -3]"


def nz(rng: random.Random, lo: int, hi: int) -> int:
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def vec(v) -> list[str]:
    return [fmt(e) for e in v]


def sgn(v) -> int:
    return 1 if v > 0 else -1


def _spd(rng, lo=2, hi=6):
    while True:
        a, d, b = rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(-2, 2)
        if a * d - b * b > 0:
            return sp.Matrix([[a, b], [b, d]])


# --------------------------------------------------------------------------- convexity
CONVEX = ["x^2 + 3x", "e^{x}", "|x|", "-\\ln x\\ \\ (x>0)", "x\\ln x\\ \\ (x>0)", "x^4", "e^{-2x}",
          "\\tfrac{1}{x}\\ \\ (x>0)", "\\max(x, 0)", "\\sqrt{x^2+1}", "(x-1)^2 + |x|"]
NONCONVEX = ["x^3", "\\sin x", "\\ln x\\ \\ (x>0)", "-x^2", "\\sqrt{x}\\ \\ (x\\ge 0)", "x^4 - x^2", "e^{-x^2}"]


def _convexity_base(rng, level):
    if level == 1:
        opts = rng.sample(CONVEX, 3) + [rng.choice(NONCONVEX)]
        bad = opts[3]
        rng.shuffle(opts)
        letter = "ABCD"[opts.index(bad)]
        listing = "\n".join(f"- **{'ABCD'[i]}.** $f(x) = {o}$" for i, o in enumerate(opts))
        return Problem(
            prompt="Exactly one of these functions is **not** convex on the stated domain (all of $\\mathbb R$ if none is given). Which?\n\n" + listing,
            answer=letter, kind="choice",
            rubric=["For smooth f: check f''(x) ≥ 0 on the whole domain", "For non-smooth f: use max/sum/composition rules or the chord definition",
                    "Find a point where f'' < 0 (or a chord below the graph) for the odd one out"],
            solution=f"**{letter}**: $f(x) = {bad}$ fails — its second derivative is negative somewhere on the domain (or it is concave). The others are convex.",
        )
    if level == 2:
        a, c = rng.choice([(1, 1), (1, 4), (4, 1), (2, 2), (2, 8), (8, 2), (1, 9), (9, 1), (3, 3), (3, 12), (4, 9), (9, 4), (5, 5)])
        ans = 2 * sp.sqrt(a * c)
        return Problem(
            prompt=(f"For which $k$ is $f(x,y) = {L(a * x ** 2)} + k\\,xy + {L(c * y ** 2)}$ convex on $\\mathbb R^2$? "
                    "Type the **largest** such $k$."),
            answer=fmt(ans), kind="num",
            rubric=["Hessian H = [[2a, k], [k, 2c]]", "Convex iff H is positive semidefinite",
                    "2x2 PSD test: diagonal ≥ 0 and det = 4ac - k² ≥ 0", "So |k| ≤ 2√(ac)"],
            solution=f"$H = \\begin{{pmatrix}}{2 * a} & k\\\\ k & {2 * c}\\end{{pmatrix}}$, $\\det H = {4 * a * c} - k^2 \\ge 0 \\iff |k| \\le {ans}$.",
            misconceptions=[(fmt(sp.sqrt(a * c)), "forgot-factor-two"), (fmt(4 * a * c), "forgot-square-root")],
        )
    m = rng.randint(1, 3)
    c = 6 * m * m
    a = sp.Symbol("a")
    return Problem(
        prompt=(f"For which real $a$ is $f(x) = x^4 + a x^3 + {c} x^2$ convex on all of $\\mathbb R$? "
                "Type the **largest** such $a$."),
        answer=str(4 * m), kind="num",
        rubric=["f''(x) = 12x² + 6a x + 2c", "Convex iff f''(x) ≥ 0 for all x", "Quadratic in x is ≥ 0 iff discriminant ≤ 0",
                "36a² ≤ 96c ⇒ |a| ≤ √(8c/3)"],
        solution=(f"$f''(x) = 12x^2 + 6ax + {2 * c}$; discriminant $36a^2 - {96 * c} \\le 0 \\iff a^2 \\le {16 * m * m}$, "
                  f"so $a_{{\\max}} = {4 * m}$."),
        misconceptions=[(fmt(2 * sp.sqrt(c)), "used-f-not-second-derivative")],
    )


# --------------------------------------------------------------------------- gradient descent
def _gd_1d_base(rng, level):
    if level in (1, 2):
        while True:
            a, c = rng.randint(1, 5), rng.randint(-5, 5)
            x0 = rng.randint(-6, 6)
            eta = R(1, rng.choice([4, 5, 8, 10, 20]))
            r = 1 - 2 * a * eta
            if x0 != c and r != 0:
                break
        f = a * (x - c) ** 2
        steps = 1 if level == 1 else 3
        ans = c + r ** steps * (x0 - c)
        wrong_sign = c + (1 + 2 * a * eta) ** steps * (x0 - c)
        mis = [(fmt(wrong_sign), "sign-error-gradient-step")]
        if level == 1:
            mis += [(fmt(x0 - 2 * a * (x0 - c)), "forgot-learning-rate"), (fmt(x0 - eta * a * (x0 - c)), "derivative-missing-factor-2")]
        else:
            mis += [(fmt(c + r * (x0 - c)), "stopped-early")]
        return Problem(
            prompt=(f"Run gradient descent $x_{{t+1}} = x_t - \\eta f'(x_t)$ on $f(x) = {L(f)}$ with $\\eta = {L(eta)}$ "
                    f"from $x_0 = {x0}$. Give $x_{{{steps}}}$."),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["f'(x) = 2a(x - c)", "Apply the update (minus sign!)"] +
                   (["Note x_t - c shrinks by the factor (1 - 2aη) each step", "Iterate three times"] if level == 2 else ["Evaluate"]),
            solution=(f"$f'(x) = {L(sp.diff(f, x))}$; $x_t - ({c}) = ({L(r)})^t\\,(x_0 - ({c}))$, so $x_{{{steps}}} = {L(ans)}$."),
            misconceptions=mis,
        )
    while True:
        x0 = rng.choice([1, 2, -1, -2])
        eta = R(1, rng.choice([4, 5, 8, 10]))
        b = rng.randint(-6, 6)
        x1 = x0 - eta * (x0 ** 3 - b)
        if sp.denom(x1) <= 5 and abs(x1) <= 3 and x1 != x0:
            x2 = x1 - eta * (x1 ** 3 - b)
            if x2 != x1:
                break
    f = x ** 4 / 4 - b * x
    return Problem(
        prompt=(f"Gradient descent on the non-quadratic $f(x) = {L(f)}$ with $\\eta = {L(eta)}$, $x_0 = {x0}$. "
                "Compute $x_2$ exactly. On paper, also check whether $f$ decreased at each step."),
        answer=fmt(x2), kind="num", answer_hint="an exact fraction, e.g. -37/125",
        rubric=[f"f'(x) = x³ - ({b})", "x_1 = x_0 - η f'(x_0)", "x_2 = x_1 - η f'(x_1)", "Compare f(x_0), f(x_1), f(x_2)"],
        solution=f"$x_1 = {x0} - {L(eta)}(({x0})^3 - ({b})) = {L(x1)}$, $x_2 = {L(x1)} - {L(eta)}\\left(({L(x1)})^3 - ({b})\\right) = {L(x2)}$.",
        misconceptions=[(fmt(x1), "stopped-early"),
                        (fmt(x1 + eta * (x1 ** 3 - b)), "sign-error-gradient-step")],
    )


def _gd_2d_base(rng, level):
    if level == 1:
        a, b, c = rng.randint(1, 4), rng.randint(1, 4), rng.randint(-3, 3)
        x0, y0 = nz(rng, -4, 4), nz(rng, -4, 4)
        eta = R(1, rng.choice([2, 4, 5, 10]))
        f = a * x ** 2 + b * y ** 2 + c * x * y
        g = [sp.diff(f, v).subs({x: x0, y: y0}) for v in (x, y)]
        new = [x0 - eta * g[0], y0 - eta * g[1]]
        return Problem(
            prompt=(f"One gradient-descent step on $f(x,y) = {L(f)}$ from $(x_0, y_0) = ({x0}, {y0})$ with $\\eta = {L(eta)}$. "
                    "Give $(x_1, y_1)$."),
            answer=vec(new), kind="vector", answer_hint=VEC_HINT,
            rubric=["Compute ∇f = (∂f/∂x, ∂f/∂y)", "Evaluate at the start point", "Subtract η∇f"],
            solution=f"$\\nabla f({x0},{y0}) = ({g[0]}, {g[1]})$, so $(x_1,y_1) = ({L(new[0])}, {L(new[1])})$.",
            misconceptions=[(vec([x0 + eta * g[0], y0 + eta * g[1]]), "sign-error-gradient-step"),
                            (vec([x0 - eta * 2 * a * x0, y0 - eta * 2 * b * y0]), "dropped-cross-term")],
        )
    if level == 2:
        A = _spd(rng)
        bv = sp.Matrix([rng.randint(-4, 4), rng.randint(-4, 4)])
        while bv == sp.zeros(2, 1):
            bv = sp.Matrix([rng.randint(-4, 4), rng.randint(-4, 4)])
        eta = R(1, rng.choice([5, 8, 10]))
        w = sp.zeros(2, 1)
        hist = []
        for _ in range(2):
            w = w - eta * (A * w - bv)
            hist.append(w)
        wrong = sp.zeros(2, 1)
        for _ in range(2):
            wrong = wrong + eta * (A * wrong - bv)
        return Problem(
            prompt=(f"$f(\\mathbf w) = \\tfrac12\\mathbf w^\\top A\\mathbf w - \\mathbf b^\\top\\mathbf w$ with $A = {L(A)}$, "
                    f"$\\mathbf b = {L(bv)}$. Starting at $\\mathbf w_0 = \\mathbf 0$ with $\\eta = {L(eta)}$, give $\\mathbf w_2$."),
            answer=vec(hist[1]), kind="vector", answer_hint=VEC_HINT,
            rubric=["∇f(w) = A w - b (A symmetric)", "w_1 = w_0 - η(A w_0 - b)", "w_2 = w_1 - η(A w_1 - b)"],
            solution=f"$\\mathbf w_1 = {L(hist[0])}$, $\\mathbf w_2 = {L(hist[1])}$.",
            misconceptions=[(vec(hist[0]), "stopped-early"), (vec(wrong), "sign-error-gradient-step")],
        )
    while True:
        m = rng.choice([4, 6, 8, 12])
        p, q = rng.sample(range(1, m), 2)
        eta = R(1, m)
        k = rng.randint(3, 5)
        x0, y0 = nz(rng, -4, 4), nz(rng, -4, 4)
        r1, r2 = 1 - 2 * eta * p, 1 - 2 * eta * q
        if abs(r1) < 1 and abs(r2) < 1:
            break
    ans = [r1 ** k * x0, r2 ** k * y0]
    return Problem(
        prompt=(f"Gradient descent on $f(x,y) = {L(p * x ** 2 + q * y ** 2)}$ with $\\eta = {L(eta)}$ from $({x0}, {y0})$. "
                f"Derive a closed form for the iterates and give $(x_{{{k}}}, y_{{{k}}})$ exactly. "
                "On paper: which coordinate converges faster, and why?"),
        answer=vec(ans), kind="vector", answer_hint=VEC_HINT,
        rubric=["The update decouples: x ← (1 - 2ηp) x, y ← (1 - 2ηq) y", "So x_k = (1 - 2ηp)^k x_0 (same for y)",
                f"Evaluate the powers for k = {k}", "Faster coordinate = smaller |1 - 2η·curvature|"],
        solution=f"$x_k = ({L(r1)})^k\\cdot({x0})$, $y_k = ({L(r2)})^k\\cdot({y0})$, giving $({L(ans[0])}, {L(ans[1])})$.",
        misconceptions=[(vec([(1 - eta * p) ** k * x0, (1 - eta * q) ** k * y0]), "forgot-factor-two-gradient")],
    )


# --------------------------------------------------------------------------- conditioning
def _conditioning_base(rng, level):
    if level == 1:
        l1, l2 = rng.sample(range(1, 21), 2)
        lmax, lmin = max(l1, l2), min(l1, l2)
        return Problem(
            prompt=(f"Gradient descent with fixed step $\\eta$ on $f(x,y) = \\tfrac{{{l1}}}{{2}}x^2 + \\tfrac{{{l2}}}{{2}}y^2$. "
                    "For which $\\eta$ does it converge from every start? Type the threshold $\\eta^\\ast$ (converges iff $0<\\eta<\\eta^\\ast$)."),
            answer=fmt(R(2, lmax)), kind="num", answer_hint=FRAC_HINT,
            rubric=["Each coordinate is multiplied by (1 - ηλ_i) per step", "Need |1 - ηλ_i| < 1 for every i",
                    "The largest eigenvalue is the binding constraint: η < 2/λ_max"],
            solution=f"$|1-\\eta\\lambda| < 1 \\iff 0 < \\eta < 2/\\lambda$; binding for $\\lambda_{{\\max}} = {lmax}$: $\\eta^\\ast = {L(R(2, lmax))}$.",
            misconceptions=[(fmt(R(2, lmin)), "used-lambda-min"), (fmt(R(1, lmax)), "confused-1-over-L-with-threshold")],
        )

    def int_eig_matrix():
        while True:
            a, d, b = rng.randint(1, 10), rng.randint(1, 10), rng.randint(1, 5)
            disc = (a - d) ** 2 + 4 * b * b
            s = sp.sqrt(disc)
            if s.is_integer and a * d - b * b > 0 and (a + d + s) % 2 == 0:
                A = sp.Matrix([[a, b], [b, d]])
                lmin, lmax = (a + d - s) / 2, (a + d + s) / 2
                return A, sp.Integer(lmin), sp.Integer(lmax)

    if level == 2:
        A, lmin, lmax = int_eig_matrix()
        eta = R(2, 1) / (lmin + lmax)
        rho = (lmax - lmin) / (lmax + lmin)
        return Problem(
            prompt=(f"For $f(\\mathbf x) = \\tfrac12\\mathbf x^\\top A\\mathbf x$ with $A = {L(A)}$, find the step size $\\eta^\\star$ "
                    "minimising the worst-case GD contraction factor, and that factor $\\rho = \\max_i|1-\\eta^\\star\\lambda_i|$."),
            answer=[fmt(eta), fmt(rho)], kind="list", answer_hint="eta, rho   e.g. 1/5, 3/5",
            rubric=["Eigenvalues of A (characteristic polynomial)", "Balance |1 - ηλ_min| = |1 - ηλ_max| ⇒ η* = 2/(λ_min+λ_max)",
                    "ρ = (κ - 1)/(κ + 1) with κ = λ_max/λ_min"],
            solution=(f"$\\lambda = {lmin}, {lmax}$; $\\eta^\\star = \\frac{{2}}{{{lmin + lmax}}} = {L(eta)}$, "
                      f"$\\rho = \\frac{{{lmax} - {lmin}}}{{{lmax} + {lmin}}} = {L(rho)}$."),
            misconceptions=[([fmt(1 / lmax), fmt(1 - lmin / lmax)], "used-1-over-L-step"),
                            ([fmt(2 / lmax), fmt(rho)], "used-stability-threshold")],
        )
    l1, l2 = rng.choice([(1, 3), (2, 6), (3, 9), (1, 2), (2, 4), (3, 6), (3, 5), (6, 10), (2, 3), (4, 6)])
    if (l1 + l2) % 2 == 0 and rng.random() < 0.6:
        p, q = (l1 + l2) // 2, (l2 - l1) // 2
        A = sp.Matrix([[p, q], [q, p]])
    else:
        A = sp.diag(l1, l2)
    d = rng.choice([3, 6])
    rho = R(l2 - l1, l2 + l1)
    target = R(1, 10 ** d)
    k = 0
    while rho ** k > target:
        k += 1
    rho_bad = 1 - R(l1, l2)
    kb = 0
    while rho_bad ** kb > target:
        kb += 1
    return Problem(
        prompt=(f"GD with the optimal fixed step $\\eta = 2/(\\lambda_{{\\min}}+\\lambda_{{\\max}})$ on "
                f"$f(\\mathbf x) = \\tfrac12\\mathbf x^\\top A\\mathbf x$, $A = {L(A)}$, shrinks the error by a factor $\\rho$ every step. "
                f"What is the smallest number of steps $k$ that guarantees $\\|\\mathbf x_k\\| \\le 10^{{-{d}}}\\,\\|\\mathbf x_0\\|$?"),
        answer=str(k), kind="num", answer_hint="an integer",
        rubric=["Eigenvalues of A → condition number κ", "ρ = (κ-1)/(κ+1)", f"Solve ρ^k ≤ 10^-{d} (take logs or tabulate powers)",
                "Round up to an integer"],
        solution=f"$\\lambda = {l1}, {l2}$, $\\kappa = {L(R(l2, l1))}$, $\\rho = {L(rho)}$; smallest $k$ with $\\rho^k \\le 10^{{-{d}}}$ is ${k}$.",
        misconceptions=[(str(kb), "used-1-over-L-step")],
    )


# --------------------------------------------------------------------------- least squares
def _least_squares_base(rng, level):
    if level == 1:
        xs = rng.sample([-3, -2, -1, 1, 2, 3, 4], 3)
        ys = [rng.randint(-6, 9) for _ in xs]
        sxy, sxx = sum(a * b for a, b in zip(xs, ys)), sum(a * a for a in xs)
        ans = R(sxy, sxx)
        mis = [(fmt(R(sum(ys), sum(xs))), "ratio-of-means")] if sum(xs) != 0 else []
        pts = ", ".join(f"({a}, {b})" for a, b in zip(xs, ys))
        return Problem(
            prompt=f"Fit the line through the origin $y = wx$ to the points {pts} by least squares. Give $\\hat w$.",
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["Minimise Σ(y_i - w x_i)²", "Set the derivative to 0: Σ x_i(y_i - w x_i) = 0", "ŵ = Σxy / Σx²"],
            solution=f"$\\sum xy = {sxy}$, $\\sum x^2 = {sxx}$, $\\hat w = {L(ans)}$.",
            misconceptions=mis,
        )
    if level == 2:
        xs = sorted(rng.sample(range(0, 7), 4))
        ys = [rng.randint(-5, 10) for _ in xs]
        X = sp.Matrix([[1, v] for v in xs])
        Y = sp.Matrix(ys)
        beta = (X.T * X).inv() * X.T * Y
        pts = ", ".join(f"({a}, {b})" for a, b in zip(xs, ys))
        return Problem(
            prompt=f"Fit $y = \\beta_0 + \\beta_1 x$ by least squares to {pts}. Give $\\beta_0, \\beta_1$.",
            answer=vec(beta), kind="list", answer_hint="intercept, slope   e.g. 1/2, 3/4",
            rubric=["Build X with a column of ones", "Normal equations XᵀX β = Xᵀy (or slope = Sxy/Sxx)", "Solve the 2×2 system",
                    "Intercept: β0 = ȳ - β1 x̄"],
            solution=f"$X^\\top X = {L(X.T * X)}$, $X^\\top y = {L(X.T * Y)}$, so $\\beta_0 = {L(beta[0])}$, $\\beta_1 = {L(beta[1])}$.",
            misconceptions=[([fmt(beta[1]), fmt(beta[0])], "swapped-intercept-slope")],
        )
    while True:
        A = sp.Matrix(3, 2, [rng.randint(-3, 3) for _ in range(6)])
        b = sp.Matrix([rng.randint(-5, 5) for _ in range(3)])
        if (A.T * A).det() == 0:
            continue
        xh = (A.T * A).inv() * A.T * b
        if A * xh != b and max(abs(sp.denom(v)) for v in xh) <= 60:
            break
    mis = []
    if A[:2, :].det() != 0:
        mis.append((vec(A[:2, :].inv() * b[:2, :]), "solved-subsystem"))
    return Problem(
        prompt=(f"The system $A\\mathbf x = \\mathbf b$ with $A = {L(A)}$, $\\mathbf b = {L(b)}$ has no exact solution. "
                "Find the least-squares solution $\\hat{\\mathbf x}$. On paper, verify the residual is orthogonal to the columns of $A$."),
        answer=vec(xh), kind="vector", answer_hint=VEC_HINT,
        rubric=["Form AᵀA and Aᵀb", "Solve AᵀA x̂ = Aᵀb", "Check Aᵀ(b - A x̂) = 0"],
        solution=f"$A^\\top A = {L(A.T * A)}$, $A^\\top\\mathbf b = {L(A.T * b)}$, $\\hat{{\\mathbf x}} = {L(xh)}$.",
        misconceptions=mis,
    )


# --------------------------------------------------------------------------- Lagrange multipliers
def _lagrange_base(rng, level):
    if level == 1:
        a, b, c = rng.randint(1, 5), rng.randint(1, 5), rng.randint(2, 20)
        ans = R(c * c, 4 * a * b)
        return Problem(
            prompt=f"Maximise $f(x,y) = xy$ subject to ${L(a * x + b * y)} = {c}$ with $x, y > 0$. Give the maximum value.",
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["∇f = λ∇g: y = λa, x = λb", "Substitute into the constraint to find λ", "x = c/(2a), y = c/(2b)", "Evaluate xy"],
            solution=f"$x = \\frac{{{c}}}{{{2 * a}}}$, $y = \\frac{{{c}}}{{{2 * b}}}$, $f_{{\\max}} = {L(ans)}$.",
            misconceptions=[(fmt(R(c * c, 4)), "ignored-coefficients"), (fmt(R(c * c, a * b)), "dropped-factor-4")],
        )
    if level == 2:
        a, b, c = nz(rng, -4, 4), nz(rng, -4, 4), nz(rng, -10, 10)
        s = a * a + b * b
        pt = [R(a * c, s), R(b * c, s)]
        return Problem(
            prompt=f"Find the point on the line ${L(a * x + b * y)} = {c}$ closest to the origin (minimise $x^2+y^2$ with a Lagrange multiplier).",
            answer=vec(pt), kind="vector", answer_hint=VEC_HINT,
            rubric=["2x = λa, 2y = λb", "Substitute into the constraint: λ = 2c/(a²+b²)", "Back-substitute for x, y"],
            solution=f"$\\lambda = \\frac{{2\\cdot{c}}}{{{s}}}$, point $= ({L(pt[0])}, {L(pt[1])})$, min value ${L(R(c * c, s))}$.",
            misconceptions=[(vec([pt[1], pt[0]]), "swapped-coordinates")],
        )
    r = rng.randint(1, 5)
    if rng.random() < 0.5:
        coeffs = list(rng.choice([(3, 4), (5, 12), (6, 8), (8, 15), (4, 3), (12, 5)]))
        norm = int(sp.sqrt(sum(c * c for c in coeffs)))
        coeffs = [c * rng.choice([1, -1]) for c in coeffs]
        obj = coeffs[0] * x + coeffs[1] * y
        cons = f"x^2 + y^2 = {r * r}"
    else:
        z = sp.Symbol("z")
        coeffs = list(rng.choice([(1, 2, 2), (2, 3, 6), (1, 4, 8), (2, 6, 9), (4, 4, 7)]))
        norm = int(sp.sqrt(sum(c * c for c in coeffs)))
        coeffs = [c * rng.choice([1, -1]) for c in coeffs]
        obj = coeffs[0] * x + coeffs[1] * y + coeffs[2] * z
        cons = f"x^2 + y^2 + z^2 = {r * r}"
    ans = r * norm
    return Problem(
        prompt=f"Maximise ${L(obj)}$ subject to ${cons}$. Give the maximum value (and, on paper, the maximiser and $\\lambda$).",
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=["∇f = λ∇g gives the point x = c/(2λ) (parallel to the coefficient vector c)",
                "Constraint fixes |λ|: x = r c/‖c‖", "Choose the sign that maximises", "Max = r‖c‖ (Cauchy–Schwarz check)"],
        solution=f"Maximiser $\\mathbf x^\\ast = {r}\\,\\mathbf c/\\|\\mathbf c\\|$ with $\\|\\mathbf c\\| = {norm}$; max $= {r}\\cdot{norm} = {ans}$.",
        misconceptions=[(str(r * sum(abs(c) for c in coeffs)), "l1-not-l2-norm"), (str(r * r * norm), "squared-radius")],
    )


# --------------------------------------------------------------------------- Newton's method
def _newton_base(rng, level):
    if level == 1:
        while True:
            x0, a = rng.choice([1, 2, -1, -2, 3]), rng.randint(-10, 10)
            if x0 ** 3 != a:
                break
        f = x ** 4 / 4 - a * x
        g, h = x0 ** 3 - a, 3 * x0 ** 2
        ans = x0 - R(g, h)
        return Problem(
            prompt=f"Take one Newton step to minimise $f(x) = {L(f)}$ from $x_0 = {x0}$: $x_1 = x_0 - f'(x_0)/f''(x_0)$.",
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["f'(x) = x³ - a", "f''(x) = 3x²", "x_1 = x_0 - f'/f''"],
            solution=f"$f'({x0}) = {g}$, $f''({x0}) = {h}$, $x_1 = {x0} - \\frac{{{g}}}{{{h}}} = {L(ans)}$.",
            misconceptions=[(fmt(x0 - g), "used-gradient-not-newton"), (fmt(x0 + R(g, h)), "sign-error")],
        )
    if level == 2:
        A = _spd(rng)
        bv = sp.Matrix([rng.randint(-5, 5), rng.randint(-5, 5)])
        w0 = sp.Matrix([rng.randint(-3, 3), rng.randint(-3, 3)])
        ans = A.inv() * bv
        return Problem(
            prompt=(f"Apply one Newton step to $f(\\mathbf w) = \\tfrac12\\mathbf w^\\top A\\mathbf w - \\mathbf b^\\top\\mathbf w$, "
                    f"$A = {L(A)}$, $\\mathbf b = {L(bv)}$, from $\\mathbf w_0 = {L(w0)}$. Give $\\mathbf w_1$ and explain on paper "
                    "why the starting point does not matter."),
            answer=vec(ans), kind="vector", answer_hint=VEC_HINT,
            rubric=["Gradient A w - b, Hessian A", "w_1 = w_0 - A^{-1}(A w_0 - b) = A^{-1} b", "Invert the 2×2 (or solve A w = b)"],
            solution=f"$\\mathbf w_1 = A^{{-1}}\\mathbf b = {L(ans)}$ — Newton is exact on quadratics.",
            misconceptions=[(vec(w0 - (A * w0 - bv)), "used-gradient-not-newton")],
        )
    while True:
        c, x0 = rng.choice([2, 3, 5, 6, 7, 10, 11]), rng.choice([1, 2, 3])
        if x0 * x0 != c:
            break
    x1 = (x0 + R(c, x0)) / 2
    x2 = (x1 + c / x1) / 2
    return Problem(
        prompt=(f"Minimise $f(x) = \\tfrac13 x^3 - {c}x$ on $x > 0$ with Newton's method from $x_0 = {x0}$. "
                f"Show the update simplifies to $x \\leftarrow \\tfrac12(x + {c}/x)$ and give $x_2$ exactly. "
                f"On paper: compare $|x_1 - \\sqrt{{{c}}}|$ and $|x_2 - \\sqrt{{{c}}}|$ (quadratic convergence)."),
        answer=fmt(x2), kind="num", answer_hint=FRAC_HINT,
        rubric=[f"f'(x) = x² - {c}, f''(x) = 2x", "x - (x² - c)/(2x) = (x + c/x)/2", "Two iterations with exact fractions",
                "Error roughly squares each step"],
        solution=f"$x_1 = {L(x1)}$, $x_2 = {L(x2)} \\approx {float(x2):.6f}$ vs $\\sqrt{{{c}}} \\approx {float(sp.sqrt(c)):.6f}$.",
        misconceptions=[(fmt(x1), "stopped-early")],
    )


# --------------------------------------------------------------------------- SGD & momentum
def _sgd_momentum_base(rng, level):
    if level == 1:
        n = 6
        data = [rng.randint(-5, 9) for _ in range(n)]
        bsz = rng.choice([2, 3])
        idx = sorted(rng.sample(range(1, n + 1), bsz))
        w0 = rng.randint(-3, 5)
        eta = R(1, rng.choice([2, 4, 5, 10]))
        gb = w0 - R(sum(data[i - 1] for i in idx), bsz)
        gfull = w0 - R(sum(data), n)
        ans = w0 - eta * gb
        return Problem(
            prompt=(f"Loss $\\mathcal L(w) = \\frac1{{{n}}}\\sum_{{i=1}}^{{{n}}} \\tfrac12 (w - a_i)^2$ with data "
                    f"$a = ({', '.join(map(str, data))})$. One SGD step uses the mini-batch of indices "
                    f"$\\{{{', '.join(map(str, idx))}\\}}$ (averaged gradient), $\\eta = {L(eta)}$, $w_0 = {w0}$. Give $w_1$."),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["Per-example gradient: w - a_i", "Average over the mini-batch only", "w_1 = w_0 - η g_B"],
            solution=f"$g_B = {w0} - \\text{{mean}}(a_B) = {L(gb)}$, $w_1 = {w0} - {L(eta)}\\cdot{L(gb)} = {L(ans)}$.",
            misconceptions=[(fmt(w0 - eta * gfull), "used-full-batch"), (fmt(w0 + eta * gb), "sign-error-gradient-step")],
        )
    if level == 2:
        a = rng.randint(1, 4)
        beta = rng.choice([R(1, 2), R(9, 10), R(1, 4)])
        eta = R(1, rng.choice([4, 5, 8, 10]))
        x0 = nz(rng, -4, 4)
        xx, v = sp.Integer(x0), sp.Integer(0)
        xg = sp.Integer(x0)
        xe, ve = sp.Integer(x0), sp.Integer(0)
        for _ in range(3):
            v = beta * v + a * xx
            xx = xx - eta * v
            xg = xg - eta * a * xg
            ve = beta * ve + (1 - beta) * a * xe
            xe = xe - eta * ve
        return Problem(
            prompt=(f"Heavy-ball momentum on $f(x) = {L(R(a, 2) * x ** 2)}$: $v_{{t+1}} = \\beta v_t + f'(x_t)$, "
                    f"$x_{{t+1}} = x_t - \\eta v_{{t+1}}$, with $\\beta = {L(beta)}$, $\\eta = {L(eta)}$, $x_0 = {x0}$, $v_0 = 0$. "
                    "Give $x_3$."),
            answer=fmt(xx), kind="num", answer_hint=FRAC_HINT,
            rubric=["f'(x) = a x", "Update v first, then x (keep a table of t, v_t, x_t)", "Three iterations with exact fractions"],
            solution=f"Iterating gives $x_3 = {L(xx)}$ (plain GD would give ${L(xg)}$).",
            misconceptions=[(fmt(xg), "ignored-momentum"), (fmt(xe), "wrong-momentum-convention")],
        )
    if rng.random() < 0.5:
        n = rng.choice([4, 5])
        while True:
            g = [rng.randint(-6, 8) for _ in range(n)]
            if len(set(g)) > 1:
                break
        B = rng.choice([2, 4, 5, 8, 10])
        mean = R(sum(g), n)
        pvar = R(sum(v * v for v in g), n) - mean ** 2
        svar = pvar * R(n, n - 1)
        ans = pvar / B
        return Problem(
            prompt=(f"A dataset of {n} examples has per-example gradients (at the current $w$) $g = ({', '.join(map(str, g))})$; "
                    f"the full gradient is their mean. SGD averages a mini-batch of $B = {B}$ indices drawn uniformly **with replacement**. "
                    "What is the variance of the mini-batch gradient?"),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["One random index gives g_I with E[g_I] = mean(g)", "Var(g_I) = mean(g²) - mean(g)² (population variance)",
                    "B i.i.d. draws: variance divides by B"],
            solution=f"$\\mathrm{{Var}}(g_I) = {L(pvar)}$, so $\\mathrm{{Var}}(\\bar g_B) = {L(pvar)}/{B} = {L(ans)}$.",
            misconceptions=[(fmt(svar / B), "used-sample-variance"), (fmt(pvar), "forgot-divide-by-B")],
        )
    beta = rng.choice([R(1, 2), R(3, 4), R(9, 10), R(4, 5)])
    eta = R(1, rng.choice([10, 20, 100]))
    gg = rng.randint(1, 9)
    ans = eta * gg / (1 - beta)
    return Problem(
        prompt=(f"Heavy-ball momentum $v_{{t+1}} = \\beta v_t + g$, $x_{{t+1}} = x_t - \\eta v_{{t+1}}$ runs on a long linear slope with "
                f"constant gradient $g = {gg}$, $\\beta = {L(beta)}$, $\\eta = {L(eta)}$, $v_0 = 0$. As $t\\to\\infty$, how far does $x$ "
                "move per step (the magnitude $|x_{t+1}-x_t|$)?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["v_t = g(1 + β + … + β^{t-1})", "Geometric series → g/(1-β)", "Step = η g/(1-β): momentum multiplies the effective lr by 1/(1-β)"],
        solution=f"$v_\\infty = \\frac{{g}}{{1-\\beta}} = {L(gg / (1 - beta))}$, step $= \\eta v_\\infty = {L(ans)}$.",
        misconceptions=[(fmt(eta * gg), "ignored-momentum"), (fmt(eta * gg / beta), "divided-by-beta")],
    )


# --------------------------------------------------------------------------- Adam
B1, B2 = R(9, 10), R(999, 1000)


def _adam_base(rng, level):
    if level == 1:
        while True:
            a, c, x0 = rng.randint(1, 4), rng.randint(-3, 3), rng.randint(-5, 5)
            if x0 != c:
                break
        eta = rng.choice([R(1, 10), R(1, 100), R(1, 2), R(1, 5)])
        g = a * (x0 - c)
        ans = x0 - eta * sgn(g)
        mis = [(str(x0 - eta * sgn(g) * sp.sqrt(10)), "forgot-bias-correction")]
        if abs(g) != 1:
            mis.append((fmt(x0 - eta * g), "used-plain-gd"))
        return Problem(
            prompt=(f"Adam ($\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $\\epsilon = 0$, $m_0 = v_0 = 0$, with bias correction) on "
                    f"$f(x) = {L(R(a, 2) * (x - c) ** 2)}$ from $x_0 = {x0}$ with $\\eta = {L(eta)}$. Give $x_1$."),
            answer=fmt(ans), kind="num", answer_hint="exact, e.g. 29/10",
            rubric=["g_1 = f'(x_0)", "m_1 = (1-β1) g, v_1 = (1-β2) g²", "Bias-correct: m̂ = g, v̂ = g²",
                    "Step η·m̂/√v̂ = η·sign(g): the first Adam step has length η regardless of |g|"],
            solution=f"$g = {g}$, $\\hat m = {g}$, $\\hat v = {g * g}$, $x_1 = {x0} - {L(eta)}\\cdot\\frac{{{g}}}{{{abs(g)}}} = {L(ans)}$.",
            misconceptions=mis,
        )
    if level == 2:
        g1, g2 = nz(rng, -5, 5), nz(rng, -5, 5)
        m2 = B1 * (1 - B1) * g1 + (1 - B1) * g2
        v2 = B2 * (1 - B2) * g1 ** 2 + (1 - B2) * g2 ** 2
        mh, vh = m2 / (1 - B1 ** 2), v2 / (1 - B2 ** 2)
        return Problem(
            prompt=(f"Adam with $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $m_0 = v_0 = 0$ sees gradients $g_1 = {g1}$, $g_2 = {g2}$. "
                    "Give the bias-corrected moments $\\hat m_2$ and $\\hat v_2$ exactly."),
            answer=[fmt(mh), fmt(vh)], kind="list", answer_hint="m_hat, v_hat   exact fractions, e.g. 50/19, 1000/1999",
            rubric=["m_t = β1 m_{t-1} + (1-β1) g_t", "v_t = β2 v_{t-1} + (1-β2) g_t²", "Divide by (1 - β^t) with t = 2",
                    "Interpretation: m̂ is a weighted average of g1, g2 (weights 0.09 and 0.1, normalised)"],
            solution=(f"$m_2 = {L(m2)}$, $v_2 = {L(v2)}$; $\\hat m_2 = m_2/0.19 = {L(mh)}$, "
                      f"$\\hat v_2 = v_2/(1-0.999^2) = {L(vh)}$."),
            misconceptions=[([fmt(m2), fmt(v2)], "forgot-bias-correction")],
        )
    while True:
        x0, g = nz(rng, -5, 5), nz(rng, -6, 6)
        lam = rng.choice([R(1, 10), R(1, 2), 1, 2])
        if g + lam * x0 != 0:
            break
    eta = rng.choice([R(1, 10), R(1, 100)])
    w = x0 - eta * (sgn(g) + lam * x0)
    l2 = x0 - eta * sgn(g + lam * x0)
    return Problem(
        prompt=(f"A weight $x_0 = {x0}$ has loss gradient $g = {g}$ at the first step. Compare (a) **AdamW** with decoupled weight "
                f"decay $\\lambda = {L(sp.nsimplify(lam))}$: $x_1 = x_0 - \\eta(\\hat m/\\sqrt{{\\hat v}} + \\lambda x_0)$, and (b) **Adam with an L2 "
                f"penalty** $\\tfrac\\lambda2 x^2$ added to the loss. Both: $\\eta = {L(eta)}$, $\\epsilon = 0$, bias-corrected first step. "
                "Give $x_1$ for (a) then (b)."),
        answer=[fmt(w), fmt(l2)], kind="list", answer_hint="AdamW x1, Adam+L2 x1   e.g. 29/10, 3",
        rubric=["First bias-corrected Adam step is η·sign(gradient)", "AdamW: decay term λx_0 added outside the normalisation",
                "Adam+L2: gradient becomes g + λx_0, then gets normalised to its sign", "Explain why the decay is 'lost' in (b)"],
        solution=(f"(a) $x_1 = {x0} - {L(eta)}({sgn(g)} + {L(sp.nsimplify(lam))}\\cdot{x0}) = {L(w)}$; "
                  f"(b) gradient $= {L(sp.nsimplify(g + lam * x0))}$, $x_1 = {x0} - {L(eta)}\\cdot({sgn(g + lam * x0)}) = {L(l2)}$."),
        misconceptions=[([fmt(l2), fmt(w)], "swapped-adamw-l2")],
    )


# --------------------------------------------------------------------------- regularization
def _regularization_base(rng, level):
    if level == 1:
        xs = rng.sample([-3, -2, -1, 1, 2, 3], 3)
        ys = [rng.randint(-6, 9) for _ in xs]
        lam = rng.randint(1, 6)
        sxy, sxx = sum(a * b for a, b in zip(xs, ys)), sum(a * a for a in xs)
        while sxy == 0:
            ys = [rng.randint(-6, 9) for _ in xs]
            sxy = sum(a * b for a, b in zip(xs, ys))
        ans = R(sxy, sxx + lam)
        pts = ", ".join(f"({a}, {b})" for a, b in zip(xs, ys))
        return Problem(
            prompt=(f"Ridge regression through the origin: minimise $\\sum_i (y_i - w x_i)^2 + {lam} w^2$ for the points {pts}. "
                    "Give $\\hat w$."),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["Differentiate: -2Σx(y - wx) + 2λw = 0", "ŵ = Σxy / (Σx² + λ)", "Compare with OLS: shrunk toward 0"],
            solution=f"$\\hat w = \\frac{{{sxy}}}{{{sxx} + {lam}}} = {L(ans)}$ (OLS would give ${L(R(sxy, sxx))}$).",
            misconceptions=[(fmt(R(sxy, sxx)), "forgot-penalty")],
        )
    if level == 2:
        lam = rng.choice([R(1, 2), 1, R(3, 2), 2])
        while True:
            z = [R(rng.randint(-8, 8), 2) for _ in range(4)]
            soft = [sp.sign(v) * max(abs(v) - lam, 0) for v in z]
            if any(s == 0 for s in soft) and any(s != 0 for s in soft):
                break
        hard = [v if abs(v) > lam else 0 for v in z]
        ridge = [v / (1 + lam) for v in z]
        return Problem(
            prompt=(f"Lasso with an orthonormal design reduces to solving, for each coordinate, "
                    f"$\\min_w \\tfrac12 (w - z)^2 + \\lambda |w|$ with $\\lambda = {L(sp.nsimplify(lam))}$. "
                    f"Apply it to $z = ({', '.join(L(v) for v in z)})$ and give the solution vector."),
            answer=vec(soft), kind="vector", answer_hint=VEC_HINT,
            rubric=["Subgradient condition: w - z + λ·∂|w| ∋ 0", "If |z| ≤ λ the solution is exactly 0",
                    "Otherwise w = z - λ·sign(z) (soft-thresholding)"],
            solution=f"$w = \\operatorname{{sign}}(z)\\max(|z|-\\lambda,0) = ({', '.join(L(v) for v in soft)})$.",
            misconceptions=[(vec(hard), "hard-threshold"), (vec(ridge), "ridge-shrinkage-not-soft-threshold")],
        )
    while True:
        nrow = rng.choice([3, 4])
        X = sp.Matrix(nrow, 2, [rng.randint(-2, 3) for _ in range(2 * nrow)])
        yv = sp.Matrix([rng.randint(-4, 6) for _ in range(nrow)])
        lam = rng.randint(1, 3)
        M = X.T * X + lam * sp.eye(2)
        w = M.inv() * X.T * yv
        if w != sp.zeros(2, 1) and max(abs(sp.denom(v)) for v in w) <= 80:
            break
    mis = []
    if (X.T * X).det() != 0:
        mis.append((vec((X.T * X).inv() * X.T * yv), "forgot-penalty"))
    return Problem(
        prompt=(f"Ridge regression: minimise $\\|X\\mathbf w - \\mathbf y\\|^2 + {lam}\\|\\mathbf w\\|^2$ with $X = {L(X)}$, "
                f"$\\mathbf y = {L(yv)}$. Derive the closed form and give $\\hat{{\\mathbf w}}$."),
        answer=vec(w), kind="vector", answer_hint=VEC_HINT,
        rubric=["Gradient: 2Xᵀ(Xw - y) + 2λw = 0", "ŵ = (XᵀX + λI)^{-1} Xᵀy", "Compute XᵀX + λI and Xᵀy", "Invert the 2×2 and multiply"],
        solution=f"$X^\\top X + {lam}I = {L(M)}$, $X^\\top\\mathbf y = {L(X.T * yv)}$, $\\hat{{\\mathbf w}} = {L(w)}$.",
        misconceptions=mis,
    )


# ===========================================================================
# Additional problem types (variety): each skill-level draws from several genuinely different tasks.
# The original generators above are kept as the first variant of each skill; the @gen-registered
# dispatchers at the bottom keep the original function names / gen_ids.
# ===========================================================================
z = sp.Symbol("z")
THETAS = [R(1, 2), R(1, 3), R(1, 4), R(2, 3), R(3, 4)]
PYTH = [(3, 4), (5, 12), (6, 8), (8, 15), (4, 3), (12, 5), (8, 6), (15, 8)]


def dec(r) -> str:
    return f"{float(r):g}"


def soft(v, lam):
    return sp.sign(v) * sp.Max(sp.Abs(v) - lam, 0)


def pts_str(xs, ys) -> str:
    return ", ".join(f"$({a}, {b})$" for a, b in zip(xs, ys))


def tup(vals) -> str:
    return "(" + ", ".join(L(sp.nsimplify(v)) for v in vals) + ")"


# --------------------------------------------------------------------------- convexity variants
def _conv_inflection(rng, level):
    while True:
        a, p, q = rng.choice([1, 2, 3]), nz(rng, -9, 9), rng.randint(-6, 6)
        f = a * x ** 3 + p * x ** 2 + q * x
        f2 = sp.diff(f, x, 2)
        xs = sp.solve(f2, x)[0]
        if sp.denom(xs) <= 9:
            break
    return Problem(
        prompt=(f"Find where the cubic $f(x) = {L(f)}$ switches from concave to convex: it is convex exactly on "
                "$[x^\\ast, \\infty)$. Give $x^\\ast$."),
        answer=fmt(xs), kind="num", answer_hint=FRAC_HINT,
        rubric=[f"Differentiate twice: f''(x) = {sp.sstr(f2)}", "f is convex where f''(x) ≥ 0", "Solve f''(x*) = 0 (the inflection point)"],
        solution=f"$f''(x) = {L(f2)} \\ge 0 \\iff x \\ge {L(xs)}$, so $x^\\ast = {L(xs)}$.",
        misconceptions=[(fmt(-xs), "sign-error"), (fmt(R(-2 * p, 3 * a)), "power-rule-forgot-multiply")],
    )


def _conv_chord(rng, level):
    a, b = rng.randint(1, 4), rng.randint(-5, 5)
    x1, x2 = rng.sample(range(-4, 6), 2)
    th = rng.choice(THETAS)
    f = sp.Lambda(x, a * x ** 2 + b * x)
    mid = th * x1 + (1 - th) * x2
    chord = th * f(x1) + (1 - th) * f(x2)
    gap = sp.nsimplify(chord - f(mid))
    return Problem(
        prompt=(f"Test the chord (Jensen) inequality numerically for $f(x) = {L(f(x))}$ with $x_1 = {x1}$, $x_2 = {x2}$ and "
                f"$\\theta = {L(th)}$: compute the gap $\\theta f(x_1) + (1-\\theta) f(x_2) - f(\\theta x_1 + (1-\\theta) x_2)$."),
        answer=fmt(gap), kind="num", answer_hint=FRAC_HINT,
        rubric=["Compute the mixed point θx₁ + (1-θ)x₂", "Evaluate f at x₁, x₂ and at the mixed point",
                "Chord value minus function value", "Convex ⇒ the gap is ≥ 0 (here it equals aθ(1-θ)(x₁-x₂)²)"],
        solution=(f"Mixed point ${L(mid)}$; chord value ${L(chord)}$, $f({L(mid)}) = {L(f(mid))}$; gap $= {L(gap)}$ "
                  f"$= {a}\\cdot{L(th)}\\cdot{L(1 - th)}\\cdot({x1 - x2})^2 \\ge 0$."),
        misconceptions=[(fmt(-gap), "reversed-jensen-inequality")],
    )


def _conv_strong(rng, level):
    while True:
        a, c, b = rng.randint(1, 6), rng.randint(1, 6), nz(rng, -6, 6)
        H = sp.Matrix([[2 * a, b], [b, 2 * c]])
        ev = sorted(H.eigenvals().keys())
        if all(e.is_integer for e in ev) and ev[0] > 0:
            break
    f = a * x ** 2 + b * x * y + c * y ** 2
    return Problem(
        prompt=(f"A function is $m$-strongly convex when $\\nabla^2 f \\succeq mI$ everywhere. Find the largest such $m$ for "
                f"$f(x,y) = {L(f)}$."),
        answer=str(ev[0]), kind="num", answer_hint="an integer",
        rubric=["Hessian H = [[2a, b], [b, 2c]] (constant for a quadratic)", "H ⪰ mI ⇔ every eigenvalue of H is ≥ m",
                "Characteristic polynomial λ² - tr(H)λ + det(H) = 0", "m = λ_min(H)"],
        solution=f"$H = {L(H)}$ has eigenvalues ${ev[0]}, {ev[1]}$, so the largest $m$ is $\\lambda_{{\\min}} = {ev[0]}$.",
        misconceptions=[(str(min(2 * a, 2 * c)), "used-diagonal-not-eigenvalue"), (str(ev[1]), "used-lambda-max"),
                        (str(min(a, c)), "forgot-factor-two")],
    )


def _conv_quartic_interval(rng, level):
    while True:
        p, q = sorted(rng.sample(range(-3, 5), 2))
        if p + q != 0 or p * q != 0:
            break
    f = x ** 4 - 2 * (p + q) * x ** 3 + 6 * p * q * x ** 2
    f2 = sp.factor(sp.diff(f, x, 2))
    return Problem(
        prompt=(f"Locate the non-convex part of the quartic $f(x) = {L(f)}$: it is convex everywhere except on one open "
                "interval $(\\alpha, \\beta)$. Give $\\alpha, \\beta$."),
        answer=[str(p), str(q)], kind="list", answer_hint="alpha, beta   e.g. -1, 3",
        rubric=["Differentiate twice", "Factor f''(x) as 12(x - α)(x - β)", "f'' < 0 strictly between its roots (upward parabola)"],
        solution=f"$f''(x) = {L(f2)}$, negative exactly for ${p} < x < {q}$.",
        misconceptions=[([str(q), str(p)], "swapped-endpoints")],
    )


def _conv_log_bump(rng, level):
    a = rng.randint(1, 5)
    c, u = sp.symbols("c u", real=True)
    h = 2 * (1 - u) / (1 + u) ** 2  # second derivative of ln(1+x^2) with u = x^2
    ustar = [s for s in sp.solve(sp.diff(h, u), u) if s >= 0][0]
    hmin = h.subs(u, ustar)
    ans = sp.solve(2 * a + c * hmin, c)[0]
    return Problem(
        prompt=(f"For which $c \\ge 0$ is $f(x) = {L(a * x ** 2)} + c\\ln(1+x^2)$ convex on all of $\\mathbb R$? "
                "Type the **largest** such $c$."),
        answer=fmt(ans), kind="num", answer_hint="an integer",
        rubric=["f''(x) = 2a + c·h(x) with h(x) = 2(1 - x²)/(1 + x²)²", "For c ≥ 0 the binding case is the minimum of h",
                "Substitute u = x², solve h'(u) = 0 ⇒ u = 3, h_min = -1/4", "Need 2a + c·h_min ≥ 0"],
        solution=(f"$h(x) = \\frac{{2(1-x^2)}}{{(1+x^2)^2}}$ is smallest at $x^2 = {ustar}$ with $h = {L(hmin)}$. "
                  f"Convex iff ${2 * a} - \\tfrac{{c}}{{4}} \\ge 0$, so $c_{{\\max}} = {L(ans)}$."),
        misconceptions=[(fmt(2 * a), "wrong-minimum-of-h"), (fmt(4 * a), "wrong-minimum-of-h")],
    )


def _conv_three_var(rng, level):
    a = rng.randint(1, 4)
    k = sp.Symbol("k")
    H = sp.Matrix(3, 3, lambda i, j_: 2 * a if i == j_ else k)
    evs = list(H.eigenvals().keys())
    bounds = sorted(sp.solve(ev, k)[0] for ev in evs)
    a_str = "" if a == 1 else str(a)
    return Problem(
        prompt=(f"For which $k$ is $f(x,y,z) = {a_str}(x^2+y^2+z^2) + k\\,(xy+yz+zx)$ convex on $\\mathbb R^3$? "
                "The answer is an interval $[k_{\\min}, k_{\\max}]$; give both ends."),
        answer=[fmt(bounds[0]), fmt(bounds[-1])], kind="list", answer_hint="k_min, k_max   e.g. -2, 4",
        rubric=["Hessian = 2a·I + k(J - I) (J = all-ones matrix)", "Eigenvalues of J are 3, 0, 0",
                "So the Hessian has eigenvalues 2a + 2k (once) and 2a - k (twice)", "Require both ≥ 0"],
        solution=(f"$\\nabla^2 f = {L(H)}$ has eigenvalues ${L(evs[0])}$ and ${L(evs[1])}$; both $\\ge 0$ iff "
                  f"${L(bounds[0])} \\le k \\le {L(bounds[-1])}$."),
        misconceptions=[([fmt(-2 * a), fmt(2 * a)], "checked-only-2x2-minors")],
    )


# --------------------------------------------------------------------------- gradient-descent variants
def _gd_landing_step(rng, level):
    while True:
        a, b, x0 = rng.randint(1, 6), rng.randint(-8, 8), rng.randint(-5, 5)
        f = a * x ** 2 + b * x
        xstar = R(-b, 2 * a)
        if x0 != xstar:
            break
    g0 = sp.diff(f, x).subs(x, x0)
    eta = sp.Symbol("eta")
    ans = sp.solve(x0 - eta * g0 - xstar, eta)[0]
    return Problem(
        prompt=(f"Find the learning rate $\\eta$ for which a single gradient-descent step from $x_0 = {x0}$ on "
                f"$f(x) = {L(f)}$ lands **exactly** on the minimiser."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Minimiser: f'(x*) = 0", "One step: x_1 = x_0 - η f'(x_0)", "Set x_1 = x* and solve for η",
                "Observe η = 1/f'' — independent of x_0 for a quadratic"],
        solution=(f"$x^\\ast = {L(xstar)}$, $f'({x0}) = {g0}$; ${x0} - \\eta\\cdot({g0}) = {L(xstar)} \\Rightarrow \\eta = {L(ans)}$ "
                  "$= 1/f''$."),
        misconceptions=[(fmt(R(1, a)), "forgot-factor-two"), (fmt(R(2, a)), "used-stability-threshold")],
    )


def _gd_value_convergence(rng, level):
    while True:
        a, c, d = rng.randint(1, 3), rng.randint(-3, 3), rng.randint(-5, 5)
        eta = R(1, rng.choice([4, 5, 8, 10]))
        x0, k = rng.randint(-4, 4), rng.randint(3, 4)
        r = 1 - 2 * a * eta
        if x0 != c and r != 0 and abs(r) < 1:
            break
    f = a * (x - c) ** 2 + d
    f0 = f.subs(x, x0) - d
    ans = r ** (2 * k) * f0
    return Problem(
        prompt=(f"Function-value convergence: gradient descent with $\\eta = {L(eta)}$ on $f(x) = {L(f)}$ from $x_0 = {x0}$. "
                f"Derive a closed form for the optimality gap $f(x_t) - f^\\ast$ and give it for $t = {k}$."),
        answer=fmt(ans), kind="num", answer_hint="an exact fraction, e.g. 16/625",
        rubric=["f* = min f and x* from the vertex form", "x_t - x* = (1 - 2aη)^t (x_0 - x*)",
                "f(x_t) - f* = a(x_t - x*)² = (1 - 2aη)^{2t}(f(x_0) - f*)", f"Evaluate at t = {k}"],
        solution=(f"$f^\\ast = {d}$, $f(x_0) - f^\\ast = {f0}$, factor $1 - 2a\\eta = {L(r)}$; "
                  f"$f(x_{{{k}}}) - f^\\ast = ({L(r)})^{{{2 * k}}}\\cdot {f0} = {L(ans)}$."),
        misconceptions=[(fmt(r ** k * f0), "forgot-square-in-function-value"), (fmt(ans + d), "gave-f-not-gap")],
    )


def _gd_exact_line_search(rng, level):
    while True:
        a, b, c = rng.randint(1, 4), rng.randint(1, 4), rng.randint(-3, 3)
        x0, y0 = rng.randint(-3, 3), rng.randint(-3, 3)
        if c * c >= 4 * a * b:
            continue
        f = a * x ** 2 + b * y ** 2 + c * x * y
        g = sp.Matrix([sp.diff(f, v).subs({x: x0, y: y0}) for v in (x, y)])
        if g == sp.zeros(2, 1):
            continue
        H = sp.hessian(f, (x, y))
        eta = sp.Symbol("eta")
        phi = f.subs({x: x0 - eta * g[0], y: y0 - eta * g[1]})
        ans = sp.solve(sp.diff(phi, eta), eta)[0]
        if sp.denom(ans) <= 150:
            break
    gg, gHg = (g.T * g)[0], (g.T * H * g)[0]
    return Problem(
        prompt=(f"Exact line search: from $(x_0, y_0) = ({x0}, {y0})$ on $f(x,y) = {L(f)}$, choose the step that minimises "
                "$\\varphi(\\eta) = f(\\mathbf x_0 - \\eta\\nabla f(\\mathbf x_0))$. Give $\\eta^\\ast$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Gradient g = ∇f(x_0)", "φ(η) = f(x_0) - η gᵀg + ½η² gᵀHg (exact for a quadratic)",
                "φ'(η) = 0 ⇒ η* = gᵀg / gᵀHg", "Compute H and the two scalars"],
        solution=(f"$\\mathbf g = {L(g)}$, $H = {L(H)}$, $\\mathbf g^\\top\\mathbf g = {gg}$, $\\mathbf g^\\top H\\mathbf g = {gHg}$, "
                  f"so $\\eta^\\ast = {L(ans)}$."),
        misconceptions=[(fmt(ans / 2), "dropped-half-in-taylor"), (fmt(2 * ans), "dropped-half-in-taylor")],
    )


# --------------------------------------------------------------------------- conditioning variants
def _cond_kappa_diag(rng, level):
    while True:
        cs = [rng.randint(1, 12) for _ in range(3)]
        if len(set(cs)) == 3:
            break
    f = cs[0] * x ** 2 + cs[1] * y ** 2 + cs[2] * z ** 2
    ev = [2 * v for v in cs]
    kappa = R(max(ev), min(ev))
    return Problem(
        prompt=f"Find the condition number $\\kappa = \\lambda_{{\\max}}/\\lambda_{{\\min}}$ of the Hessian of $f(x,y,z) = {L(f)}$.",
        answer=fmt(kappa), kind="num", answer_hint=FRAC_HINT,
        rubric=["Hessian is diagonal: diag(2c₁, 2c₂, 2c₃)", "Eigenvalues of a diagonal matrix = diagonal entries",
                "κ = largest / smallest (the factor 2 cancels)"],
        solution=f"$\\nabla^2 f = \\operatorname{{diag}}({', '.join(map(str, ev))})$, $\\kappa = {max(ev)}/{min(ev)} = {L(kappa)}$.",
        misconceptions=[(fmt(1 / kappa), "inverted-condition-number")],
    )


def _cond_contraction(rng, level):
    while True:
        l1, l2 = sorted(rng.sample(range(1, 13), 2))
        m = rng.choice([4, 5, 8, 10, 12, 16])
        eta = R(1, m)
        facs = [1 - eta * l1, 1 - eta * l2]
        rho = max(abs(v) for v in facs)
        if eta * l2 < 2:
            break
    return Problem(
        prompt=(f"Each GD step on $f(x,y) = \\tfrac{{{l1}}}{{2}}x^2 + \\tfrac{{{l2}}}{{2}}y^2$ multiplies coordinate $i$ by "
                f"$1 - \\eta\\lambda_i$. With $\\eta = {L(eta)}$, what is the worst-case contraction factor "
                "$\\rho = \\max_i|1 - \\eta\\lambda_i|$?"),
        answer=fmt(rho), kind="num", answer_hint=FRAC_HINT,
        rubric=["λ = curvatures = coefficients of x²/2 and y²/2", "Compute 1 - ηλ for each", "Take absolute values", "ρ = the larger one"],
        solution=f"$1 - \\eta\\lambda = {L(facs[0])}, {L(facs[1])}$; $\\rho = {L(rho)}$.",
        misconceptions=[(fmt(min(abs(v) for v in facs)), "used-best-coordinate"), (fmt(max(facs)), "forgot-absolute-value")],
    )


def _cond_int_eig(rng):
    while True:
        a, d, b = rng.randint(1, 10), rng.randint(1, 10), rng.randint(1, 5)
        s = sp.sqrt((a - d) ** 2 + 4 * b * b)
        if s.is_integer and a * d - b * b > 0 and (a + d + s) % 2 == 0:
            return sp.Matrix([[a, b], [b, d]]), sp.Integer((a + d - s) / 2), sp.Integer((a + d + s) / 2)


def _cond_one_over_L(rng, level):
    A, lmin, lmax = _cond_int_eig(rng)
    kappa = lmax / lmin
    rho = 1 - 1 / kappa
    return Problem(
        prompt=(f"The common step size $\\eta = 1/L$ ($L = \\lambda_{{\\max}}$) on $f(\\mathbf x) = \\tfrac12\\mathbf x^\\top A\\mathbf x$, "
                f"$A = {L(A)}$: compute the condition number $\\kappa$ of $A$ and the per-step contraction factor "
                "$\\rho = \\max_i |1 - \\lambda_i/L|$."),
        answer=[fmt(kappa), fmt(rho)], kind="list", answer_hint="kappa, rho   e.g. 4, 3/4",
        rubric=["Eigenvalues from λ² - tr(A)λ + det(A) = 0", "κ = λ_max/λ_min",
                "With η = 1/L the slowest coordinate shrinks by 1 - λ_min/λ_max", "ρ = 1 - 1/κ"],
        solution=(f"$\\lambda = {lmin}, {lmax}$, $\\kappa = {L(kappa)}$, $\\rho = 1 - {L(1 / kappa)} = {L(rho)}$ "
                  "(the optimal step would give $(\\kappa-1)/(\\kappa+1)$)."),
        misconceptions=[([fmt(kappa), fmt((kappa - 1) / (kappa + 1))], "used-optimal-step-rate"),
                        ([fmt(1 / kappa), fmt(rho)], "inverted-condition-number")],
    )


def _cond_rescale(rng, level):
    p0, s0 = rng.randint(1, 4), rng.randint(2, 4)
    if rng.random() < 0.5:
        p, q, s = p0, p0 * s0 * s0, sp.Integer(s0)
    else:
        p, q, s = p0 * s0 * s0, p0, R(1, s0)
    eta = R(1, 2 * q)
    pc = "" if p == 1 else str(p)
    return Problem(
        prompt=(f"Preconditioning by a change of variables: in $f(x,y) = {L(p * x ** 2 + q * y ** 2)}$ substitute $x = s\\,u$ "
                "($s > 0$) so that the Hessian in $(u, y)$ is a multiple of $I$. Give $s$, and the step $\\eta$ for which GD in "
                "$(u, y)$ reaches the minimum in one step."),
        answer=[fmt(s), fmt(eta)], kind="list", answer_hint="s, eta   e.g. 3, 1/18",
        rubric=["In (u, y): f = p s² u² + q y²", "Equal curvatures: p s² = q ⇒ s = √(q/p)", "Then the Hessian is 2q·I, κ = 1",
                "One-step GD needs η = 1/(2q)"],
        solution=(f"${pc}s^2 = {q} \\Rightarrow s = {L(s)}$; $f = {q}(u^2 + y^2)$, Hessian ${2 * q}I$, so $\\eta = {L(eta)}$."),
        misconceptions=[([fmt(1 / s), fmt(eta)], "inverted-scaling"), ([fmt(s), fmt(R(1, q))], "forgot-factor-two")],
    )


def _cond_zigzag(rng, level):
    kap = rng.choice([2, 3, 4, 5, 9])
    k = rng.choice([2, 3])
    Q = sp.diag(1, kap)
    w = sp.Matrix([kap, 1])
    for _ in range(k):
        g = Q * w
        w = w - ((g.T * g)[0] / (g.T * Q * g)[0]) * g
    rho = R(kap - 1, kap + 1)
    mis = [(vec([kap * rho, -rho]), "stopped-early")]
    if k % 2:
        mis.append((vec([kap * rho ** k, rho ** k]), "missed-zigzag-sign"))
    return Problem(
        prompt=(f"Steepest descent with exact line search zigzags on ill-conditioned problems. Run it on "
                f"$f(x,y) = \\tfrac12(x^2 + {kap}y^2)$ from $({kap}, 1)$: derive a closed form for the iterates "
                f"(compute two steps and spot the pattern) and give $(x_{{{k}}}, y_{{{k}}})$."),
        answer=vec(w), kind="vector", answer_hint=VEC_HINT,
        rubric=["Exact step η = gᵀg / gᵀQg", "First step: x_1 = κρ, y_1 = -ρ with ρ = (κ-1)/(κ+1)",
                "Pattern: x_k = κρ^k, y_k = (-ρ)^k", "The sign of y flips each step (zigzag)"],
        solution=(f"$\\rho = {L(rho)}$; $x_k = {kap}\\rho^k$, $y_k = (-\\rho)^k$, so $(x_{{{k}}}, y_{{{k}}}) = "
                  f"({L(w[0])}, {L(w[1])})$."),
        misconceptions=mis,
    )


def _cond_heavy_ball(rng, level):
    l1, l2 = rng.choice([(1, 4), (1, 9), (2, 8), (1, 16), (4, 9), (3, 12), (2, 18), (4, 25)])
    kap = R(l2, l1)
    sk = sp.sqrt(kap)
    rgd, rhb = (kap - 1) / (kap + 1), (sk - 1) / (sk + 1)
    return Problem(
        prompt=(f"Momentum beats plain GD on ill-conditioned quadratics. For $f(\\mathbf x) = \\tfrac12\\mathbf x^\\top A\\mathbf x$ with "
                f"eigenvalues $\\lambda = {l1}, {l2}$: derive the best contraction factor of GD with a fixed step, and compare with "
                "optimally tuned heavy-ball momentum, whose factor is $(\\sqrt\\kappa-1)/(\\sqrt\\kappa+1)$. Give both factors."),
        answer=[fmt(rgd), fmt(rhb)], kind="list", answer_hint="rho_GD, rho_HB   e.g. 3/5, 1/3",
        rubric=["GD multiplies eigencomponent i by 1 - ηλ_i", "Balance |1 - ηλ_min| = |1 - ηλ_max| ⇒ η = 2/(λ_min+λ_max)",
                "ρ_GD = (κ-1)/(κ+1)", "ρ_HB = (√κ-1)/(√κ+1)", "On paper: iterations scale like κ vs √κ"],
        solution=f"$\\kappa = {L(kap)}$, $\\rho_{{GD}} = {L(rgd)}$, $\\sqrt\\kappa = {L(sk)}$, $\\rho_{{HB}} = {L(rhb)}$.",
        misconceptions=[([fmt(rhb), fmt(rgd)], "swapped-methods"), ([fmt(1 - 1 / kap), fmt(rhb)], "used-1-over-L-step")],
    )


# --------------------------------------------------------------------------- least-squares variants
def _ls_weighted_mean(rng, level):
    while True:
        ys = [rng.randint(5, 20) for _ in range(3)]
        ws = [rng.randint(1, 4) for _ in range(3)]
        c = sp.Symbol("c")
        ans = sp.solve(sp.diff(sum(w_ * (yy - c) ** 2 for w_, yy in zip(ws, ys)), c), c)[0]
        if len(set(ws)) > 1 and ans != R(sum(ys), 3):
            break
    return Problem(
        prompt=(f"Three labs measure the same quantity and report ${', '.join(map(str, ys))}$ with reliability weights "
                f"${', '.join(map(str, ws))}$. Estimate it by weighted least squares, minimising $\\sum_i w_i (y_i - c)^2$. Give $\\hat c$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Differentiate: -2Σ w_i (y_i - c) = 0", "ĉ = Σ w_i y_i / Σ w_i", "Evaluate"],
        solution=f"$\\hat c = \\frac{{{sum(w_ * yy for w_, yy in zip(ws, ys))}}}{{{sum(ws)}}} = {L(ans)}$.",
        misconceptions=[(fmt(R(sum(ys), 3)), "ignored-weights")],
    )


def _ls_project_line(rng, level):
    dim = rng.choice([2, 3])
    while True:
        a = sp.Matrix([rng.randint(-3, 3) for _ in range(dim)])
        b = sp.Matrix([rng.randint(-5, 5) for _ in range(dim)])
        if a.dot(a) != 0 and a.dot(b) != 0 and sp.Matrix.hstack(a, b).rank() == 2:
            break
    p = (a.dot(b) / a.dot(a)) * a
    return Problem(
        prompt=(f"Project $\\mathbf b = {tup(b)}$ onto the line spanned by $\\mathbf a = {tup(a)}$ "
                "(the least-squares approximation of $\\mathbf b$ by a multiple $t\\mathbf a$). Give the projection $\\mathbf p$."),
        answer=vec(p), kind="vector", answer_hint=VEC_HINT,
        rubric=["Minimise ‖b - t a‖² ⇒ t = a·b / a·a", "p = t a", "Check: b - p ⟂ a"],
        solution=(f"$t = \\frac{{\\mathbf a\\cdot\\mathbf b}}{{\\mathbf a\\cdot\\mathbf a}} = \\frac{{{a.dot(b)}}}{{{a.dot(a)}}} = "
                  f"{L(a.dot(b) / a.dot(a))}$, $\\mathbf p = t\\,\\mathbf a = {tup(p)}$."),
        misconceptions=[(vec((a.dot(b) / b.dot(b)) * b), "projected-wrong-way"), (vec(a.dot(b) * a), "forgot-normalise")],
    )


def _ls_min_sse(rng, level):
    while True:
        xs = rng.sample([-3, -2, -1, 1, 2, 3], 3)
        ys = [rng.randint(-6, 9) for _ in xs]
        sxy, sxx, syy = (sum(a * b for a, b in zip(xs, ys)), sum(a * a for a in xs), sum(b * b for b in ys))
        sse = syy - R(sxy * sxy, sxx)
        if sxy != 0 and sse != 0:
            break
    w = R(sxy, sxx)
    return Problem(
        prompt=(f"How well can a line through the origin fit the data? Fit $y = wx$ by least squares to {pts_str(xs, ys)} "
                "and give the minimum residual sum of squares $\\sum_i (y_i - \\hat w x_i)^2$."),
        answer=fmt(sse), kind="num", answer_hint=FRAC_HINT,
        rubric=["ŵ = Σxy / Σx²", "Residuals y_i - ŵ x_i (or use SSE = Σy² - (Σxy)²/Σx²)", "Sum the squares"],
        solution=(f"$\\hat w = {L(w)}$; $\\text{{SSE}} = \\sum y^2 - \\frac{{(\\sum xy)^2}}{{\\sum x^2}} = {syy} - "
                  f"\\frac{{{sxy * sxy}}}{{{sxx}}} = {L(sse)}$."),
        misconceptions=[(fmt(syy), "forgot-to-subtract-explained"), (fmt(w), "gave-slope-not-sse")],
    )


def _ls_survey(rng, level):
    while True:
        a, b, c = rng.randint(10, 40), rng.randint(10, 40), rng.randint(20, 80)
        if c != a + b:
            break
    A = sp.Matrix([[1, 0], [0, 1], [1, 1]])
    bb = sp.Matrix([a, b, c])
    sol = (A.T * A).inv() * A.T * bb
    return Problem(
        prompt=(f"A surveyor measures two lengths $u$ and $v$ separately, getting ${a}$ and ${b}$, and then measures them laid end "
                f"to end, getting ${c}$. The measurements are inconsistent; find the least-squares estimates $\\hat u, \\hat v$."),
        answer=vec(sol), kind="list", answer_hint="u, v   exact fractions, e.g. 31/3, 44/3",
        rubric=["Write the overdetermined system: u = a, v = b, u + v = c", "A = [[1,0],[0,1],[1,1]]; AᵀA = [[2,1],[1,2]]",
                "Aᵀb = (a + c, b + c)", "Solve the 2×2 normal equations"],
        solution=(f"$A^\\top A = {L(A.T * A)}$, $A^\\top\\mathbf b = {L(A.T * bb)}$, so $\\hat u = {L(sol[0])}$, $\\hat v = {L(sol[1])}$ "
                  "(each estimate moves by the same amount toward consistency)."),
        misconceptions=[([str(a), str(b)], "ignored-third-equation"), ([fmt(sol[1]), fmt(sol[0])], "swapped-coordinates")],
    )


def _ls_through_point(rng, level):
    while True:
        p, q = rng.randint(-2, 2), rng.randint(-3, 3)
        xs = sorted(rng.sample(range(-2, 6), 4))
        ys = [rng.randint(-5, 9) for _ in xs]
        sxx = sum((a - p) ** 2 for a in xs)
        b1 = R(sum((a - p) * (b - q) for a, b in zip(xs, ys)), sxx) if sxx else None
        X = sp.Matrix([[1, v] for v in xs])
        ols = ((X.T * X).inv() * X.T * sp.Matrix(ys))[1]
        if b1 is not None and b1 != ols and sp.denom(b1) <= 40:
            break
    return Problem(
        prompt=(f"Constrained regression: fit a line $y = q + \\beta(x - p)$ that is **forced** to pass through $(p, q) = ({p}, {q})$ "
                f"to the data {pts_str(xs, ys)} by least squares. Derive $\\hat\\beta$ and give its value."),
        answer=fmt(b1), kind="num", answer_hint=FRAC_HINT,
        rubric=["Shift coordinates: X = x - p, Y = y - q", "The problem becomes a line through the origin: Y ≈ βX",
                "β̂ = Σ(x - p)(y - q) / Σ(x - p)²", "Evaluate"],
        solution=(f"$\\sum (x-p)(y-q) = {sum((a - p) * (b - q) for a, b in zip(xs, ys))}$, $\\sum (x-p)^2 = {sxx}$, "
                  f"$\\hat\\beta = {L(b1)}$ (the unconstrained OLS slope is ${L(ols)}$)."),
        misconceptions=[(fmt(ols), "ignored-constraint")],
    )


def _ls_loo(rng, level):
    while True:
        xs = rng.sample([-3, -2, -1, 1, 2, 3], 4)
        ys = [rng.randint(-6, 9) for _ in xs]
        kk = rng.randrange(4)
        sxx = sum(a * a for a in xs)
        sxy = sum(a * b for a, b in zip(xs, ys))
        w = R(sxy, sxx)
        w_loo = R(sxy - xs[kk] * ys[kk], sxx - xs[kk] ** 2)
        loo = ys[kk] - w_loo * xs[kk]
        e = ys[kk] - w * xs[kk]
        if e != 0 and sp.denom(loo) <= 30:
            break
    h = R(xs[kk] ** 2, sxx)
    return Problem(
        prompt=(f"Leave-one-out validation: fit $y = wx$ by least squares to the points {pts_str(xs, ys)} **except** point "
                f"${kk + 1}$, i.e. $({xs[kk]}, {ys[kk]})$, then give its prediction error $y_{kk + 1} - \\hat w_{{(-{kk + 1})}} x_{kk + 1}$. "
                "On paper: show it equals $e_k/(1 - h_{kk})$ with leverage $h_{kk} = x_k^2/\\sum x^2$."),
        answer=fmt(loo), kind="num", answer_hint=FRAC_HINT,
        rubric=["Refit without the point: ŵ₍₋ₖ₎ = (Σxy - x_k y_k)/(Σx² - x_k²)", "Prediction error y_k - ŵ₍₋ₖ₎ x_k",
                "Shortcut: in-sample residual e_k divided by 1 - h_kk", "Check both routes agree"],
        solution=(f"$\\hat w_{{(-{kk + 1})}} = {L(w_loo)}$, error $= {L(loo)}$. Check: $e_k = {L(e)}$, $h_{{kk}} = {L(h)}$, "
                  f"$e_k/(1-h_{{kk}}) = {L(e / (1 - h))}$."),
        misconceptions=[(fmt(e), "used-in-sample-residual")],
    )


# --------------------------------------------------------------------------- Lagrange variants
def _lag_two_machines(rng, level):
    a, b = rng.sample(range(1, 7), 2)
    c = rng.randint(2, 12)
    lam = sp.Symbol("lambda")
    sol = sp.solve([2 * a * x - lam, 2 * b * y - lam, x + y - c], [x, y, lam], dict=True)[0]
    cost = a * sol[x] ** 2 + b * sol[y] ** 2
    return Problem(
        prompt=(f"A load of ${c}$ units is split between two machines: $x$ units on machine 1 cost ${L(a * x ** 2)}$ and $y$ "
                f"units on machine 2 cost ${L(b * y ** 2)}$. Minimise the total cost subject to $x + y = {c}$; give the minimum cost."),
        answer=fmt(cost), kind="num", answer_hint=FRAC_HINT,
        rubric=["Lagrange: 2a x = λ, 2b y = λ (equal marginal costs)", "So x/y = b/a", "Use x + y = c to find x, y", "Evaluate the cost"],
        solution=(f"$x = {L(sol[x])}$, $y = {L(sol[y])}$ (equal marginal cost $\\lambda = {L(sol[lam])}$); minimum cost $= {L(cost)}$."),
        misconceptions=[(fmt((a + b) * R(c * c, 4)), "split-equally")],
    )


def _lag_river_fence(rng, level):
    P = rng.choice([40, 60, 80, 100, 120, 200, 240])
    lam = sp.Symbol("lambda")
    sol = sp.solve([y - 2 * lam, x - lam, 2 * x + y - P], [x, y, lam], dict=True)[0]
    area = sol[x] * sol[y]
    return Problem(
        prompt=(f"A farmer has ${P}$ m of fence to enclose a rectangular field along a straight river; the river side needs no fence. "
                "With $x$ the two sides perpendicular to the river and $y$ the side parallel to it, maximise the area $xy$. "
                "Give the maximum area."),
        answer=fmt(area), kind="num", answer_hint="a number of square metres, e.g. 450",
        rubric=["Constraint 2x + y = P", "∇(xy) = λ∇(2x + y): y = 2λ, x = λ", "So y = 2x ⇒ x = P/4, y = P/2", "Area P²/8"],
        solution=f"$x = {sol[x]}$, $y = {sol[y]}$, area $= {area}$ m².",
        misconceptions=[(fmt(R(P * P, 16)), "fenced-all-four-sides"), (fmt(R(P * P, 9)), "split-fence-in-thirds")],
    )


def _lag_open_box(rng, level):
    s = rng.choice([2, 4, 6, 8, 10])
    V = R(s ** 3, 2)
    h = sp.Symbol("h", positive=True)
    lam = sp.Symbol("lambda")
    xp, yp = sp.symbols("x y", positive=True)
    S = xp * yp + 2 * xp * h + 2 * yp * h
    eqs = [sp.diff(S, v) - lam * sp.diff(xp * yp * h, v) for v in (xp, yp, h)] + [xp * yp * h - V]
    sol = [d for d in sp.solve(eqs, [xp, yp, h, lam], dict=True) if d[xp] > 0][0]
    area = S.subs(sol)
    closed = 6 * V ** R(2, 3)
    return Problem(
        prompt=(f"An open-top box (no lid) with a rectangular base must hold ${L(V)}$ cubic units. Minimise the material "
                "used, $xy + 2xh + 2yh$, subject to $xyh = V$. Give the minimum surface area."),
        answer=fmt(area), kind="num", answer_hint="a number, e.g. 75",
        rubric=["Lagrange: y + 2h = λyh, x + 2h = λxh, 2x + 2y = λxy", "Symmetry of the first two ⇒ x = y",
                "Third equation ⇒ h = x/2", "Use xyh = V to get x, then evaluate the area"],
        solution=f"$x = y = {sol[xp]}$, $h = {sol[h]}$, area $= {L(area)}$.",
        misconceptions=[(fmt(closed), "included-a-lid")],
    )


def _lag_cobb_douglas(rng, level):
    al, be = rng.choice([(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3)])
    p, q = rng.randint(1, 5), rng.randint(1, 5)
    M = (al + be) * p * q * rng.randint(1, 4)
    ans = [R(al * M, (al + be) * p), R(be * M, (al + be) * q)]
    xp, yp = sp.symbols("x y", positive=True)
    U = xp ** al * yp ** be
    lam = sp.Symbol("lambda")
    chk = sp.solve([sp.diff(sp.log(U), xp) - lam * p, sp.diff(sp.log(U), yp) - lam * q, p * xp + q * yp - M], [xp, yp, lam], dict=True)[0]
    assert [chk[xp], chk[yp]] == ans
    return Problem(
        prompt=(f"A consumer with a budget of ${M}$ buys $x$ units of good A (price ${p}$) and $y$ units of good B (price ${q}$), "
                f"maximising the utility $U = {L(x ** al * y ** be)}$. Find the optimal bundle $(x, y)$."),
        answer=vec(ans), kind="vector", answer_hint="x, y   e.g. 4, 6",
        rubric=["Maximise ln U = α ln x + β ln y (same maximiser)", "α/x = λp, β/y = λq",
                "Budget px + qy = M ⇒ spend fraction α/(α+β) on A", "x = αM/((α+β)p), y = βM/((α+β)q)"],
        solution=f"Spend $\\tfrac{{{al}}}{{{al + be}}}$ of the budget on A: $x = {L(ans[0])}$, $y = {L(ans[1])}$.",
        misconceptions=[(vec([R(be * M, (al + be) * p), R(al * M, (al + be) * q)]), "swapped-exponents"),
                        (vec([R(M, 2 * p), R(M, 2 * q)]), "split-budget-equally")],
    )


def _lag_kkt_halfspace(rng, level):
    p, q = rng.choice([(1, 1), (1, 2), (2, 1), (1, -1), (2, 3)])
    while True:
        a, b = rng.randint(-3, 5), rng.randint(-3, 5)
        c = rng.randint(-4, 6)
        if p * a + q * b != c:
            break
    active = p * a + q * b > c
    if active:
        t = R(p * a + q * b - c, p * p + q * q)
        sol = [a - t * p, b - t * q, 2 * t]
    else:
        t = R(p * a + q * b - c, p * p + q * q)
        sol = [sp.Integer(a), sp.Integer(b), sp.Integer(0)]
    proj = [a - t * p, b - t * q, 2 * t]
    cons = L(p * x + q * y)
    return Problem(
        prompt=(f"KKT conditions with an inequality: minimise ${L((x - a) ** 2 + (y - b) ** 2)}$ subject to ${cons} \\le {c}$. "
                "Give the minimiser and the multiplier $\\mu \\ge 0$ of the constraint."),
        answer=vec(sol), kind="list", answer_hint="x, y, mu   e.g. 1/2, 3/2, 1",
        rubric=["Check the unconstrained minimiser (a, b) against the constraint", "If feasible: μ = 0 (inactive)",
                "If not: constraint active, 2(x - a) + μp = 0, 2(y - b) + μq = 0, px + qy = c",
                "Complementary slackness μ·(px + qy - c) = 0 and μ ≥ 0"],
        solution=(f"At the unconstrained minimiser $({a}, {b})$: ${cons} = {p * a + q * b}$ "
                  + (f"$> {c}$, so the constraint is active: point $= ({L(sol[0])}, {L(sol[1])})$, $\\mu = {L(sol[2])}$."
                     if active else f"$\\le {c}$: feasible, constraint inactive, $\\mu = 0$, minimiser $({a}, {b})$.")),
        misconceptions=[(vec([a, b, 0]), "ignored-constraint"), (vec(proj), "forced-constraint-active"),
                        (vec([sol[0], sol[1], -sol[2]]), "multiplier-sign-error")],
    )


def _lag_sensitivity(rng, level):
    ws = sorted(rng.sample([1, 2, 3, 4, 6], 3))
    c = rng.randint(2, 12)
    lam = sp.Symbol("lambda")
    cc = sp.Symbol("c")
    f = ws[0] * x ** 2 + ws[1] * y ** 2 + ws[2] * z ** 2
    sol = sp.solve([sp.diff(f, v) - lam for v in (x, y, z)] + [x + y + z - cc], [x, y, z, lam], dict=True)[0]
    fstar = f.subs(sol)
    lam_val = sol[lam].subs(cc, c)
    assert sp.simplify(sp.diff(fstar, cc) - sol[lam]) == 0
    return Problem(
        prompt=(f"The multiplier as a price: minimise $f = {L(f)}$ subject to $x + y + z = c$ with $c = {c}$. Find the Lagrange "
                "multiplier $\\lambda$ (convention $\\nabla f = \\lambda\\nabla g$) and, on paper, verify that "
                "$\\lambda = \\frac{d f^\\ast}{dc}$."),
        answer=fmt(lam_val), kind="num", answer_hint=FRAC_HINT,
        rubric=["2w_i x_i = λ ⇒ x_i = λ/(2w_i)", "Constraint: (λ/2)Σ1/w_i = c ⇒ λ = 2c/Σ(1/w_i)",
                "f* = c²/Σ(1/w_i)", "df*/dc = 2c/Σ(1/w_i) = λ: the multiplier is the marginal cost of the constraint"],
        solution=(f"$\\lambda(c) = {L(sol[lam])}$, $f^\\ast(c) = {L(sp.simplify(fstar))}$, and $\\frac{{df^\\ast}}{{dc}} = {L(sol[lam])}$. "
                  f"At $c = {c}$: $\\lambda = {L(lam_val)}$."),
        misconceptions=[(fmt(fstar.subs(cc, c)), "gave-optimal-value-not-lambda"), (fmt(lam_val / 2), "multiplier-scale-error")],
    )


# --------------------------------------------------------------------------- Newton variants
def _newton_log(rng, level):
    while True:
        a = rng.randint(2, 6)
        x0 = rng.choice([R(1, 2), R(1, 3), R(1, 4), sp.Integer(1), R(1, 5), R(1, 10)])
        f = a * x - sp.log(x)
        g, h = sp.diff(f, x).subs(x, x0), sp.diff(f, x, 2).subs(x, x0)
        ans = x0 - g / h
        if x0 != R(1, a) and ans > 0:
            break
    return Problem(
        prompt=f"Use one Newton step to minimise $f(x) = {a}x - \\ln x$ on $x > 0$, starting from $x_0 = {L(x0)}$. Give $x_1$.",
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["f'(x) = a - 1/x", "f''(x) = 1/x²", "x_1 = x_0 - f'(x_0)/f''(x_0) (simplifies to 2x_0 - a x_0²)"],
        solution=f"$f'({L(x0)}) = {L(g)}$, $f''({L(x0)}) = {L(h)}$, $x_1 = {L(x0)} - \\frac{{{L(g)}}}{{{L(h)}}} = {L(ans)}$.",
        misconceptions=[(fmt(x0 - g), "used-gradient-not-newton"), (fmt(x0 + g / h), "sign-error")],
    )


def _newton_2d(rng, level):
    while True:
        b, c = rng.randint(1, 3), nz(rng, -2, 2)
        x0, y0 = rng.choice([1, -1, 2]), rng.randint(-2, 2)
        f = x ** 4 / 4 + R(b, 2) * y ** 2 + c * x * y
        H = sp.hessian(f, (x, y)).subs({x: x0, y: y0})
        if H.det() > 0:
            break
    g = sp.Matrix([sp.diff(f, v).subs({x: x0, y: y0}) for v in (x, y)])
    w0 = sp.Matrix([x0, y0])
    w1 = w0 - H.inv() * g
    diag = w0 - sp.Matrix([g[0] / H[0, 0], g[1] / H[1, 1]])
    return Problem(
        prompt=(f"Take one 2-D Newton step $\\mathbf x_1 = \\mathbf x_0 - H^{{-1}}\\nabla f(\\mathbf x_0)$ on $f(x,y) = {L(f)}$ "
                f"from $(x_0, y_0) = ({x0}, {y0})$. Give $(x_1, y_1)$."),
        answer=vec(w1), kind="vector", answer_hint=VEC_HINT,
        rubric=["∇f = (x³ + c y, b y + c x)", "H = [[3x², c], [c, b]] at the start point", "Solve H d = ∇f (or invert the 2×2)",
                "x_1 = x_0 - d"],
        solution=f"$\\nabla f = {L(g)}$, $H = {L(H)}$, $\\mathbf x_1 = {L(w1)}$.",
        misconceptions=[(vec(w0 - g), "used-gradient-not-newton"), (vec(diag), "ignored-off-diagonal-hessian")],
    )


def _newton_to_max(rng, level):
    while True:
        a, x0 = rng.choice([4, 6, 9, 12]), rng.choice([1, -1])
        f = x ** 4 / 4 - R(a, 2) * x ** 2
        step = sp.Lambda(x, sp.simplify(x - sp.diff(f, x) / sp.diff(f, x, 2)))
        x1 = step(x0)
        if 3 * x1 ** 2 != a and x1 != 0:
            break
    x2 = step(x1)
    return Problem(
        prompt=(f"Newton's method only looks for $f'(x) = 0$. Apply it to $f(x) = {L(f)}$ from $x_0 = {x0}$ and give $x_2$. "
                "On paper: what kind of critical point are the iterates approaching, and why did that happen?"),
        answer=fmt(x2), kind="num", answer_hint=FRAC_HINT,
        rubric=["f'(x) = x³ - a x, f''(x) = 3x² - a", "Newton map x ← x - f'/f'' = 2x³/(3x² - a)", "Two iterations",
                "f''(x_0) < 0, so the local quadratic model is concave ⇒ heads to the local max at 0"],
        solution=f"$x \\leftarrow \\frac{{2x^3}}{{3x^2 - {a}}}$: $x_1 = {L(x1)}$, $x_2 = {L(x2)}$ → toward $0$, a local **maximum**.",
        misconceptions=[(fmt(x1), "stopped-early")],
    )


def _newton_logistic(rng, level):
    while True:
        xs = [nz(rng, -3, 3) for _ in range(4)]
        ys = [rng.choice([1, -1]) for _ in xs]
        if sum(a * b for a, b in zip(xs, ys)) != 0:
            break
    w = sp.Symbol("w")
    loss = sum(sp.log(1 + sp.exp(-yy * w * xx)) for xx, yy in zip(xs, ys))
    g0, h0 = sp.simplify(sp.diff(loss, w).subs(w, 0)), sp.simplify(sp.diff(loss, w, 2).subs(w, 0))
    ans = sp.nsimplify(-g0 / h0)
    data = ", ".join(f"$({xx}, {'+1' if yy > 0 else '-1'})$" for xx, yy in zip(xs, ys))
    return Problem(
        prompt=(f"Logistic regression with one weight and no bias: minimise $\\sum_i \\ln(1 + e^{{-y_i w x_i}})$ for the data "
                f"$(x_i, y_i)$ = {data}. Take one Newton step from $w_0 = 0$ and give $w_1$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["ℓ'(w) = -Σ y_i x_i σ(-y_i w x_i); at 0, σ = 1/2", "ℓ''(w) = Σ x_i² σ(1-σ); at 0 it is Σx_i²/4",
                "w_1 = -ℓ'(0)/ℓ''(0) = 2Σy_i x_i / Σx_i²"],
        solution=f"$\\ell'(0) = {L(g0)}$, $\\ell''(0) = {L(h0)}$, $w_1 = {L(ans)}$.",
        misconceptions=[(fmt(-g0), "used-gradient-not-newton"), (fmt(g0 / h0), "sign-error")],
    )


def _newton_error_recursion(rng, level):
    a = rng.randint(2, 9)
    e = sp.Symbol("err")
    newton_map = sp.expand(sp.cancel(x - sp.diff(a * x - sp.log(x), x) / sp.diff(a * x - sp.log(x), x, 2)))
    xnext = sp.expand(newton_map.subs(x, R(1, a) + e))
    C = sp.Poly(xnext - R(1, a), e).coeff_monomial(e ** 2)
    assert sp.expand(xnext - R(1, a) - C * e ** 2) == 0
    xmax = R(2, a)
    return Problem(
        prompt=(f"Quadratic convergence made explicit: Newton's method minimises $f(x) = {a}x - \\ln x$ ($x > 0$). With error "
                f"$e_k = x_k - x^\\ast$, show that $e_{{k+1}} = C e_k^2$ **exactly**. Give $C$, and the largest $x_{{\\max}}$ such that "
                "the method converges for every $0 < x_0 < x_{\\max}$."),
        answer=[fmt(C), fmt(xmax)], kind="list", answer_hint="C, x_max   e.g. -3, 2/3",
        rubric=["x* = 1/a", "Newton map: x ← x - (a - 1/x)·x² = 2x - a x²", "Substitute x = 1/a + e: next error = -a e²",
                "Converges iff |a e_0| < 1 ⇔ 0 < x_0 < 2/a"],
        solution=(f"$x_{{k+1}} = 2x_k - {a}x_k^2$; with $x_k = \\tfrac1{{{a}}} + e_k$: $e_{{k+1}} = {L(C)}e_k^2$. "
                  f"Then $|{a}e_{{k+1}}| = |{a}e_k|^2$, so convergence iff $|{a}e_0| < 1$, i.e. $0 < x_0 < {L(xmax)}$."),
        misconceptions=[([fmt(-C), fmt(xmax)], "sign-error"), ([fmt(C), fmt(R(1, a))], "confused-minimiser-with-boundary")],
    )


def _newton_flat_min(rng, level):
    p, x0, k = rng.choice([4, 6, 8]), rng.randint(1, 3), rng.randint(3, 5)
    step = sp.Lambda(x, sp.simplify(x - sp.diff(x ** p, x) / sp.diff(x ** p, x, 2)))
    xk = sp.Integer(x0)
    for _ in range(k):
        xk = step(xk)
    r = R(p - 2, p - 1)
    return Problem(
        prompt=(f"When Newton is only linear: minimise $f(x) = x^{{{p}}}$, whose minimiser $0$ has $f''(0) = 0$, with Newton's method "
                f"from $x_0 = {x0}$. Derive a closed form for $x_k$ and give $x_{{{k}}}$ exactly. On paper: why is the "
                "convergence not quadratic here?"),
        answer=fmt(xk), kind="num", answer_hint="an exact fraction, e.g. 16/81",
        rubric=["f'/f'' = p x^{p-1} / (p(p-1) x^{p-2}) = x/(p-1)", "Newton map x ← x(p-2)/(p-1)", "x_k = x_0 ((p-2)/(p-1))^k",
                "Degenerate minimum (f'' = 0) ⇒ linear rate"],
        solution=f"$x \\leftarrow {L(r)}\\,x$, so $x_{{{k}}} = {x0}\\cdot({L(r)})^{{{k}}} = {L(xk)}$.",
        misconceptions=[(fmt(x0 * r ** (k - 1)), "off-by-one-iterations"), (fmt(x0 * r ** (2 ** k)), "assumed-quadratic-convergence")],
    )


# --------------------------------------------------------------------------- SGD & momentum variants
def _sgd_momentum_bookkeeping(rng, level):
    beta = rng.choice([R(1, 2), R(9, 10), R(3, 4)])
    eta = R(1, rng.choice([2, 4, 5, 10]))
    xt, vt, g = rng.randint(-5, 5), nz(rng, -6, 6), nz(rng, -6, 6)
    v1 = beta * vt + g
    x1 = xt - eta * v1
    return Problem(
        prompt=(f"Momentum bookkeeping: at step $t$ an optimiser has $x_t = {xt}$, velocity $v_t = {vt}$, and sees gradient "
                f"$g_t = {g}$. With heavy-ball updates $v_{{t+1}} = \\beta v_t + g_t$, $x_{{t+1}} = x_t - \\eta v_{{t+1}}$, "
                f"$\\beta = {L(beta)}$, $\\eta = {L(eta)}$, give $v_{{t+1}}$ and $x_{{t+1}}$."),
        answer=[fmt(v1), fmt(x1)], kind="list", answer_hint="v, x   e.g. 3/2, -1/4",
        rubric=["Update the velocity first: β v_t + g_t", "Then move x by -η v_{t+1}"],
        solution=f"$v_{{t+1}} = {L(beta)}\\cdot({vt}) + ({g}) = {L(v1)}$, $x_{{t+1}} = {xt} - {L(eta)}\\cdot({L(v1)}) = {L(x1)}$.",
        misconceptions=[([fmt(g), fmt(xt - eta * g)], "ignored-momentum"), ([fmt(v1), fmt(xt + eta * v1)], "sign-error-gradient-step")],
    )


def _sgd_expected_step(rng, level):
    n = rng.choice([4, 5, 6])
    data = [rng.randint(-5, 9) for _ in range(n)]
    w0 = rng.randint(-3, 5)
    eta = R(1, rng.choice([2, 4, 5, 10]))
    ans = w0 - eta * (w0 - R(sum(data), n))
    return Problem(
        prompt=(f"Single-sample SGD on $\\mathcal L(w) = \\frac1{{{n}}}\\sum_i \\tfrac12 (w - a_i)^2$ with data $a = ({', '.join(map(str, data))})$: "
                f"one index $I$ is drawn uniformly at random and $w_1 = w_0 - \\eta(w_0 - a_I)$, with $w_0 = {w0}$, $\\eta = {L(eta)}$. "
                "What is $\\mathbb E[w_1]$?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Linearity: E[w_1] = w_0 - η(w_0 - E[a_I])", "E[a_I] = mean of the data (uniform index)",
                "So E[w_1] equals the full-batch GD step (SGD gradient is unbiased)"],
        solution=f"$\\mathbb E[a_I] = {L(R(sum(data), n))}$, $\\mathbb E[w_1] = {w0} - {L(eta)}\\cdot({L(w0 - R(sum(data), n))}) = {L(ans)}$.",
        misconceptions=[(fmt(w0 - eta * (w0 - data[0])), "used-one-example"), (fmt(w0 + eta * (w0 - R(sum(data), n))), "sign-error-gradient-step")],
    )


def _sgd_nesterov(rng, level):
    a = rng.randint(1, 4)
    beta = rng.choice([R(1, 2), R(9, 10), R(1, 4)])
    eta = R(1, rng.choice([4, 5, 8, 10]))
    x0 = nz(rng, -4, 4)
    xx, v = sp.Integer(x0), sp.Integer(0)
    xh, vh = sp.Integer(x0), sp.Integer(0)
    table = []
    for _ in range(2):
        v = beta * v - eta * a * (xx + beta * v)
        xx = xx + v
        table.append((v, xx))
        vh = beta * vh - eta * a * xh
        xh = xh + vh
    return Problem(
        prompt=(f"Nesterov momentum looks ahead before taking the gradient: $v_{{t+1}} = \\beta v_t - \\eta f'(x_t + \\beta v_t)$, "
                f"$x_{{t+1}} = x_t + v_{{t+1}}$. On $f(x) = {L(R(a, 2) * x ** 2)}$ with $\\beta = {L(beta)}$, $\\eta = {L(eta)}$, "
                f"$x_0 = {x0}$, $v_0 = 0$, give $x_2$."),
        answer=fmt(xx), kind="num", answer_hint=FRAC_HINT,
        rubric=["f'(x) = a x", "Look-ahead point x_t + β v_t", "Update v, then x; keep a table", "Two iterations"],
        solution=(f"$v_1 = {L(table[0][0])}$, $x_1 = {L(table[0][1])}$; look-ahead point $x_1 + \\beta v_1 = "
                  f"{L(table[0][1] + beta * table[0][0])}$, $v_2 = {L(table[1][0])}$, $x_2 = {L(xx)}$ "
                  f"(heavy ball, gradient at $x_t$, would give ${L(xh)}$)."),
        misconceptions=[(fmt(xh), "evaluated-gradient-at-current-point"), (fmt(table[0][1]), "stopped-early")],
    )


def _sgd_epoch(rng, level):
    data = [rng.randint(-5, 9) for _ in range(4)]
    w0 = rng.randint(-3, 5)
    eta = R(1, rng.choice([2, 4, 5]))
    w1 = w0 - eta * (w0 - R(data[0] + data[1], 2))
    w2 = w1 - eta * (w1 - R(data[2] + data[3], 2))
    full = sp.Integer(w0)
    for _ in range(2):
        full = full - eta * (full - R(sum(data), 4))
    return Problem(
        prompt=(f"One epoch of mini-batch SGD: loss $\\mathcal L(w) = \\frac14\\sum_{{i=1}}^4 \\tfrac12 (w - a_i)^2$, data "
                f"$a = ({', '.join(map(str, data))})$, batches $\\{{1,2\\}}$ then $\\{{3,4\\}}$ in that order (averaged gradients), "
                f"$\\eta = {L(eta)}$, $w_0 = {w0}$. Give $w$ at the end of the epoch."),
        answer=fmt(w2), kind="num", answer_hint=FRAC_HINT,
        rubric=["Batch gradient: w - mean(a_B)", "First step with batch {1,2}", "Second step from w_1 with batch {3,4}"],
        solution=f"$w_1 = {L(w1)}$, $w_2 = {L(w2)}$ (two full-batch steps would give ${L(full)}$).",
        misconceptions=[(fmt(w1), "stopped-early"), (fmt(full), "used-full-batch")],
    )


def _sgd_critical_damping(rng, level):
    s = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(2, 3), R(3, 4)])
    a = rng.choice([1, 2, 4])
    eta = s ** 2 / a
    beta = sp.Symbol("beta")
    roots = [r for r in sp.solve((1 + beta - eta * a) ** 2 - 4 * beta, beta) if 0 < r < 1]
    ans = roots[0]
    return Problem(
        prompt=(f"Critical damping of heavy-ball momentum: on $f(x) = \\tfrac{{a}}{{2}}x^2$ with $a = {a}$ and $\\eta = {L(eta)}$, the iterates "
                "satisfy $x_{t+1} = (1 + \\beta - \\eta a)\\,x_t - \\beta x_{t-1}$. Find the $\\beta \\in (0, 1)$ for which the "
                "characteristic polynomial has a repeated root (fastest non-oscillating convergence)."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Characteristic polynomial r² - (1 + β - ηa) r + β = 0", "Repeated root ⇔ discriminant (1 + β - ηa)² - 4β = 0",
                "Solve: √β = 1 ± √(ηa)", "Pick the root in (0, 1): β = (1 - √(ηa))²"],
        solution=f"$\\eta a = {L(eta * a)}$, $\\sqrt{{\\eta a}} = {L(s)}$, $\\beta = (1 - {L(s)})^2 = {L(ans)}$ (double root $r = {L(sp.sqrt(ans))}$).",
        misconceptions=[(fmt(1 - eta * a), "linearised-condition"), (fmt((1 + s) ** 2), "picked-wrong-root")],
    )


# --------------------------------------------------------------------------- Adam-family variants
def _adam_first_moments(rng, level):
    b1, b2 = rng.choice([(R(9, 10), R(999, 1000)), (R(9, 10), R(99, 100)), (R(4, 5), R(99, 100)), (R(1, 2), R(3, 4))])
    g = nz(rng, -8, 8)
    m1, v1 = (1 - b1) * g, (1 - b2) * g * g
    return Problem(
        prompt=(f"First moment updates of Adam: with $\\beta_1 = {dec(b1)}$, $\\beta_2 = {dec(b2)}$ and $m_0 = v_0 = 0$, the first "
                f"gradient is $g_1 = {g}$. Give the raw (not bias-corrected) moments $m_1$ and $v_1$."),
        answer=[fmt(m1), fmt(v1)], kind="list", answer_hint="m1, v1   e.g. 0.3, 0.009",
        rubric=["m_1 = β1·0 + (1 - β1) g_1", "v_1 = β2·0 + (1 - β2) g_1²", "Square the gradient for v (sign disappears)"],
        solution=f"$m_1 = {dec(1 - b1)}\\cdot({g}) = {dec(m1)}$, $v_1 = {dec(1 - b2)}\\cdot{g * g} = {dec(v1)}$.",
        misconceptions=[([fmt(g), fmt(g * g)], "applied-bias-correction"), ([fmt(b1 * g), fmt(b2 * g * g)], "swapped-ema-weights")],
    )


def _adam_rmsprop(rng, level):
    rho = rng.choice([R(3, 4), R(99, 100), R(15, 16), R(8, 9), R(24, 25)])
    while True:
        a, c, x0 = rng.randint(1, 4), rng.randint(-3, 3), rng.randint(-5, 5)
        if x0 != c:
            break
    eta = rng.choice([R(1, 10), R(1, 100), R(1, 20)])
    g = a * (x0 - c)
    ans = x0 - eta * g / sp.sqrt((1 - rho) * g * g)
    return Problem(
        prompt=(f"RMSProp without bias correction: $v_1 = \\rho v_0 + (1-\\rho) g_1^2$, $x_1 = x_0 - \\eta\\, g_1/\\sqrt{{v_1}}$, with "
                f"$v_0 = 0$, $\\rho = {L(rho)}$, $\\eta = {L(eta)}$, on $f(x) = {L(R(a, 2) * (x - c) ** 2)}$ from $x_0 = {x0}$. Give $x_1$."),
        answer=fmt(ans), kind="num", answer_hint="exact, e.g. 29/10",
        rubric=["g_1 = f'(x_0)", "v_1 = (1 - ρ) g_1²", "√v_1 = √(1-ρ)·|g_1|", "Step = η·sign(g)/√(1-ρ): larger than η at the start"],
        solution=(f"$g_1 = {g}$, $\\sqrt{{v_1}} = {L(sp.sqrt(1 - rho))}\\cdot{abs(g)}$, step $= {L(eta)}\\cdot\\left({L(sp.sign(g) / sp.sqrt(1 - rho))}\\right)$, "
                  f"$x_1 = {L(ans)}$."),
        misconceptions=[(fmt(x0 - eta * sp.sign(g)), "applied-bias-correction"), (fmt(x0 - eta * g), "used-plain-gd")],
    )


def _adam_epsilon(rng, level):
    eps = rng.choice([R(1, 10), R(1, 2), sp.Integer(1)])
    big = nz(rng, -9, 9) * (1 if eps < 1 else 4)
    small = rng.choice([1, -1]) * eps * rng.choice([1, R(1, 2), 2])
    eta = rng.choice([R(1, 10), R(1, 100)])
    w0 = sp.Matrix([rng.randint(-3, 3), rng.randint(-3, 3)])
    gv = sp.Matrix([big, small])
    w1 = w0 - eta * sp.Matrix([gi / (abs(gi) + eps) for gi in gv])
    w_noeps = w0 - eta * sp.Matrix([sp.sign(gi) for gi in gv])
    return Problem(
        prompt=(f"Why Adam's $\\epsilon$ matters for tiny gradients: two parameters start at $\\mathbf w_0 = {tup(w0)}$ with gradients "
                f"$\\mathbf g = {tup(gv)}$. The first bias-corrected Adam step is $\\eta\\,g/(|g| + \\epsilon)$ per coordinate; with "
                f"$\\eta = {L(eta)}$, $\\epsilon = {L(eps)}$, give $\\mathbf w_1$."),
        answer=vec(w1), kind="vector", answer_hint=VEC_HINT,
        rubric=["Bias-corrected first step: m̂ = g, √v̂ = |g|", "Per coordinate: η g/(|g| + ε)",
                "Large |g|: step ≈ η; |g| comparable to ε: step clearly smaller", "Subtract from w_0"],
        solution=(f"Steps $= {L(eta)}\\cdot\\left({L(gv[0] / (abs(gv[0]) + eps))}, {L(gv[1] / (abs(gv[1]) + eps))}\\right)$, "
                  f"$\\mathbf w_1 = {tup(w1)}$."),
        misconceptions=[(vec(w_noeps), "ignored-epsilon"), (vec(w0 - eta * gv), "used-plain-gd")],
    )


def _adam_adagrad(rng, level):
    g1, g2 = rng.choice(PYTH)
    g1, g2 = g1 * rng.choice([1, -1]), g2 * rng.choice([1, -1])
    x0 = rng.randint(-3, 3)
    eta = rng.choice([R(1, 10), R(1, 2), sp.Integer(1)])
    x1 = x0 - eta * g1 / sp.sqrt(g1 ** 2)
    x2 = x1 - eta * g2 / sp.sqrt(g1 ** 2 + g2 ** 2)
    return Problem(
        prompt=(f"AdaGrad accumulates squared gradients: $G_t = G_{{t-1}} + g_t^2$, $x_{{t+1}} = x_t - \\eta\\,g_t/\\sqrt{{G_t}}$, "
                f"$G_0 = 0$. A parameter starts at $x_0 = {x0}$ and receives gradients $g_1 = {g1}$, $g_2 = {g2}$; with "
                f"$\\eta = {L(eta)}$, give $x_2$."),
        answer=fmt(x2), kind="num", answer_hint=FRAC_HINT,
        rubric=["G_1 = g_1², first step = η·sign(g_1)", "G_2 = g_1² + g_2²", "Second step = η g_2/√G_2", "Note the step sizes only shrink"],
        solution=(f"$G_1 = {g1 ** 2}$, $x_1 = {L(x1)}$; $G_2 = {g1 ** 2 + g2 ** 2}$, $\\sqrt{{G_2}} = {L(sp.sqrt(g1 ** 2 + g2 ** 2))}$, "
                  f"$x_2 = {L(x1)} - {L(eta)}\\cdot\\frac{{{g2}}}{{{L(sp.sqrt(g1 ** 2 + g2 ** 2))}}} = {L(x2)}$."),
        misconceptions=[(fmt(x0 - eta * (sp.sign(g1) + sp.sign(g2))), "normalised-by-current-gradient-only"),
                        (fmt(x1), "stopped-early")],
    )


def _adam_bias_ratio(rng, level):
    b1, b2 = rng.choice([(R(9, 10), R(999, 1000)), (R(9, 10), R(9999, 10000)), (R(4, 5), R(99, 100)),
                         (R(1, 2), R(99, 100)), (R(9, 10), R(99, 100))])
    t = rng.choice([1, 2])
    ratio = sp.nsimplify((1 - b1 ** t) / sp.sqrt(1 - b2 ** t))
    return Problem(
        prompt=(f"What bias correction fixes: Adam sees a constant gradient $g > 0$ from step 1 on, with $\\beta_1 = {dec(b1)}$, "
                f"$\\beta_2 = {dec(b2)}$, $m_0 = v_0 = 0$, $\\epsilon = 0$. Derive $m_t$ and $v_t$, and give the length of step "
                f"$t = {t}$ **without** bias correction, $\\eta\\,m_t/\\sqrt{{v_t}}$, in units of $\\eta$."),
        answer=fmt(ratio), kind="num", answer_hint="exact, e.g. sqrt(10) or 3/sqrt(7), or a decimal to 5 significant figures",
        rubric=["m_t = (1 - β1^t) g (geometric sum)", "v_t = (1 - β2^t) g²", "Ratio m_t/√v_t = (1 - β1^t)/√(1 - β2^t) — g cancels",
                "Bias correction divides these factors out, making the step exactly η"],
        solution=(f"$m_{t} = (1 - {dec(b1)}^{t})g$, $v_{t} = (1 - {dec(b2)}^{t})g^2$, so the step is "
                  f"$\\frac{{{L(1 - b1 ** t)}}}{{\\sqrt{{{L(1 - b2 ** t)}}}}}\\,\\eta = {L(ratio)}\\,\\eta \\approx {float(ratio):.4f}\\,\\eta$."),
        misconceptions=[(fmt((1 - b1 ** t) / (1 - b2 ** t)), "forgot-square-root"), ("1", "applied-bias-correction")],
        tol=1e-4,
    )


def _adam_alternating(rng, level):
    b1 = rng.choice([R(9, 10), R(1, 2), R(4, 5), R(3, 4)])
    ans = (1 - b1) / (1 + b1)
    return Problem(
        prompt=(f"A noisy coordinate whose gradient flips sign every step, $g_t = (-1)^t G$, is optimised with Adam "
                f"($\\beta_1 = {dec(b1)}$, $\\beta_2$ close to 1, $\\epsilon = 0$, bias-corrected). In the long run ($t \\to \\infty$), "
                "how large is each step $|x_{t+1} - x_t|$, in units of $\\eta$?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Bias corrections → 1 as t → ∞", "v_t → G² since g_t² = G² every step",
                "Steady state of m alternates: m = -β1 m + (1-β1)G (sign flips) ⇒ |m| = (1-β1)G/(1+β1)",
                "Step = η|m|/√v = η(1-β1)/(1+β1)"],
        solution=(f"Alternating fixed point: $|m_\\infty| = \\frac{{1-\\beta_1}}{{1+\\beta_1}}G = {L(ans)}G$, $\\sqrt{{v_\\infty}} = G$, "
                  f"so steps shrink to ${L(ans)}\\,\\eta$ — Adam damps directions where the gradient keeps reversing."),
        misconceptions=[(fmt(1 - b1), "ignored-sign-flips-in-ema"), ("1", "assumed-step-always-eta")],
    )


# --------------------------------------------------------------------------- regularization variants
def _reg_weight_decay(rng, level):
    w0, g = nz(rng, -6, 6), nz(rng, -8, 8)
    lam = rng.choice([R(1, 10), R(1, 2), sp.Integer(1), sp.Integer(2)])
    eta = R(1, rng.choice([2, 4, 5, 10]))
    ans = w0 - eta * (g + lam * w0)
    return Problem(
        prompt=(f"Weight decay in one SGD step: a weight $w = {w0}$ has loss gradient $g = {g}$; the objective adds "
                f"$\\tfrac{{\\lambda}}{{2}}w^2$ with $\\lambda = {L(lam)}$. With $\\eta = {L(eta)}$, give the updated weight."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Gradient of the penalty: λw", "Total gradient g + λw", "w ← w - η(g + λw) = (1 - ηλ)w - ηg"],
        solution=f"$w \\leftarrow {w0} - {L(eta)}\\left({g} + {L(lam)}\\cdot({w0})\\right) = {L(ans)}$.",
        misconceptions=[(fmt(w0 - eta * g), "forgot-penalty"), (fmt(w0 - eta * g - lam * w0), "decay-not-scaled-by-lr")],
    )


def _reg_ball_projection(rng, level):
    a, b = rng.choice(PYTH)
    k = rng.randint(1, 3)
    zz = sp.Matrix([a * k * rng.choice([1, -1]), b * k * rng.choice([1, -1])])
    nrm = sp.sqrt(zz.dot(zz))
    r = rng.randint(1, int(nrm) - 1)
    w = r * zz / nrm
    clip = sp.Matrix([sp.Max(-r, sp.Min(r, v)) for v in zz])
    return Problem(
        prompt=(f"Norm-constrained (ridge-style) estimate: minimise $\\|\\mathbf w - \\mathbf z\\|^2$ subject to $\\|\\mathbf w\\|_2 \\le {r}$, "
                f"with $\\mathbf z = {tup(zz)}$. Give $\\hat{{\\mathbf w}}$."),
        answer=vec(w), kind="vector", answer_hint=VEC_HINT,
        rubric=["‖z‖ > r, so the constraint is active", "The closest point of a ball is along the ray to z", "ŵ = r·z/‖z‖"],
        solution=f"$\\|\\mathbf z\\| = {nrm} > {r}$, so $\\hat{{\\mathbf w}} = \\frac{{{r}}}{{{nrm}}}\\mathbf z = {tup(w)}$.",
        misconceptions=[(vec(clip), "clipped-each-coordinate"), (vec(zz), "ignored-constraint")],
    )


def _reg_choose_lambda(rng, level):
    while True:
        xs = rng.sample([-3, -2, -1, 1, 2, 3], 3)
        ys = [rng.randint(-6, 9) for _ in xs]
        if sum(a * b for a, b in zip(xs, ys)) != 0:
            break
    fr = rng.choice([R(1, 2), R(2, 3), R(3, 4), R(1, 3), R(4, 5)])
    sxx, sxy = sum(a * a for a in xs), sum(a * b for a, b in zip(xs, ys))
    lam = sp.Symbol("lambda")
    ans = sp.solve(R(sxy) / (sxx + lam) - fr * R(sxy, sxx), lam)[0]
    return Problem(
        prompt=(f"Tuning the ridge penalty: for the points {pts_str(xs, ys)}, the ridge estimate of $y = wx$ is "
                f"$\\hat w_\\lambda = \\arg\\min_w \\sum_i (y_i - wx_i)^2 + \\lambda w^2$. Find $\\lambda$ so that $\\hat w_\\lambda$ is exactly "
                f"${L(fr)}$ of the least-squares estimate."),
        answer=fmt(ans), kind="num", answer_hint="an exact number, e.g. 14 or 7/2",
        rubric=["OLS: ŵ = Σxy/Σx²", "Ridge: ŵ_λ = Σxy/(Σx² + λ)", "Ratio Σx²/(Σx² + λ) = target fraction", "Solve for λ"],
        solution=f"$\\frac{{{sxx}}}{{{sxx} + \\lambda}} = {L(fr)} \\Rightarrow \\lambda = {L(ans)}$.",
        misconceptions=[(fmt(fr * sxx), "inverted-fraction"), (fmt(sxx * (1 - fr)), "wrong-shrinkage-ratio")],
    )


def _reg_elastic_net(rng, level):
    l1 = rng.choice([R(1, 2), sp.Integer(1)])
    l2 = rng.choice([R(1, 2), sp.Integer(1), sp.Integer(3)])
    while True:
        zz = [R(rng.randint(-8, 8), 2) for _ in range(3)]
        sol = [soft(v, l1) / (1 + l2) for v in zz]
        if any(s == 0 for s in sol) and any(s != 0 for s in sol):
            break
    only_soft = [soft(v, l1) for v in zz]
    wrong_order = [soft(v / (1 + l2), l1) for v in zz]
    return Problem(
        prompt=(f"Elastic net, coordinate by coordinate: minimise $\\tfrac12(w - z)^2 + \\lambda_1|w| + \\tfrac{{\\lambda_2}}{{2}}w^2$ with "
                f"$\\lambda_1 = {L(l1)}$, $\\lambda_2 = {L(l2)}$, for each entry of $z = {tup(zz)}$. Give the solution vector."),
        answer=vec(sol), kind="vector", answer_hint=VEC_HINT,
        rubric=["Subgradient: (1 + λ2) w - z + λ1·∂|w| ∋ 0", "|z| ≤ λ1 ⇒ w = 0", "Otherwise w = (z - λ1 sign z)/(1 + λ2)",
                "Soft-threshold first, then shrink"],
        solution=f"$w = \\frac{{\\operatorname{{sign}}(z)\\max(|z| - \\lambda_1, 0)}}{{1 + \\lambda_2}} = {tup(sol)}$.",
        misconceptions=[(vec(only_soft), "forgot-l2-shrink"), (vec(wrong_order), "wrong-order-of-shrinkage")],
    )


def _reg_lasso_1d(rng, level):
    while True:
        xs = rng.sample([-3, -2, -1, 1, 2, 3], 3)
        ys = [rng.randint(-6, 9) for _ in xs]
        sxx, sxy = sum(a * a for a in xs), sum(a * b for a, b in zip(xs, ys))
        if abs(sxy) >= 3:
            break
    lam = rng.randint(1, abs(sxy) - 1)
    w = sp.sign(sxy) * R(abs(sxy) - lam, sxx)
    return Problem(
        prompt=(f"Lasso with a single feature: minimise $\\tfrac12\\sum_i (y_i - wx_i)^2 + \\lambda|w|$ for the points {pts_str(xs, ys)}. "
                f"Derive (a) the smallest $\\lambda$ for which $\\hat w = 0$, and (b) $\\hat w$ at $\\lambda = {lam}$."),
        answer=[fmt(abs(sxy)), fmt(w)], kind="list", answer_hint="lambda_max, w_hat   e.g. 13, 5/14",
        rubric=["Subgradient: w Σx² - Σxy + λ·∂|w| ∋ 0", "w = 0 is optimal iff |Σxy| ≤ λ ⇒ λ_max = |Σxy|",
                "Otherwise ŵ = sign(Σxy)(|Σxy| - λ)/Σx²"],
        solution=f"$\\sum xy = {sxy}$, $\\sum x^2 = {sxx}$: $\\lambda_{{\\max}} = {abs(sxy)}$, $\\hat w = {L(w)}$.",
        misconceptions=[([fmt(abs(sxy)), fmt(R(sxy, sxx + lam))], "ridge-shrinkage-not-soft-threshold"),
                        ([fmt(R(abs(sxy), sxx)), fmt(w)], "confused-lambda-max-with-ols")],
    )


def _reg_bias_variance(rng, level):
    while True:
        xs = rng.sample([-3, -2, -1, 1, 2, 3], rng.choice([2, 3]))
        wt, s2 = rng.randint(1, 3), rng.randint(1, 8)
        S = sum(a * a for a in xs)
        lam = sp.Symbol("lambda", positive=True)
        mse = (lam * wt / (S + lam)) ** 2 + s2 * S / (S + lam) ** 2
        crit = [c for c in sp.solve(sp.diff(mse, lam), lam) if c > 0]
        if crit:
            break
    lstar = crit[0]
    mstar = sp.simplify(mse.subs(lam, lstar))
    return Problem(
        prompt=(f"Bias–variance trade-off of ridge: $y_i = w x_i + \\varepsilon_i$ with true $w = {wt}$, noise variance $\\sigma^2 = {s2}$, "
                f"design $x = ({', '.join(map(str, xs))})$, and $\\hat w_\\lambda = \\sum x_iy_i/(\\sum x_i^2 + \\lambda)$. Derive "
                "$\\mathrm{MSE}(\\lambda) = \\text{bias}^2 + \\text{variance}$ and give the minimising $\\lambda^\\ast$ and $\\mathrm{MSE}(\\lambda^\\ast)$."),
        answer=[fmt(lstar), fmt(mstar)], kind="list", answer_hint="lambda*, MSE   e.g. 2, 1/8",
        rubric=["E[ŵ] = S w/(S + λ) with S = Σx² ⇒ bias = -λw/(S + λ)", "Var(ŵ) = σ² S/(S + λ)²",
                "Differentiate MSE in λ ⇒ λ* = σ²/w²", "MSE(λ*) = σ²/(S + λ*) < σ²/S (the OLS MSE)"],
        solution=(f"$S = {S}$; $\\mathrm{{MSE}}(\\lambda) = \\frac{{{wt * wt}\\lambda^2 + {s2 * S}}}{{({S} + \\lambda)^2}}$, minimised at "
                  f"$\\lambda^\\ast = \\sigma^2/w^2 = {L(lstar)}$ with $\\mathrm{{MSE}} = {L(mstar)}$ (OLS: ${L(R(s2, S))}$)."),
        misconceptions=[([fmt(lstar), fmt(R(s2, S))], "gave-ols-mse"), ([fmt(R(wt * wt, s2)), fmt(mstar)], "inverted-lambda-star")],
    )


# --------------------------------------------------------------------------- registered dispatchers
def _pick(rng, level, table):
    return rng.choice(table[level])(rng, level)


@gen("opt.convexity")
def convexity(rng, level):
    return _pick(rng, level, {1: [_convexity_base, _conv_inflection, _conv_chord],
                              2: [_convexity_base, _conv_strong, _conv_quartic_interval],
                              3: [_convexity_base, _conv_log_bump, _conv_three_var]})


@gen("opt.gradient_descent", name="gd_1d")
def gd_1d(rng, level):
    return _pick(rng, level, {1: [_gd_1d_base, _gd_landing_step], 2: [_gd_1d_base],
                              3: [_gd_1d_base, _gd_value_convergence]})


@gen("opt.gradient_descent", name="gd_2d")
def gd_2d(rng, level):
    return _pick(rng, level, {1: [_gd_2d_base], 2: [_gd_2d_base, _gd_exact_line_search], 3: [_gd_2d_base]})


@gen("opt.conditioning")
def conditioning(rng, level):
    return _pick(rng, level, {1: [_conditioning_base, _cond_kappa_diag, _cond_contraction],
                              2: [_conditioning_base, _cond_one_over_L, _cond_rescale],
                              3: [_conditioning_base, _cond_zigzag, _cond_heavy_ball]})


@gen("opt.least_squares")
def least_squares(rng, level):
    return _pick(rng, level, {1: [_least_squares_base, _ls_weighted_mean, _ls_project_line],
                              2: [_least_squares_base, _ls_min_sse, _ls_survey],
                              3: [_least_squares_base, _ls_through_point, _ls_loo]})


@gen("opt.lagrange")
def lagrange(rng, level):
    return _pick(rng, level, {1: [_lagrange_base, _lag_two_machines, _lag_river_fence],
                              2: [_lagrange_base, _lag_open_box, _lag_cobb_douglas],
                              3: [_lagrange_base, _lag_kkt_halfspace, _lag_sensitivity]})


@gen("opt.newton")
def newton(rng, level):
    return _pick(rng, level, {1: [_newton_base, _newton_log, _newton_2d],
                              2: [_newton_base, _newton_to_max, _newton_logistic],
                              3: [_newton_base, _newton_error_recursion, _newton_flat_min]})


@gen("opt.sgd_momentum")
def sgd_momentum(rng, level):
    # the base L3 already mixes two types (mini-batch variance / terminal velocity) 50:50
    return _pick(rng, level, {1: [_sgd_momentum_base, _sgd_momentum_bookkeeping, _sgd_expected_step],
                              2: [_sgd_momentum_base, _sgd_nesterov, _sgd_epoch],
                              3: [_sgd_momentum_base, _sgd_momentum_base, _sgd_critical_damping]})


@gen("opt.adam")
def adam(rng, level):
    return _pick(rng, level, {1: [_adam_base, _adam_first_moments, _adam_rmsprop],
                              2: [_adam_base, _adam_epsilon, _adam_adagrad],
                              3: [_adam_base, _adam_bias_ratio, _adam_alternating]})


@gen("opt.regularization")
def regularization(rng, level):
    return _pick(rng, level, {1: [_regularization_base, _reg_weight_decay, _reg_ball_projection],
                              2: [_regularization_base, _reg_choose_lambda, _reg_elastic_net],
                              3: [_regularization_base, _reg_lasso_1d, _reg_bias_variance]})
