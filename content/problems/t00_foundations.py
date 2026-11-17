"""Track 0 — Foundations. Reference implementation for problem generators.

Conventions (see engine/content.py):
  * Each generator: (rng, level) -> Problem, deterministic given rng.
  * Levels: 1 = mechanical, 2 = multi-step, 3 = derivation / reasoning with a checkable final result.
  * Use sympy to compute the answer; never hand-type answers that can be computed.
"""
from __future__ import annotations

import random

import sympy as sp

from engine.checker import fmt
from engine.content import Problem, gen

x, y, n, k = sp.symbols("x y n k")
L = sp.latex


def proper(rng: random.Random, dens=(2, 3, 4, 5, 6, 7, 8, 9)) -> tuple[int, int]:
    """A proper fraction a/b in lowest terms (no 2/2, 3/6 ...)."""
    while True:
        b = rng.choice(dens)
        a = rng.randint(1, b - 1)
        if sp.gcd(a, b) == 1:
            return a, b


def nz(rng: random.Random, lo: int, hi: int) -> int:
    """Random non-zero integer in [lo, hi]."""
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


# --------------------------------------------------------------------------- fractions
@gen("found.fractions")
def fractions(rng, level):
    if level == 1:
        while True:
            (a, b), (c, d) = proper(rng), proper(rng)
            if b != d:
                break
        expr = sp.Rational(a, b) + sp.Rational(c, d)
        return Problem(
            prompt=f"Compute and simplify $\\dfrac{{{a}}}{{{b}}} + \\dfrac{{{c}}}{{{d}}}$.",
            answer=fmt(expr), kind="num",
            rubric=["Find a common denominator", "Add numerators", "Reduce to lowest terms"],
            solution=f"Common denominator ${b*d}$ (or the LCM ${sp.ilcm(b, d)}$): result $= {L(expr)}$.",
            misconceptions=[(fmt(sp.Rational(a + c, b + d)), "added-denominators")],
        )
    if level == 2:
        (a, b), (c, d), (e, f) = proper(rng), proper(rng), proper(rng)
        a += b * rng.randint(0, 1)  # sometimes an improper first fraction
        expr = (sp.Rational(a, b) - sp.Rational(c, d)) / sp.Rational(e, f)
        return Problem(
            prompt=f"Simplify $\\left(\\dfrac{{{a}}}{{{b}}} - \\dfrac{{{c}}}{{{d}}}\\right) \\div \\dfrac{{{e}}}{{{f}}}$.",
            answer=fmt(expr), kind="num",
            rubric=["Subtract with a common denominator", "Divide by multiplying by the reciprocal", "Reduce"],
            solution=f"$= {L(sp.Rational(a, b) - sp.Rational(c, d))} \\cdot \\frac{{{f}}}{{{e}}} = {L(expr)}$.",
        )
    a, b, c = rng.randint(2, 6), rng.randint(2, 6), rng.randint(2, 6)
    expr = sp.Rational(1, a) + sp.Rational(1, b) + sp.Rational(1, c)
    res = 1 / expr
    return Problem(
        prompt=(f"Three resistors $R_1={a}, R_2={b}, R_3={c}$ are in parallel: "
                f"$\\frac{{1}}{{R}} = \\frac{{1}}{{R_1}}+\\frac{{1}}{{R_2}}+\\frac{{1}}{{R_3}}$. Find $R$ exactly."),
        answer=fmt(res), kind="num",
        rubric=["Sum the reciprocals over a common denominator", "Invert the result", "Simplify"],
        solution=f"$\\frac1R = {L(expr)}$, so $R = {L(res)}$.",
        misconceptions=[(fmt(sp.Integer(a + b + c)), "added-resistances")],
    )


# --------------------------------------------------------------------------- exponents
@gen("found.exponents")
def exponents(rng, level):
    if level == 1:
        a, m, nn = rng.randint(2, 5), rng.randint(2, 6), rng.randint(2, 6)
        return Problem(
            prompt=f"Write as a single power of ${a}$: $\\;{a}^{{{m}}} \\cdot {a}^{{{nn}}} \\div {a}^{{{m - 1}}}$. Give the exponent.",
            answer=str(nn + 1), kind="num",
            rubric=["Add exponents when multiplying", "Subtract exponents when dividing"],
            solution=f"${m}+{nn}-({m - 1}) = {nn + 1}$.",
            misconceptions=[(str(m * nn - (m - 1)), "multiplied-exponents")],
        )
    if level == 2:
        base = rng.choice([4, 8, 9, 16, 27, 32])
        root = {4: 2, 8: 3, 9: 2, 16: 4, 27: 3, 32: 5}[base]
        p = rng.randint(1, 3)
        expr = sp.Integer(base) ** sp.Rational(-p, root)
        return Problem(
            prompt=f"Evaluate exactly: ${base}^{{-{p}/{root}}}$.",
            answer=fmt(sp.nsimplify(expr)), kind="num",
            rubric=["Fractional exponent = root", "Negative exponent = reciprocal", "Evaluate"],
            solution=f"$\\left(\\sqrt[{root}]{{{base}}}\\right)^{{-{p}}} = {L(sp.nsimplify(expr))}$.",
        )
    a, b = rng.randint(2, 5), rng.randint(2, 5)
    expr = (x ** a * y ** (-b)) ** 2 / (x ** (-1) * y ** b)
    simp = sp.simplify(expr)
    return Problem(
        prompt=f"Simplify $\\dfrac{{(x^{{{a}}}y^{{-{b}}})^2}}{{x^{{-1}}y^{{{b}}}}}$ (positive exponents are fine).",
        answer=fmt(simp), kind="expr",
        rubric=["Distribute the outer power", "Combine x powers", "Combine y powers"],
        solution=f"$= x^{{{2*a+1}}} y^{{{-3*b}}} = {L(simp)}$.",
    )


# --------------------------------------------------------------------------- linear equations
@gen("found.linear_eq")
def linear_eq(rng, level):
    if level == 1:
        a, b, sol = nz(rng, -9, 9), rng.randint(-20, 20), rng.randint(-10, 10)
        c = a * sol + b
        return Problem(
            prompt=f"Solve for $x$: $\\;{L(a*x + b)} = {c}$.",
            answer=str(sol), kind="num",
            rubric=["Isolate the x term", "Divide by the coefficient", "Check by substitution"],
            solution=f"$x = ({c} - ({b}))/{a} = {sol}$.",
        )
    if level == 2:
        xs, ys = rng.randint(-6, 6), rng.randint(-6, 6)
        while True:
            a, b, c, d = (nz(rng, -6, 6) for _ in range(4))
            if a * d - b * c != 0:
                break
        e, f = a * xs + b * ys, c * xs + d * ys
        return Problem(
            prompt=f"Solve the system: $\\;{L(a*x+b*y)} = {e},\\quad {L(c*x+d*y)} = {f}$. Answer as $x, y$.",
            answer=[str(xs), str(ys)], kind="list", answer_hint="x, y   e.g. 2, -1",
            rubric=["Eliminate or substitute one variable", "Solve for the remaining variable", "Back-substitute", "Check both equations"],
            solution=f"$x={xs},\\ y={ys}$.",
            misconceptions=[(f"{ys}, {xs}", "swapped-x-y")],
        )
    # level 3: word problem
    r1, r2, t = rng.randint(40, 70), rng.randint(71, 110), rng.randint(2, 5)
    dist = (r1 + r2) * t
    return Problem(
        prompt=(f"Two trains start {dist} km apart and drive toward each other at {r1} km/h and {r2} km/h. "
                "After how many hours do they meet? Set up the equation on paper first."),
        answer=str(t), kind="num",
        rubric=["Define the unknown time t", f"Write ({r1}+{r2})t = {dist}", "Solve"],
        solution=f"$({r1}+{r2})t = {dist} \\Rightarrow t = {t}$ h.",
    )


# --------------------------------------------------------------------------- polynomials
@gen("found.polynomials")
def polynomials(rng, level):
    if level == 1:
        a, b, c, d = nz(rng, -5, 5), nz(rng, -7, 7), nz(rng, -5, 5), nz(rng, -7, 7)
        expr = sp.expand((a * x + b) * (c * x + d))
        return Problem(
            prompt=f"Expand $({L(a*x+b)})({L(c*x+d)})$.",
            answer=fmt(expr), kind="expr",
            rubric=["Distribute every term (FOIL)", "Collect like terms"],
            solution=f"$= {L(expr)}$.",
        )
    if level == 2:
        r1, r2 = nz(rng, -8, 8), nz(rng, -8, 8)
        expr = sp.expand((x - r1) * (x - r2))
        return Problem(
            prompt=f"Factor completely: ${L(expr)}$.",
            answer=fmt(sp.factor(expr)), kind="expr",
            rubric=["Find two numbers with the right product and sum", "Write as a product of linear factors"],
            solution=f"$= {L(sp.factor(expr))}$.",
        )
    r = nz(rng, -4, 4)
    q = sp.expand((x ** 2 + nz(rng, -3, 3) * x + nz(rng, -5, 5)))
    rem = rng.randint(-6, 6)
    p = sp.expand(q * (x - r) + rem)
    return Problem(
        prompt=f"Divide ${L(p)}$ by $(x - ({r}))$. Give the **remainder**, then verify it with the remainder theorem on paper.",
        answer=str(rem), kind="num",
        rubric=["Polynomial long (or synthetic) division", f"Remainder theorem: p({r})", "Both agree"],
        solution=f"Quotient ${L(q)}$, remainder $p({r}) = {rem}$.",
    )


# --------------------------------------------------------------------------- quadratics
@gen("found.quadratics")
def quadratics(rng, level):
    if level == 1:
        r1, r2 = rng.randint(-9, 9), rng.randint(-9, 9)
        expr = sp.expand((x - r1) * (x - r2))
        return Problem(
            prompt=f"Solve ${L(expr)} = 0$. Give all real roots.",
            answer=[str(r1), str(r2)] if r1 != r2 else [str(r1)], kind="set",
            rubric=["Factor or use the quadratic formula", "State all roots"],
            solution=f"$x \\in \\{{{r1}, {r2}\\}}$.",
            misconceptions=[(f"{-r1}, {-r2}" if r1 != r2 else str(-r1), "sign-of-roots")],
        )
    if level == 2:
        while True:
            a, b, c = nz(rng, -4, 4), rng.randint(-9, 9), rng.randint(-9, 9)
            disc = b * b - 4 * a * c
            if disc > 0 and sp.sqrt(disc).is_rational is False:
                break
        roots = sp.solve(a * x ** 2 + b * x + c, x)
        return Problem(
            prompt=f"Use the quadratic formula to solve ${L(a*x**2+b*x+c)} = 0$ exactly.",
            answer=[fmt(r) for r in roots], kind="set", answer_hint="exact values, e.g. (-3+sqrt(17))/4, (-3-sqrt(17))/4",
            rubric=["Compute the discriminant", "Apply x = (-b ± √Δ)/(2a)", "Simplify"],
            solution=f"$\\Delta = {disc}$, $x = {L(roots[0])},\\ {L(roots[1])}$.",
        )
    a, b = nz(rng, 1, 4), nz(rng, -8, 8) * 2
    c = rng.randint(-9, 9)
    h = sp.Rational(-b, 2 * a)
    kk = c - sp.Rational(b * b, 4 * a)
    return Problem(
        prompt=(f"Complete the square for $f(x) = {L(a*x**2+b*x+c)}$ and give the vertex $(h, k)$ "
                "of the parabola."),
        answer=[fmt(h), fmt(kk)], kind="vector", answer_hint="h, k",
        rubric=["Factor out the leading coefficient", "Add and subtract (b/2a)^2", "Read off vertex"],
        solution=f"$f(x) = {a}(x - ({L(h)}))^2 + {L(kk)}$, vertex $({L(h)}, {L(kk)})$.",
    )


# --------------------------------------------------------------------------- functions
@gen("found.functions")
def functions(rng, level):
    a, b, c, d = nz(rng, -5, 5), rng.randint(-6, 6), nz(rng, -4, 4), rng.randint(-6, 6)
    f = a * x + b
    g = c * x ** 2 + d
    if level == 1:
        x0 = rng.randint(-3, 3)
        val = f.subs(x, g.subs(x, x0))
        return Problem(
            prompt=f"Let $f(x) = {L(f)}$ and $g(x) = {L(g)}$. Compute $f(g({x0}))$.",
            answer=str(val), kind="num",
            rubric=[f"Compute g({x0}) first", "Feed the result into f"],
            solution=f"$g({x0}) = {g.subs(x, x0)}$, $f({g.subs(x, x0)}) = {val}$.",
            misconceptions=[(str(g.subs(x, f.subs(x, x0))), "composition-order")],
        )
    if level == 2:
        inv = sp.solve(sp.Eq(y, f), x)[0].subs(y, x)
        return Problem(
            prompt=f"Find the inverse function of $f(x) = {L(f)}$. Write $f^{{-1}}(x)$ in terms of $x$.",
            answer=fmt(inv), kind="expr",
            rubric=["Write y = f(x)", "Solve for x", "Swap names x <-> y", "Check f(f^{-1}(x)) = x"],
            solution=f"$f^{{-1}}(x) = {L(inv)}$.",
            misconceptions=[(fmt(1 / f), "inverse-as-reciprocal")],
        )
    while True:  # avoid the degenerate case where h is constant (p*c + q == 0)
        p, q, cc = nz(rng, 1, 4), rng.randint(-5, 5), rng.randint(-3, 3)
        if p * cc + q != 0:
            break
    h = (p * x + q) / (x - cc)
    pole = sp.solve(sp.denom(h), x)[0]
    inv = sp.simplify(sp.solve(sp.Eq(y, h), x)[0].subs(y, x))
    return Problem(
        prompt=f"For $h(x) = {L(h)}$ find $h^{{-1}}(x)$. (On paper: also state the domain of $h$ and the range.)",
        answer=fmt(inv), kind="expr",
        rubric=["Cross-multiply y(x - c) = px + q", "Collect x terms", "Solve and swap", f"Domain excludes x = {pole}"],
        solution=f"$h^{{-1}}(x) = {L(inv)}$; domain $x \\ne {pole}$, range $y \\ne {p}$.",
    )


# --------------------------------------------------------------------------- logarithms
@gen("found.logs")
def logs(rng, level):
    if level == 1:
        b = rng.choice([2, 3, 5, 10])
        e = rng.randint(2, 5)
        return Problem(
            prompt=f"Evaluate $\\log_{{{b}}} {b**e}$ and then $\\log_{{{b}}} \\frac{{1}}{{{b**e}}}$. Give the **sum** of the two.",
            answer="0", kind="num",
            rubric=[f"log_b(b^k) = k", "log of reciprocal is negative", "Add"],
            solution=f"${e} + (-{e}) = 0$.",
        )
    if level == 2:
        a, b, c = rng.randint(2, 5), rng.randint(2, 9), rng.randint(3, 40)
        sol = sp.log(sp.Rational(c, a)) / sp.log(b)
        return Problem(
            prompt=f"Solve ${a}\\cdot {b}^x = {c}$ exactly (use natural logs).",
            answer=fmt(sp.log(sp.Rational(c, a)) / sp.log(b)), kind="expr",
            rubric=[f"Divide by {a}", "Take logs of both sides", "Use log(b^x) = x log b"],
            solution=f"$x = \\dfrac{{\\ln({L(sp.Rational(c, a))})}}{{\\ln {b}}} \\approx {float(sol):.4f}$.",
        )
    r = rng.choice([2, 3, 4, 5, 6, 8])
    return Problem(
        prompt=(f"A quantity grows continuously: $N(t) = N_0 e^{{kt}}$ with growth rate ${r}\\%$ per year ($k = {r/100}$). "
                "Find the doubling time $T$ in years (exact form)."),
        answer=f"ln(2)/{r/100}", kind="expr",
        rubric=["Set N(T) = 2 N_0", "Cancel N_0, take ln", "Solve for T"],
        solution=f"$e^{{kT}} = 2 \\Rightarrow T = \\ln 2 / {r/100} \\approx {sp.log(2).evalf()/(r/100):.2f}$ years.",
    )


# --------------------------------------------------------------------------- trig
_UNIT = [(0, "0"), (sp.pi / 6, "\\pi/6"), (sp.pi / 4, "\\pi/4"), (sp.pi / 3, "\\pi/3"), (sp.pi / 2, "\\pi/2"),
         (2 * sp.pi / 3, "2\\pi/3"), (3 * sp.pi / 4, "3\\pi/4"), (5 * sp.pi / 6, "5\\pi/6"), (sp.pi, "\\pi"),
         (7 * sp.pi / 6, "7\\pi/6"), (5 * sp.pi / 4, "5\\pi/4"), (4 * sp.pi / 3, "4\\pi/3"), (3 * sp.pi / 2, "3\\pi/2"),
         (5 * sp.pi / 3, "5\\pi/3"), (7 * sp.pi / 4, "7\\pi/4"), (11 * sp.pi / 6, "11\\pi/6")]


@gen("found.trig")
def trig(rng, level):
    if level == 1:
        ang, t = rng.choice(_UNIT)
        fn = rng.choice(["sin", "cos"])
        val = getattr(sp, fn)(ang)
        return Problem(
            prompt=f"Without a calculator: $\\{fn}({t})$ = ?",
            answer=fmt(val), kind="num",
            rubric=["Locate the angle on the unit circle", "Reference angle value", "Correct sign for the quadrant"],
            solution=f"$\\{fn}({t}) = {L(val)}$.",
            misconceptions=[(fmt(-val), "wrong-quadrant-sign")] if val != 0 else [],
        )
    if level == 2:
        ang, t = rng.choice(_UNIT[1:8])
        val = sp.tan(ang) if ang != sp.pi / 2 else None
        if val is None:
            ang, t = _UNIT[1]
            val = sp.tan(ang)
        deg = int(ang * 180 / sp.pi)
        return Problem(
            prompt=f"Convert ${deg}^\\circ$ to radians and compute $\\tan$ of it exactly. Give $\\tan({deg}^\\circ)$.",
            answer=fmt(val), kind="num",
            rubric=["Multiply by π/180", "tan = sin/cos", "Rationalize if needed"],
            solution=f"${deg}^\\circ = {t}$, $\\tan = {L(val)}$.",
        )
    a = rng.choice([sp.pi / 6, sp.pi / 4, sp.pi / 3])
    b = rng.choice([sp.pi / 6, sp.pi / 4, sp.pi / 3])
    val = sp.radsimp(sp.expand(sp.cos(a) * sp.cos(b) - sp.sin(a) * sp.sin(b)))
    return Problem(
        prompt=(f"Use the angle-addition formula $\\cos(a+b)=\\cos a\\cos b-\\sin a\\sin b$ to compute "
                f"$\\cos\\left({L(a)} + {L(b)}\\right)$ exactly."),
        answer=fmt(val), kind="num",
        rubric=["Write each sin/cos value", "Apply the formula", "Simplify the surds"],
        solution=f"$= {L(sp.cos(a))}\\cdot{L(sp.cos(b))} - {L(sp.sin(a))}\\cdot{L(sp.sin(b))} = {L(val)}$.",
    )


# --------------------------------------------------------------------------- sums
@gen("found.sums")
def sums(rng, level):
    if level == 1:
        N = rng.randint(10, 100)
        s = N * (N + 1) // 2
        return Problem(
            prompt=f"Compute $\\displaystyle\\sum_{{k=1}}^{{{N}}} k$.",
            answer=str(s), kind="num",
            rubric=["Use n(n+1)/2 (Gauss pairing)", "Evaluate"],
            solution=f"${N}\\cdot{N+1}/2 = {s}$.",
        )
    if level == 2:
        a, r, N = rng.randint(1, 5), rng.choice([2, 3, sp.Rational(1, 2)]), rng.randint(4, 8)
        s = sum(a * r ** j for j in range(N))
        return Problem(
            prompt=f"Compute the geometric sum $\\displaystyle\\sum_{{j=0}}^{{{N-1}}} {a}\\cdot\\left({L(r)}\\right)^j$ exactly.",
            answer=fmt(sp.nsimplify(s)), kind="num",
            rubric=["Identify first term a and ratio r", "Use a(1 - r^n)/(1 - r)", "Simplify"],
            solution=f"$= {a}\\frac{{1-{L(r)}^{{{N}}}}}{{1-{L(r)}}} = {L(sp.nsimplify(s))}$.",
        )
    r = sp.Rational(1, rng.choice([2, 3, 4, 5]))
    a = rng.randint(1, 6)
    s = a / (1 - r)
    return Problem(
        prompt=f"Find $\\displaystyle\\sum_{{j=0}}^{{\\infty}} {a}\\left({L(r)}\\right)^j$. On paper, justify why it converges.",
        answer=fmt(s), kind="num",
        rubric=["|r| < 1 so the series converges", "Partial sums a(1-r^n)/(1-r)", "Take n → ∞"],
        solution=f"$= {a}/(1-{L(r)}) = {L(s)}$.",
    )


# --------------------------------------------------------------------------- induction
@gen("found.induction")
def induction(rng, level):
    if level == 1:
        return Problem(
            prompt=("Prove by induction on paper that $\\sum_{k=1}^n (2k-1) = n^2$. "
                    "Then type the expression you add when going from $n$ to $n+1$ (the $(n+1)$-th term)."),
            answer="2*n+1", kind="expr",
            rubric=["Base case n = 1", "State the inductive hypothesis", "Add the (n+1)th term 2n+1", "Show n^2 + 2n + 1 = (n+1)^2"],
            solution="$(n+1)$-th term is $2(n+1)-1 = 2n+1$, and $n^2+2n+1=(n+1)^2$.",
            misconceptions=[("2*n-1", "used-nth-term")],
        )
    if level == 2:
        p = rng.choice([2, 3])
        closed = sp.factor(sp.summation(k ** p, (k, 1, n)))
        return Problem(
            prompt=(f"Find a closed form for $\\displaystyle\\sum_{{k=1}}^n k^{{{p}}}$ in terms of $n$ "
                    "and prove it by induction on paper."),
            answer=fmt(closed), kind="expr",
            rubric=["Guess the form (try small n)", "Base case", "Inductive step algebra", "Conclude"],
            solution=f"$\\sum k^{{{p}}} = {L(closed)}$.",
        )
    a = rng.choice([3, 4, 5, 7])
    return Problem(
        prompt=(f"Prove by induction that ${a}^n - 1$ is divisible by ${a-1}$ for all $n\\ge 1$. "
                f"In the inductive step you write ${a}^{{n+1}} - 1 = {a}({a}^n - 1) + c$. What is $c$?"),
        answer=str(a - 1), kind="num",
        rubric=["Base case", f"Rewrite {a}^(n+1) - 1", "Both terms divisible by the divisor", "Conclude"],
        solution=f"${a}^{{n+1}}-1 = {a}({a}^n-1) + {a-1}$.",
    )


# --------------------------------------------------------------------------- complex numbers
@gen("found.complex")
def complex_numbers(rng, level):
    I = sp.I
    if level == 1:
        a, b, c, d = (nz(rng, -6, 6) for _ in range(4))
        z = sp.expand((a + b * I) * (c + d * I))
        return Problem(
            prompt=f"Compute $({a} {'+' if b>0 else '-'} {abs(b)}i)({c} {'+' if d>0 else '-'} {abs(d)}i)$ in the form $a+bi$.",
            answer=fmt(z), kind="expr", answer_hint="like 3 + 4*i",
            rubric=["Distribute", "Use i^2 = -1", "Collect real and imaginary parts"],
            solution=f"$= {L(z)}$.",
            misconceptions=[(fmt(a * c + b * d + (a * d + b * c) * I), "i-squared-plus-one")],
        )
    if level == 2:
        a, b, c, d = (nz(rng, -6, 6) for _ in range(4))
        z = sp.simplify((a + b * I) / (c + d * I))
        z = sp.expand(z)
        return Problem(
            prompt=f"Write $\\dfrac{{{L(a+b*I)}}}{{{L(c+d*I)}}}$ in the form $a+bi$ (exact).",
            answer=fmt(z), kind="expr",
            rubric=["Multiply top and bottom by the conjugate", "Denominator becomes |w|^2", "Split real/imag parts"],
            solution=f"$= {L(z)}$.",
        )
    ang, t = rng.choice(_UNIT[1:])
    r = rng.choice([1, 2, 3])
    nn = rng.choice([3, 4, 6])
    z = r * sp.exp(I * ang)
    zn = sp.nsimplify(sp.expand_complex(z ** nn))
    return Problem(
        prompt=(f"Let $z = {r}e^{{i{t}}}$. Using de Moivre / Euler, compute $z^{{{nn}}}$ in the form $a+bi$. "
                "On paper also write z in Cartesian form."),
        answer=fmt(zn), kind="expr",
        rubric=["Raise modulus to the power", "Multiply the angle", "Convert back with cos + i sin"],
        solution=f"$z^{{{nn}}} = {r**nn}e^{{i\\cdot {nn}\\cdot {t}}} = {L(zn)}$.",
    )
