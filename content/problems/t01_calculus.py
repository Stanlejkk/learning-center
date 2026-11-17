"""Track 1 — Calculus (single & multivariable).

Conventions (see engine/content.py and t00_foundations.py):
  * Each generator: (rng, level) -> Problem, deterministic given rng.
  * Levels: 1 = mechanical, 2 = multi-step, 3 = derivation / reasoning with a checkable final result
    (often connected to ML: logistic loss, softplus, backprop, gradient descent step sizes).
  * Every answer is computed with sympy.
  * Never use the symbol names e, E, i, I, j, C as variables: the checker maps them to constants.
"""
from __future__ import annotations

import random

import sympy as sp

from engine.checker import fmt
from engine.content import Problem, gen

x, y, t, w, s, u = sp.symbols("x y t w s u")


def L(e) -> str:
    """LaTeX with ln instead of log."""
    return sp.latex(e, ln_notation=True)


TRIPLES = [(3, 4, 5), (5, 12, 13), (6, 8, 10), (8, 15, 17), (9, 12, 15)]


def nz(rng: random.Random, lo: int, hi: int) -> int:
    """Random non-zero integer in [lo, hi]."""
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def vec(vals) -> list[str]:
    """Vector answer as a list of canonical strings."""
    return [fmt(sp.nsimplify(v)) for v in vals]


def tup(vals) -> str:
    """LaTeX for a tuple (a, b, ...)."""
    return "(" + ", ".join(L(sp.nsimplify(v)) for v in vals) + ")"


# =========================================================================== limits
@gen("calc.limits")
def limits(rng, level):
    if level == 1:
        a = rng.randint(-6, 6)
        b = rng.randint(-6, 6)
        while b == a:
            b = rng.randint(-6, 6)
        num = sp.expand((x - a) * (x - b))
        lim = sp.limit(num / (x - a), x, a)
        return Problem(
            prompt=f"Evaluate $\\displaystyle\\lim_{{x\\to {a}}} \\frac{{{L(num)}}}{{{L(x - a)}}}$.",
            answer=fmt(lim), kind="num",
            rubric=["Notice that direct substitution gives 0/0",
                    f"Factor the numerator: ({x - a})({x - b})",
                    "Cancel the common factor (valid because x ≠ a in the limit)",
                    f"Substitute x = {a}"],
            solution=(f"${L(num)} = ({L(x - a)})({L(x - b)})$, so for $x \\ne {a}$ the quotient equals ${L(x - b)}$. "
                      f"Limit $= {a} - ({b}) = {L(lim)}$."),
            misconceptions=[("0", "plugged-in-0-over-0"), (fmt(-lim), "sign-error-in-factor")],
        )
    if level == 2:
        if rng.random() < 0.5:
            c, a = rng.randint(1, 5), nz(rng, -8, 8)
            expr = (sp.sqrt(c ** 2 + a * x) - c) / x
            lim = sp.limit(expr, x, 0)
            return Problem(
                prompt=f"Evaluate $\\displaystyle\\lim_{{x\\to 0}} \\frac{{\\sqrt{{{c**2} {'+' if a > 0 else '-'} {abs(a)}x}} - {c}}}{{x}}$.",
                answer=fmt(lim), kind="num",
                rubric=["Direct substitution gives 0/0",
                        f"Multiply numerator and denominator by the conjugate √({c**2}{a:+d}x) + {c}",
                        "Numerator simplifies to a multiple of x; cancel x",
                        "Substitute x = 0"],
                solution=(f"$\\frac{{(\\sqrt{{{c**2}{a:+d}x}}-{c})(\\sqrt{{{c**2}{a:+d}x}}+{c})}}{{x(\\sqrt{{{c**2}{a:+d}x}}+{c})}}"
                          f" = \\frac{{{a}}}{{\\sqrt{{{c**2}{a:+d}x}}+{c}}} \\to \\frac{{{a}}}{{{2*c}}} = {L(lim)}$."),
                misconceptions=[("0", "plugged-in-0-over-0"), (fmt(sp.Rational(a, c)), "conjugate-forgot-factor-2")],
            )
        a, b = nz(rng, -8, 8), rng.randint(-5, 9)
        expr = sp.sqrt(x ** 2 + a * x + b) - x
        lim = sp.limit(expr, x, sp.oo)
        return Problem(
            prompt=f"Evaluate $\\displaystyle\\lim_{{x\\to\\infty}} \\left(\\sqrt{{{L(x**2 + a*x + b)}}} - x\\right)$.",
            answer=fmt(lim), kind="num",
            rubric=["Recognise the ∞ − ∞ form", "Multiply and divide by the conjugate √(…) + x",
                    "Divide numerator and denominator by x", "Let x → ∞"],
            solution=(f"$= \\frac{{{L(a*x + b)}}}{{\\sqrt{{{L(x**2 + a*x + b)}}} + x}}"
                      f" = \\frac{{{a} + {b}/x}}{{\\sqrt{{1 + {a}/x + {b}/x^2}} + 1}} \\to {L(lim)}$."),
            misconceptions=[("0", "infinity-minus-infinity-is-zero"), (str(a), "conjugate-forgot-factor-2")],
        )
    a = nz(rng, -4, 4)
    if rng.random() < 0.5:
        g = (sp.exp(a * x) - 1 - a * x) / x ** 2
        sgn = "-" if a > 0 else "+"
        shown = f"\\frac{{{L(sp.exp(a * x))} - 1 {sgn} {L(abs(a) * x)}}}{{x^2}}"
        series = f"{L(sp.exp(a * x))} = 1 + {L(a * x + a**2 * x**2 / 2)} + O(x^3)"
        wrongs = [(fmt(a ** 2), "taylor-forgot-factorial"), ("0", "plugged-in-0-over-0")]
    else:
        g = (sp.sin(a * x) - a * x) / x ** 3
        sgn = "-" if a > 0 else "+"
        shown = f"\\frac{{{L(sp.sin(a * x))} {sgn} {L(abs(a) * x)}}}{{x^3}}"
        series = f"{L(sp.sin(a * x))} = {L(a * x)} - \\frac{{({a})^3x^3}}{{6}} + O(x^5)"
        wrongs = [(fmt(sp.Rational(a ** 3, 6)), "taylor-sign-error"), ("0", "plugged-in-0-over-0")]
    k = sp.limit(g, x, 0)
    return Problem(
        prompt=(f"Define $f(x) = {shown}$ for $x \\neq 0$ and $f(0) = k$. "
                "Find the value of $k$ that makes $f$ continuous at $0$. "
                "(Use L'Hôpital's rule or a Taylor expansion, and explain on paper why that value is forced.)"),
        answer=fmt(k), kind="num",
        rubric=["Continuity at 0 means k = lim_{x→0} f(x)", "Identify the 0/0 form",
                "Expand the transcendental function to enough terms (or apply L'Hôpital repeatedly)",
                "Cancel the power of x and take the limit"],
        solution=f"${series}$, so the leading terms cancel and $\\lim_{{x\\to0}} f(x) = {L(k)}$. Hence $k = {L(k)}$.",
        misconceptions=wrongs,
    )


# =========================================================================== derivative rules
@gen("calc.derivative_rules")
def derivative_rules(rng, level):
    if level == 1:
        n1, n2 = rng.randint(3, 7), rng.randint(1, 2)
        a, b, c = nz(rng, -9, 9), nz(rng, -9, 9), rng.randint(-9, 9)
        f = a * x ** n1 + b * x ** n2 + c
        d = sp.diff(f, x)
        return Problem(
            prompt=f"Differentiate $f(x) = {L(f)}$.",
            answer=fmt(d), kind="expr",
            rubric=["Apply the power rule d/dx x^n = n x^(n-1) to each term", "Keep the coefficients (linearity)",
                    "The constant term differentiates to 0"],
            solution=f"$f'(x) = {L(d)}$.",
            misconceptions=[(fmt(a * x ** (n1 - 1) + b * x ** (n2 - 1)), "power-rule-forgot-multiply"),
                            (fmt(a * n1 * x ** n1 + b * n2 * x ** n2), "power-rule-kept-exponent")],
        )
    if level == 2:
        while True:
            a, b, c, d = nz(rng, -6, 6), rng.randint(-6, 6), nz(rng, -5, 5), rng.randint(-6, 6)
            if a * d - b * c != 0:
                break
        f = (a * x + b) / (c * x + d)
        fp = sp.factor(sp.diff(f, x))
        return Problem(
            prompt=f"Differentiate $f(x) = \\dfrac{{{L(a*x + b)}}}{{{L(c*x + d)}}}$ and simplify.",
            answer=fmt(fp), kind="expr",
            rubric=["Quotient rule (N'D − N D')/D²", f"N' = {a}, D' = {c}",
                    "Expand the numerator: the x terms cancel", "Leave the denominator squared"],
            solution=(f"$f'(x) = \\dfrac{{{a}({L(c*x + d)}) - ({L(a*x + b)})\\cdot ({c})}}{{({L(c*x + d)})^2}}"
                      f" = {L(fp)}$."),
            misconceptions=[(fmt(-fp), "quotient-rule-sign"), (fmt(sp.Rational(a, c)), "quotient-of-derivatives")],
        )
    a, b = nz(rng, -6, 6), rng.randint(-4, 4)
    x0 = rng.randint(-3, 3)
    while x0 + b == 0:
        x0 = rng.randint(-3, 3)
    f = a / (x + b)
    fp = sp.diff(f, x)
    val = fp.subs(x, x0)
    hh = sp.Symbol("h")
    return Problem(
        prompt=(f"Using **only** the limit definition $f'(x) = \\lim_{{h\\to 0}} \\frac{{f(x+h)-f(x)}}{{h}}$ "
                f"(no derivative rules), derive $f'(x)$ for $f(x) = {L(f)}$. Then type the value $f'({x0})$."),
        answer=fmt(val), kind="num",
        rubric=["Write the difference quotient with f(x+h) and f(x)",
                "Combine the two fractions over the common denominator (x+h+b)(x+b)",
                "The numerator becomes a multiple of h; cancel h",
                "Let h → 0 to get f'(x) = −a/(x+b)²", f"Evaluate at x = {x0}"],
        solution=(f"$\\frac{{f(x+h)-f(x)}}{{h}} = \\frac{{{a}}}{{h}}\\left(\\frac{{1}}{{{L(x + hh + b)}}} - \\frac{{1}}{{{L(x + b)}}}\\right)"
                  f" = \\frac{{{-a}}}{{({L(x + hh + b)})({L(x + b)})}} \\to {L(fp)}$. "
                  f"At $x={x0}$: ${L(val)}$."),
        misconceptions=[(fmt(-val), "lost-minus-sign"), (fmt(sp.Rational(a, x0 + b)), "evaluated-f-not-derivative")],
    )


@gen("calc.derivative_rules", name="points_and_extrema")
def derivative_points(rng, level):
    if level == 1:
        a, b = nz(rng, -8, 8), nz(rng, -8, 8)
        x0 = rng.choice([1, 4, 9])
        f = a * sp.sqrt(x) + b / x
        val = sp.diff(f, x).subs(x, x0)
        return Problem(
            prompt=f"Let $f(x) = {L(a * sp.sqrt(x))} {'+' if b > 0 else '-'} \\dfrac{{{abs(b)}}}{{x}}$. Compute $f'({x0})$.",
            answer=fmt(val), kind="num",
            rubric=["Rewrite with exponents: x^(1/2) and x^(-1)", "Power rule on each term",
                    f"Substitute x = {x0}"],
            solution=(f"$f'(x) = \\frac{{{a}}}{{2\\sqrt{{x}}}} {'-' if b > 0 else '+'} \\frac{{{abs(b)}}}{{x^2}}$, "
                      f"so $f'({x0}) = {L(val)}$."),
            misconceptions=[(fmt(sp.Rational(a, 2) / sp.sqrt(x0) + sp.Rational(b, x0 ** 2)), "negative-exponent-sign")],
        )
    if level == 2:
        a, b, c = rng.randint(-5, 5), nz(rng, -4, 4), rng.randint(-6, 6)
        x0 = nz(rng, -2, 2)
        f = (x ** 2 + a) * (b * x ** 3 + c)
        val = sp.diff(f, x).subs(x, x0)
        return Problem(
            prompt=f"Let $f(x) = ({L(x**2 + a)})({L(b*x**3 + c)})$. Use the product rule to compute $f'({x0})$.",
            answer=fmt(val), kind="num",
            rubric=["Product rule (uv)' = u'v + uv'", "u' = 2x, v' = 3b x^2", f"Substitute x = {x0} into both terms"],
            solution=(f"$f'(x) = 2x({L(b*x**3 + c)}) + ({L(x**2 + a)})\\cdot {L(3*b*x**2)}$; "
                      f"at $x = {x0}$: $f'({x0}) = {L(val)}$."),
            misconceptions=[(fmt(2 * x0 * 3 * b * x0 ** 2), "product-of-derivatives")],
        )
    r1, r2 = sorted(rng.sample(range(-4, 5), 2))
    d = rng.randint(-5, 5)
    f = 2 * x ** 3 - 3 * (r1 + r2) * x ** 2 + 6 * r1 * r2 * x + d
    fp = sp.factor(sp.diff(f, x))
    return Problem(
        prompt=(f"For $f(x) = {L(f)}$, find all points with a horizontal tangent and decide which is a local "
                "minimum. Type the $x$-coordinate of the **local minimum**."),
        answer=str(r2), kind="num",
        rubric=["Compute f'(x) and set it to 0", f"Factor: f'(x) = {fp}",
                "Classify each critical point with f''(x) or a sign chart", "Pick the one where f' goes − to +"],
        solution=(f"$f'(x) = {L(fp)}$, critical points $x = {r1}, {r2}$. $f''(x) = {L(sp.diff(f, x, 2))}$ is "
                  f"${sp.diff(f, x, 2).subs(x, r1)} < 0$ at ${r1}$ (local max) and ${sp.diff(f, x, 2).subs(x, r2)} > 0$ at ${r2}$ (local min)."),
        misconceptions=[(str(r1), "max-min-swapped")],
    )


# =========================================================================== chain rule
@gen("calc.chain_rule")
def chain_rule(rng, level):
    if level == 1:
        a, b, n = nz(rng, -6, 6), nz(rng, -7, 7), rng.randint(3, 8)
        f = (a * x + b) ** n
        d = sp.diff(f, x)
        return Problem(
            prompt=f"Differentiate $f(x) = ({L(a*x + b)})^{{{n}}}$ (no need to expand).",
            answer=fmt(d), kind="expr",
            rubric=["Outer function u^n, inner u = ax + b", "Outer derivative n u^(n-1)", "Multiply by the inner derivative a"],
            solution=f"$f'(x) = {n}({L(a*x + b)})^{{{n-1}}}\\cdot {a} = {L(d)}$.",
            misconceptions=[(fmt(n * (a * x + b) ** (n - 1)), "chain-forgot-inner-derivative")],
        )
    if level == 2:
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        if rng.random() < 0.5:
            f = sp.sqrt(a * x ** 2 + b)
            shown = f"\\sqrt{{{L(a*x**2 + b)}}}"
            wrong = 1 / (2 * sp.sqrt(a * x ** 2 + b))
        else:
            n = rng.randint(2, 4)
            f = 1 / (a * x ** 2 + b) ** n
            shown = f"\\dfrac{{1}}{{({L(a*x**2 + b)})^{{{n}}}}}"
            wrong = -n / (a * x ** 2 + b) ** (n + 1)
        d = sp.simplify(sp.diff(f, x))
        return Problem(
            prompt=f"Differentiate $f(x) = {shown}$ and simplify.",
            answer=fmt(d), kind="expr",
            rubric=["Rewrite as a power of the inner function u = a x² + b", "Differentiate the outer power",
                    "Multiply by u' = 2a x", "Simplify"],
            solution=f"$f'(x) = {L(d)}$.",
            misconceptions=[(fmt(wrong), "chain-forgot-inner-derivative")],
        )
    x0 = rng.randint(1, 3)
    g0 = rng.randint(1, 4)
    while g0 == x0:
        g0 = rng.randint(1, 4)
    g1, f0, f1 = nz(rng, -4, 4), nz(rng, -5, 5), nz(rng, -4, 4)
    fx0, fpx0, gg0, gpg0 = nz(rng, -5, 5), nz(rng, -4, 4), rng.randint(-3, 5), nz(rng, -4, 4)
    ans = 2 * f0 * f1 * g1
    table = (f"| | $x={x0}$ | $x={g0}$ |\n|---|---|---|\n"
             f"| $f(x)$ | ${fx0}$ | ${f0}$ |\n| $f'(x)$ | ${fpx0}$ | ${f1}$ |\n"
             f"| $g(x)$ | ${g0}$ | ${gg0}$ |\n| $g'(x)$ | ${g1}$ | ${gpg0}$ |")
    return Problem(
        prompt=(f"Let $h(x) = \\big[f(g(x))\\big]^2$. Using the table below, compute $h'({x0})$.\n\n{table}"),
        answer=str(ans), kind="num",
        rubric=["Identify three layers: square ∘ f ∘ g", "h'(x) = 2 f(g(x)) · f'(g(x)) · g'(x)",
                f"Evaluate g({x0}) = {g0} first", f"Read f({g0}), f'({g0}) and g'({x0}) from the table"],
        solution=(f"$h'({x0}) = 2 f(g({x0}))\\, f'(g({x0}))\\, g'({x0}) = 2 f({g0}) f'({g0}) g'({x0})"
                  f" = 2\\cdot({f0})({f1})({g1}) = {ans}$."),
        misconceptions=[(str(2 * f0 * f1), "chain-forgot-inner-derivative"), (str(f1 * g1), "chain-forgot-outer-power"),
                        (str(2 * fx0 * fpx0 * g1), "evaluated-f-at-x-not-g-x")],
    )


@gen("calc.chain_rule", name="implicit_and_related_rates")
def implicit(rng, level):
    if level == 1:
        p, q, r = rng.choice(TRIPLES)
        if rng.random() < 0.5:
            p, q = q, p
        p, q = p * rng.choice([-1, 1]), q * rng.choice([-1, 1])
        slope = sp.Rational(-p, q)
        return Problem(
            prompt=f"Use implicit differentiation on $x^2 + y^2 = {r**2}$ to find $\\frac{{dy}}{{dx}}$ at the point $({p}, {q})$.",
            answer=fmt(slope), kind="num",
            rubric=["Differentiate both sides w.r.t. x, treating y as y(x)", "d/dx y² = 2y y' (chain rule)",
                    "Solve: y' = −x/y", f"Substitute ({p}, {q})"],
            solution=f"$2x + 2y y' = 0 \\Rightarrow y' = -x/y = {L(slope)}$.",
            misconceptions=[(fmt(sp.Rational(-q, p)), "inverted-slope"), (fmt(-slope), "sign-error")],
        )
    if level == 2:
        while True:
            a, p, q = nz(rng, -3, 3), nz(rng, -3, 3), nz(rng, -3, 3)
            if a * p + 2 * q != 0:
                break
        c = p ** 2 + a * p * q + q ** 2
        slope = sp.Rational(-(2 * p + a * q), a * p + 2 * q)
        return Problem(
            prompt=(f"The curve ${L(x**2 + a*x*y + y**2)} = {c}$ passes through $({p}, {q})$. "
                    "Find $\\frac{dy}{dx}$ there."),
            answer=fmt(slope), kind="num",
            rubric=["Differentiate implicitly; use the product rule on the xy term",
                    "Collect the y' terms on one side", "Solve for y'", f"Substitute ({p}, {q})"],
            solution=(f"$2x {a:+d}(y + x y') + 2y y' = 0 \\Rightarrow y' = -\\frac{{{L(2*x + a*y)}}}{{{L(a*x + 2*y)}}}$; "
                      f"at $({p},{q})$: $y' = {L(slope)}$."),
            misconceptions=[(fmt(sp.Rational(-p, q)), "dropped-product-rule-term")],
        )
    p, q, ld = rng.choice(TRIPLES)
    if rng.random() < 0.5:
        p, q = q, p
    v = rng.randint(1, 4)
    rate = sp.Rational(-p * v, q)
    return Problem(
        prompt=(f"A {ld} m ladder leans against a vertical wall. Its foot slides away from the wall at {v} m/s. "
                f"How fast is the top of the ladder moving when the foot is {p} m from the wall? "
                "Give $dy/dt$ in m/s, where $y$ is the height of the top (negative means it is moving down)."),
        answer=fmt(rate), kind="num",
        rubric=[f"Relation x² + y² = {ld}²", "Differentiate in t: 2x x' + 2y y' = 0",
                f"Find y = {q} when x = {p}", f"Solve y' = −x x'/y with x' = {v}"],
        solution=f"$y = \\sqrt{{{ld}^2 - {p}^2}} = {q}$, $\\dot y = -\\frac{{x\\dot x}}{{y}} = -\\frac{{{p}\\cdot{v}}}{{{q}}} = {L(rate)}$ m/s.",
        misconceptions=[(fmt(-rate), "related-rates-sign"), (fmt(sp.Rational(-q * v, p)), "swapped-legs")],
    )


# =========================================================================== transcendental derivatives
@gen("calc.transcendental_deriv")
def transcendental(rng, level):
    if level == 1:
        a, b = nz(rng, -6, 6), nz(rng, -5, 5)
        kind = rng.choice(["exp", "sin", "cos", "ln"])
        if kind == "exp":
            f, wrong, tag = a * sp.exp(b * x), a * sp.exp(b * x), "chain-forgot-inner-derivative"
        elif kind == "sin":
            f, wrong, tag = a * sp.sin(b * x), a * sp.cos(b * x), "chain-forgot-inner-derivative"
        elif kind == "cos":
            f, wrong, tag = a * sp.cos(b * x), a * b * sp.sin(b * x), "derivative-of-cos-sign"
        else:
            c = rng.randint(1, 6)
            b = abs(b)
            f, wrong, tag = sp.log(b * x + c), 1 / (b * x + c), "chain-forgot-inner-derivative"
        d = sp.diff(f, x)
        return Problem(
            prompt=f"Differentiate $f(x) = {L(f)}$.",
            answer=fmt(d), kind="expr",
            rubric=["Recall the basic derivative (eˣ, sin, cos, ln)", "Multiply by the derivative of the inner linear function"],
            solution=f"$f'(x) = {L(d)}$.",
            misconceptions=[(fmt(wrong), tag)],
        )
    if level == 2:
        kind = rng.choice(["xexp", "lnquad", "expsin"])
        a = nz(rng, -3, 3)
        if kind == "xexp":
            n = rng.randint(1, 3)
            f = x ** n * sp.exp(a * x)
            wrong, tag = n * x ** (n - 1) * a * sp.exp(a * x), "product-of-derivatives"
            steps = ["Product rule with u = xⁿ, v = e^(ax)", "v' = a e^(ax) (chain rule)", "Factor out e^(ax)"]
        elif kind == "lnquad":
            a = abs(a)
            f = sp.log(x ** 2 + a)
            wrong, tag = 1 / (x ** 2 + a), "chain-forgot-inner-derivative"
            steps = ["Outer ln u → 1/u", "Inner u = x² + a → u' = 2x", "Multiply"]
        else:
            b = nz(rng, 1, 4)
            f = sp.exp(a * x) * sp.sin(b * x)
            wrong, tag = a * b * sp.exp(a * x) * sp.cos(b * x), "product-of-derivatives"
            steps = ["Product rule", "d/dx e^(ax) = a e^(ax), d/dx sin(bx) = b cos(bx)", "Factor out e^(ax)"]
        d = sp.diff(f, x)
        return Problem(
            prompt=f"Differentiate $f(x) = {L(f)}$.",
            answer=fmt(d), kind="expr",
            rubric=steps,
            solution=f"$f'(x) = {L(sp.factor(d))}$.",
            misconceptions=[(fmt(wrong), tag)],
        )
    x0 = rng.choice([-2, -1, 1, 2, 3])
    m = rng.choice([2, 3, 4])
    yv = rng.choice([0, 1])
    sig = 1 / (1 + sp.exp(-w * x0))
    ell = -(yv * sp.log(sig) + (1 - yv) * sp.log(1 - sig))
    val = sp.nsimplify(sp.simplify(sp.diff(ell, w).subs(w, sp.log(m) / x0)))
    p = sp.Rational(m, m + 1)
    return Problem(
        prompt=(f"Logistic regression with one weight: $\\sigma(s) = \\dfrac{{1}}{{1+e^{{-s}}}}$ and, for one training example "
                f"$(x, y) = ({x0}, {yv})$, the cross-entropy loss is "
                f"$\\ell(w) = -\\big[y\\ln\\sigma(wx) + (1-y)\\ln(1-\\sigma(wx))\\big]$.\n\n"
                "1. Show that $\\sigma'(s) = \\sigma(s)(1-\\sigma(s))$.\n"
                "2. Derive $\\ell'(w)$ in closed form.\n"
                f"3. Evaluate $\\ell'(w)$ at $w = \\dfrac{{\\ln {m}}}{{{x0}}}$ and type the number."),
        answer=fmt(val), kind="num",
        rubric=["σ'(s) = e^(-s)/(1+e^(-s))² = σ(s)(1−σ(s))",
                "d/dw ln σ(wx) = (1 − σ(wx)) x and d/dw ln(1−σ(wx)) = −σ(wx) x",
                "Combine: ℓ'(w) = (σ(wx) − y) x",
                f"At w x = ln {m}: σ = {m}/{m+1}", "Substitute and simplify"],
        solution=(f"$\\ell'(w) = (\\sigma(wx) - y)\\,x$. With $wx = \\ln {m}$, $\\sigma = \\frac{{{m}}}{{{m+1}}}$, so "
                  f"$\\ell' = \\left({L(p)} - {yv}\\right)\\cdot({x0}) = {L(val)}$."),
        misconceptions=[(fmt(p - yv), "chain-forgot-inner-derivative"), (fmt(-val), "loss-gradient-sign")],
    )


# =========================================================================== applications
@gen("calc.applications")
def applications(rng, level):
    if level == 1:
        a, b, c, x0 = nz(rng, -4, 4), rng.randint(-6, 6), rng.randint(-6, 6), rng.randint(-3, 3)
        f = a * x ** 2 + b * x + c
        m = sp.diff(f, x).subs(x, x0)
        y0 = f.subs(x, x0)
        line = sp.expand(m * (x - x0) + y0)
        return Problem(
            prompt=f"Find the tangent line to $y = {L(f)}$ at $x = {x0}$. Answer as an expression in $x$ (e.g. 3*x - 2).",
            answer=fmt(line), kind="expr", answer_hint="the line as an expression in x, e.g. 3*x - 2",
            rubric=[f"Point: f({x0}) = {y0}", f"Slope: f'({x0}) = {m}", "Point-slope form y − y0 = m(x − x0)", "Simplify"],
            solution=f"$y = {m}(x - ({x0})) + ({y0}) = {L(line)}$.",
            misconceptions=[(fmt(m * x), "tangent-missing-intercept"), (fmt(sp.diff(f, x)), "derivative-function-not-line")],
        )
    if level == 2:
        if rng.random() < 0.5:
            a, xs = rng.randint(1, 6), rng.randint(1, 5)
            b = a * xs ** 2
            mn = 2 * a * xs
            return Problem(
                prompt=f"Find the minimum value of $f(x) = {a}x + \\dfrac{{{b}}}{{x}}$ for $x > 0$.",
                answer=str(mn), kind="num",
                rubric=["Set f'(x) = a − b/x² = 0", f"Solve: x* = {xs} (only the positive root)",
                        "Check it is a minimum (f'' > 0 for x > 0)", "Evaluate f(x*)"],
                solution=f"$f'(x) = {a} - {b}/x^2 = 0 \\Rightarrow x^* = {xs}$; $f''>0$; $f({xs}) = {a*xs} + {b//xs} = {mn}$.",
                misconceptions=[(str(xs), "gave-minimiser-not-minimum")],
            )
        P = 4 * rng.randint(5, 30)
        A = sp.Rational(P * P, 8)
        return Problem(
            prompt=(f"A farmer has {P} m of fence to enclose a rectangular field along a straight river; the river side "
                    "needs no fence. What is the largest area (in m²) she can enclose?"),
            answer=fmt(A), kind="num",
            rubric=["Variables: x = side perpendicular to river, y = side parallel", f"Constraint 2x + y = {P}",
                    f"Area A(x) = x({P} − 2x)", "Set A'(x) = 0, solve", "Evaluate the area"],
            solution=f"$A = x({P}-2x)$, $A' = {P} - 4x = 0 \\Rightarrow x = {P//4}$, $y = {P//2}$, $A = {L(A)}$ m².",
            misconceptions=[(fmt(sp.Rational(P * P, 16)), "assumed-square-four-sides")],
        )
    while True:
        xs = [nz(rng, -3, 3) for _ in range(3)]
        ys = [rng.randint(-5, 5) for _ in range(3)]
        sxy, sxx = sum(a * b for a, b in zip(xs, ys)), sum(a * a for a in xs)
        if sxy != 0:
            break
    E = sum((w * a - b) ** 2 for a, b in zip(xs, ys))
    ws = sp.solve(sp.diff(E, w), w)[0]
    pts = ", ".join(f"({a}, {b})" for a, b in zip(xs, ys))
    wrongs = []
    if sum(xs) != 0:
        wrongs.append((fmt(sp.Rational(sum(ys), sum(xs))), "ratio-of-means"))
    return Problem(
        prompt=(f"Fit the model $\\hat y = w x$ (a line through the origin) to the data points {pts} by minimising "
                "the squared error $E(w) = \\sum_k (w x_k - y_k)^2$. Derive the minimiser on paper and type $w^*$."),
        answer=fmt(ws), kind="num",
        rubric=["E'(w) = 2 Σ x_k (w x_k − y_k)", "Set E'(w) = 0: w* = Σ x_k y_k / Σ x_k²",
                f"Σ x y = {sxy}, Σ x² = {sxx}", "E''(w) = 2 Σ x_k² > 0 so it is a minimum"],
        solution=f"$w^* = \\dfrac{{\\sum x_k y_k}}{{\\sum x_k^2}} = \\dfrac{{{sxy}}}{{{sxx}}} = {L(ws)}$.",
        misconceptions=wrongs,
    )


# =========================================================================== basic integrals
@gen("calc.integrals_basic")
def integrals_basic(rng, level):
    if level == 1:
        a = rng.randint(-2, 2)
        b = rng.randint(a + 1, 3)
        p, q, r = nz(rng, -4, 4), rng.randint(-5, 5), rng.randint(-5, 5)
        f = 3 * p * x ** 2 + 2 * q * x + r
        F = sp.integrate(f, x)
        val = sp.integrate(f, (x, a, b))
        return Problem(
            prompt=f"Compute $\\displaystyle\\int_{{{a}}}^{{{b}}} \\left({L(f)}\\right)\\,dx$.",
            answer=fmt(val), kind="num",
            rubric=["Find an antiderivative F (reverse power rule)", "F(b) − F(a)", "Careful with signs at the lower limit"],
            solution=f"$F(x) = {L(F)}$, so $F({b}) - F({a}) = {F.subs(x, b)} - ({F.subs(x, a)}) = {L(val)}$.",
            misconceptions=[(fmt(F.subs(x, b)), "forgot-lower-limit")],
        )
    if level == 2:
        r1, r2 = sorted(rng.sample(range(-3, 5), 2))
        line = sp.expand((r1 + r2) * x - r1 * r2)
        area = sp.integrate(line - x ** 2, (x, r1, r2))
        return Problem(
            prompt=f"Find the area of the region enclosed by $y = x^2$ and $y = {L(line)}$.",
            answer=fmt(area), kind="num",
            rubric=["Find intersection points by solving x² = line", f"x = {r1} and x = {r2}",
                    "Determine which curve is on top in between", "Integrate (top − bottom)"],
            solution=(f"Intersections at $x = {r1}, {r2}$; the line is above the parabola there. "
                      f"$\\int_{{{r1}}}^{{{r2}}} ({L(line)} - x^2)\\,dx = {L(area)}$."),
            misconceptions=[(fmt(-area), "subtracted-wrong-way")],
        )
    a = rng.randint(0, 2)
    b, c = rng.randint(-3, 3), rng.randint(-3, 3)
    x0 = nz(rng, -2, 2)
    g = t ** 2 + b * t + c
    ans = sp.diff(sp.integrate(g, (t, a, x ** 2)), x).subs(x, x0)
    return Problem(
        prompt=(f"Let $F(x) = \\displaystyle\\int_{{{a}}}^{{x^2}} \\left({L(g)}\\right)\\,dt$. "
                f"Using the Fundamental Theorem of Calculus (without computing the integral), find $F'({x0})$."),
        answer=fmt(ans), kind="num",
        rubric=["FTC part 1: d/du ∫_a^u g(t) dt = g(u)", "Chain rule with u = x², du/dx = 2x",
                "F'(x) = g(x²) · 2x", f"Evaluate at x = {x0}"],
        solution=f"$F'(x) = g(x^2)\\cdot 2x = ({L(g.subs(t, x**2))})\\cdot 2x$, so $F'({x0}) = {L(ans)}$.",
        misconceptions=[(fmt(g.subs(t, x0 ** 2)), "ftc-forgot-chain-factor"), (fmt(g.subs(t, x0) * 2 * x0), "ftc-wrong-argument")],
    )


# =========================================================================== integration techniques
@gen("calc.integration_techniques")
def integration_techniques(rng, level):
    if level == 1:
        a, b, n = rng.randint(0, 3), rng.randint(1, 2), rng.randint(1, 3)
        f = x * (x ** 2 + a) ** n
        val = sp.integrate(f, (x, 0, b))
        return Problem(
            prompt=f"Compute $\\displaystyle\\int_0^{{{b}}} x\\left({L(x**2 + a)}\\right)^{{{n}}}\\,dx$ using a substitution.",
            answer=fmt(val), kind="num",
            rubric=[f"u = x² + {a}, du = 2x dx", f"Change the limits: u from {a} to {b*b + a}",
                    "Integrate u^n / 2", "Evaluate"],
            solution=(f"$= \\frac12\\int_{{{a}}}^{{{b*b + a}}} u^{{{n}}}\\,du = \\frac{{1}}{{{2*(n+1)}}}\\left({b*b + a}^{{{n+1}}} - {a}^{{{n+1}}}\\right) = {L(val)}$."),
            misconceptions=[(fmt(2 * val), "u-sub-forgot-du-factor")],
        )
    if level == 2:
        kind = rng.choice(["xexp", "xlnx", "xsin"])
        if kind == "xexp":
            a = rng.choice([1, 2, 3, -1])
            f, lo, hi, lo_t, hi_t = x * sp.exp(a * x), 0, 1, "0", "1"
            uv = (x * sp.exp(a * x) / a)
            steps = ["u = x, dv = e^(ax) dx", "v = e^(ax)/a", "uv| − ∫ v du"]
        elif kind == "xlnx":
            n = rng.randint(1, 3)
            f, lo, hi, lo_t, hi_t = x ** n * sp.log(x), 1, sp.E, "1", "e"
            uv = x ** (n + 1) / (n + 1) * sp.log(x)
            steps = ["u = ln x, dv = xⁿ dx", "v = x^(n+1)/(n+1), du = dx/x", "uv| − ∫ v du"]
        else:
            k = rng.randint(1, 3)
            f, lo, hi, lo_t, hi_t = x * sp.sin(k * x), 0, sp.pi, "0", "\\pi"
            uv = -x * sp.cos(k * x) / k
            steps = ["u = x, dv = sin(kx) dx", "v = −cos(kx)/k", "uv| − ∫ v du", "cos(kπ) = (−1)^k"]
        val = sp.simplify(sp.integrate(f, (x, lo, hi)))
        boundary = sp.simplify(uv.subs(x, hi) - uv.subs(x, lo))
        return Problem(
            prompt=f"Compute exactly $\\displaystyle\\int_{{{lo_t}}}^{{{hi_t}}} {L(f)}\\,dx$ (integration by parts).",
            answer=fmt(val), kind="num", answer_hint="exact form, e.g. (e^2+1)/4 or pi/2 - 1",
            rubric=steps + ["Evaluate between the limits"],
            solution=f"Boundary term $uv\\big|_{{{lo_t}}}^{{{hi_t}}} = {L(boundary)}$; total $= {L(val)} \\approx {float(val):.4f}$.",
            misconceptions=[(fmt(boundary), "parts-dropped-integral-term")],
        )
    n = rng.randint(1, 3)
    lam = rng.choice([2, 3, sp.Rational(1, 2)])
    val = sp.integrate(x ** n * lam * sp.exp(-lam * x), (x, 0, sp.oo))
    return Problem(
        prompt=(f"Let $X \\sim \\text{{Exponential}}(\\lambda = {L(lam)})$ with density $p(x) = \\lambda e^{{-\\lambda x}}$, $x\\ge 0$. "
                f"Compute $\\mathbb{{E}}[X^{{{n}}}] = \\displaystyle\\int_0^\\infty x^{{{n}}}\\,\\lambda e^{{-\\lambda x}}\\,dx$ "
                "exactly. On paper, derive the reduction formula by integration by parts."),
        answer=fmt(val), kind="num",
        rubric=["Parts with u = xⁿ, dv = λ e^(−λx) dx", "Boundary term vanishes at 0 and ∞",
                "Reduction: E[Xⁿ] = (n/λ) E[X^(n−1)], E[X⁰] = 1", f"So E[X^{n}] = {n}!/λ^{n}"],
        solution=f"$\\mathbb{{E}}[X^n] = \\frac{{n!}}{{\\lambda^n}} = \\frac{{{sp.factorial(n)}}}{{{L(lam)}^{{{n}}}}} = {L(val)}$.",
        misconceptions=[(fmt(sp.factorial(n) / lam ** (n + 1)), "dropped-density-lambda"),
                        (fmt(sp.factorial(n) / lam), "power-of-lambda")],
    )


# =========================================================================== series / Taylor
@gen("calc.series_taylor")
def series_taylor(rng, level):
    if level == 1:
        a = nz(rng, -3, 3)
        kind = rng.choice(["exp", "cos", "geom", "log"])
        if kind == "exp":
            f, k = sp.exp(a * x), rng.randint(2, 4)
        elif kind == "cos":
            a = nz(rng, -2, 2)
            f, k = sp.cos(a * x), rng.choice([2, 4])
        elif kind == "geom":
            f, k = 1 / (1 - a * x), rng.randint(2, 4)
        else:
            f, k = sp.log(1 + a * x), rng.randint(2, 4)
        coef = sp.series(f, x, 0, k + 1).removeO().coeff(x, k)
        wrongs = [(fmt(-coef), "taylor-alternating-sign")]
        if kind in ("exp", "cos"):
            wrongs.append((fmt(coef * sp.factorial(k)), "taylor-forgot-factorial"))
        return Problem(
            prompt=f"What is the coefficient of $x^{{{k}}}$ in the Maclaurin series of $f(x) = {L(f)}$?",
            answer=fmt(coef), kind="num",
            rubric=["Recall the standard series (eᵘ, cos u, 1/(1−u), ln(1+u))", f"Substitute u = {a}x",
                    f"Pick out the x^{k} term (including (a)^{k} and any factorial)"],
            solution=f"${L(f)} = {L(sp.series(f, x, 0, k + 2).removeO())} + \\dots$, so the coefficient is ${L(coef)}$.",
            misconceptions=wrongs,
        )
    if level == 2:
        kind = rng.choice(["expmul", "log", "sqrt"])
        a = nz(rng, -2, 2)
        if kind == "expmul":
            b = nz(rng, -3, 3)
            f = sp.exp(a * x) * (1 + b * x)
        elif kind == "log":
            f = sp.log(1 + a * x) - a * x ** 2 if rng.random() < 0.5 else sp.log(1 + a * x)
        else:
            a = rng.choice([1, 2, 4, -4])
            f = sp.sqrt(1 + a * x)
        poly = sp.series(f, x, 0, 4).removeO()
        return Problem(
            prompt=f"Find the Maclaurin polynomial of degree 3 of $f(x) = {L(f)}$.",
            answer=fmt(sp.expand(poly)), kind="expr",
            rubric=["Either compute f(0), f'(0), f''(0), f'''(0) or combine known series",
                    "Coefficient of xᵏ is f⁽ᵏ⁾(0)/k!", "Collect terms up to x³ and discard higher powers"],
            solution=f"$T_3(x) = {L(sp.expand(poly))}$.",
            misconceptions=[(fmt(sp.series(f, x, 0, 3).removeO()), "truncated-too-early")],
        )
    kind = rng.choice(["softplus", "limit", "sqrt"])
    if kind == "softplus":
        a = rng.choice([1, 2, 3, -1, -2])
        f = sp.log(1 + sp.exp(a * s))
        poly = sp.expand(sp.series(f, s, 0, 3).removeO())
        return Problem(
            prompt=(f"The softplus function $\\mathrm{{sp}}(s) = \\ln(1+e^{{s}})$ is a smooth ReLU. Find the second-order "
                    f"Taylor polynomial of $f(s) = \\ln\\left(1 + e^{{{a}s}}\\right)$ about $s = 0$ (an expression in $s$)."),
            answer=fmt(poly), kind="expr", answer_hint="an expression in s, e.g. log(2) + s/2 + s^2/8",
            rubric=["f(0) = ln 2", "f'(s) = a σ(as), so f'(0) = a/2",
                    "f''(s) = a² σ(as)(1 − σ(as)), so f''(0) = a²/4", "T₂ = f(0) + f'(0)s + f''(0)s²/2"],
            solution=f"$T_2(s) = {L(poly)}$.",
            misconceptions=[(fmt(sp.log(2) + sp.Rational(a, 2) * s + sp.Rational(a * a, 4) * s ** 2), "taylor-forgot-factorial")],
        )
    if kind == "limit":
        a = nz(rng, -3, 3)
        lim = sp.limit((sp.sin(a * x) - a * x) / x ** 3, x, 0)
        return Problem(
            prompt=f"Use Taylor series (not L'Hôpital) to evaluate $\\displaystyle\\lim_{{x\\to0}} \\frac{{\\sin({a}x) - ({a})x}}{{x^3}}$.",
            answer=fmt(lim), kind="num",
            rubric=["sin u = u − u³/6 + O(u⁵)", f"Substitute u = {a}x", "Linear terms cancel", "Divide by x³ and let x → 0"],
            solution=f"$\\sin({a}x) - ({a})x = -\\frac{{({a})^3 x^3}}{{6}} + O(x^5)$, limit $= {L(lim)}$.",
            misconceptions=[(fmt(-lim), "taylor-sign-error"), (fmt(-lim * 6), "taylor-forgot-factorial")],
        )
    m = rng.choice([2, 3, 4, 5, 10])
    d = rng.choice([-2, -1, 1, 2])
    approx = sp.series(sp.sqrt(m ** 2 + u), u, 0, 3).removeO().subs(u, d)
    return Problem(
        prompt=(f"Approximate $\\sqrt{{{m**2 + d}}}$ using the **second-order** Taylor polynomial of $g(x) = \\sqrt{{x}}$ "
                f"about $x = {m**2}$. Give the exact rational value of the approximation."),
        answer=fmt(approx), kind="num",
        rubric=[f"g({m**2}) = {m}", f"g'(x) = 1/(2√x) → 1/{2*m}", f"g''(x) = −1/(4 x^(3/2)) → −1/{4*m**3}",
                f"T₂ = g + g'·({d}) + g''·({d})²/2"],
        solution=(f"$T_2 = {m} + \\frac{{{d}}}{{{2*m}}} - \\frac{{({d})^2}}{{{8*m**3}}} = {L(approx)} \\approx {float(approx):.6f}$ "
                  f"(true value ${float(sp.sqrt(m**2 + d)):.6f}$)."),
        misconceptions=[(fmt(m + sp.Rational(d, 2 * m)), "stopped-at-first-order")],
    )


# =========================================================================== partial derivatives
@gen("calc.partial_derivatives")
def partials(rng, level):
    if level == 1:
        a, b, c, d = nz(rng, -5, 5), nz(rng, -6, 6), rng.randint(-5, 5), rng.randint(-5, 5)
        p, q = rng.randint(2, 3), rng.randint(1, 3)
        f = a * x ** p * y ** q + b * x * y + c * y ** 2 + d * x
        var, other = (x, y) if rng.random() < 0.5 else (y, x)
        dd = sp.diff(f, var)
        return Problem(
            prompt=f"Let $f(x, y) = {L(f)}$. Compute $\\dfrac{{\\partial f}}{{\\partial {var}}}$.",
            answer=fmt(dd), kind="expr",
            rubric=[f"Treat {other} as a constant", f"Differentiate each term with respect to {var}",
                    f"Terms without {var} vanish"],
            solution=f"$\\partial f/\\partial {var} = {L(dd)}$.",
            misconceptions=[(fmt(sp.diff(f, other)), "differentiated-wrong-variable")],
        )
    if level == 2:
        a = nz(rng, -3, 3)
        kind = rng.choice(["exp", "log", "trig"])
        if kind == "exp":
            f = sp.exp(a * x * y)
            wrong, tag = a ** 2 * x * y * sp.exp(a * x * y), "product-rule-dropped-term"
        elif kind == "log":
            a = abs(a)
            f = sp.log(x ** 2 + a * y ** 2)
            wrong, tag = sp.Integer(0), "mixed-partial-assumed-zero"
        else:
            p = rng.randint(2, 3)
            f = x ** p * sp.sin(a * y)
            wrong, tag = p * x ** (p - 1) * sp.sin(a * y), "only-one-partial-taken"
        fxy = sp.simplify(sp.diff(f, x, y))
        return Problem(
            prompt=f"Let $f(x, y) = {L(f)}$. Compute the mixed partial $f_{{xy}} = \\dfrac{{\\partial^2 f}}{{\\partial y\\,\\partial x}}$.",
            answer=fmt(fxy), kind="expr",
            rubric=["First compute f_x (y constant)", "Differentiate f_x with respect to y (product/chain rule as needed)",
                    "Simplify; optionally verify f_xy = f_yx (Clairaut)"],
            solution=f"$f_x = {L(sp.diff(f, x))}$, $f_{{xy}} = {L(fxy)}$.",
            misconceptions=[(fmt(wrong), tag)],
        )
    if rng.random() < 0.5:
        D = rng.choice([1, 2, 3, sp.Rational(1, 2)])
        a = rng.randint(1, 4)
        k = D * a ** 2
        return Problem(
            prompt=(f"For which constant $k$ does $u(x, t) = e^{{-kt}}\\sin({a}x)$ solve the heat equation "
                    f"$u_t = {L(D)}\\,u_{{xx}}$?"),
            answer=fmt(k), kind="num",
            rubric=["u_t = −k e^(−kt) sin(ax)", "u_xx = −a² e^(−kt) sin(ax)", "Substitute and cancel the common factor", "Solve for k"],
            solution=f"$-k\\,u = {L(D)}\\cdot(-{a**2})\\,u \\Rightarrow k = {L(k)}$.",
            misconceptions=[(fmt(D * a), "forgot-square-from-second-derivative"), (fmt(-k), "sign-error")],
        )
    p = nz(rng, -4, 4)
    k = -3 * p
    return Problem(
        prompt=(f"Find the constant $k$ such that $f(x, y) = {p}x^3 + k\\,x y^2$ is harmonic, i.e. "
                "$f_{xx} + f_{yy} = 0$ everywhere."),
        answer=str(k), kind="num",
        rubric=[f"f_xx = {6*p}x", "f_yy = 2k x", "Sum must vanish for all x", "Solve for k"],
        solution=f"${6*p}x + 2kx = 0 \\Rightarrow k = {k}$.",
        misconceptions=[(str(-k), "sign-error"), (str(-p), "forgot-derivative-factors")],
    )


# =========================================================================== gradients
@gen("calc.gradient")
def gradient(rng, level):
    if level == 1:
        a, b, c = nz(rng, -4, 4), nz(rng, -4, 4), rng.randint(-5, 5)
        x0, y0 = rng.randint(-3, 3), rng.randint(-3, 3)
        f = a * x ** 2 * y + b * y ** 2 + c * x
        g = [sp.diff(f, v).subs({x: x0, y: y0}) for v in (x, y)]
        return Problem(
            prompt=f"Compute the gradient $\\nabla f({x0}, {y0})$ for $f(x, y) = {L(f)}$.",
            answer=vec(g), kind="vector", answer_hint="[f_x, f_y], e.g. [3, -2]",
            rubric=["f_x (y constant)", "f_y (x constant)", f"Evaluate both at ({x0}, {y0})"],
            solution=f"$\\nabla f = ({L(sp.diff(f, x))},\\ {L(sp.diff(f, y))})$, at the point: ${tup(g)}$.",
            misconceptions=[(fmt(list(reversed(g))), "gradient-components-swapped")],
        )
    if level == 2:
        a, b, c = nz(rng, -3, 3), rng.randint(-4, 4), nz(rng, -3, 3)
        x0, y0 = rng.randint(-2, 2), rng.randint(-2, 2)
        p, q, n_ = rng.choice(TRIPLES[:2])
        p, q = p * rng.choice([-1, 1]), q * rng.choice([-1, 1])
        f = a * x ** 2 + b * x * y + c * y ** 2
        g = [sp.diff(f, v).subs({x: x0, y: y0}) for v in (x, y)]
        while g == [0, 0]:
            x0, y0 = rng.randint(-2, 2), rng.randint(-2, 2)
            g = [sp.diff(f, v).subs({x: x0, y: y0}) for v in (x, y)]
        dot = g[0] * p + g[1] * q
        D = sp.Rational(dot, n_)
        return Problem(
            prompt=(f"Let $f(x, y) = {L(f)}$. Find the directional derivative of $f$ at $({x0}, {y0})$ "
                    f"in the direction of the vector $({p}, {q})$."),
            answer=fmt(D), kind="num",
            rubric=["Compute ∇f at the point", f"Normalise the direction: ‖({p},{q})‖ = {n_}",
                    "D_u f = ∇f · u"],
            solution=f"$\\nabla f = {tup(g)}$, $u = \\frac{{1}}{{{n_}}}({p}, {q})$, $D_u f = \\frac{{{dot}}}{{{n_}}} = {L(D)}$.",
            misconceptions=[(str(dot), "forgot-unit-vector")],
        )
    while True:
        xs = [rng.randint(-2, 3) for _ in range(3)]
        ys = [rng.randint(-3, 4) for _ in range(3)]
        w0, b0 = rng.randint(-2, 2), rng.randint(-2, 2)
        res = [w0 * a + b0 - b for a, b in zip(xs, ys)]
        gw, gb = sum(r * a for r, a in zip(res, xs)), sum(res)
        if (gw, gb) != (0, 0) and len(set(xs)) > 1:
            break
    bb = sp.Symbol("b")
    Lf = sp.Rational(1, 2) * sum((w * a + bb - b) ** 2 for a, b in zip(xs, ys))
    g = [sp.diff(Lf, v).subs({w: w0, bb: b0}) for v in (w, bb)]
    pts = ", ".join(f"({a}, {b})" for a, b in zip(xs, ys))
    return Problem(
        prompt=(f"Linear regression $\\hat y = w x + b$ on the data $(x_k, y_k)$ = {pts} with loss "
                "$\\mathcal{L}(w, b) = \\frac12\\sum_k (w x_k + b - y_k)^2$. Derive $\\nabla\\mathcal{L}$ in general "
                f"(in terms of the residuals $r_k = w x_k + b - y_k$), then evaluate it at $(w, b) = ({w0}, {b0})$."),
        answer=vec(g), kind="vector", answer_hint="[dL/dw, dL/db]",
        rubric=["∂L/∂w = Σ r_k x_k (chain rule)", "∂L/∂b = Σ r_k", f"Residuals at ({w0},{b0}): {res}",
                "Sum them up"],
        solution=(f"Residuals $r = {tup(res)}$. $\\partial_w \\mathcal{{L}} = \\sum r_k x_k = {gw}$, "
                  f"$\\partial_b \\mathcal{{L}} = \\sum r_k = {gb}$."),
        misconceptions=[(fmt([gb, gb]), "chain-forgot-inner-derivative"), (fmt([-gw, -gb]), "loss-gradient-sign")],
    )


def _sym2(rng, lo=1, hi=8):
    """Symmetric 2x2 integer matrix with integer eigenvalues in [lo, hi] (distinct)."""
    while True:
        dd, bb = rng.choice([(0, 1), (0, 2), (3, 2), (6, 4), (8, 3), (0, 3)])
        dd *= rng.choice([-1, 1])
        bb *= rng.choice([-1, 1])
        c = rng.randint(-3, 6)
        A = sp.Matrix([[c + dd, bb], [bb, c]])
        ev = sorted(A.eigenvals().keys())
        if all(v.is_integer for v in ev) and lo <= ev[0] and ev[-1] <= hi and len(ev) == 2:
            return A, ev


@gen("calc.gradient", name="gradient_descent")
def gradient_descent(rng, level):
    if level == 1:
        a, b = rng.randint(1, 4), rng.randint(1, 4)
        x0, y0 = nz(rng, -4, 4), nz(rng, -4, 4)
        eta = rng.choice([sp.Rational(1, 10), sp.Rational(1, 4), sp.Rational(1, 8)])
        f = a * x ** 2 + b * y ** 2
        g = [sp.diff(f, v).subs({x: x0, y: y0}) for v in (x, y)]
        new = [x0 - eta * g[0], y0 - eta * g[1]]
        return Problem(
            prompt=(f"Take **one** gradient-descent step with learning rate $\\eta = {L(eta)}$ on $f(x, y) = {L(f)}$ "
                    f"starting from $({x0}, {y0})$. Give the new point."),
            answer=vec(new), kind="vector", answer_hint="[x1, y1]",
            rubric=["∇f = (2a x, 2b y)", f"Evaluate ∇f at ({x0},{y0})", "x_new = x − η ∇f"],
            solution=f"$\\nabla f = {tup(g)}$; new point $= ({x0}, {y0}) - {L(eta)}{tup(g)} = {tup(new)}$.",
            misconceptions=[(fmt([x0 + eta * g[0], y0 + eta * g[1]]), "gradient-ascent-not-descent")],
        )
    if level == 2:
        a = rng.choice([1, 2, 3])
        p, q = rng.randint(-3, 3), rng.randint(-3, 3)
        x0, y0 = rng.randint(-2, 2), rng.randint(-2, 2)
        while (x0, y0) == (p, q):
            x0, y0 = rng.randint(-2, 2), rng.randint(-2, 2)
        eta = rng.choice([sp.Rational(1, 4), sp.Rational(1, 8)])
        f = (x - p) ** 2 + a * (y - q) ** 2
        pt = [sp.Integer(x0), sp.Integer(y0)]
        g0 = [sp.diff(f, v).subs({x: x0, y: y0}) for v in (x, y)]
        stale = [x0 - 2 * eta * g0[0], y0 - 2 * eta * g0[1]]
        hist = [tup(pt)]
        for _ in range(2):
            g = [sp.diff(f, v).subs({x: pt[0], y: pt[1]}) for v in (x, y)]
            pt = [pt[0] - eta * g[0], pt[1] - eta * g[1]]
            hist.append(tup(pt))
        return Problem(
            prompt=(f"Run **two** steps of gradient descent with $\\eta = {L(eta)}$ on $f(x, y) = {L(f)}$, starting at "
                    f"$({x0}, {y0})$. Give the point after the second step (exact fractions)."),
            answer=vec(pt), kind="vector", answer_hint="[x2, y2] with fractions like 3/4",
            rubric=[f"∇f = (2(x − {p}), {2*a}(y − {q}))", "Step 1: evaluate the gradient, update",
                    "Step 2: re-evaluate the gradient at the new point (not the old one!)",
                    "Notice each coordinate shrinks its distance to the minimiser by a constant factor"],
            solution=" → ".join(f"${h}$" for h in hist) + ".",
            misconceptions=[(fmt(stale), "gd-reused-stale-gradient")],
        )
    A, ev = _sym2(rng, 1, 8)
    eta_max = sp.Rational(2, ev[-1])
    return Problem(
        prompt=(f"Gradient descent $\\mathbf{{x}}_{{k+1}} = \\mathbf{{x}}_k - \\eta\\nabla f(\\mathbf{{x}}_k)$ is run on the quadratic "
                f"$f(\\mathbf{{x}}) = \\tfrac12\\mathbf{{x}}^\\top A\\mathbf{{x}}$ with $A = {L(A)}$. "
                "Find the supremum $\\eta_{\\max}$ of step sizes $\\eta > 0$ for which the iteration converges to $0$ "
                "from **every** starting point."),
        answer=fmt(eta_max), kind="num",
        rubric=["∇f(x) = A x, so x_{k+1} = (I − ηA) x_k",
                "Converges for all x₀ iff every eigenvalue of I − ηA has |1 − ηλ| < 1",
                f"Eigenvalues of A: {ev[0]}, {ev[1]}", "Binding constraint: η < 2/λ_max"],
        solution=(f"$\\lambda(A) = {ev[0]}, {ev[1]}$. Need $|1-\\eta\\lambda| < 1 \\iff 0 < \\eta < 2/\\lambda$ for both, "
                  f"so $\\eta_{{\\max}} = 2/{ev[-1]} = {L(eta_max)}$."),
        misconceptions=[(fmt(sp.Rational(2, ev[0])), "used-smallest-curvature"),
                        (fmt(sp.Rational(1, ev[-1])), "forgot-factor-2-stability")],
    )


# =========================================================================== multivariable chain rule
@gen("calc.multivar_chain")
def multivar_chain(rng, level):
    if level == 1:
        a = nz(rng, -3, 3)
        p, q, r = rng.randint(-2, 2), nz(rng, -3, 3), rng.randint(-2, 2)
        t0 = rng.randint(-2, 2)
        zf = x ** 2 * y + a * y ** 2
        xt, yt = t ** 2 + p, q * t + r
        ans = sp.diff(zf.subs({x: xt, y: yt}), t).subs(t, t0)
        only_x = (sp.diff(zf, x) * sp.diff(xt, t)).subs({x: xt, y: yt}).subs(t, t0)
        return Problem(
            prompt=(f"Let $z = {L(zf)}$ with $x(t) = {L(xt)}$ and $y(t) = {L(yt)}$. "
                    f"Use the multivariable chain rule to compute $\\frac{{dz}}{{dt}}$ at $t = {t0}$."),
            answer=fmt(ans), kind="num",
            rubric=["dz/dt = z_x · dx/dt + z_y · dy/dt", f"Evaluate x({t0}), y({t0})",
                    "Compute each partial and each derivative at that point", "Add both paths"],
            solution=(f"At $t={t0}$: $x = {xt.subs(t, t0)}$, $y = {yt.subs(t, t0)}$. "
                      f"$z_x = 2xy$, $z_y = x^2 {2*a:+d}y$, $x' = 2t$, $y' = {q}$. $\\frac{{dz}}{{dt}} = {ans}$."),
            misconceptions=[(fmt(only_x), "chain-missed-a-path")],
        )
    if level == 2:
        a, b, c, d = (nz(rng, -3, 3) for _ in range(4))
        F = sp.Matrix([a * x ** 2 * y + b * y, c * x + d * x * y ** 2])
        x0, y0 = nz(rng, -2, 2), nz(rng, -2, 2)
        J = F.jacobian([x, y]).subs({x: x0, y: y0})
        return Problem(
            prompt=(f"Compute the Jacobian matrix of $F(x, y) = \\left({L(F[0])},\\ {L(F[1])}\\right)$ at $({x0}, {y0})$. "
                    "Row $i$ holds the partial derivatives of component $F_i$."),
            answer=fmt(J), kind="matrix", answer_hint="[[dF1/dx, dF1/dy], [dF2/dx, dF2/dy]]",
            rubric=["Row 1: ∂F₁/∂x, ∂F₁/∂y", "Row 2: ∂F₂/∂x, ∂F₂/∂y", f"Evaluate at ({x0}, {y0})"],
            solution=f"$J_F = {L(F.jacobian([x, y]))}$, at the point $J = {L(J)}$.",
            misconceptions=[(fmt(J.T), "jacobian-transposed")],
        )
    while True:
        x0 = nz(rng, -2, 2)
        w1, b1, w2, b2 = nz(rng, -3, 3), rng.randint(-2, 2), nz(rng, -3, 3), rng.randint(-2, 2)
        yt = rng.randint(-3, 3)
        z1 = w1 * x0 + b1
        r = w2 * z1 + b2 - yt
        if z1 > 0 and r != 0:
            break
    W1, B1, W2, B2 = sp.symbols("W1 B1 W2 B2")
    loss = sp.Rational(1, 2) * (W2 * sp.Max(0, W1 * x0 + B1) + B2 - yt) ** 2
    sub = {W1: w1, B1: b1, W2: w2, B2: b2}
    g = [sp.diff(loss, v).subs(sub) for v in (W1, B1, W2)]
    return Problem(
        prompt=(f"A one-hidden-unit network: $z = w_1 x + b_1$, $h = \\mathrm{{ReLU}}(z)$, $\\hat y = w_2 h + b_2$, "
                "$\\mathcal{L} = \\frac12(\\hat y - y)^2$. "
                f"With $x = {x0}$, $y = {yt}$, $w_1 = {w1}$, $b_1 = {b1}$, $w_2 = {w2}$, $b_2 = {b2}$, "
                "do a forward pass and then backpropagate to get "
                "$\\left(\\frac{\\partial\\mathcal{L}}{\\partial w_1}, \\frac{\\partial\\mathcal{L}}{\\partial b_1}, "
                "\\frac{\\partial\\mathcal{L}}{\\partial w_2}\\right)$."),
        answer=vec(g), kind="vector", answer_hint="[dL/dw1, dL/db1, dL/dw2]",
        rubric=[f"Forward: z = {z1}, h = {z1}, ŷ = {w2*z1 + b2}", f"δ = ∂L/∂ŷ = ŷ − y = {r}",
                "∂L/∂w₂ = δ·h", "∂L/∂h = δ·w₂; ReLU'(z) = 1 since z > 0",
                "∂L/∂w₁ = δ w₂ x, ∂L/∂b₁ = δ w₂"],
        solution=(f"$z = {z1}$, $h = {z1}$, $\\hat y = {w2*z1 + b2}$, $\\delta = {r}$. "
                  f"$\\partial_{{w_1}} = \\delta w_2 x = {g[0]}$, $\\partial_{{b_1}} = \\delta w_2 = {g[1]}$, "
                  f"$\\partial_{{w_2}} = \\delta h = {g[2]}$."),
        misconceptions=[(fmt([-v for v in g]), "loss-gradient-sign"),
                        (fmt([r * x0, r, r * z1]), "backprop-skipped-downstream-weight")],
    )


# =========================================================================== Hessians
@gen("calc.hessian")
def hessian(rng, level):
    if level == 1:
        a, b, c, d = nz(rng, -3, 3), nz(rng, -3, 3), rng.randint(-4, 4), rng.randint(-4, 4)
        f = a * x ** 3 + b * x ** 2 * y + c * y ** 2 + d * x * y
        x0, y0 = rng.randint(-2, 2), rng.randint(-2, 2)
        H = sp.hessian(f, (x, y))
        Hv = H.subs({x: x0, y: y0})
        wrong = sp.diag(Hv[0, 0], Hv[1, 1])
        return Problem(
            prompt=f"Compute the Hessian matrix of $f(x, y) = {L(f)}$ at $({x0}, {y0})$.",
            answer=fmt(Hv), kind="matrix", answer_hint="[[f_xx, f_xy], [f_yx, f_yy]]",
            rubric=["Compute f_x and f_y", "Second derivatives f_xx, f_xy, f_yy", "Hessian is symmetric: f_xy = f_yx",
                    f"Evaluate at ({x0}, {y0})"],
            solution=f"$H(x,y) = {L(H)}$, so $H({x0},{y0}) = {L(Hv)}$.",
            misconceptions=[(fmt(wrong), "forgot-mixed-partials")],
        )
    if level == 2:
        while True:
            a, c, b = nz(rng, -3, 3), nz(rng, -3, 3), rng.randint(-4, 4)
            p, q = rng.randint(-3, 3), rng.randint(-3, 3)
            H = sp.Matrix([[2 * a, b], [b, 2 * c]])
            if H.det() != 0 and p != q and (p, q) != (0, 0) and p != -q:
                break
        lin = -H * sp.Matrix([p, q])
        f = a * x ** 2 + b * x * y + c * y ** 2 + lin[0] * x + lin[1] * y
        det = H.det()
        typ = "saddle point" if det < 0 else ("local minimum" if a > 0 else "local maximum")
        types = ["local minimum", "local maximum", "saddle point"]
        other = [tt for tt in types if tt != typ]
        opts = [((p, q), typ), ((p, q), other[0]), ((-p, -q), typ), ((q, p), typ)]
        order = list(range(4))
        rng.shuffle(order)
        letters = "ABCD"
        lines, ans, mis = [], "", []
        for k, idx in enumerate(order):
            (pp, qq), tt = opts[idx]
            lines.append(f"- **{letters[k]}**: $({pp}, {qq})$, {tt}")
            if idx == 0:
                ans = letters[k]
            elif idx == 1:
                mis.append((letters[k], "misclassified-critical-point"))
            elif idx == 2:
                mis.append((letters[k], "sign-error-solving-gradient"))
        return Problem(
            prompt=(f"$f(x, y) = {L(f)}$ has exactly one critical point. Find it and classify it with the "
                    "second-derivative test. Which is correct?\n\n" + "\n".join(lines)),
            answer=ans, kind="choice",
            rubric=["Set ∇f = 0 and solve the 2×2 linear system", f"Hessian H = {H.tolist()}",
                    f"det H = {det}" + (" < 0 → saddle" if det < 0 else f" > 0, f_xx = {2*a} → " + typ),
                    "State the conclusion"],
            solution=f"Critical point $({p}, {q})$; $H = {L(H)}$, $\\det H = {det}$, so it is a **{typ}**.",
            misconceptions=mis,
        )
    a, b = nz(rng, -3, 3), nz(rng, -3, 3)
    kind = rng.choice(["expcos", "explin", "log"])
    if kind == "expcos":
        f = sp.exp(a * x) * sp.cos(b * y)
    elif kind == "explin":
        f = sp.exp(a * x + b * y)
    else:
        f = sp.log(1 + a * x + b * y)
    origin = {x: 0, y: 0}
    g = [sp.diff(f, v).subs(origin) for v in (x, y)]
    H = sp.hessian(f, (x, y)).subs(origin)
    vv = sp.Matrix([x, y])
    Q = sp.expand(f.subs(origin) + g[0] * x + g[1] * y + sp.Rational(1, 2) * (vv.T * H * vv)[0])
    Q_nohalf = sp.expand(f.subs(origin) + g[0] * x + g[1] * y + (vv.T * H * vv)[0])
    return Problem(
        prompt=(f"Find the second-order Taylor polynomial of $f(x, y) = {L(f)}$ about $(0, 0)$, i.e. "
                "$Q(\\mathbf{h}) = f(\\mathbf{0}) + \\nabla f(\\mathbf{0})^\\top\\mathbf{h} + \\tfrac12\\mathbf{h}^\\top H(\\mathbf{0})\\mathbf{h}$ "
                "with $\\mathbf{h} = (x, y)$. Type it as a polynomial in $x, y$."),
        answer=fmt(Q), kind="expr", answer_hint="a polynomial in x and y, e.g. 1 + 2*x - x^2/2 + 3*x*y",
        rubric=["f(0,0)", f"∇f(0,0) = {g}", f"Hessian at the origin = {H.tolist()}",
                "Expand ½ hᵀHh = ½(f_xx x² + 2 f_xy xy + f_yy y²)"],
        solution=f"$\\nabla f(0) = {tup(g)}$, $H(0) = {L(H)}$, so $Q = {L(Q)}$.",
        misconceptions=[(fmt(Q_nohalf), "forgot-half-in-quadratic-term")],
    )


# =========================================================================== multiple integrals
@gen("calc.multiple_integrals")
def multiple_integrals(rng, level):
    if level == 1:
        p, q = rng.randint(1, 3), rng.randint(1, 3)
        while q == p:
            q = rng.randint(1, 3)
        m, n_, a = rng.randint(0, 2), rng.randint(1, 2), rng.randint(1, 6)
        f = a * x ** m * y ** n_
        val = sp.integrate(f, (y, 0, q), (x, 0, p))
        swapped = sp.integrate(f, (y, 0, p), (x, 0, q))
        return Problem(
            prompt=f"Compute $\\displaystyle\\int_0^{{{p}}}\\!\\int_0^{{{q}}} {L(f)}\\,dy\\,dx$.",
            answer=fmt(val), kind="num",
            rubric=["Inner integral in y with x held constant", "Outer integral in x",
                    "(Shortcut: separable integrand → product of 1-D integrals)"],
            solution=f"$= {a}\\left(\\int_0^{{{p}}} x^{{{m}}}dx\\right)\\left(\\int_0^{{{q}}} y^{{{n_}}}dy\\right) = {L(val)}$.",
            misconceptions=[(fmt(swapped), "swapped-limits")],
        )
    if level == 2:
        if rng.random() < 0.5:
            p, a, b = rng.randint(1, 3), rng.randint(1, 4), rng.randint(-3, 3)
            f = a * x + b * y
            val = sp.integrate(f, (y, 0, x), (x, 0, p))
            return Problem(
                prompt=(f"Compute $\\iint_T ({L(f)})\\,dA$ over the triangle $T$ with vertices $(0,0)$, $({p},0)$, $({p},{p})$."),
                answer=fmt(val), kind="num",
                rubric=["Sketch T: 0 ≤ y ≤ x, 0 ≤ x ≤ p", "Set up ∫₀ᵖ ∫₀ˣ … dy dx", "Inner integral in y", "Outer integral in x"],
                solution=f"$\\int_0^{{{p}}}\\int_0^x ({L(f)})\\,dy\\,dx = {L(val)}$.",
                misconceptions=[(fmt(sp.integrate(f, (y, 0, p), (x, 0, p))), "used-full-square")],
            )
        if rng.random() < 0.5:
            sv = rng.choice([1, 2])
            inner_f, shown_s = sp.exp(x ** 2), str(sv)
        else:
            sv = rng.choice([sp.sqrt(sp.pi), sp.sqrt(sp.pi / 2)])
            inner_f, shown_s = sp.sin(x ** 2), L(sv)
        val = sp.simplify(sp.integrate(x * inner_f, (x, 0, sv)))
        return Problem(
            prompt=(f"Evaluate $\\displaystyle\\int_0^{{{shown_s}}}\\!\\int_y^{{{shown_s}}} {L(inner_f)}\\,dx\\,dy$ exactly. "
                    "(Hint: the inner integral has no elementary antiderivative — reverse the order.)"),
            answer=fmt(val), kind="num", answer_hint="exact, e.g. (e - 1)/2",
            rubric=["Sketch the region: 0 ≤ y ≤ x ≤ s", "Reverse: ∫₀ˢ ∫₀ˣ … dy dx", "Inner integral gives x·g(x²)",
                    "Substitute u = x²"],
            solution=f"$= \\int_0^{{{shown_s}}} x\\,{L(inner_f)}\\,dx = {L(val)}$.",
            misconceptions=[(fmt(2 * val), "u-sub-forgot-du-factor")],
        )
    if rng.random() < 0.5:
        R = rng.randint(1, 3)
        val = sp.pi * (1 - sp.exp(-R ** 2))
        return Problem(
            prompt=(f"Using polar coordinates, evaluate $\\displaystyle\\iint_{{x^2+y^2\\le {R**2}}} e^{{-(x^2+y^2)}}\\,dA$."),
            answer=fmt(val), kind="num", answer_hint="exact, e.g. pi*(1 - exp(-4))",
            rubric=["dA = r dr dθ", f"Limits 0 ≤ r ≤ {R}, 0 ≤ θ ≤ 2π", "Substitute u = r²", "Multiply by 2π"],
            solution=f"$\\int_0^{{2\\pi}}\\int_0^{{{R}}} e^{{-r^2}} r\\,dr\\,d\\theta = 2\\pi\\cdot\\frac{{1-e^{{-{R**2}}}}}{{2}} = {L(val)}$.",
            misconceptions=[(fmt(2 * sp.pi * (1 - sp.exp(-R ** 2))), "polar-forgot-half-from-substitution"),
                            (fmt(sp.pi * R ** 2), "forgot-jacobian-r")],
        )
    a = rng.choice([1, 2, 3, 4, sp.Rational(1, 2)])
    val = sp.sqrt(sp.pi / a)
    return Problem(
        prompt=(f"Let $I = \\displaystyle\\int_{{-\\infty}}^{{\\infty}} e^{{-{L(a)}x^2}}\\,dx$. Write $I^2$ as a double integral "
                "over the plane, switch to polar coordinates, and compute $I$ exactly. "
                "(This is the normalising constant of a Gaussian.)"),
        answer=fmt(val), kind="num", answer_hint="exact, e.g. sqrt(pi/2)",
        rubric=["I² = ∬ e^(−a(x²+y²)) dx dy", "Polar: dA = r dr dθ", "∫₀^∞ e^(−a r²) r dr = 1/(2a)",
                "I² = π/a, take the positive root"],
        solution=f"$I^2 = \\int_0^{{2\\pi}}\\int_0^\\infty e^{{-{L(a)}r^2}}r\\,dr\\,d\\theta = \\frac{{\\pi}}{{{L(a)}}}$, so $I = {L(val)}$.",
        misconceptions=[(fmt(sp.pi / a), "forgot-square-root"), (fmt(sp.sqrt(sp.pi) / a), "scaled-wrongly")],
    )


# ===========================================================================
# Extra problem TYPES per skill-level (variety): each generator below asks for a different *task* than the
# original generators of the same skill, not just new numbers.
# ===========================================================================
PI_ANGLES = [sp.pi / 6, sp.pi / 4, sp.pi / 3, sp.pi / 2]


@gen("calc.limits", name="limits_more")
def limits_more(rng, level):
    if level == 1:
        if rng.choice(["asymptote", "sinx"]) == "asymptote":
            a, b, c = nz(rng, -6, 6), rng.randint(-6, 6), rng.randint(-6, 6)
            d, q = nz(rng, -5, 5), rng.randint(1, 6)
            num, den = a * x ** 2 + b * x + c, d * x ** 2 + q
            lim = sp.limit(num / den, x, sp.oo)
            wr = [("0", "limit-at-infinity-is-zero")]
            if c != 0:
                wr.append((fmt(sp.Rational(c, q)), "ratio-of-constant-terms"))
            return Problem(
                prompt=(f"Find the horizontal asymptote of $y = \\dfrac{{{L(num)}}}{{{L(den)}}}$, i.e. compute "
                        "$\\displaystyle\\lim_{x\\to\\infty} y$, by dividing top and bottom by the highest power of $x$. "
                        "Type the limit."),
                answer=fmt(lim), kind="num",
                rubric=["Highest power in the denominator is x²", "Divide every term by x²",
                        "Terms like b/x and c/x² tend to 0", "Limit = ratio of the leading coefficients"],
                solution=(f"After dividing by $x^2$ every term except the $x^2$-coefficients tends to $0$, so the limit is "
                          f"$\\dfrac{{{a}}}{{{d}}} = {L(lim)}$; the asymptote is $y = {L(lim)}$."),
                misconceptions=wr,
            )
        a, b = nz(rng, -6, 6), rng.randint(1, 6)
        while abs(a) == b:
            b = rng.randint(1, 6)
        lim = sp.limit(sp.sin(a * x) / (b * x), x, 0)
        sin_ax = f"\\sin({L(a * x)})"
        return Problem(
            prompt=(f"Using the standard limit $\\lim_{{u\\to 0}}\\frac{{\\sin u}}{{u}} = 1$, evaluate "
                    f"$\\displaystyle\\lim_{{x\\to 0}} \\frac{{{sin_ax}}}{{{L(b * x)}}}$."),
            answer=fmt(lim), kind="num",
            rubric=[f"Multiply and divide by {a}: sin({a}x)/({b}x) = ({a}/{b}) · sin({a}x)/({a}x)",
                    f"With u = {a}x → 0, sin u / u → 1", "Multiply the constants"],
            solution=(f"$\\frac{{{sin_ax}}}{{{L(b * x)}}} = {L(sp.Rational(a, b))}\\cdot\\frac{{{sin_ax}}}{{{L(a * x)}}} "
                      f"\\to {L(lim)}\\cdot 1 = {L(lim)}$."),
            misconceptions=[("1", "ignored-inner-scaling"), (fmt(sp.Rational(b, a)), "inverted-ratio")],
        )
    if level == 2:
        if rng.choice(["piecewise", "exp"]) == "piecewise":
            x0, p = nz(rng, -3, 3), rng.randint(-5, 5)
            m, q = sp.symbols("m q")
            sol = sp.solve([sp.Eq((x ** 2 + p).subs(x, x0), m * x0 + q), sp.Eq(2 * x0, m)], [m, q], dict=True)[0]
            mv, qv = sol[m], sol[q]
            return Problem(
                prompt=(f"Let $f(x) = \\begin{{cases}} {L(x**2 + p)} & x < {x0} \\\\ mx + q & x \\ge {x0} \\end{{cases}}$. "
                        f"Find the constants $m$ and $q$ that make $f$ **differentiable** at $x = {x0}$. Answer as $m, q$."),
                answer=[fmt(mv), fmt(qv)], kind="list", answer_hint="m, q   e.g. 4, -1",
                rubric=[f"Continuity at {x0}: {x0**2 + p} = {x0}m + q",
                        f"Equal one-sided derivatives: 2·({x0}) = m", "Solve m first, then q"],
                solution=(f"Derivatives must match: $m = 2\\cdot({x0}) = {mv}$. Values must match: "
                          f"${x0**2 + p} = {mv}\\cdot({x0}) + q \\Rightarrow q = {qv}$."),
                misconceptions=[([fmt(qv), fmt(mv)], "swapped-order"),
                                ([fmt(mv), str(x0 ** 2 + p)], "intercept-equals-function-value")],
            )
        a, b = nz(rng, -3, 3), rng.randint(1, 3)
        lim = sp.limit((1 + sp.Integer(a) / x) ** (b * x), x, sp.oo)
        inner = f"1 + \\frac{{{a}}}{{x}}" if a > 0 else f"1 - \\frac{{{-a}}}{{x}}"
        expo = "x" if b == 1 else f"{b}x"
        return Problem(
            prompt=(f"Evaluate $\\displaystyle\\lim_{{x\\to\\infty}}\\left({inner}\\right)^{{{expo}}}$. "
                    "(Take the logarithm first: this is a $1^\\infty$ form.)"),
            answer=fmt(lim), kind="num", answer_hint="exact, e.g. exp(6) or e^-2",
            rubric=[f"ln y = {expo} · ln(1 + ({a})/x)", "Rewrite as a 0/0 quotient ln(1 + a/x) / (1/x)",
                    f"L'Hôpital (or ln(1+u) ≈ u) gives ln y → {a*b}", "Exponentiate"],
            solution=(f"$\\ln y = {'' if b == 1 else str(b) + chr(92) + 'cdot'}\\dfrac{{\\ln\\left({inner}\\right)}}{{1/x}} "
                      f"\\to {'' if b == 1 else str(b) + chr(92) + 'cdot'}({a}) = {a*b}$, "
                      f"so the limit is ${L(lim)}$."),
            misconceptions=[("1", "one-to-the-infinity-is-one"), (fmt(sp.exp(a)), "forgot-exponent-factor")],
        )
    if rng.choice(["find_a", "riemann"]) == "find_a":
        c = nz(rng, -4, 4)
        av = sp.Symbol("a_c")
        ser = sp.series(sp.exp(c * x) - 1 - av * x, x, 0, 3).removeO()
        a_sol = sp.solve(ser.coeff(x, 1), av)[0]
        lim = sp.limit((sp.exp(c * x) - 1 - a_sol * x) / x ** 2, x, 0)
        return Problem(
            prompt=(f"Find the constant $a$ for which $\\displaystyle\\lim_{{x\\to0}}\\frac{{{L(sp.exp(c*x))} - 1 - ax}}{{x^2}}$ "
                    "is finite, and the value $L$ of the limit for that $a$. Answer as $a, L$."),
            answer=[fmt(a_sol), fmt(lim)], kind="list", answer_hint="a, L   e.g. 2, 2",
            rubric=[f"Expand e^({c}x) = 1 + ({c})x + ({c}x)²/2 + O(x³)",
                    "The 1/x term must vanish: the x-coefficient of the numerator is 0", f"So a = {a_sol}",
                    f"The remaining x² coefficient ({c})²/2 gives L"],
            solution=(f"Numerator $= ({c} - a)x + \\frac{{{c**2}}}{{2}}x^2 + O(x^3)$. Divided by $x^2$ this blows up unless "
                      f"$a = {a_sol}$; then the limit is ${L(lim)}$."),
            misconceptions=[([fmt(a_sol), fmt(c ** 2)], "taylor-forgot-factorial"), ([fmt(-a_sol), fmt(lim)], "sign-error")],
        )
    a, b = rng.randint(0, 3), rng.randint(1, 3)
    val = sp.integrate((a + b * t) ** 2, (t, 0, 1))
    frac = "\\frac{k}{n}" if b == 1 else f"\\frac{{{b}k}}{{n}}"
    inner = f"{a} + {frac}" if a else frac
    return Problem(
        prompt=(f"Evaluate $\\displaystyle\\lim_{{n\\to\\infty}} \\frac1n\\sum_{{k=1}}^{{n}}\\left({inner}\\right)^2$ "
                "by recognising the sum as a Riemann sum of a definite integral."),
        answer=fmt(val), kind="num",
        rubric=["Sample points t_k = k/n with width Δt = 1/n on [0, 1]",
                f"The sum → ∫₀¹ ({L(a + b*t)})² dt", "Expand the square and integrate term by term"],
        solution=(f"$\\frac1n\\sum_k g(k/n) \\to \\int_0^1 \\left({L(a + b*t)}\\right)^2\\,dt = {L(val)}$."),
        misconceptions=[(fmt(sp.Integer(a + b) ** 2), "took-last-term-only"),
                        (fmt(sp.Integer(a) ** 2 + sp.Rational(b * b, 3)), "dropped-cross-term")],
    )


@gen("calc.derivative_rules", name="derivative_rules_more")
def derivative_rules_more(rng, level):
    if level == 1:
        n1 = rng.randint(4, 6)
        n2 = rng.randint(2, n1 - 1)
        a, b, c = nz(rng, -5, 5), nz(rng, -6, 6), nz(rng, -9, 9)
        f = a * x ** n1 + b * x ** n2 + c * x
        d2 = sp.expand(sp.diff(f, x, 2))
        return Problem(
            prompt=f"Find the **second** derivative $f''(x)$ of $f(x) = {L(f)}$.",
            answer=fmt(d2), kind="expr",
            rubric=["Differentiate once with the power rule", "Differentiate the result again",
                    "The linear term disappears after two derivatives"],
            solution=f"$f'(x) = {L(sp.diff(f, x))}$, $f''(x) = {L(d2)}$.",
            misconceptions=[(fmt(sp.diff(f, x)), "gave-first-derivative")],
        )
    if level == 2:
        x0 = rng.randint(1, 4)
        while True:
            f0, f1, g0, g1 = nz(rng, -6, 6), nz(rng, -5, 5), nz(rng, -4, 4), nz(rng, -4, 4)
            if f1 * g0 - f0 * g1 != 0:
                break
        F, G = sp.Function("f"), sp.Function("g")
        expr = sp.diff(F(x) / G(x), x)
        val = sp.nsimplify(expr.subs({sp.Derivative(F(x), x): f1, sp.Derivative(G(x), x): g1}).subs({F(x): f0, G(x): g0}))
        return Problem(
            prompt=(f"Suppose $f({x0}) = {f0}$, $f'({x0}) = {f1}$, $g({x0}) = {g0}$ and $g'({x0}) = {g1}$. "
                    f"Find $h'({x0})$ for $h(x) = \\dfrac{{f(x)}}{{g(x)}}$."),
            answer=fmt(val), kind="num",
            rubric=["Quotient rule: h' = (f'g − f g')/g²", f"Plug in the values at x = {x0}",
                    "Square the denominator", "Simplify the fraction"],
            solution=(f"$h'({x0}) = \\dfrac{{({f1})({g0}) - ({f0})({g1})}}{{({g0})^2}} = {L(val)}$."),
            misconceptions=[(fmt(sp.Rational(f1, g1)), "quotient-of-derivatives"), (fmt(-val), "quotient-rule-sign")],
        )
    x0, m, c = nz(rng, -3, 3), nz(rng, -5, 5), rng.randint(-5, 5)
    av, bv = sp.symbols("a_t b_t")
    par = x ** 2 + av * x + bv
    sol = sp.solve([sp.Eq(par.subs(x, x0), m * x0 + c), sp.Eq(sp.diff(par, x).subs(x, x0), m)], [av, bv], dict=True)[0]
    A, B = sol[av], sol[bv]
    return Problem(
        prompt=(f"Find constants $a$ and $b$ such that the parabola $y = x^2 + ax + b$ is **tangent** to the line "
                f"$y = {L(m*x + c)}$ at $x = {x0}$. Answer as $a, b$."),
        answer=[fmt(A), fmt(B)], kind="list", answer_hint="a, b   e.g. -2, 5",
        rubric=[f"Same slope at x = {x0}: 2·({x0}) + a = {m}", f"Same height at x = {x0}: {x0**2} + ({x0})a + b = {m*x0 + c}",
                "Solve the two equations", "Check: parabola − line = (x − x₀)²"],
        solution=(f"Slope: ${2*x0} + a = {m} \\Rightarrow a = {A}$. Height: ${x0**2} + ({A})({x0}) + b = {m*x0 + c} "
                  f"\\Rightarrow b = {B}$. Check: $\\left(x^2 + ({A})x + ({B})\\right) - \\left({L(m*x + c)}\\right) = "
                  f"{L(sp.factor(par.subs(sol) - m*x - c))}$."),
        misconceptions=[([str(m), str(c)], "matched-coefficients-not-tangency"), ([fmt(B), fmt(A)], "swapped-order")],
    )


@gen("calc.chain_rule", name="chain_rule_more")
def chain_rule_more(rng, level):
    if level == 1:
        if rng.choice(["exp", "table"]) == "exp":
            a, b = nz(rng, -3, 3), rng.randint(-4, 4)
            f = sp.exp(a * x ** 2 + b * x)
            d = sp.diff(f, x)
            return Problem(
                prompt=f"Use the chain rule to find $f'(x)$ for $f(x) = {L(f)}$.",
                answer=fmt(d), kind="expr",
                rubric=["Outer function e^u, inner u = a x² + b x", "The derivative of e^u is e^u (unchanged)",
                        f"Multiply by u' = {L(2*a*x + b)}"],
                solution=f"$f'(x) = {L(f)}\\cdot\\left({L(2*a*x + b)}\\right) = {L(d)}$.",
                misconceptions=[(fmt(f), "chain-forgot-inner-derivative"),
                                (fmt((a * x ** 2 + b * x) * sp.exp(a * x ** 2 + b * x - 1)), "power-rule-on-exponential")],
            )
        x0 = rng.randint(1, 3)
        g0 = rng.randint(-2, 5)
        while g0 == x0:
            g0 = rng.randint(-2, 5)
        g1, fp_g0, fp_x0 = nz(rng, -4, 4), nz(rng, -5, 5), nz(rng, -5, 5)
        while fp_x0 == fp_g0:
            fp_x0 = nz(rng, -5, 5)
        ans = fp_g0 * g1
        return Problem(
            prompt=(f"You are told $g({x0}) = {g0}$, $g'({x0}) = {g1}$, $f'({g0}) = {fp_g0}$ and $f'({x0}) = {fp_x0}$. "
                    f"Compute $(f\\circ g)'({x0})$, the derivative of $f(g(x))$ at $x = {x0}$."),
            answer=str(ans), kind="num",
            rubric=["(f∘g)'(x) = f'(g(x)) · g'(x)", f"Inner value first: g({x0}) = {g0}",
                    f"Use f' at {g0} (not at {x0})", "Multiply"],
            solution=f"$(f\\circ g)'({x0}) = f'(g({x0}))\\,g'({x0}) = f'({g0})\\cdot({g1}) = ({fp_g0})({g1}) = {ans}$.",
            misconceptions=[(str(fp_x0 * g1), "evaluated-f-at-x-not-g-x"), (str(fp_g0), "chain-forgot-inner-derivative")],
        )
    if level == 2:
        n, a = rng.randint(2, 4), nz(rng, -3, 3)
        f = (1 + sp.exp(a * x)) ** n
        val = sp.diff(f, x).subs(x, 0)
        return Problem(
            prompt=(f"Let $f(x) = \\left(1 + {L(sp.exp(a*x))}\\right)^{{{n}}}$. Identify the three layers of this composition "
                    "and compute $f'(0)$."),
            answer=fmt(val), kind="num",
            rubric=["Layers: power ∘ (1 + exp) ∘ linear", f"Outer: {n}(1 + e^({a}x))^{n-1}",
                    f"Middle: e^({a}x); inner: {a}", "At x = 0, e⁰ = 1 so 1 + e⁰ = 2"],
            solution=(f"$f'(x) = {n}\\left(1 + {L(sp.exp(a*x))}\\right)^{{{n-1}}}\\cdot {L(sp.exp(a*x))}\\cdot({a})$; at $0$: "
                      f"${n}\\cdot 2^{{{n-1}}}\\cdot 1\\cdot({a}) = {val}$."),
            misconceptions=[(str(n * 2 ** (n - 1)), "chain-forgot-inner-derivative"), (str(n * a), "chain-forgot-outer-power")],
        )
    a, b, x0 = rng.randint(1, 4), rng.randint(-5, 5), nz(rng, -2, 2)
    f = x ** 3 + a * x + b
    y0 = f.subs(x, x0)
    fp = sp.diff(f, x)
    val = 1 / fp.subs(x, x0)
    return Problem(
        prompt=(f"The function $f(x) = {L(f)}$ is strictly increasing, so it has an inverse $f^{{-1}}$. "
                f"Without solving for $f^{{-1}}$, compute $(f^{{-1}})'({y0})$. "
                "(Differentiate the identity $f(f^{-1}(y)) = y$ on paper.)"),
        answer=fmt(val), kind="num",
        rubric=["Chain rule on f(f⁻¹(y)) = y: f'(f⁻¹(y)) · (f⁻¹)'(y) = 1", f"Find x with f(x) = {y0}: x = {x0}",
                f"f'(x) = {fp}, so f'({x0}) = {fp.subs(x, x0)}", "Take the reciprocal"],
        solution=(f"$f({x0}) = {y0}$, so $f^{{-1}}({y0}) = {x0}$. With $f'(x) = {L(fp)}$, $f'({x0}) = {fp.subs(x, x0)}$ and "
                  f"$(f^{{-1}})'({y0}) = \\frac{{1}}{{f'({x0})}} = {L(val)}$."),
        misconceptions=[(fmt(fp.subs(x, x0)), "inverse-not-reciprocal"), (fmt(1 / fp.subs(x, y0)), "evaluated-at-y-not-x")],
    )


@gen("calc.transcendental_deriv", name="transcendental_more")
def transcendental_more(rng, level):
    if level == 1:
        if rng.choice(["trig_point", "base"]) == "trig_point":
            a, b = nz(rng, -5, 5), nz(rng, -5, 5)
            x0 = rng.choice([sp.Integer(0), sp.pi] + PI_ANGLES)
            f = a * sp.sin(x) + b * sp.cos(x)
            val = sp.nsimplify(sp.diff(f, x).subs(x, x0))
            return Problem(
                prompt=f"Let $f(x) = {L(f)}$. Compute $f'\\left({L(x0)}\\right)$ exactly.",
                answer=fmt(val), kind="num", answer_hint="exact, e.g. 3*sqrt(3)/2 - 1",
                rubric=["(sin x)' = cos x", "(cos x)' = −sin x", f"Use exact values of sin and cos at {x0}"],
                solution=(f"$f'(x) = {L(sp.diff(f, x))}$, so $f'\\left({L(x0)}\\right) = {L(val)}$."),
                misconceptions=[(fmt(sp.nsimplify((a * sp.cos(x) + b * sp.sin(x)).subs(x, x0))), "derivative-of-cos-sign"),
                                (fmt(sp.nsimplify(f.subs(x, x0))), "evaluated-f-not-derivative")],
            )
        base, c = rng.choice([2, 3, 5, 10]), nz(rng, -4, 4)
        f = c * sp.Integer(base) ** x
        d = sp.diff(f, x)
        return Problem(
            prompt=f"Differentiate $f(x) = {L(f)}$ by first rewriting ${base}^x = e^{{x\\ln {base}}}$.",
            answer=fmt(d), kind="expr", answer_hint="use log(...) for ln, e.g. 2^x*log(2)",
            rubric=[f"{base}^x = e^(x ln {base})", f"Chain rule: the derivative is e^(x ln {base}) · ln {base}",
                    "Keep the constant factor"],
            solution=f"$f(x) = {L(f)}$ so $f'(x) = {L(f)}\\cdot\\ln {base} = {L(d)}$.",
            misconceptions=[(fmt(c * x * sp.Integer(base) ** (x - 1)), "power-rule-on-exponential"),
                            (fmt(f), "forgot-ln-base")],
        )
    if level == 2:
        if rng.choice(["logdiff", "arctan"]) == "logdiff":
            k, x0 = rng.choice([1, 2]), rng.choice([1, 2, 3])
            f = x ** (k * x)
            val = sp.simplify(sp.diff(f, x).subs(x, x0))
            shown = "x^{x}" if k == 1 else f"x^{{{k}x}}"
            kk = "" if k == 1 else str(k)
            return Problem(
                prompt=(f"Use logarithmic differentiation (take $\\ln$ of both sides, then differentiate) to compute "
                        f"$f'({x0})$ for $f(x) = {shown}$, $x > 0$."),
                answer=fmt(val), kind="num", answer_hint="exact, e.g. 4*(log(2) + 1)",
                rubric=[f"ln f = {kk}x ln x", f"f'/f = {kk}(ln x + 1) (product rule)", f"f' = f · {kk}(ln x + 1)",
                        f"Evaluate at x = {x0}"],
                solution=(f"$\\ln f = {kk}x\\ln x \\Rightarrow \\frac{{f'}}{{f}} = {kk}(\\ln x + 1)$. "
                          f"At $x = {x0}$: $f'({x0}) = {L(val)}$."),
                misconceptions=[(fmt((k * x * x ** (k * x - 1)).subs(x, x0)), "power-rule-on-variable-exponent")],
            )
        a = rng.randint(2, 4)
        f = x * sp.atan(a * x)
        d = sp.diff(f, x)
        return Problem(
            prompt=f"Find $f'(x)$ for the inverse-trig function $f(x) = x\\arctan({a}x)$ (product rule and chain rule).",
            answer=fmt(d), kind="expr", answer_hint="e.g. atan(2*x) + 2*x/(1 + 4*x^2)",
            rubric=["(arctan u)' = u'/(1 + u²)", "Product rule with u = x, v = arctan(ax)", f"Inner derivative {a}"],
            solution=f"$f'(x) = \\arctan({a}x) + x\\cdot\\frac{{{a}}}{{1 + {a**2}x^2}} = \\arctan({a}x) + \\frac{{{a}x}}{{1 + {a**2}x^2}}$.",
            misconceptions=[(fmt(sp.atan(a * x) + x / (1 + a ** 2 * x ** 2)), "chain-forgot-inner-derivative"),
                            (fmt(a / (1 + a ** 2 * x ** 2)), "product-rule-dropped-term")],
        )
    if rng.choice(["tanh", "gauss"]) == "tanh":
        m, x0 = rng.choice([2, 3]), rng.choice([-2, -1, 1, 2, 3])
        val = sp.nsimplify(sp.simplify(sp.diff(sp.tanh(w * x0), w).subs(w, sp.log(m) / x0).rewrite(sp.exp)))
        th = sp.Rational(m * m - 1, m * m + 1)
        return Problem(
            prompt=(f"A tanh unit computes $h(w) = \\tanh(w x)$ for a fixed input $x = {x0}$.\n\n"
                    "1. Starting from $\\tanh s = \\dfrac{e^s - e^{-s}}{e^s + e^{-s}}$, show $\\tanh'(s) = 1 - \\tanh^2 s$.\n"
                    f"2. Evaluate $\\dfrac{{dh}}{{dw}}$ at $w = {'-' if x0 < 0 else ''}\\dfrac{{\\ln {m}}}{{{abs(x0)}}}$ "
                    "and type the exact number."),
            answer=fmt(val), kind="num",
            rubric=["Quotient rule on (eˢ − e⁻ˢ)/(eˢ + e⁻ˢ)", "Recognise 1 − tanh² s",
                    "Chain rule: dh/dw = x(1 − tanh²(wx))", f"tanh(ln {m}) = ({m}² − 1)/({m}² + 1) = {th}"],
            solution=(f"$wx = \\ln {m}$, $\\tanh(\\ln {m}) = \\frac{{{m} - 1/{m}}}{{{m} + 1/{m}}} = {L(th)}$, so "
                      f"$\\frac{{dh}}{{dw}} = ({x0})\\left(1 - \\left({L(th)}\\right)^2\\right) = {L(val)}$."),
            misconceptions=[(fmt(1 - th ** 2), "chain-forgot-inner-derivative"), (fmt(x0 * (1 - th)), "tanh-derivative-not-squared")],
        )
    a, mu = rng.choice([sp.Rational(1, 2), sp.Integer(1), sp.Integer(2), sp.Rational(1, 8)]), rng.randint(-3, 3)
    f = sp.exp(-a * (x - mu) ** 2)
    ratio = sp.expand(sp.simplify(sp.diff(f, x, 2) / f))
    pts = sp.solve(ratio, x)
    return Problem(
        prompt=(f"The bell curve $f(x) = {L(f)}$ (an unnormalised Gaussian) is concave near its peak and convex in its tails. "
                "Find both inflection points (where $f''$ changes sign)."),
        answer=[fmt(p_) for p_ in pts], kind="set", answer_hint="both x-values, e.g. 1, 3  or  2 - sqrt(2)/2, 2 + sqrt(2)/2",
        rubric=["f' = −2a(x − μ) f", "f'' = (4a²(x − μ)² − 2a) f (product rule)", "f > 0, so solve 4a²(x − μ)² = 2a",
                "x = μ ± 1/√(2a)"],
        solution=(f"$f''(x) = \\left({L(ratio)}\\right)f(x)$, which changes sign at "
                  f"$x = {',\\ '.join(L(p_) for p_ in pts)}$."),
        misconceptions=[([fmt(mu - 1 / sp.sqrt(a)), fmt(mu + 1 / sp.sqrt(a))], "inflection-forgot-factor-2"),
                        ([str(mu)], "gave-maximum")],
    )


@gen("calc.applications", name="applications_more")
def applications_more(rng, level):
    if level == 1:
        if rng.choice(["linapprox", "motion"]) == "linapprox":
            nn, d = rng.randint(2, 5), nz(rng, -3, 3)
            g = sp.cbrt(x)
            approx = g.subs(x, nn ** 3) + sp.diff(g, x).subs(x, nn ** 3) * d
            return Problem(
                prompt=(f"Use the linear approximation (tangent line) of $g(x) = \\sqrt[3]{{x}}$ at $x = {nn**3}$ to "
                        f"estimate $\\sqrt[3]{{{nn**3 + d}}}$. Give the exact fraction."),
                answer=fmt(approx), kind="num",
                rubric=[f"g({nn**3}) = {nn}", f"g'(x) = 1/(3x^(2/3)), so g'({nn**3}) = 1/{3*nn*nn}",
                        f"L(x) = g(a) + g'(a)(x − a) with x − a = {d}"],
                solution=(f"$\\sqrt[3]{{{nn**3 + d}}} \\approx {nn} + \\frac{{1}}{{{3*nn*nn}}}\\cdot({d}) = {L(approx)} "
                          f"\\approx {float(approx):.5f}$ (true value ${float(sp.cbrt(nn**3 + d)):.5f}$)."),
                misconceptions=[(fmt(nn + sp.Rational(d, 3 * nn)), "derivative-exponent-slip"),
                                (fmt(nn + sp.Rational(d, 3)), "forgot-to-evaluate-derivative")],
            )
        a, b, c, d = nz(rng, -2, 2), rng.randint(-6, 6), rng.randint(-9, 9), rng.randint(-5, 5)
        t0 = rng.randint(1, 4)
        pos = a * t ** 3 + b * t ** 2 + c * t + d
        v, acc = sp.diff(pos, t).subs(t, t0), sp.diff(pos, t, 2).subs(t, t0)
        return Problem(
            prompt=(f"A particle moves along a line with position $s(t) = {L(pos)}$ (metres, $t$ in seconds). "
                    f"Find its velocity and acceleration at $t = {t0}$. Answer as $v, a$."),
            answer=[fmt(v), fmt(acc)], kind="list", answer_hint="v, a   e.g. 12, -4",
            rubric=["v(t) = s'(t)", "a(t) = v'(t) = s''(t)", f"Evaluate both at t = {t0}"],
            solution=(f"$v(t) = {L(sp.diff(pos, t))}$, $a(t) = {L(sp.expand(sp.diff(pos, t, 2)))}$; at $t = {t0}$: "
                      f"$v = {v}$ m/s, $a = {acc}$ m/s²."),
            misconceptions=[([fmt(pos.subs(t, t0)), fmt(v)], "position-instead-of-velocity"),
                            ([fmt(acc), fmt(v)], "swapped-order")],
        )
    if level == 2:
        N = rng.choice([2, 3, 5, 6, 7, 10, 11])
        x0 = int(sp.floor(sp.sqrt(N) + sp.Rational(1, 2)))
        f = x ** 2 - N
        step = x - f / sp.diff(f, x)
        x1 = sp.nsimplify(step.subs(x, x0))
        x2 = sp.nsimplify(step.subs(x, x1))
        return Problem(
            prompt=(f"Apply **two** steps of Newton's method to $f(x) = {L(f)}$ starting from $x_0 = {x0}$ to approximate "
                    f"$\\sqrt{{{N}}}$. Give $x_2$ as an exact fraction."),
            answer=fmt(x2), kind="num",
            rubric=["Newton update x_{n+1} = x_n − f(x_n)/f'(x_n)", f"f'(x) = 2x, so x_{{n+1}} = (x_n + {N}/x_n)/2",
                    f"x₁ = {x1}", "Repeat once more with x₁ (exact fractions)"],
            solution=(f"$x_1 = \\frac12\\left({x0} + \\frac{{{N}}}{{{x0}}}\\right) = {L(x1)}$, "
                      f"$x_2 = \\frac12\\left(x_1 + \\frac{{{N}}}{{x_1}}\\right) = {L(x2)} \\approx {float(x2):.6f}$ "
                      f"($\\sqrt{{{N}}} \\approx {float(sp.sqrt(N)):.6f}$)."),
            misconceptions=[(fmt(x1), "stopped-after-one-step"), (fmt(sp.nsimplify(x0 + f.subs(x, x0) / (2 * x0))), "newton-sign-error")],
        )
    if rng.choice(["ellipse", "mle"]) == "ellipse":
        A, B = rng.sample(range(2, 7), 2)
        p = sp.Symbol("p", positive=True)
        area = 4 * p * B * sp.sqrt(1 - p ** 2 / A ** 2)
        ps = next(r for r in sp.solve(sp.diff(area, p), p) if r.is_positive)
        best = sp.simplify(area.subs(p, ps))
        return Problem(
            prompt=(f"A rectangle with sides parallel to the axes is inscribed in the ellipse "
                    f"$\\dfrac{{x^2}}{{{A**2}}} + \\dfrac{{y^2}}{{{B**2}}} = 1$. Find the **largest possible area** of the rectangle."),
            answer=fmt(best), kind="num",
            rubric=["Corner (x, y) in the first quadrant: area = 4xy", f"Constraint gives y = {B}·√(1 − x²/{A**2})",
                    "Maximise (easier: maximise area² or use x = A cos θ, y = B sin θ)",
                    f"Optimum at x = {A}/√2, y = {B}/√2"],
            solution=(f"With $x = {A}\\cos\\theta$, $y = {B}\\sin\\theta$: area $= 4\\cdot{A*B}\\sin\\theta\\cos\\theta = "
                      f"{2*A*B}\\sin 2\\theta$, maximal at $\\theta = \\pi/4$: area $= {L(best)}$."),
            misconceptions=[(str(A * B), "forgot-four-quadrants"), (str(4 * A * B), "area-of-bounding-box"),
                            (fmt(sp.pi * A * B), "area-of-ellipse")],
        )
    n = rng.randint(3, 5)
    data = [rng.randint(1, 6) for _ in range(n)]
    lam = sp.Symbol("lambda", positive=True)
    ell = n * sp.log(lam) - lam * sum(data)
    lam_hat = sp.solve(sp.diff(ell, lam), lam)[0]
    return Problem(
        prompt=(f"Waiting times {', '.join(map(str, data))} (minutes) are modelled as independent draws from an "
                "Exponential distribution with density $p(x) = \\lambda e^{-\\lambda x}$. Write the log-likelihood "
                "$\\ell(\\lambda)$, maximise it, and type the maximum-likelihood estimate $\\hat\\lambda$."),
        answer=fmt(lam_hat), kind="num",
        rubric=["Likelihood = Π λ e^(−λ x_k)", "ℓ(λ) = n ln λ − λ Σ x_k", "ℓ'(λ) = n/λ − Σ x_k = 0",
                f"λ̂ = n / Σ x_k = {n}/{sum(data)}", "ℓ''(λ) = −n/λ² < 0, so it is a maximum"],
        solution=(f"$\\ell(\\lambda) = {n}\\ln\\lambda - {sum(data)}\\lambda$, $\\ell'(\\lambda) = \\frac{{{n}}}{{\\lambda}} - {sum(data)} = 0 "
                  f"\\Rightarrow \\hat\\lambda = {L(lam_hat)}$ (one over the sample mean)."),
        misconceptions=[(fmt(sp.Rational(sum(data), n)), "gave-mean-not-rate")],
    )


@gen("calc.integrals_basic", name="integrals_basic_more")
def integrals_basic_more(rng, level):
    if level == 1:
        if rng.choice(["ivp", "log"]) == "ivp":
            p, q, r = nz(rng, -3, 3), rng.randint(-4, 4), nz(rng, -6, 6)
            x0, y0 = rng.randint(-2, 2), rng.randint(-6, 6)
            fprime = 3 * p * x ** 2 + 2 * q * x + r
            K = sp.Symbol("K")
            F0 = sp.integrate(fprime, x)
            Kv = sp.solve(sp.Eq(F0.subs(x, x0) + K, y0), K)[0]
            F = F0 + Kv
            return Problem(
                prompt=(f"Find the function $F$ with $F'(x) = {L(fprime)}$ and $F({x0}) = {y0}$."),
                answer=fmt(F), kind="expr", answer_hint="an expression in x, e.g. x^3 - 2*x + 5",
                rubric=["Antidifferentiate term by term", "Add an unknown constant K",
                        f"Use F({x0}) = {y0} to solve for K"],
                solution=f"$F(x) = {L(F0)} + K$; $F({x0}) = {F0.subs(x, x0)} + K = {y0} \\Rightarrow K = {Kv}$, so $F(x) = {L(F)}$.",
                misconceptions=[(fmt(F0), "forgot-constant-of-integration"), (fmt(F0 + y0), "constant-equals-given-value")],
            )
        a, c = nz(rng, -5, 5), rng.randint(-3, 3)
        b = rng.choice([2, 3, 4, sp.E, sp.E ** 2])
        g = a / x + c
        val = sp.simplify(sp.integrate(g, (x, 1, b)))
        return Problem(
            prompt=(f"Using $\\int \\frac{{dx}}{{x}} = \\ln|x|$, compute $\\displaystyle\\int_1^{{{L(b)}}} \\left({L(g)}\\right)dx$ exactly."),
            answer=fmt(val), kind="num", answer_hint="exact, e.g. 3*log(2) + 2",
            rubric=[f"Antiderivative {L(a * sp.log(x) + c * x)} (ln|x| in general)", "Evaluate at the upper limit", "ln 1 = 0 at the lower limit"],
            solution=f"$\\left[{L(a * sp.log(x) + c * x)}\\right]_1^{{{L(b)}}} = {L(val)}$ (using $\\ln 1 = 0$).",
            misconceptions=[(fmt(sp.simplify(val + c)), "forgot-lower-limit"),
                            (fmt(sp.nsimplify(a * (1 - 1 / b) + c * (b - 1))), "antiderivative-of-1-over-x-wrong")],
        )
    if level == 2:
        if rng.choice(["average", "total_area"]) == "average":
            a, b, c = nz(rng, -3, 3), rng.randint(-4, 4), rng.randint(-5, 5)
            lo = rng.randint(-2, 1)
            hi = lo + rng.randint(2, 4)
            f = 3 * a * x ** 2 + 2 * b * x + c
            I = sp.integrate(f, (x, lo, hi))
            avg = I / (hi - lo)
            return Problem(
                prompt=f"Find the average value of $f(x) = {L(f)}$ on the interval $[{lo}, {hi}]$.",
                answer=fmt(avg), kind="num",
                rubric=["Average = (1/(b − a)) ∫_a^b f(x) dx", "Antiderivative and F(b) − F(a)", f"Divide by the length {hi - lo}"],
                solution=f"$\\int_{{{lo}}}^{{{hi}}} f\\,dx = {L(I)}$, average $= \\frac{{{L(I)}}}{{{hi - lo}}} = {L(avg)}$.",
                misconceptions=[(fmt(I), "forgot-divide-by-length"),
                                (fmt((f.subs(x, lo) + f.subs(x, hi)) / 2), "averaged-endpoints")],
            )
        k = rng.randint(1, 3)
        m = k + rng.randint(1, 2)
        f = x ** 2 - k ** 2
        total = sp.integrate(-f, (x, 0, k)) + sp.integrate(f, (x, k, m))
        net = sp.integrate(f, (x, 0, m))
        return Problem(
            prompt=(f"Find the **total** area between the curve $y = {L(f)}$ and the $x$-axis for $0 \\le x \\le {m}$ "
                    "(regions below the axis count as positive area)."),
            answer=fmt(total), kind="num",
            rubric=[f"Find where the curve crosses the axis: x = {k}", f"On [0, {k}] the curve is below the axis: integrate −f",
                    f"On [{k}, {m}] it is above: integrate f", "Add the two areas"],
            solution=(f"$\\int_0^{{{k}}} ({L(-f)})\\,dx + \\int_{{{k}}}^{{{m}}} ({L(f)})\\,dx = "
                      f"{L(sp.integrate(-f, (x, 0, k)))} + {L(sp.integrate(f, (x, k, m)))} = {L(total)}$ "
                      f"(the signed integral would give ${L(net)}$)."),
            misconceptions=[(fmt(net), "net-vs-total-area"), (fmt(abs(net)), "net-vs-total-area")],
        )
    if rng.choice(["density", "halving"]) == "density":
        nn = rng.randint(1, 3)
        cc = sp.Symbol("c", positive=True)
        cval = sp.solve(sp.Eq(cc * sp.integrate(x ** nn * (1 - x), (x, 0, 1)), 1), cc)[0]
        prob = sp.integrate(cval * x ** nn * (1 - x), (x, 0, sp.Rational(1, 2)))
        xn = "x" if nn == 1 else f"x^{{{nn}}}"
        return Problem(
            prompt=(f"Let $p(x) = c\\,{xn}(1 - x)$ for $0 \\le x \\le 1$ and $0$ elsewhere. Find the constant $c$ that makes $p$ "
                    "a probability density, then compute $P(X \\le \\tfrac12)$. Type the probability."),
            answer=fmt(prob), kind="num",
            rubric=["Normalise: c ∫₀¹ xⁿ(1 − x) dx = 1", f"∫₀¹ xⁿ(1−x) dx = 1/(n+1) − 1/(n+2), so c = {cval}",
                    "P(X ≤ 1/2) = ∫₀^(1/2) p(x) dx", "Evaluate with exact fractions"],
            solution=(f"$c = {cval}$. $P(X\\le\\tfrac12) = {cval}\\int_0^{{1/2}} {xn}(1-x)\\,dx = {L(prob)}$."),
            misconceptions=[(fmt(prob / cval), "forgot-normalising-constant"), (fmt(sp.Rational(1, 2)), "assumed-symmetry")],
        )
    nn, bnd = rng.randint(1, 3), rng.randint(1, 3)
    cc = sp.Symbol("c", positive=True)
    half = sp.integrate(x ** nn, (x, 0, bnd)) / 2
    cval = sp.solve(sp.Eq(sp.integrate(x ** nn, (x, 0, cc)), half), cc)[0]
    xn = "x" if nn == 1 else f"x^{{{nn}}}"
    return Problem(
        prompt=(f"The region under $y = {xn}$ for $0 \\le x \\le {bnd}$ is cut by a vertical line $x = c$ into two parts of "
                "**equal area**. Find $c$ exactly."),
        answer=fmt(cval), kind="num", answer_hint="exact, e.g. sqrt(2) or 3/2^(1/3)",
        rubric=[f"Total area ∫₀^{bnd} {xn} dx = {2*half}", f"Set ∫₀^c {xn} dx = c^{nn+1}/{nn+1} equal to half of it",
                "Solve for c > 0"],
        solution=(f"Total area $= {L(2*half)}$. $\\frac{{c^{{{nn+1}}}}}{{{nn+1}}} = {L(half)} \\Rightarrow c = {L(cval)} "
                  f"\\approx {float(cval):.4f}$."),
        misconceptions=[(fmt(sp.Rational(bnd, 2)), "halved-the-interval")],
    )


@gen("calc.integration_techniques", name="integration_techniques_more")
def integration_techniques_more(rng, level):
    if level == 1:
        if rng.choice(["logderiv", "halfangle"]) == "logderiv":
            while True:
                p, q, b = rng.randint(-4, 4), rng.randint(1, 6), rng.randint(1, 3)
                g = x ** 2 + p * x + q
                if all(g.subs(x, sp.Rational(k, 10)) > 0 for k in range(10 * b + 1)):
                    break
            val = sp.simplify(sp.integrate(sp.diff(g, x) / g, (x, 0, b)))
            return Problem(
                prompt=(f"Compute $\\displaystyle\\int_0^{{{b}}} \\frac{{{L(sp.diff(g, x))}}}{{{L(g)}}}\\,dx$ exactly. "
                        "(Look for the pattern $\\int \\frac{g'(x)}{g(x)}\\,dx$.)"),
                answer=fmt(val), kind="num", answer_hint="exact, e.g. log(7/3)",
                rubric=["Numerator is the derivative of the denominator", "Antiderivative ln|g(x)|",
                        f"Evaluate ln g({b}) − ln g(0)"],
                solution=f"$\\left[\\ln({L(g)})\\right]_0^{{{b}}} = \\ln {g.subs(x, b)} - \\ln {q} = {L(val)}$.",
                misconceptions=[(fmt(sp.log(g.subs(x, b))), "forgot-lower-limit")],
            )
        fn = rng.choice([sp.sin, sp.cos])
        c = rng.choice(PI_ANGLES + [sp.pi])
        val = sp.simplify(sp.integrate(fn(x) ** 2, (x, 0, c)))
        ident = "\\sin^2 x = \\tfrac12(1 - \\cos 2x)" if fn is sp.sin else "\\cos^2 x = \\tfrac12(1 + \\cos 2x)"
        return Problem(
            prompt=(f"Use the double-angle identity ${ident}$ to compute "
                    f"$\\displaystyle\\int_0^{{{L(c)}}} {L(fn(x)**2)}\\,dx$ exactly."),
            answer=fmt(val), kind="num", answer_hint="exact, e.g. pi/8 + 1/4",
            rubric=["Rewrite the square with the identity", "∫ cos 2x dx = sin(2x)/2", "Evaluate between the limits"],
            solution=f"$= \\left[\\tfrac{{x}}{{2}} {'-' if fn is sp.sin else '+'} \\tfrac{{\\sin 2x}}{{4}}\\right]_0^{{{L(c)}}} = {L(val)}$.",
            misconceptions=[(fmt(c / 2), "dropped-oscillating-term"),
                            (fmt(sp.simplify(fn(c) ** 3 / 3 - fn(0) ** 3 / 3)), "power-rule-on-trig")],
        )
    if level == 2:
        if rng.choice(["partial", "trigsub"]) == "partial":
            a, b = sorted(rng.sample(range(1, 5), 2))
            f = 1 / ((x + a) * (x + b))
            val = sp.simplify(sp.expand_log(sp.integrate(f, (x, 0, 1)), force=True))
            pf = sp.apart(f, x)
            return Problem(
                prompt=(f"Use partial fractions to compute $\\displaystyle\\int_0^1 \\frac{{dx}}{{({L(x + a)})({L(x + b)})}}$ exactly."),
                answer=fmt(val), kind="num", answer_hint="exact, using log(...), e.g. log(4/3)",
                rubric=["Write 1/((x+a)(x+b)) = A/(x+a) + B/(x+b)", f"A = 1/{b - a}, B = −1/{b - a} (cover-up)",
                        "Integrate to logarithms", "Evaluate and combine the logs"],
                solution=f"$\\frac{{1}}{{({L(x + a)})({L(x + b)})}} = {L(pf)}$, so the integral is ${L(val)} \\approx {float(val):.5f}$.",
                misconceptions=[(fmt(sp.simplify(val * (b - a))), "partial-fractions-forgot-coefficient")],
            )
        a = rng.randint(2, 4)
        if rng.random() < 0.5:
            f, shown = sp.sqrt(a ** 2 - x ** 2), f"\\sqrt{{{a**2} - x^2}}"
            sub, sub_txt = f"x = {a}\\sin\\theta", f"x = {a} sin θ, dx = {a} cos θ dθ"
            theta = f"{a**2}\\int_0^{{\\pi/2}}\\cos^2\\theta\\,d\\theta = {a**2}\\cdot\\frac{{\\pi}}{{4}}"
            wrong = sp.pi * a / 4
        else:
            f, shown = 1 / (a ** 2 + x ** 2), f"\\frac{{1}}{{{a**2} + x^2}}"
            sub, sub_txt = f"x = {a}\\tan\\theta", f"x = {a} tan θ, dx = {a} sec²θ dθ"
            theta = f"\\frac{{1}}{{{a}}}\\int_0^{{\\pi/4}} d\\theta"
            wrong = sp.pi / 4
        val = sp.integrate(f, (x, 0, a))
        return Problem(
            prompt=(f"Use a trigonometric substitution to compute $\\displaystyle\\int_0^{{{a}}} {shown}\\,dx$ exactly."),
            answer=fmt(val), kind="num", answer_hint="exact, e.g. pi/8",
            rubric=[f"Substitute {sub_txt}", "Express dx and the integrand in θ",
                    "New limits θ = 0 to π/4 or π/2", "Integrate and simplify"],
            solution=f"With ${sub}$ the integral becomes ${theta} = {L(val)}$.",
            misconceptions=[(fmt(wrong), "trig-sub-forgot-scaling")],
        )
    if rng.choice(["gauss4", "wallis"]) == "gauss4":
        sig = rng.choice([1, 2, 3, sp.Rational(1, 2)])
        dens = sp.exp(-x ** 2 / (2 * sig ** 2)) / (sig * sp.sqrt(2 * sp.pi))
        m4 = sp.simplify(sp.integrate(x ** 4 * dens, (x, -sp.oo, sp.oo)))
        return Problem(
            prompt=(f"$X \\sim \\mathcal{{N}}(0, \\sigma^2)$ with $\\sigma = {L(sig)}$ has density "
                    "$p(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-x^2/(2\\sigma^2)}$, and you may use $\\int p = 1$. "
                    "Using integration by parts (note $x\\,e^{-x^2/(2\\sigma^2)}$ has an elementary antiderivative), derive "
                    "$\\mathbb{E}[X^2] = \\sigma^2$ and then $\\mathbb{E}[X^4]$. Type $\\mathbb{E}[X^4]$."),
            answer=fmt(m4), kind="num",
            rubric=["Write x² p(x) = x · (x p(x)) and integrate by parts: E[X²] = σ² ∫p = σ²",
                    "Write x⁴ p(x) = x³ · (x p(x)); boundary terms vanish", "E[X⁴] = 3σ² E[X²] = 3σ⁴",
                    "Substitute σ"],
            solution=(f"$\\int x^{{n}} p\\,dx$: with $u = x^{{n-1}}$, $dv = x p\\,dx$, $v = -\\sigma^2 p$, so "
                      f"$\\mathbb{{E}}[X^n] = (n-1)\\sigma^2\\mathbb{{E}}[X^{{n-2}}]$. Hence $\\mathbb{{E}}[X^4] = 3\\sigma^4 = {L(m4)}$."),
            misconceptions=[(fmt(sig ** 4), "forgot-factor-3"), (fmt(sig ** 2), "gave-variance")],
        )
    nn = rng.randint(3, 6)
    val = sp.integrate(sp.sin(x) ** nn, (x, 0, sp.pi / 2))
    return Problem(
        prompt=("Let $I_n = \\displaystyle\\int_0^{\\pi/2} \\sin^n x\\,dx$. Integrate by parts with $u = \\sin^{n-1}x$ to "
                f"derive the reduction formula $I_n = \\frac{{n-1}}{{n}} I_{{n-2}}$, then compute $I_{{{nn}}}$ exactly."),
        answer=fmt(val), kind="num", answer_hint="exact, e.g. 3*pi/16 or 8/15",
        rubric=["Parts: u = sin^(n−1)x, dv = sin x dx; boundary term vanishes",
                "Use cos²x = 1 − sin²x to get n I_n = (n−1) I_(n−2)", "Base cases I₀ = π/2, I₁ = 1",
                f"Apply the recursion down to the base case for n = {nn}"],
        solution=(f"$I_{{{nn}}} = " + "\\cdot".join(f"\\frac{{{k - 1}}}{{{k}}}" for k in range(nn, 1, -2))
                  + (" \\cdot \\frac{\\pi}{2}" if nn % 2 == 0 else " \\cdot 1") + f" = {L(val)}$."),
        misconceptions=[(fmt(sp.Rational(nn - 1, nn)), "stopped-reduction-early"),
                        (fmt(val * sp.pi / 2 if nn % 2 else val * 2 / sp.pi), "wrong-base-case")],
    )


@gen("calc.series_taylor", name="series_taylor_more", levels=(1, 2))
def series_taylor_more(rng, level):
    if level == 1:
        if rng.choice(["geometric", "recognise"]) == "geometric":
            a = nz(rng, -6, 6)
            r = rng.choice([sp.Rational(1, 2), sp.Rational(-1, 2), sp.Rational(1, 3), sp.Rational(-1, 3),
                            sp.Rational(2, 3), sp.Rational(-2, 3), sp.Rational(3, 4), sp.Rational(1, 4)])
            nn = sp.Symbol("n", integer=True, nonnegative=True)
            total = sp.summation(a * r ** nn, (nn, 0, sp.oo))
            return Problem(
                prompt=f"Find the sum of the geometric series $\\displaystyle\\sum_{{n=0}}^{{\\infty}} {a}\\left({L(r)}\\right)^n$.",
                answer=fmt(total), kind="num",
                rubric=["First term a (the n = 0 term)", "Ratio r with |r| < 1, so it converges", "Sum = a/(1 − r)"],
                solution=f"$\\frac{{{a}}}{{1 - \\left({L(r)}\\right)}} = {L(total)}$.",
                misconceptions=[(fmt(a / (1 + r)), "geometric-sign"), (fmt(a * r / (1 - r)), "started-at-n-1")],
            )
        if rng.random() < 0.5:
            a = rng.choice([2, 3, -1, -2])
            body = f"\\frac{{{a}^n}}{{n!}}" if a > 0 else f"\\frac{{({a})^n}}{{n!}}"
            total, start, wrong = sp.exp(a), 0, sp.exp(-a)
            known = f"e^u = \\sum u^n/n!$ with $u = {a}"
        else:
            r = rng.choice([sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 4)])
            body = f"(-1)^{{n+1}}\\frac{{({L(r)})^n}}{{n}}"
            total, start, wrong = sp.log(1 + r), 1, -sp.log(1 - r)
            known = f"\\ln(1+u) = \\sum_{{n\\ge1}} (-1)^{{n+1}}u^n/n$ with $u = {L(r)}"
        return Problem(
            prompt=(f"Find the exact value of $\\displaystyle\\sum_{{n={start}}}^{{\\infty}} {body}$ by recognising a known "
                    "Maclaurin series."),
            answer=fmt(total), kind="num", answer_hint="exact, e.g. exp(3) or log(3/2)",
            rubric=["Compare with the standard series eᵘ, ln(1+u), cos u, 1/(1−u)", "Identify the value of u",
                    "Evaluate the function at that u"],
            solution=f"This is ${known}$, so the sum is ${L(total)}$.",
            misconceptions=[(fmt(wrong), "wrong-sign-pattern")],
        )
    if rng.choice(["radius", "integral"]) == "radius":
        p, q = rng.choice([(1, 2), (1, 3), (2, 3), (3, 2), (2, 1), (3, 1), (1, 4), (3, 4)])
        c = rng.randint(-3, 3)
        R = sp.Rational(q, p)
        ratio = f"\\frac{{{p}^n}}{{{q}^n\\,n}}" if p != 1 else f"\\frac{{1}}{{{q}^n\\,n}}"
        if q == 1:
            ratio = f"\\frac{{{p}^n}}{{n}}"
        xc = L(x - c)
        return Problem(
            prompt=(f"Use the ratio test to find where $\\displaystyle\\sum_{{n=1}}^{{\\infty}} {ratio}\\,({xc})^n$ converges. "
                    "Type the endpoints of the open interval of convergence, left endpoint first."),
            answer=[fmt(c - R), fmt(c + R)], kind="list", answer_hint="left, right   e.g. -1/2, 3/2",
            rubric=["|a_(n+1)/a_n| → (p/q)|x − c|", "Converges when (p/q)|x − c| < 1",
                    f"Radius R = {R}", f"Interval centred at {c}"],
            solution=(f"$\\left|\\frac{{a_{{n+1}}}}{{a_n}}\\right| = \\frac{{{p}}}{{{q}}}\\cdot\\frac{{n}}{{n+1}}|{xc}| \\to "
                      f"\\frac{{{p}}}{{{q}}}|{xc}|$, so $R = {L(R)}$ and the open interval is "
                      f"$({L(c - R)}, {L(c + R)})$."),
            misconceptions=[([fmt(c - 1 / R), fmt(c + 1 / R)], "radius-inverted"), ([fmt(-R), fmt(R)], "forgot-centre")],
        )
    h = rng.choice([sp.Rational(1, 2), sp.Integer(1), sp.Rational(1, 3)])
    if rng.random() < 0.5:
        f, shown = sp.exp(-x ** 2), "e^{-x^2}"
    else:
        f, shown = sp.sin(x) / x, "\\frac{\\sin x}{x}"
    ser = sp.series(f, x, 0, 6).removeO()
    terms = sp.Add.make_args(sp.expand(ser))
    approx = sp.integrate(ser, (x, 0, h))
    two = sp.integrate(sum(tm for tm in terms if sp.degree(tm, x) <= 2), (x, 0, h))
    return Problem(
        prompt=(f"Approximate $\\displaystyle\\int_0^{{{L(h)}}} {shown}\\,dx$ by integrating the first **three** non-zero "
                "terms of the Maclaurin series of the integrand. Give the exact fraction."),
        answer=fmt(approx), kind="num",
        rubric=["Write the series of the integrand from a known series (substitute or divide)",
                f"Keep three terms: {sp.latex(ser, order='rev-lex')}", "Integrate term by term", f"Evaluate at {h}"],
        solution=(f"${shown} \\approx {sp.latex(ser, order='rev-lex')}$, so the integral $\\approx {L(approx)} \\approx {float(approx):.6f}$ "
                  f"(true value ${float(sp.Integral(f, (x, 0, h)).evalf()):.6f}$)."),
        misconceptions=[(fmt(two), "truncated-too-early")],
    )


@gen("calc.partial_derivatives", name="partials_more")
def partials_more(rng, level):
    if level == 1:
        if rng.choice(["second", "at_point"]) == "second":
            a, b, c = nz(rng, -4, 4), nz(rng, -5, 5), nz(rng, -5, 5)
            p, q = rng.randint(2, 4), rng.randint(1, 3)
            f = a * x ** p * y ** q + b * x * y ** 2 + c * y
            var = rng.choice([x, y])
            d2 = sp.expand(sp.diff(f, var, 2))
            return Problem(
                prompt=(f"For $f(x, y) = {L(f)}$, compute the second partial derivative "
                        f"$f_{{{var}{var}}} = \\dfrac{{\\partial^2 f}}{{\\partial {var}^2}}$."),
                answer=fmt(d2), kind="expr",
                rubric=[f"Differentiate once with respect to {var} (other variable constant)",
                        f"Differentiate the result again with respect to {var}", "Terms linear in that variable vanish"],
                solution=f"$f_{{{var}}} = {L(sp.diff(f, var))}$, $f_{{{var}{var}}} = {L(d2)}$.",
                misconceptions=[(fmt(sp.diff(f, var)), "gave-first-partial"), (fmt(sp.diff(f, x, y)), "computed-mixed-partial")],
            )
        a, b, n = nz(rng, -3, 3), nz(rng, -3, 3), rng.randint(2, 3)
        while True:
            x0, y0 = rng.randint(-2, 2), rng.randint(-2, 2)
            if a * x0 + b * y0 != 0:
                break
        f = (a * x + b * y) ** n
        val = sp.diff(f, y).subs({x: x0, y: y0})
        return Problem(
            prompt=(f"Let $f(x, y) = \\left({L(a*x + b*y)}\\right)^{{{n}}}$. Evaluate $\\dfrac{{\\partial f}}{{\\partial y}}$ "
                    f"at the point $({x0}, {y0})$."),
            answer=fmt(val), kind="num",
            rubric=["Treat x as a constant", f"Chain rule: {n}({L(a*x + b*y)})^{n-1} · ({b})", f"Substitute ({x0}, {y0})"],
            solution=(f"$f_y = {n}\\left({L(a*x + b*y)}\\right){'' if n == 2 else '^{' + str(n - 1) + '}'}\\cdot({b})$; "
                      f"at the point the bracket is "
                      f"${a*x0 + b*y0}$, so $f_y = {val}$."),
            misconceptions=[(fmt(sp.Rational(val, b)), "chain-forgot-inner-derivative"),
                            (fmt(sp.diff(f, x).subs({x: x0, y: y0})), "differentiated-wrong-variable")],
        )
    if level == 2:
        if rng.choice(["linapprox", "implicit"]) == "linapprox":
            pa, pb = rng.choice([(2, 1), (1, 2), (2, 2), (3, 1)])
            x0, y0 = rng.randint(1, 3), rng.randint(1, 3)
            dx, dy = (sp.Rational(nz(rng, -2, 2), 10) for _ in range(2))
            f = x ** pa * y ** pb
            fx, fy = sp.diff(f, x).subs({x: x0, y: y0}), sp.diff(f, y).subs({x: x0, y: y0})
            approx = f.subs({x: x0, y: y0}) + fx * dx + fy * dy
            X1, Y1 = x0 + dx, y0 + dy
            return Problem(
                prompt=(f"Use the linear approximation (tangent plane) of $f(x, y) = {L(f)}$ at $({x0}, {y0})$ to estimate "
                        f"$f({float(X1):g}, {float(Y1):g})$."),
                answer=fmt(approx), kind="num", answer_hint="a decimal or fraction, e.g. 4.6",
                rubric=[f"f({x0},{y0}) = {f.subs({x: x0, y: y0})}", f"f_x = {fx}, f_y = {fy} at the point",
                        f"Δx = {float(dx):g}, Δy = {float(dy):g}", "f ≈ f₀ + f_x Δx + f_y Δy"],
                solution=(f"$f \\approx {f.subs({x: x0, y: y0})} + ({fx})({float(dx):g}) + ({fy})({float(dy):g}) = {float(approx):g}$ "
                          f"(exact value ${float(f.subs({x: X1, y: Y1})):.4f}$)."),
                misconceptions=[(fmt(f.subs({x: x0, y: y0}) + fx * dx), "linear-approx-missed-a-direction")],
            )
        px, py, pz, r = rng.choice([(1, 2, 2, 3), (2, 3, 6, 7), (1, 4, 8, 9), (2, 6, 9, 11), (4, 4, 7, 9)])
        px, py, pz = (v_ * rng.choice([-1, 1]) for v_ in (px, py, pz))
        z = sp.Symbol("z")
        F = x ** 2 + y ** 2 + z ** 2 - r ** 2
        zx = sp.Rational(-px, pz)
        assert -sp.diff(F, x).subs({x: px, z: pz}) / sp.diff(F, z).subs({x: px, z: pz}) == zx
        return Problem(
            prompt=(f"Near the point $({px}, {py}, {pz})$ the sphere $x^2 + y^2 + z^2 = {r**2}$ defines $z$ as a function of "
                    "$(x, y)$. Use implicit differentiation to find $\\dfrac{\\partial z}{\\partial x}$ at that point."),
            answer=fmt(zx), kind="num",
            rubric=["Differentiate both sides with respect to x, y held constant", "2x + 2z ∂z/∂x = 0",
                    "∂z/∂x = −x/z", f"Substitute x = {px}, z = {pz}"],
            solution=f"$2x + 2z\\,z_x = 0 \\Rightarrow z_x = -\\frac{{x}}{{z}} = -\\frac{{{px}}}{{{pz}}} = {L(zx)}$.",
            misconceptions=[(fmt(-zx), "implicit-sign"), (fmt(sp.Rational(-pz, px)), "inverted-implicit-ratio")],
        )
    if rng.choice(["euler", "wave"]) == "euler":
        p, q = rng.randint(1, 3), rng.randint(1, 3)
        dg = p + q
        a, b = nz(rng, -4, 4), nz(rng, -4, 4)
        m_ = rng.choice([v_ for v_ in range(dg + 1) if v_ != p])
        f = a * x ** p * y ** q + b * x ** m_ * y ** (dg - m_)
        k = sp.simplify((x * sp.diff(f, x) + y * sp.diff(f, y)) / f)
        return Problem(
            prompt=(f"For $f(x, y) = {L(f)}$, compute $x f_x + y f_y$ and show it equals $k\\,f$ for a constant $k$ "
                    "(Euler's identity for homogeneous functions). Type $k$."),
            answer=fmt(k), kind="num",
            rubric=["Compute f_x and f_y", "Multiply by x and y respectively", "Each monomial xᵐyⁿ gets the factor m + n",
                    "All monomials have the same total degree, so the sum is k f"],
            solution=(f"$x f_x + y f_y = {L(sp.expand(x * sp.diff(f, x) + y * sp.diff(f, y)))} = {k}\\,f$: "
                      f"$f$ is homogeneous of degree ${k}$."),
            misconceptions=[(str(k - 1), "degree-minus-one"), (str(p), "used-only-one-exponent")],
        )
    a, c = rng.randint(1, 4), rng.randint(1, 6)
    while c == a:
        c = rng.randint(1, 6)
    v = sp.Symbol("v", positive=True)
    uu = sp.sin(a * x) * sp.cos(c * t)
    vs = next(r_ for r_ in sp.solve(sp.Eq(sp.diff(uu, t, 2), v ** 2 * sp.diff(uu, x, 2)), v) if r_.is_positive)
    return Problem(
        prompt=(f"For which speed $v > 0$ does $u(x, t) = {L(uu)}$ satisfy the wave equation $u_{{tt}} = v^2\\,u_{{xx}}$?"),
        answer=fmt(vs), kind="num",
        rubric=[f"u_tt = −{c*c} u", f"u_xx = −{a*a} u", f"Equation: −{c*c} = −v²·{a*a}", "Take the positive root"],
        solution=f"$u_{{tt}} = -{c**2}u$, $u_{{xx}} = -{a**2}u$, so $v^2 = \\frac{{{c**2}}}{{{a**2}}}$ and $v = {L(vs)}$.",
        misconceptions=[(fmt(sp.Rational(c * c, a * a)), "forgot-square-root"), (fmt(sp.Rational(a, c)), "inverted-ratio")],
    )


@gen("calc.gradient", name="gradient_more")
def gradient_more(rng, level):
    if level == 1:
        a, b = nz(rng, -3, 3), nz(rng, -5, 5)
        x0, y0 = nz(rng, -3, 3), rng.randint(-3, 3)
        f = a * x ** 2 + b * y + x * y
        g = [sp.diff(f, v_).subs({x: x0, y: y0}) for v_ in (x, y)]
        while g == [0, 0]:
            x0 = nz(rng, -3, 3)
            g = [sp.diff(f, v_).subs({x: x0, y: y0}) for v_ in (x, y)]
        rate = sp.sqrt(g[0] ** 2 + g[1] ** 2)
        return Problem(
            prompt=(f"At the point $({x0}, {y0})$, what is the **maximum rate of increase** of $f(x, y) = {L(f)}$, i.e. the "
                    "largest directional derivative over all unit directions?"),
            answer=fmt(rate), kind="num", answer_hint="exact, e.g. sqrt(13)",
            rubric=["The fastest increase is in the direction of ∇f", f"∇f({x0},{y0}) = ({g[0]}, {g[1]})",
                    "Maximum rate = ‖∇f‖"],
            solution=f"$\\nabla f = {tup(g)}$, maximum rate $\\|\\nabla f\\| = \\sqrt{{{g[0]**2} + {g[1]**2}}} = {L(rate)}$.",
            misconceptions=[(fmt(g[0] ** 2 + g[1] ** 2), "forgot-square-root"), (fmt(g[0] + g[1]), "added-components")],
        )
    if level == 2:
        a, b, c = nz(rng, -3, 3), rng.randint(-3, 3), nz(rng, -3, 3)
        x0, y0 = nz(rng, -2, 2), nz(rng, -2, 2)
        f = a * x ** 2 + b * x * y + c * y ** 2
        f0 = f.subs({x: x0, y: y0})
        fx, fy = sp.diff(f, x).subs({x: x0, y: y0}), sp.diff(f, y).subs({x: x0, y: y0})
        plane = sp.expand(f0 + fx * (x - x0) + fy * (y - y0))
        return Problem(
            prompt=(f"Find the tangent plane to the surface $z = {L(f)}$ at the point above $({x0}, {y0})$. "
                    "Give $z$ as an expression in $x$ and $y$."),
            answer=fmt(plane), kind="expr", answer_hint="z as an expression in x and y, e.g. 3*x - 2*y + 1",
            rubric=[f"Height: f({x0},{y0}) = {f0}", f"Slopes: f_x = {fx}, f_y = {fy} at the point",
                    "z = f₀ + f_x (x − x₀) + f_y (y − y₀)", "Expand"],
            solution=f"$z = {f0} + ({fx})(x - ({x0})) + ({fy})(y - ({y0})) = {L(plane)}$.",
            misconceptions=[(fmt(sp.expand(fx * (x - x0) + fy * (y - y0))), "tangent-missing-intercept"),
                            (fmt(sp.expand(f0 + fx * x + fy * y)), "tangent-plane-not-shifted")],
        )
    a, b, c = nz(rng, -4, 4), nz(rng, -4, 4), nz(rng, -10, 10)
    lam = sp.Symbol("lambda")
    sol = sp.solve([2 * x - lam * a, 2 * y - lam * b, a * x + b * y - c], [x, y, lam], dict=True)[0]
    mn = sp.simplify((x ** 2 + y ** 2).subs(sol))
    return Problem(
        prompt=(f"Use Lagrange multipliers to find the **minimum value** of $x^2 + y^2$ subject to the constraint "
                f"${L(a*x + b*y)} = {c}$. (Geometrically: the squared distance from the origin to a line.)"),
        answer=fmt(mn), kind="num",
        rubric=["∇(x² + y²) = λ ∇(constraint): 2x = λa, 2y = λb", "Substitute x = λa/2, y = λb/2 into the constraint",
                f"λ = 2c/(a² + b²) = {sol[lam]}", f"Minimiser ({sol[x]}, {sol[y]}); evaluate x² + y²"],
        solution=(f"$\\lambda = {L(sol[lam])}$, $(x, y) = ({L(sol[x])}, {L(sol[y])})$, minimum "
                  f"$= \\frac{{c^2}}{{a^2 + b^2}} = \\frac{{{c**2}}}{{{a*a + b*b}}} = {L(mn)}$."),
        misconceptions=[(fmt(sp.sqrt(mn)), "gave-distance-not-squared"), (fmt(sol[lam]), "gave-multiplier-not-value")],
    )


@gen("calc.multivar_chain", name="multivar_chain_more")
def multivar_chain_more(rng, level):
    if level == 1:
        if rng.choice(["linear_inner", "polar"]) == "linear_inner":
            a, b, c, d = (nz(rng, -4, 4) for _ in range(4))
            p, q = nz(rng, -5, 5), nz(rng, -5, 5)
            while (a, c) == (b, d):
                b, d = nz(rng, -4, 4), nz(rng, -4, 4)
            ans = p * b + q * d
            return Problem(
                prompt=(f"Let $z = f(u, v)$ with $u = {L(a*x + b*y)}$ and $v = {L(c*x + d*y)}$. At the point of interest "
                        f"$f_u = {p}$ and $f_v = {q}$. Compute $\\dfrac{{\\partial z}}{{\\partial y}}$ there."),
                answer=str(ans), kind="num",
                rubric=["∂z/∂y = f_u ∂u/∂y + f_v ∂v/∂y", f"∂u/∂y = {b}, ∂v/∂y = {d}", "Multiply and add both paths"],
                solution=f"$\\frac{{\\partial z}}{{\\partial y}} = ({p})({b}) + ({q})({d}) = {ans}$.",
                misconceptions=[(str(p * a + q * c), "used-x-coefficients"), (str(p + q), "chain-forgot-inner-derivative")],
            )
        k, r0 = rng.randint(2, 4), rng.randint(1, 3)
        th0 = rng.choice([sp.pi / 6, sp.pi / 4, sp.pi / 3])
        rr, th = sp.symbols("r theta")
        zf = x ** 2 + k * y ** 2
        zpolar = zf.subs({x: rr * sp.cos(th), y: rr * sp.sin(th)})
        val = sp.nsimplify(sp.simplify(sp.diff(zpolar, th).subs({rr: r0, th: th0})))
        only_x = sp.nsimplify((2 * x * (-rr * sp.sin(th))).subs({x: rr * sp.cos(th)}).subs({rr: r0, th: th0}))
        return Problem(
            prompt=(f"Let $z = {L(zf)}$ where $x = r\\cos\\theta$, $y = r\\sin\\theta$. Use the chain rule to compute "
                    f"$\\dfrac{{\\partial z}}{{\\partial \\theta}}$ at $r = {r0}$, $\\theta = {L(th0)}$."),
            answer=fmt(val), kind="num", answer_hint="exact, e.g. 3*sqrt(3)/2",
            rubric=["∂z/∂θ = z_x ∂x/∂θ + z_y ∂y/∂θ", "∂x/∂θ = −r sin θ, ∂y/∂θ = r cos θ",
                    f"z_x = 2x, z_y = {2*k}y", "Evaluate at the given r, θ"],
            solution=(f"$\\frac{{\\partial z}}{{\\partial\\theta}} = 2x(-r\\sin\\theta) + {2*k}y(r\\cos\\theta) = "
                      f"{2*(k-1)}r^2\\sin\\theta\\cos\\theta = {L(val)}$."),
            misconceptions=[(fmt(only_x), "chain-missed-a-path"), (fmt(-val), "sign-error")],
        )
    if level == 2:
        if rng.choice(["jacdet", "matrix_chain"]) == "jacdet":
            while True:
                a, b, c = nz(rng, -3, 3), nz(rng, -3, 3), nz(rng, -3, 3)
                x0, y0 = nz(rng, -2, 2), nz(rng, -2, 2)
                U, V = a * x ** 2 + b * y, c * x * y
                J = sp.Matrix([U, V]).jacobian([x, y])
                det = J.det().subs({x: x0, y: y0})
                if det != 0:
                    break
            Jv = J.subs({x: x0, y: y0})
            return Problem(
                prompt=(f"Compute the Jacobian determinant $\\dfrac{{\\partial(u, v)}}{{\\partial(x, y)}}$ of the map "
                        f"$u = {L(U)}$, $v = {L(V)}$ at $(x, y) = ({x0}, {y0})$. (It is the local area-scaling factor.)"),
                answer=fmt(det), kind="num",
                rubric=["J = [[u_x, u_y], [v_x, v_y]]", f"Evaluate at ({x0}, {y0}): {Jv.tolist()}", "det = u_x v_y − u_y v_x"],
                solution=f"$J = {L(J)}$, at the point $J = {L(Jv)}$, $\\det J = {det}$.",
                misconceptions=[(fmt(-det), "jacobian-determinant-sign"), (fmt(Jv.trace()), "trace-not-determinant")],
            )
        while True:
            JH = sp.Matrix(2, 2, [rng.randint(-3, 3) for _ in range(4)])
            JG = sp.Matrix(2, 2, [rng.randint(-3, 3) for _ in range(4)])
            if JG * JH != JH * JG and JH.det() != 0 and JG.det() != 0:
                break
        return Problem(
            prompt=(f"Let $F = G\\circ H$ for maps $H, G: \\mathbb{{R}}^2 \\to \\mathbb{{R}}^2$. At a point $\\mathbf{{p}}$ the "
                    f"Jacobian of $H$ is $J_H(\\mathbf{{p}}) = {L(JH)}$ and the Jacobian of $G$ at $H(\\mathbf{{p}})$ is "
                    f"$J_G = {L(JG)}$. Compute $J_F(\\mathbf{{p}})$."),
            answer=fmt(JG * JH), kind="matrix", answer_hint="a 2x2 matrix, e.g. [[1, 2], [3, 4]]",
            rubric=["Multivariable chain rule: J_F(p) = J_G(H(p)) · J_H(p)", "Order matters: outer Jacobian on the left",
                    "Multiply row-by-column"],
            solution=f"$J_F = J_G J_H = {L(JG)}{L(JH)} = {L(JG * JH)}$.",
            misconceptions=[(fmt(JH * JG), "chain-matrix-order")],
        )
    if rng.choice(["implicit3", "sigmoid_backprop"]) == "implicit3":
        z = sp.Symbol("z")
        while True:
            a = nz(rng, -3, 3)
            x0, y0, z0 = nz(rng, -2, 2), nz(rng, -2, 2), nz(rng, -2, 2)
            G = x ** 2 * y + y * z ** 2 + a * z * x
            pt = {x: x0, y: y0, z: z0}
            Fz = sp.diff(G, z).subs(pt)
            if Fz != 0:
                break
        k = G.subs(pt)
        zx, zy = -sp.diff(G, x).subs(pt) / Fz, -sp.diff(G, y).subs(pt) / Fz
        return Problem(
            prompt=(f"The equation ${L(G)} = {k}$ defines $z$ implicitly as a function of $(x, y)$ near the point "
                    f"$({x0}, {y0}, {z0})$. Derive the formula for the partials of an implicit function from the chain rule, "
                    "then compute $\\left(\\dfrac{\\partial z}{\\partial x}, \\dfrac{\\partial z}{\\partial y}\\right)$ there."),
            answer=vec([zx, zy]), kind="vector", answer_hint="[dz/dx, dz/dy] with fractions, e.g. [1/2, -3]",
            rubric=["Differentiate F(x, y, z(x, y)) = const: F_x + F_z z_x = 0", "So z_x = −F_x/F_z and z_y = −F_y/F_z",
                    f"F_x, F_y, F_z at the point: {sp.diff(G, x).subs(pt)}, {sp.diff(G, y).subs(pt)}, {Fz}", "Divide"],
            solution=(f"$F_x = {L(sp.diff(G, x))}$, $F_y = {L(sp.diff(G, y))}$, $F_z = {L(sp.diff(G, z))}$. At the point: "
                      f"$z_x = -\\frac{{{sp.diff(G, x).subs(pt)}}}{{{Fz}}} = {L(zx)}$, $z_y = -\\frac{{{sp.diff(G, y).subs(pt)}}}{{{Fz}}} = {L(zy)}$."),
            misconceptions=[(fmt([-zx, -zy]), "implicit-sign")],
        )
    while True:
        x0, w1 = nz(rng, -3, 3), nz(rng, -2, 2)
        b1 = -w1 * x0
        w2, b2, yt = nz(rng, -3, 3), rng.randint(-2, 2), rng.randint(-3, 3)
        yhat = sp.Rational(w2, 2) + b2
        if yhat != yt:
            break
    W1, W2 = sp.symbols("W1 W2")
    sig = 1 / (1 + sp.exp(-(W1 * x0 + b1)))
    loss = sp.Rational(1, 2) * (W2 * sig + b2 - yt) ** 2
    g = [sp.nsimplify(sp.simplify(sp.diff(loss, v_).subs({W1: w1, W2: w2}))) for v_ in (W1, W2)]
    delta = yhat - yt
    return Problem(
        prompt=(f"A sigmoid unit: $z = w_1 x + b_1$, $h = \\sigma(z)$, $\\hat y = w_2 h + b_2$, $\\mathcal{{L}} = \\frac12(\\hat y - y)^2$. "
                f"With $x = {x0}$, $w_1 = {w1}$, $b_1 = {b1}$, $w_2 = {w2}$, $b_2 = {b2}$, $y = {yt}$: do the forward pass, "
                "then backpropagate to get $\\left(\\frac{\\partial\\mathcal{L}}{\\partial w_1}, \\frac{\\partial\\mathcal{L}}{\\partial w_2}\\right)$. "
                "Use $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$."),
        answer=vec(g), kind="vector", answer_hint="[dL/dw1, dL/dw2], fractions allowed",
        rubric=["Forward: z = 0, h = σ(0) = 1/2", f"ŷ = {yhat}, δ = ŷ − y = {delta}", "∂L/∂w₂ = δ · h",
                "∂L/∂w₁ = δ · w₂ · σ'(z) · x with σ'(0) = 1/4"],
        solution=(f"$z = 0$, $h = \\tfrac12$, $\\hat y = {L(yhat)}$, $\\delta = {L(delta)}$. "
                  f"$\\partial_{{w_2}}\\mathcal{{L}} = \\delta h = {L(g[1])}$, "
                  f"$\\partial_{{w_1}}\\mathcal{{L}} = \\delta\\,w_2\\,\\tfrac14\\,x = {L(g[0])}$."),
        misconceptions=[(fmt([delta * w2 * x0, delta / 2]), "backprop-forgot-activation-derivative"),
                        (fmt([-v_ for v_ in g]), "loss-gradient-sign")],
    )


@gen("calc.hessian", name="hessian_more")
def hessian_more(rng, level):
    if level == 1:
        if rng.choice(["classify", "critical"]) == "classify":
            kind = rng.choice(["min", "max", "saddle", "flat"])
            while True:
                a, b, c = nz(rng, -6, 6), rng.randint(-4, 4), nz(rng, -6, 6)
                det = a * c - b * b
                ok = {"min": det > 0 and a > 0, "max": det > 0 and a < 0, "saddle": det < 0, "flat": det == 0}[kind]
                if ok:
                    break
            H = sp.Matrix([[a, b], [b, c]])
            ans = {"min": "A", "max": "B", "saddle": "C", "flat": "D"}[kind]
            mis = []
            if kind == "saddle":
                mis.append(("A" if a > 0 else "B", "ignored-determinant"))
            elif kind in ("min", "max"):
                mis.append(("B" if kind == "min" else "A", "max-min-swapped"))
            else:
                mis.append(("A" if a > 0 else "B", "ignored-zero-determinant"))
            return Problem(
                prompt=(f"At a critical point of $f(x, y)$ the Hessian is $H = {L(H)}$. What does the second-derivative test say?\n\n"
                        "- **A**: local minimum\n- **B**: local maximum\n- **C**: saddle point\n- **D**: the test is inconclusive"),
                answer=ans, kind="choice",
                rubric=["Compute det H = f_xx f_yy − f_xy²", "det < 0 → saddle; det = 0 → inconclusive",
                        "det > 0: sign of f_xx decides min (+) or max (−)"],
                solution=(f"$\\det H = ({a})({c}) - ({b})^2 = {det}$" +
                          {"min": f", positive, and $f_{{xx}} = {a} > 0$: local minimum (A).",
                           "max": f", positive, and $f_{{xx}} = {a} < 0$: local maximum (B).",
                           "saddle": ", negative: saddle point (C).",
                           "flat": ": the test is inconclusive (D)."}[kind]),
                misconceptions=mis,
            )
        while True:
            a, c, b = nz(rng, -3, 3), nz(rng, -3, 3), rng.randint(-3, 3)
            H = sp.Matrix([[2 * a, b], [b, 2 * c]])
            if H.det() != 0:
                break
        p, q = rng.randint(-3, 3), rng.randint(-3, 3)
        lin = -H * sp.Matrix([p, q])
        f = a * x ** 2 + b * x * y + c * y ** 2 + lin[0] * x + lin[1] * y
        sol = sp.solve([sp.diff(f, x), sp.diff(f, y)], [x, y], dict=True)[0]
        return Problem(
            prompt=f"Find the critical point of $f(x, y) = {L(f)}$, i.e. solve $\\nabla f = \\mathbf{{0}}$.",
            answer=vec([sol[x], sol[y]]), kind="vector", answer_hint="[x, y], e.g. [1, -2]",
            rubric=["f_x = 0 and f_y = 0 give a 2×2 linear system", "Solve by elimination or substitution",
                    "Check both equations"],
            solution=(f"$f_x = {L(sp.diff(f, x))} = 0$, $f_y = {L(sp.diff(f, y))} = 0$ $\\Rightarrow (x, y) = ({p}, {q})$."),
            misconceptions=[(fmt([q, p]), "swapped-x-y"), (fmt([-p, -q]), "sign-error-solving-gradient")],
        )
    if level == 2:
        if rng.choice(["convex_k", "saddle"]) == "convex_k":
            a, c = rng.choice([(1, 1), (1, 4), (4, 1), (2, 2), (1, 9), (9, 1), (2, 8), (8, 2), (3, 3), (4, 4), (3, 12)])
            k = sp.Symbol("k")
            H = sp.hessian(a * x ** 2 + k * x * y + c * y ** 2, (x, y))
            kmax = max(sp.solve(sp.Eq(H.det(), 0), k))
            return Problem(
                prompt=(f"For which values of $k$ is $f(x, y) = {a if a > 1 else ''}x^2 + kxy + {c if c > 1 else ''}y^2$ convex? "
                        "Type the **largest** such $k$."),
                answer=fmt(kmax), kind="num",
                rubric=[f"Hessian H = [[{2*a}, k], [k, {2*c}]] (constant)", "Convex ⇔ H positive semidefinite",
                        f"Diagonal entries > 0; need det H = {4*a*c} − k² ≥ 0", f"|k| ≤ {kmax}"],
                solution=(f"$H = {L(H)}$; $\\det H = {4*a*c} - k^2 \\ge 0 \\iff |k| \\le {L(kmax)}$. Largest $k = {L(kmax)}$."),
                misconceptions=[(fmt(kmax / 2), "forgot-factor-2-in-hessian"), (fmt(kmax ** 2), "forgot-square-root")],
            )
        p, c, q = rng.randint(1, 3), nz(rng, -3, 3), rng.randint(-2, 2)
        f = x ** 3 - 3 * p ** 2 * x + c * (y - q) ** 2
        crit = sp.solve([sp.diff(f, x), sp.diff(f, y)], [x, y], dict=True)
        H = sp.hessian(f, (x, y))
        saddle = next(pt for pt in crit if H.subs(pt).det() < 0)
        other = next(pt for pt in crit if pt != saddle)
        return Problem(
            prompt=(f"$f(x, y) = {L(f)}$ has two critical points. Find both, classify them with the Hessian, and type the "
                    "coordinates of the **saddle point**."),
            answer=vec([saddle[x], saddle[y]]), kind="vector", answer_hint="[x, y]",
            rubric=[f"f_x = 3x² − {3*p*p} = 0 → x = ±{p}", f"f_y = {2*c}(y − {q}) = 0 → y = {q}",
                    f"H = diag(6x, {2*c})", "det H < 0 at the saddle"],
            solution=(f"Critical points $(\\pm{p}, {q})$. $\\det H = 6x\\cdot({2*c})$: negative at $({saddle[x]}, {q})$ (saddle), "
                      f"positive at $({other[x]}, {q})$ ({'minimum' if c > 0 else 'maximum'})."),
            misconceptions=[(fmt([other[x], other[y]]), "misclassified-critical-point")],
        )
    if rng.choice(["newton", "logistic"]) == "newton":
        a, q = rng.randint(1, 3), rng.randint(-3, 3)
        x0, y0 = nz(rng, -3, 3), rng.randint(-3, 3)
        while y0 == q:
            y0 = rng.randint(-3, 3)
        f = x ** 4 + a * (y - q) ** 2
        pt = {x: x0, y: y0}
        gv = sp.Matrix([sp.diff(f, v_) for v_ in (x, y)]).subs(pt)
        Hv = sp.hessian(f, (x, y)).subs(pt)
        new = sp.Matrix([x0, y0]) - Hv.inv() * gv
        gd = sp.Matrix([x0, y0]) - gv
        return Problem(
            prompt=(f"Take one step of **Newton's method** $\\mathbf{{x}}_1 = \\mathbf{{x}}_0 - H(\\mathbf{{x}}_0)^{{-1}}\\nabla f(\\mathbf{{x}}_0)$ "
                    f"for minimising $f(x, y) = {L(f)}$ from $\\mathbf{{x}}_0 = ({x0}, {y0})$. Explain on paper why the "
                    "$y$-coordinate lands exactly on the minimiser while $x$ does not."),
            answer=vec(list(new)), kind="vector", answer_hint="[x1, y1], fractions allowed",
            rubric=[f"∇f(x₀) = ({gv[0]}, {gv[1]})", f"H(x₀) = diag({Hv[0, 0]}, {Hv[1, 1]})",
                    "H is diagonal: divide each gradient component by its curvature",
                    "y-part is quadratic, so Newton is exact there; x⁴ is not quadratic"],
            solution=(f"$\\nabla f = {tup(list(gv))}$, $H = {L(Hv)}$, so $\\mathbf{{x}}_1 = ({x0}, {y0}) - "
                      f"\\left(\\tfrac{{{gv[0]}}}{{{Hv[0, 0]}}}, \\tfrac{{{gv[1]}}}{{{Hv[1, 1]}}}\\right) = {tup(list(new))}$."),
            misconceptions=[(fmt(list(gd)), "used-gradient-not-newton"), (fmt(list(sp.Matrix([x0, y0]) + Hv.inv() * gv)), "newton-sign-error")],
        )
    while True:
        p, q = nz(rng, -3, 3), rng.randint(-3, 3)
        if (p, q) != (0, 0):
            break
    yl = rng.choice([-1, 1])
    W1, W2 = sp.symbols("W1 W2")
    loss = sp.log(1 + sp.exp(-yl * (W1 * p + W2 * q)))
    H = sp.simplify(sp.hessian(loss, (W1, W2)).subs({W1: 0, W2: 0}))
    xx = sp.Matrix([p, q]) * sp.Matrix([[p, q]])
    return Problem(
        prompt=(f"Logistic loss for one example with features $\\mathbf{{x}} = ({p}, {q})$ and label $y = {yl}$: "
                "$\\ell(\\mathbf{w}) = \\ln\\left(1 + e^{-y\\,\\mathbf{w}^\\top\\mathbf{x}}\\right)$. "
                "Derive $\\nabla\\ell$ and the Hessian $\\nabla^2\\ell = \\sigma(m)(1-\\sigma(m))\\,\\mathbf{x}\\mathbf{x}^\\top$ "
                "(with $m = y\\,\\mathbf{w}^\\top\\mathbf{x}$), then evaluate the Hessian at $\\mathbf{w} = \\mathbf{0}$."),
        answer=fmt(H), kind="matrix", answer_hint="a 2x2 matrix, e.g. [[1/4, 0], [0, 1/4]]",
        rubric=["∇ℓ = −y σ(−m) x", "Differentiate again: ∇²ℓ = y² σ(m)(1 − σ(m)) x xᵀ, and y² = 1",
                "At w = 0: σ(0)(1 − σ(0)) = 1/4", "Multiply out x xᵀ"],
        solution=(f"$\\nabla^2\\ell(\\mathbf{{0}}) = \\tfrac14\\mathbf{{x}}\\mathbf{{x}}^\\top = \\tfrac14{L(xx)} = {L(H)}$; "
                  "it is positive semidefinite, which is why logistic regression is convex."),
        misconceptions=[(fmt(xx), "forgot-sigma-prime-factor"), (fmt(xx / 2), "used-sigma-not-sigma-prime")],
    )


@gen("calc.multiple_integrals", name="multiple_integrals_more")
def multiple_integrals_more(rng, level):
    if level == 1:
        if rng.choice(["average", "triple"]) == "average":
            a, b = nz(rng, -4, 4), nz(rng, -3, 3)
            p, q = rng.randint(1, 3), rng.randint(1, 3)
            f = a * x + b * y ** 2
            I = sp.integrate(f, (x, 0, p), (y, 0, q))
            avg = I / (p * q)
            return Problem(
                prompt=f"Find the average value of $f(x, y) = {L(f)}$ over the rectangle $[0, {p}] \\times [0, {q}]$.",
                answer=fmt(avg), kind="num",
                rubric=["Average = (1/area) ∬_R f dA", f"Area = {p*q}", "Integrate term by term (separable pieces)",
                        "Divide by the area"],
                solution=f"$\\iint_R f\\,dA = {L(I)}$, average $= \\frac{{{L(I)}}}{{{p*q}}} = {L(avg)}$.",
                misconceptions=[(fmt(I), "forgot-divide-by-area")],
            )
        k, p, q, r = rng.randint(1, 6), rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3)
        z = sp.Symbol("z")
        val = sp.integrate(k * x * y * z, (x, 0, p), (y, 0, q), (z, 0, r))
        return Problem(
            prompt=(f"Compute the triple integral $\\displaystyle\\iiint_B {k if k > 1 else ''}xyz\\,dV$ over the box "
                    f"$B = [0, {p}] \\times [0, {q}] \\times [0, {r}]$."),
            answer=fmt(val), kind="num",
            rubric=["Iterate: integrate in x, then y, then z (any order for a box)",
                    "Separable integrand: product of three 1-D integrals", "∫₀ᵃ s ds = a²/2"],
            solution=(f"$= {k}\\cdot\\frac{{{p}^2}}{{2}}\\cdot\\frac{{{q}^2}}{{2}}\\cdot\\frac{{{r}^2}}{{2}} = {L(val)}$."),
            misconceptions=[(fmt(sp.Integer(k * p * p * q * q * r * r) / 2), "dropped-factors-of-half"),
                            (fmt(sp.Integer(k * p * q * r)), "forgot-to-integrate")],
        )
    if level == 2:
        m_, n_ = rng.sample(range(4), 2)
        cc = sp.Symbol("c", positive=True)
        cval = sp.solve(sp.Eq(cc * sp.integrate(x ** m_ * y ** n_, (x, 0, 1), (y, 0, 1)), 1), cc)[0]
        prob = sp.integrate(cval * x ** m_ * y ** n_, (x, 0, y), (y, 0, 1))
        body = L(x ** m_ * y ** n_)
        return Problem(
            prompt=(f"$(X, Y)$ has joint density $p(x, y) = c\\,{body}$ on the unit square $[0,1]^2$ (zero elsewhere). "
                    "Find $c$, then compute $P(X < Y)$. Type the probability."),
            answer=fmt(prob), kind="num",
            rubric=["Normalise: c ∬ p = 1 over the square", f"c = {cval}", "Region X < Y: 0 ≤ x ≤ y, 0 ≤ y ≤ 1",
                    "Inner integral in x from 0 to y, outer in y"],
            solution=(f"$c = {cval}$. $P(X<Y) = \\int_0^1\\int_0^y {cval}\\,{body}\\,dx\\,dy = {L(prob)}$."),
            misconceptions=[(fmt(sp.Rational(1, 2)), "assumed-symmetry"), (fmt(1 - prob), "integrated-wrong-region")],
        )
    m_, A, B = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 4)
    uu, vv = sp.symbols("u v")
    jac = abs(sp.Matrix([(uu + vv) / 2, (uu - vv) / 2]).jacobian([uu, vv]).det())
    val = sp.integrate(uu ** m_ * jac, (uu, 0, A), (vv, 0, B))
    return Problem(
        prompt=(f"Let $D$ be the parallelogram $0 \\le x + y \\le {A}$, $0 \\le x - y \\le {B}$. Use the change of variables "
                f"$u = x + y$, $v = x - y$ to evaluate $\\displaystyle\\iint_D (x + y)^{{{m_}}}\\,dA$."),
        answer=fmt(val), kind="num",
        rubric=["Invert: x = (u + v)/2, y = (u − v)/2", "|∂(x,y)/∂(u,v)| = 1/2, so dA = ½ du dv",
                f"New region is the rectangle [0, {A}] × [0, {B}]", "Integrate u^m over it"],
        solution=(f"$\\iint_D (x+y)^{{{m_}}}\\,dA = \\int_0^{{{B}}}\\int_0^{{{A}}} u^{{{m_}}}\\cdot\\tfrac12\\,du\\,dv = {L(val)}$."),
        misconceptions=[(fmt(2 * val), "forgot-jacobian"), (fmt(4 * val), "inverted-jacobian")],
    )
