"""Track 0 — Foundations: additional problem types so every level has 3+ different kinds of task."""
from __future__ import annotations

import sympy as sp

from engine.checker import fmt
from engine.content import Problem, gen

x, y, n, k, t = sp.symbols("x y n k t")
L = sp.latex


def nz(rng, lo, hi):
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def signed(c, var=""):
    """'+ 3x' / '- 3x' for building readable LaTeX."""
    return f"{'+' if c >= 0 else '-'} {abs(c)}{var}"


# =========================================================================== fractions
@gen("found.fractions", name="fraction_of_quantity")
def fraction_of_quantity(rng, level):
    if level == 1:
        d = rng.choice([3, 4, 5, 6, 8]); a = rng.choice([v for v in range(1, d) if sp.gcd(v, d) == 1]); q = d * rng.randint(3, 12)
        ans = sp.Rational(a, d) * q
        return Problem(
            prompt=f"What is $\\dfrac{{{a}}}{{{d}}}$ of ${q}$?",
            answer=fmt(ans), kind="num",
            rubric=["Divide by the denominator", "Multiply by the numerator"],
            solution=f"${q}/{d} = {q // d}$, then $\\times {a} = {L(ans)}$.",
            misconceptions=[(fmt(sp.Rational(d, a) * q), "inverted-fraction")])
    if level == 2:
        b = rng.choice([5, 6, 8, 10]); a = rng.choice([v for v in range(1, 5) if sp.gcd(v, b) == 1])
        d = rng.choice([3, 4]); c = rng.choice([v for v in range(1, d) if sp.gcd(v, d) == 1])
        total = b * d * rng.randint(2, 6)
        left = total - sp.Rational(a, b) * total - sp.Rational(c, d) * (total - sp.Rational(a, b) * total)
        return Problem(
            prompt=(f"You have {total} €. You spend $\\frac{{{a}}}{{{b}}}$ of it on rent, then $\\frac{{{c}}}{{{d}}}$ "
                    "of what remains on food. How much money is left?"),
            answer=fmt(left), kind="num",
            rubric=["Compute the rent", "Compute what remains", "Take the second fraction of the remainder", "Subtract"],
            solution=(f"Rent ${L(sp.Rational(a, b) * total)}$, remaining ${L(total - sp.Rational(a, b) * total)}$; "
                      f"food ${L(sp.Rational(c, d) * (total - sp.Rational(a, b) * total))}$; left ${L(left)}$."),
            misconceptions=[(fmt(total - sp.Rational(a, b) * total - sp.Rational(c, d) * total), "fraction-of-wrong-whole")])
    a, b = rng.randint(1, 5), rng.randint(2, 7)
    c, d = rng.randint(1, 5), rng.randint(2, 7)
    expr = (sp.Rational(a, b) + sp.Rational(c, d)) / (1 - sp.Rational(a, b) * sp.Rational(c, d))
    if sp.Rational(a, b) * sp.Rational(c, d) == 1:
        a += 1
        expr = (sp.Rational(a, b) + sp.Rational(c, d)) / (1 - sp.Rational(a, b) * sp.Rational(c, d))
    return Problem(
        prompt=(f"Simplify the complex fraction $\\dfrac{{\\frac{{{a}}}{{{b}}} + \\frac{{{c}}}{{{d}}}}}"
                f"{{1 - \\frac{{{a}}}{{{b}}}\\cdot\\frac{{{c}}}{{{d}}}}}$ to a single fraction."),
        answer=fmt(expr), kind="num",
        rubric=["Simplify the numerator", "Simplify the denominator", "Divide = multiply by the reciprocal", "Reduce"],
        solution=f"$= {L(sp.Rational(a, b) + sp.Rational(c, d))} \\div {L(1 - sp.Rational(a * c, b * d))} = {L(expr)}$.")


@gen("found.fractions", name="decimals_percent")
def decimals_percent(rng, level):
    if level == 1:
        d = rng.choice([4, 5, 8, 20, 25]); a = rng.randint(1, d - 1)
        while sp.gcd(a, d) != 1:
            a = rng.randint(1, d - 1)
        return Problem(
            prompt=f"Write $\\dfrac{{{a}}}{{{d}}}$ as a percentage (number only, e.g. 37.5 for 37.5%).",
            answer=fmt(sp.Rational(100 * a, d)), kind="num",
            rubric=["Divide numerator by denominator", "Multiply by 100"],
            solution=f"${a}/{d} = {float(sp.Rational(a, d))}$, i.e. ${float(sp.Rational(100 * a, d))}\\%$.")
    if level == 2:
        p1, p2 = rng.choice([10, 20, 25, 50]), rng.choice([10, 20, 25, 40])
        net = (1 + sp.Rational(p1, 100)) * (1 - sp.Rational(p2, 100))
        return Problem(
            prompt=(f"A price rises by {p1}% and then falls by {p2}%. Give the overall change as a percentage "
                    "of the original price (negative if it went down)."),
            answer=fmt((net - 1) * 100), kind="num",
            rubric=["Write each change as a multiplier", "Multiply the multipliers", "Convert back to a percent change"],
            solution=f"$(1 + {p1}/100)(1 - {p2}/100) = {L(net)}$, change $= {L((net - 1) * 100)}\\%$.",
            misconceptions=[(str(p1 - p2), "added-percentages")])
    num = rng.randint(1, 9); rep = rng.randint(1, 9)
    while rep == 0:
        rep = rng.randint(1, 9)
    val = sp.Rational(num, 10) + sp.Rational(rep, 90)
    return Problem(
        prompt=f"Write the repeating decimal $0.{num}\\overline{{{rep}}} = 0.{num}{rep}{rep}{rep}\\ldots$ as a fraction in lowest terms.",
        answer=fmt(val), kind="num",
        rubric=["Let x be the number", "Multiply by powers of 10 to line up the repeating part", "Subtract and solve for x", "Reduce"],
        solution=f"$100x - 10x = {num}{rep}.\\overline{{{rep}}} - {num}.\\overline{{{rep}}}$, so $90x = {10 * num + rep - num}$, $x = {L(val)}$.")


# =========================================================================== exponents
@gen("found.exponents", name="evaluate_powers")
def evaluate_powers(rng, level):
    if level == 1:
        a, m = rng.choice([2, 3, 5, 10]), rng.randint(2, 4)
        return Problem(
            prompt=f"Evaluate $\\left(\\tfrac{{1}}{{{a}}}\\right)^{{-{m}}}$.",
            answer=str(a ** m), kind="num",
            rubric=["Negative exponent flips the fraction", "Raise to the power"],
            solution=f"$= {a}^{{{m}}} = {a ** m}$.",
            misconceptions=[(fmt(sp.Rational(1, a ** m)), "ignored-negative-exponent")])
    if level == 2:
        b = rng.choice([2, 3]); p, q, r = rng.randint(2, 5), rng.randint(2, 4), rng.randint(1, 4)
        big = b ** 2
        expo = p + 2 * q - r
        return Problem(
            prompt=f"Write $\\dfrac{{{b}^{{{p}}}\\cdot {big}^{{{q}}}}}{{{b}^{{{r}}}}}$ as a single power of ${b}$ and give the exponent.",
            answer=str(expo), kind="num",
            rubric=[f"Rewrite {big} as {b}^2", "Use (a^m)^n = a^(mn)", "Add / subtract exponents"],
            solution=f"${big}^{{{q}}} = {b}^{{{2 * q}}}$, so the exponent is ${p} + {2 * q} - {r} = {expo}$.",
            misconceptions=[(str(p + q - r), "forgot-to-rewrite-base")])
    c = rng.choice([2, 3, 4, 5]); e1 = rng.randint(3, 6)
    return Problem(
        prompt=(f"A GPU does ${c}\\times 10^{{{e1 + 9}}}$ operations per second. How many seconds does it need for "
                f"${c * 6}\\times 10^{{{e1 + 12}}}$ operations?"),
        answer=str(6 * 1000), kind="num",
        rubric=["Divide the mantissas", "Subtract the exponents", "Write the result as a plain number"],
        solution=f"$\\frac{{{c * 6}}}{{{c}}}\\times10^{{{e1 + 12}-{e1 + 9}}} = 6\\times10^3 = 6000$ s.",
        misconceptions=[("6000000", "added-exponents")])


@gen("found.exponents", name="exponent_equations")
def exponent_equations(rng, level):
    if level == 1:
        b = rng.choice([2, 3, 5]); m = rng.randint(2, 5)
        return Problem(
            prompt=f"Solve for $x$: ${b}^x = {b ** m}$.",
            answer=str(m), kind="num",
            rubric=[f"Write {b ** m} as a power of {b}", "Equate exponents"],
            solution=f"${b ** m} = {b}^{{{m}}}$, so $x = {m}$.")
    if level == 2:
        b = rng.choice([2, 3]); a1, a2 = rng.randint(1, 3), rng.randint(1, 3); c = rng.randint(-3, 4)
        # b^(a1 x) = (b^2)^(x + c)  ->  a1 x = 2x + 2c
        if a1 == 2:
            a1 = 3
        sol = sp.Rational(2 * c, a1 - 2)
        return Problem(
            prompt=f"Solve for $x$: ${b}^{{{a1}x}} = {b ** 2}^{{x {signed(c)}}}$.",
            answer=fmt(sol), kind="num",
            rubric=[f"Rewrite {b ** 2} as {b}^2", "Equate exponents", "Solve the linear equation"],
            solution=f"${a1}x = 2(x {signed(c)})$ so $x = {L(sol)}$.")
    r = rng.choice([2, 3]); n0 = rng.randint(2, 9) * 100; hours = rng.choice([2, 3, 4])
    k_ = rng.randint(3, 6)
    return Problem(
        prompt=(f"A bacteria culture of {n0} cells multiplies by {r} every {hours} hours. "
                f"After how many hours are there ${n0}\\cdot {r}^{{{k_}}}$ cells? Model it as $N(t) = N_0\\, {r}^{{t/{hours}}}$."),
        answer=str(k_ * hours), kind="num",
        rubric=["Write N(t) = N0 r^(t/h)", "Cancel N0", "Equate exponents t/h = k", "Solve for t"],
        solution=f"${r}^{{t/{hours}}} = {r}^{{{k_}}} \\Rightarrow t = {k_ * hours}$ h.",
        misconceptions=[(str(k_), "forgot-period")])


# =========================================================================== linear equations
@gen("found.linear_eq", name="linear_with_fractions")
def linear_with_fractions(rng, level):
    if level == 1:
        a, b, sol = rng.choice([2, 3, 4, 5]), rng.randint(-9, 9), rng.randint(-6, 6)
        c = a * sol + b * a  # (x + b) * ... keep integer
        return Problem(
            prompt=f"Clear the fraction, then find $x$: $\\dfrac{{x {signed(b * a)}}}{{{a}}} = {sol + b}$.",
            answer=str(sol), kind="num",
            rubric=[f"Multiply both sides by {a}", "Isolate x"],
            solution=f"$x {signed(b * a)} = {a * (sol + b)}$, so $x = {sol}$.")
    if level == 2:
        a, b, c, d = rng.randint(2, 6), rng.randint(2, 6), rng.randint(-5, 5), rng.randint(-5, 5)
        sol = sp.solve(sp.Rational(1, a) * (x + c) - sp.Rational(1, b) * (x - d) - 1, x)
        if not sol:
            a += 1
            sol = sp.solve(sp.Rational(1, a) * (x + c) - sp.Rational(1, b) * (x - d) - 1, x)
        return Problem(
            prompt=f"Solve for $x$: $\\dfrac{{x {signed(c)}}}{{{a}}} - \\dfrac{{x {signed(-d)}}}{{{b}}} = 1$.",
            answer=fmt(sol[0]), kind="num",
            rubric=[f"Multiply through by the LCD {sp.ilcm(a, b)}", "Expand, watching the minus sign", "Collect x terms and solve"],
            solution=f"$x = {L(sol[0])}$.",
            misconceptions=[(fmt(sp.solve(sp.Rational(1, a) * (x + c) - sp.Rational(1, b) * (x + d) - 1, x)[0]) if sp.solve(sp.Rational(1, a) * (x + c) - sp.Rational(1, b) * (x + d) - 1, x) else "999999", "minus-sign-distribution")])
    p1, p2 = rng.choice([10, 20, 30]), rng.choice([50, 60, 80])
    target, total = rng.choice([25, 40, 45]), rng.choice([10, 20, 30])
    while not (p1 < target < p2):
        target = rng.choice([25, 40, 45])
    a_amt = sp.Rational(total * (p2 - target), p2 - p1)
    return Problem(
        prompt=(f"How many litres of a {p1}% acid solution must be mixed with a {p2}% solution to get "
                f"{total} litres of a {target}% solution? (Give the litres of the {p1}% solution.)"),
        answer=fmt(a_amt), kind="num",
        rubric=["Let a = litres of the first solution", f"Acid balance: {p1}a + {p2}({total} - a) = {target}·{total}", "Solve for a"],
        solution=f"${p1}a + {p2}({total}-a) = {target * total} \\Rightarrow a = {L(a_amt)}$ L.")


@gen("found.linear_eq", name="lines")
def lines(rng, level):
    if level == 1:
        x1, y1 = rng.randint(-5, 5), rng.randint(-5, 5)
        x2, y2 = x1 + nz(rng, -4, 4), rng.randint(-5, 5)
        m = sp.Rational(y2 - y1, x2 - x1)
        return Problem(
            prompt=f"Find the slope of the line through $({x1}, {y1})$ and $({x2}, {y2})$.",
            answer=fmt(m), kind="num",
            rubric=["Rise over run: (y2 - y1)/(x2 - x1)"],
            solution=f"$m = \\frac{{{y2} - ({y1})}}{{{x2} - ({x1})}} = {L(m)}$.",
            misconceptions=[(fmt(sp.Rational(x2 - x1, y2 - y1)) if y2 != y1 else "999999", "run-over-rise")])
    if level == 2:
        x1, y1 = rng.randint(-5, 5), rng.randint(-5, 5)
        x2, y2 = x1 + nz(rng, -4, 4), rng.randint(-5, 5)
        m = sp.Rational(y2 - y1, x2 - x1)
        b = y1 - m * x1
        return Problem(
            prompt=f"Find the equation $y = mx + b$ of the line through $({x1}, {y1})$ and $({x2}, {y2})$. Type the right-hand side.",
            answer=fmt(m * x + b), kind="expr", answer_hint="an expression in x, e.g. 2/3*x - 1",
            rubric=["Compute the slope", "Substitute a point to find b", "Check with the other point"],
            solution=f"$m = {L(m)}$, $b = {L(b)}$, so $y = {L(m * x + b)}$.")
    m1, b1 = nz(rng, -3, 3), rng.randint(-6, 6)
    m2 = nz(rng, -3, 3)
    while m2 == m1:
        m2 = nz(rng, -3, 3)
    xs = rng.randint(-4, 4)
    b2 = (m1 - m2) * xs + b1
    ys = m1 * xs + b1
    return Problem(
        prompt=(f"Where do the lines $y = {L(m1 * x + b1)}$ and $y = {L(m2 * x + b2)}$ intersect? "
                "Also decide on paper whether a third line with slope equal to the first can ever meet it. Give the point $x, y$."),
        answer=[str(xs), str(ys)], kind="list", answer_hint="x, y",
        rubric=["Set the right-hand sides equal", "Solve for x", "Substitute back for y", "Parallel lines (same slope) never meet unless identical"],
        solution=f"${L(m1 * x + b1)} = {L(m2 * x + b2)} \\Rightarrow x = {xs}$, $y = {ys}$.")


# =========================================================================== polynomials
@gen("found.polynomials", name="special_products")
def special_products(rng, level):
    if level == 1:
        a, b = rng.randint(1, 5), nz(rng, -7, 7)
        expr = sp.expand((a * x + b) ** 2)
        return Problem(
            prompt=f"Use the square-of-a-sum identity to multiply out $({L(a * x + b)})^2$.",
            answer=fmt(expr), kind="expr",
            rubric=["Use (p + q)^2 = p^2 + 2pq + q^2", "Collect terms"],
            solution=f"$= {L(expr)}$.",
            misconceptions=[(fmt(a ** 2 * x ** 2 + b ** 2), "forgot-middle-term")])
    if level == 2:
        a, b = rng.randint(1, 6), rng.randint(1, 9)
        expr = a ** 2 * x ** 2 - b ** 2
        return Problem(
            prompt=f"Factor completely: ${L(expr)}$ (difference of squares).",
            answer=fmt(sp.factor(expr)), kind="expr",
            rubric=["Recognise p^2 - q^2", "Write (p - q)(p + q)"],
            solution=f"$= {L(sp.factor(expr))}$.")
    r = rng.randint(1, 4); c = rng.randint(1, 3)
    expr = sp.expand((x - r) * (x ** 2 + r * x + r ** 2) * c)
    return Problem(
        prompt=f"Factor completely over the reals: ${L(expr)}$.",
        answer=fmt(sp.factor(expr)), kind="expr",
        rubric=["Take out the common factor", "Recognise a difference of cubes", "Check the quadratic factor has no real roots"],
        solution=f"$= {L(sp.factor(expr))}$ since $a^3 - b^3 = (a-b)(a^2+ab+b^2)$.")


@gen("found.polynomials", name="poly_eval_roots")
def poly_eval_roots(rng, level):
    if level == 1:
        coeffs = [nz(rng, -4, 4), rng.randint(-5, 5), rng.randint(-5, 5), rng.randint(-5, 5)]
        p = sum(c * x ** (3 - i) for i, c in enumerate(coeffs))
        x0 = rng.randint(-2, 3)
        return Problem(
            prompt=f"Evaluate $p(x) = {L(p)}$ at $x = {x0}$.",
            answer=str(p.subs(x, x0)), kind="num",
            rubric=["Substitute carefully (watch negative signs in powers)", "Add up"],
            solution=f"$p({x0}) = {p.subs(x, x0)}$.")
    if level == 2:
        r1, r2, r3 = rng.sample([-3, -2, -1, 1, 2, 3, 4], 3)
        p = sp.expand((x - r1) * (x - r2) * (x - r3))
        return Problem(
            prompt=(f"One root of $p(x) = {L(p)}$ is $x = {r1}$. Find all roots "
                    "(divide out the known root, then solve the quadratic)."),
            answer=[str(r1), str(r2), str(r3)], kind="set",
            rubric=[f"Divide by (x - ({r1}))", "Factor the quadratic quotient", "List all three roots"],
            solution=f"$p(x) = {L(sp.factor(p))}$.")
    r1, r2 = rng.sample([-3, -2, -1, 1, 2, 3], 2); c = rng.randint(-5, 5)
    # find k so that (x - r1) divides x^3 + k x + c ... choose polynomial x^3 + a x^2 + k x + c
    a = rng.randint(-3, 3)
    kk = sp.solve((x ** 3 + a * x ** 2 + sp.Symbol("kk") * x + c).subs(x, r1), sp.Symbol("kk"))[0]
    return Problem(
        prompt=f"For which value of $k$ is $(x {signed(-r1)})$ a factor of $x^3 {signed(a, 'x^2')} + kx {signed(c)}$?",
        answer=fmt(kk), kind="num",
        rubric=[f"Factor theorem: p({r1}) = 0", "Substitute and solve for k"],
        solution=f"$p({r1}) = 0 \\Rightarrow k = {L(kk)}$.")


# =========================================================================== quadratics
@gen("found.quadratics", name="quadratic_models")
def quadratic_models(rng, level):
    if level == 1:
        a = rng.randint(1, 5); c = a * rng.choice([1, 4, 9, 16, 25])
        r = sp.sqrt(sp.Rational(c, a))
        return Problem(
            prompt=f"Solve ${a}x^2 - {c} = 0$.",
            answer=[fmt(r), fmt(-r)], kind="set",
            rubric=["Isolate x^2", "Take ± square root"],
            solution=f"$x^2 = {L(sp.Rational(c, a))}$, $x = \\pm {L(r)}$.",
            misconceptions=[(fmt(r), "forgot-negative-root")])
    if level == 2:
        v0 = rng.choice([10, 15, 20, 25, 30]); g = 5
        tmax = sp.Rational(v0, 2 * g); hmax = v0 * tmax - g * tmax ** 2
        return Problem(
            prompt=(f"A ball is thrown upward; its height is $h(t) = {v0}t - {g}t^2$ metres. "
                    "What is the maximum height?"),
            answer=fmt(hmax), kind="num",
            rubric=["The vertex is at t = -b/(2a)", "Evaluate h at the vertex"],
            solution=f"$t^* = {L(tmax)}$, $h(t^*) = {L(hmax)}$ m.")
    p = rng.choice([20, 24, 28, 32, 40])
    return Problem(
        prompt=(f"A rectangle has perimeter {p} m. Write its area as a quadratic in the width $w$ and find the "
                "largest possible area."),
        answer=fmt(sp.Rational(p, 4) ** 2), kind="num",
        rubric=[f"Length = {p}/2 - w", "Area A(w) = w(p/2 - w)", "Vertex at w = p/4", "Evaluate the area"],
        solution=f"$A = w({p // 2} - w)$, max at $w = {L(sp.Rational(p, 4))}$: $A = {L(sp.Rational(p, 4) ** 2)}$ m².")


@gen("found.quadratics", name="discriminant")
def discriminant(rng, level):
    if level == 1:
        a, b, c = nz(rng, -4, 4), rng.randint(-8, 8), rng.randint(-8, 8)
        d = b * b - 4 * a * c
        return Problem(
            prompt=f"Compute the discriminant of ${L(a * x ** 2 + b * x + c)} = 0$ and decide how many real roots it has. Type the discriminant.",
            answer=str(d), kind="num",
            rubric=["Δ = b² - 4ac", ">0 two roots, =0 one, <0 none"],
            solution=f"$\\Delta = {b}^2 - 4({a})({c}) = {d}$: {'two real roots' if d > 0 else 'one real root' if d == 0 else 'no real roots'}.",
            misconceptions=[(str(b * b + 4 * a * c), "sign-of-4ac")])
    if level == 2:
        b = rng.choice([2, 4, 6, 8, 10]); a = rng.randint(1, 3)
        kk = sp.Rational(b * b, 4 * a)
        return Problem(
            prompt=f"For which $k$ does ${a}x^2 + {b}x + k = 0$ have exactly one (repeated) real root?",
            answer=fmt(kk), kind="num",
            rubric=["Exactly one root means Δ = 0", "Solve b² - 4ak = 0 for k"],
            solution=f"${b}^2 - 4\\cdot{a}k = 0 \\Rightarrow k = {L(kk)}$.")
    s, p = rng.randint(-6, 6), rng.randint(-9, 9)
    r = sp.symbols("r")
    return Problem(
        prompt=(f"The roots $r_1, r_2$ of a quadratic satisfy $r_1 + r_2 = {s}$ and $r_1 r_2 = {p}$. "
                "Without finding the roots, compute $r_1^2 + r_2^2$."),
        answer=str(s * s - 2 * p), kind="num",
        rubric=["Use (r1 + r2)^2 = r1^2 + 2 r1 r2 + r2^2", "Rearrange for r1^2 + r2^2"],
        solution=f"$r_1^2 + r_2^2 = (r_1+r_2)^2 - 2r_1r_2 = {s * s} - {2 * p} = {s * s - 2 * p}$.",
        misconceptions=[(str(s * s), "forgot-cross-term")])


# =========================================================================== functions
@gen("found.functions", name="domain_range")
def domain_range(rng, level):
    if level == 1:
        a = rng.randint(-6, 6)
        return Problem(
            prompt=f"Which value of $x$ must be excluded from the domain of $f(x) = \\dfrac{{3}}{{x {signed(-a)}}}$?",
            answer=str(a), kind="num",
            rubric=["The denominator cannot be zero", "Solve denominator = 0"],
            solution=f"$x {signed(-a)} = 0 \\Rightarrow x = {a}$.",
            misconceptions=[(str(-a), "sign-of-excluded-value")])
    if level == 2:
        a = nz(rng, 1, 4); b = rng.randint(-8, 8)
        bound = sp.Rational(-b, a)
        return Problem(
            prompt=f"The domain of $f(x) = \\sqrt{{{L(a * x + b)}}}$ is $x \\ge c$. Find $c$.",
            answer=fmt(bound), kind="num",
            rubric=["The expression under the root must be ≥ 0", "Solve the inequality"],
            solution=f"${L(a * x + b)} \\ge 0 \\Rightarrow x \\ge {L(bound)}$.")
    a, b = rng.randint(1, 3), rng.randint(-4, 4)
    return Problem(
        prompt=f"Is $f(x) = x^{{{2 * a + 1}}} {signed(b)}x$ even, odd, or neither? (A) even (B) odd (C) neither",
        answer="B", kind="choice",
        rubric=["Compute f(-x)", "Compare with f(x) and -f(x)"],
        solution=f"$f(-x) = -x^{{{2 * a + 1}}} {signed(-b)}x = -f(x)$: odd (only odd powers).")


@gen("found.functions", name="transformations")
def transformations(rng, level):
    if level == 1:
        h, kk = rng.randint(-5, 5), rng.randint(-5, 5)
        return Problem(
            prompt=f"The graph of $y = x^2$ is shifted so its vertex moves to $({h}, {kk})$. Write the new function (in $x$).",
            answer=fmt((x - h) ** 2 + kk), kind="expr",
            rubric=["Horizontal shift: replace x by (x - h)", "Vertical shift: add k"],
            solution=f"$y = {L((x - h) ** 2 + kk)}$.",
            misconceptions=[(fmt((x + h) ** 2 + kk), "horizontal-shift-sign")] if h != 0 else [])
    if level == 2:
        a, b = rng.randint(2, 4), rng.randint(-5, 5)
        f = sp.sqrt(x)
        g = a * f.subs(x, x - b)
        x0 = b + rng.choice([1, 4, 9])
        return Problem(
            prompt=f"Let $g(x) = {a}\\sqrt{{x {signed(-b)}}}$. Compute $g({x0})$, then describe on paper how $g$ is built from $\\sqrt{{x}}$.",
            answer=fmt(g.subs(x, x0)), kind="num",
            rubric=[f"Shift right by {b}", f"Stretch vertically by {a}", "Evaluate"],
            solution=f"$g({x0}) = {a}\\sqrt{{{x0 - b}}} = {L(g.subs(x, x0))}$.")
    a, b = nz(rng, -3, 3), rng.randint(-5, 5)
    c, d = nz(rng, -3, 3), rng.randint(-5, 5)
    f, g = a * x + b, c * x + d
    comp = sp.expand(f.subs(x, g))
    return Problem(
        prompt=f"Let $f(x) = {L(f)}$ and $g(x) = {L(g)}$. Find $(f\\circ g)(x) - (g\\circ f)(x)$ as an expression in $x$.",
        answer=fmt(sp.expand(comp - g.subs(x, f))), kind="expr",
        rubric=["Compute f(g(x))", "Compute g(f(x))", "Subtract and simplify"],
        solution=f"$f(g(x)) = {L(comp)}$, $g(f(x)) = {L(sp.expand(g.subs(x, f)))}$, difference $= {L(sp.expand(comp - g.subs(x, f)))}$.")


# =========================================================================== logarithms
@gen("found.logs", name="log_rules")
def log_rules(rng, level):
    if level == 1:
        b = rng.choice([2, 3, 10]); m, nn = rng.randint(1, 4), rng.randint(1, 4)
        return Problem(
            prompt=f"Simplify $\\log_{{{b}}}({b ** m}) + \\log_{{{b}}}({b ** nn})$ to a number.",
            answer=str(m + nn), kind="num",
            rubric=["log_b(b^k) = k", "Add"],
            solution=f"${m} + {nn} = {m + nn}$.")
    if level == 2:
        b = rng.choice([2, 3]); p, q = rng.randint(2, 5), rng.randint(1, 3)
        val = sp.Rational(p, 1) + q * 2 - 1
        return Problem(
            prompt=(f"Given $\\log_{{{b}}} A = {p}$, compute $\\log_{{{b}}}\\!\\left(\\dfrac{{A\\cdot {b ** (2 * q)}}}{{{b}}}\\right)$."),
            answer=fmt(val), kind="num",
            rubric=["Split the product/quotient into sums/differences", f"log_b({b ** (2 * q)}) = {2 * q}", "Add up"],
            solution=f"$= {p} + {2 * q} - 1 = {L(val)}$.")
    a = rng.randint(2, 6)
    sol = sp.solve(sp.log(x) + sp.log(x - a + 1) - sp.log(a * (a + 0)), x)
    # x(x - a + 1) = a  -> pick a so a root is nice: use x = a: a*1 = a
    return Problem(
        prompt=f"Solve $\\ln x + \\ln(x - {a - 1}) = \\ln {a}$. (Check which solutions are allowed.)",
        answer=str(a), kind="num",
        rubric=["Combine: ln(x(x - c)) = ln a", "Solve the quadratic", "Reject solutions outside the domain (x > c)"],
        solution=f"$x(x-{a - 1}) = {a} \\Rightarrow x^2 - {a - 1}x - {a} = 0 \\Rightarrow x = {a}$ or $x = -1$; reject $-1$.",
        misconceptions=[(f"{a}, -1", "kept-invalid-root"), ("-1", "kept-invalid-root")])


@gen("found.logs", name="log_applications")
def log_applications(rng, level):
    if level == 1:
        nbits = rng.choice([8, 16, 32, 64, 128, 256, 1024])
        return Problem(
            prompt=f"How many bits are needed to label ${nbits}$ equally likely items? (Compute $\\log_2 {nbits}$.)",
            answer=str(sp.log(nbits, 2)), kind="num",
            rubric=[f"Write {nbits} as a power of 2", "Read off the exponent"],
            solution=f"$\\log_2 {nbits} = {sp.log(nbits, 2)}$.")
    if level == 2:
        db1 = rng.choice([30, 40, 50, 60]); ratio = rng.choice([10, 100, 1000])
        return Problem(
            prompt=f"Loudness in decibels is $L = 10\\log_{{10}}(I/I_0)$. A sound is {ratio} times more intense than a {db1} dB sound. How loud is it in dB?",
            answer=str(db1 + 10 * int(sp.log(ratio, 10))), kind="num",
            rubric=["Write the new intensity as ratio · I", "Use log(ab) = log a + log b", "Add"],
            solution=f"$L = {db1} + 10\\log_{{10}}{ratio} = {db1 + 10 * int(sp.log(ratio, 10))}$ dB.",
            misconceptions=[(str(db1 * ratio), "multiplied-decibels")])
    r = rng.choice([5, 8, 10, 12]); mult = rng.choice([2, 3, 4])
    T = sp.log(mult) / sp.log(1 + sp.Rational(r, 100))
    return Problem(
        prompt=(f"An investment grows {r}% per year (compounded yearly). After how many years does it become {mult}× "
                "its starting value? Give the exact form using ln, or a decimal to 2 places."),
        answer=fmt(T), kind="num", tol=1e-3,
        rubric=[f"Set (1 + {r}/100)^t = {mult}", "Take ln of both sides", "Solve for t"],
        solution=f"$t = \\dfrac{{\\ln {mult}}}{{\\ln {L(1 + sp.Rational(r, 100))}}} \\approx {float(T):.2f}$ years.",
        answer_hint="e.g. ln(2)/ln(1.08) or 9.01")


# =========================================================================== trig
@gen("found.trig", name="right_triangles")
def right_triangles(rng, level):
    if level == 1:
        a, b, c = rng.choice([(3, 4, 5), (5, 12, 13), (8, 15, 17), (6, 8, 10), (7, 24, 25)])
        which = rng.choice(["sin", "cos", "tan"])
        val = {"sin": sp.Rational(a, c), "cos": sp.Rational(b, c), "tan": sp.Rational(a, b)}[which]
        return Problem(
            prompt=(f"A right triangle has legs {a} (opposite angle $\\theta$) and {b} (adjacent), hypotenuse {c}. "
                    f"Find $\\{which}\\theta$."),
            answer=fmt(val), kind="num",
            rubric=["SOH-CAH-TOA", "Reduce the fraction"],
            solution=f"$\\{which}\\theta = {L(val)}$.")
    if level == 2:
        sval = rng.choice([sp.Rational(3, 5), sp.Rational(5, 13), sp.Rational(8, 17), sp.Rational(7, 25)])
        cval = -sp.sqrt(1 - sval ** 2)
        return Problem(
            prompt=(f"$\\sin\\theta = {L(sval)}$ and $\\theta$ is in the second quadrant ($\\pi/2 < \\theta < \\pi$). "
                    "Find $\\cos\\theta$."),
            answer=fmt(cval), kind="num",
            rubric=["sin² + cos² = 1", "Choose the sign from the quadrant"],
            solution=f"$\\cos\\theta = -\\sqrt{{1 - {L(sval ** 2)}}} = {L(cval)}$.",
            misconceptions=[(fmt(-cval), "wrong-quadrant-sign")])
    h = rng.choice([10, 20, 30, 50]); ang = rng.choice([30, 45, 60])
    d = h / sp.tan(sp.rad(ang))
    return Problem(
        prompt=(f"From a point on the ground, the top of a {h} m tower is seen at an elevation angle of ${ang}^\\circ$. "
                "How far is the point from the base of the tower? (Exact.)"),
        answer=fmt(sp.nsimplify(d)), kind="num",
        rubric=["Draw the right triangle", "tan(angle) = height / distance", "Solve for the distance"],
        solution=f"$d = {h}/\\tan {ang}^\\circ = {L(sp.nsimplify(d))}$ m.")


@gen("found.trig", name="trig_equations")
def trig_equations(rng, level):
    if level == 1:
        val, sols = rng.choice([("1/2", ["pi/6", "5*pi/6"]), ("sqrt(2)/2", ["pi/4", "3*pi/4"]),
                                ("sqrt(3)/2", ["pi/3", "2*pi/3"])])
        return Problem(
            prompt=f"Find all solutions of $\\sin x = {L(sp.sympify(val))}$ with $0 \\le x < 2\\pi$.",
            answer=sols, kind="set",
            rubric=["Reference angle from the unit circle", "sin is positive in quadrants I and II"],
            solution=f"$x \\in \\{{{', '.join(L(sp.sympify(s)) for s in sols)}\\}}$.",
            misconceptions=[(sols[0], "only-one-solution")])
    if level == 2:
        a = rng.choice([sp.pi / 6, sp.pi / 4, sp.pi / 3])
        val = sp.nsimplify(sp.cos(2 * a))
        return Problem(
            prompt=f"Use a double-angle formula to compute $\\cos\\left(2\\cdot {L(a)}\\right)$ from the values of $\\sin {L(a)}$ and $\\cos {L(a)}$.",
            answer=fmt(val), kind="num",
            rubric=["cos 2a = cos²a - sin²a", "Substitute the unit-circle values", "Simplify"],
            solution=f"$= {L(sp.cos(a) ** 2)} - {L(sp.sin(a) ** 2)} = {L(val)}$.")
    return Problem(
        prompt="Solve $2\\cos^2 x - \\cos x - 1 = 0$ for $0 \\le x < 2\\pi$ (treat it as a quadratic in $\\cos x$).",
        answer=["0", "2*pi/3", "4*pi/3"], kind="set",
        rubric=["Substitute u = cos x", "Factor (2u + 1)(u - 1) = 0", "Solve cos x = 1 and cos x = -1/2", "List all solutions in [0, 2π)"],
        solution="$\\cos x = 1 \\Rightarrow x = 0$; $\\cos x = -\\tfrac12 \\Rightarrow x = \\tfrac{2\\pi}{3}, \\tfrac{4\\pi}{3}$.")


# =========================================================================== sums
@gen("found.sums", name="sigma_notation")
def sigma_notation(rng, level):
    if level == 1:
        N = rng.randint(4, 7); a, b = rng.randint(1, 4), rng.randint(-3, 3)
        s = sum(a * kk + b for kk in range(1, N + 1))
        return Problem(
            prompt=f"Split the sum into two simpler sums and evaluate: $\\displaystyle\\sum_{{k=1}}^{{{N}}} ({L(a * k + b)})$.",
            answer=str(s), kind="num",
            rubric=["Split into a·Σk + b·N (or list the terms)", "Evaluate"],
            solution=f"$= {a}\\cdot\\frac{{{N}\\cdot{N + 1}}}{{2}} {signed(b * N)} = {s}$.",
            misconceptions=[(str(s - b * N + b), "constant-counted-once")])
    if level == 2:
        a1, d, N = rng.randint(1, 9), rng.randint(2, 6), rng.randint(10, 30)
        s = N * (2 * a1 + (N - 1) * d) // 2
        return Problem(
            prompt=f"An arithmetic sequence starts {a1}, {a1 + d}, {a1 + 2 * d}, … Find the sum of its first {N} terms.",
            answer=str(s), kind="num",
            rubric=["Identify a1 and d", "Last term a_N = a1 + (N - 1)d", "Sum = N(a1 + a_N)/2"],
            solution=f"$a_{{{N}}} = {a1 + (N - 1) * d}$, sum $= {N}({a1} + {a1 + (N - 1) * d})/2 = {s}$.")
    p = rng.randint(2, 5)
    s = sp.summation(sp.Rational(1, 1) / (k * (k + 1)), (k, 1, p * 10))
    return Problem(
        prompt=(f"Compute $\\displaystyle\\sum_{{k=1}}^{{{p * 10}}} \\frac{{1}}{{k(k+1)}}$ exactly "
                "(hint: partial fractions make it telescope)."),
        answer=fmt(s), kind="num",
        rubric=["Write 1/(k(k+1)) = 1/k - 1/(k+1)", "Cancel the middle terms", "Evaluate 1 - 1/(N+1)"],
        solution=f"$= 1 - \\frac{{1}}{{{p * 10 + 1}}} = {L(s)}$.")


@gen("found.sums", name="series_applications")
def series_applications(rng, level):
    if level == 1:
        a, r, N = rng.randint(1, 3), rng.choice([2, 3]), rng.randint(4, 6)
        return Problem(
            prompt=f"A sequence starts at {a} and is multiplied by {r} each step. What is its term number {N} (the first term is term 1)?",
            answer=str(a * r ** (N - 1)), kind="num",
            rubric=["Geometric term a·r^(n-1)", "Evaluate"],
            solution=f"$a_{{{N}}} = {a}\\cdot{r}^{{{N - 1}}} = {a * r ** (N - 1)}$.",
            misconceptions=[(str(a * r ** N), "off-by-one-exponent")])
    if level == 2:
        dep, rate, yrs = rng.choice([100, 200, 500]), rng.choice([sp.Rational(1, 10), sp.Rational(1, 20)]), rng.randint(3, 5)
        total = sum(dep * (1 + rate) ** j for j in range(1, yrs + 1))
        return Problem(
            prompt=(f"You deposit {dep} € at the start of each year for {yrs} years at {int(rate * 100)}% interest "
                    "(compounded yearly). How much is in the account at the end of the last year? (Exact.)"),
            answer=fmt(total), kind="num",
            rubric=["The k-th deposit grows for a different number of years", "Write the sum as a geometric series", "Use the geometric sum formula"],
            solution=f"$\\sum_{{j=1}}^{{{yrs}}} {dep}({L(1 + rate)})^j = {L(total)}$ €.")
    h, r = rng.choice([4, 6, 8, 10]), rng.choice([sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(3, 4)])
    dist = h + 2 * h * r / (1 - r)
    return Problem(
        prompt=(f"A ball is dropped from {h} m and each bounce reaches ${L(r)}$ of the previous height. "
                "What total vertical distance does it travel before coming to rest?"),
        answer=fmt(dist), kind="num",
        rubric=["First drop h", "Each bounce goes up AND down: 2·h·r^k", "Sum the infinite geometric series"],
        solution=f"$D = {h} + 2\\cdot\\frac{{{h}\\cdot{L(r)}}}{{1 - {L(r)}}} = {L(dist)}$ m.",
        misconceptions=[(fmt(h / (1 - r)), "counted-bounces-once")])


# =========================================================================== induction
@gen("found.induction", name="induction_steps")
def induction_steps(rng, level):
    if level == 1:
        a = rng.randint(2, 5)
        return Problem(
            prompt=(f"We prove $\\sum_{{k=0}}^{{n}} {a}^k = \\frac{{{a}^{{n+1}}-1}}{{{a - 1}}}$ by induction. "
                    "What is the value of both sides in the base case $n = 0$?"),
            answer="1", kind="num",
            rubric=["Left side with n = 0 has one term", "Right side: (a - 1)/(a - 1)", "Both equal"],
            solution=f"Left: ${a}^0 = 1$. Right: $\\frac{{{a} - 1}}{{{a - 1}}} = 1$.")
    if level == 2:
        c = rng.choice([2, 3])
        # claim: n! > c^n for n >= N0
        N0 = next(m for m in range(1, 20) if sp.factorial(m) > c ** m)
        return Problem(
            prompt=(f"Find the smallest $n \\ge 1$ with $n! > {c}^n$ (the base case for proving it for all larger $n$). "
                    f"On paper, also do the inductive step: $(n+1)! = (n+1)\\,n! > (n+1){c}^n \\ge {c}^{{n+1}}$."),
            answer=str(N0), kind="num",
            rubric=["Tabulate n! and c^n for small n", "Base case", "Inductive step uses n + 1 ≥ c"],
            solution=f"$n = {N0}$: ${sp.factorial(N0)} > {c ** N0}$, and it fails for $n = {N0 - 1}$.")
    m = rng.choice([3, 5, 6])
    return Problem(
        prompt=(f"Prove by induction that $n^3 - n$ is divisible by {m if m != 5 else 6} for all $n \\ge 1$. "
                "In the inductive step, $(n+1)^3 - (n+1) = (n^3 - n) + Q(n)$. Find $Q(n)$."),
        answer="3*n^2 + 3*n", kind="expr",
        rubric=["Expand (n+1)^3 - (n+1)", "Group n^3 - n", "Show Q(n) = 3n(n+1) is divisible by 6"],
        solution="$(n+1)^3-(n+1) = n^3-n + 3n^2+3n$, and $3n(n+1)$ is a multiple of 6.")


@gen("found.induction", name="recurrences")
def recurrences(rng, level):
    if level == 1:
        a0, d = rng.randint(1, 5), rng.randint(2, 5)
        N = rng.randint(4, 7)
        return Problem(
            prompt=f"A sequence has $a_0 = {a0}$ and $a_{{n+1}} = a_n + {d}$. Compute $a_{{{N}}}$.",
            answer=str(a0 + N * d), kind="num",
            rubric=["List terms or spot a_n = a0 + n d", "Evaluate"],
            solution=f"$a_n = {a0} + {d}n$, so $a_{{{N}}} = {a0 + N * d}$.")
    if level == 2:
        r, c = rng.choice([2, 3]), rng.randint(1, 4)
        N = rng.randint(3, 5)
        a = [1]
        for _ in range(N):
            a.append(r * a[-1] + c)
        return Problem(
            prompt=f"Let $a_0 = 1$ and $a_{{n+1}} = {r}a_n + {c}$. Compute $a_{{{N}}}$.",
            answer=str(a[N]), kind="num",
            rubric=["Apply the recurrence step by step", "Keep the arithmetic tidy"],
            solution=", ".join(f"$a_{{{i}}} = {v}$" for i, v in enumerate(a)) + ".")
    r, c = rng.choice([2, 3]), rng.randint(1, 4)
    closed = (1 + sp.Rational(c, r - 1)) * r ** n - sp.Rational(c, r - 1)
    return Problem(
        prompt=(f"Guess and prove by induction a closed form for $a_0 = 1$, $a_{{n+1}} = {r}a_n + {c}$. "
                "Type $a_n$ as an expression in $n$."),
        answer=fmt(closed), kind="expr",
        rubric=["Compute a few terms", "Guess a_n = A r^n + B (fixed point B = -c/(r - 1))", "Base case", "Inductive step"],
        solution=f"$a_n = {L(sp.simplify(closed))}$.")


# =========================================================================== complex numbers
@gen("found.complex", name="complex_forms")
def complex_forms(rng, level):
    I = sp.I
    if level == 1:
        a, b = rng.choice([(3, 4), (5, 12), (8, 15), (6, 8), (1, 1), (2, 2)])
        a *= rng.choice([1, -1]); b *= rng.choice([1, -1])
        return Problem(
            prompt=f"Find the modulus $|z|$ of $z = {L(a + b * I)}$.",
            answer=fmt(sp.sqrt(a * a + b * b)), kind="num",
            rubric=["|a + bi| = sqrt(a² + b²)"],
            solution=f"$|z| = \\sqrt{{{a * a} + {b * b}}} = {L(sp.sqrt(a * a + b * b))}$.",
            misconceptions=[(str(abs(a) + abs(b)), "added-components")])
    if level == 2:
        a, b = nz(rng, -5, 5), nz(rng, -5, 5)
        z = a + b * I
        return Problem(
            prompt=f"For $z = {L(z)}$ compute $z\\,\\bar z$ and $z + \\bar z$. Type $z\\bar z + (z + \\bar z)$.",
            answer=str(a * a + b * b + 2 * a), kind="num",
            rubric=["Conjugate flips the sign of the imaginary part", "z z̄ = |z|²", "z + z̄ = 2 Re z"],
            solution=f"$z\\bar z = {a * a + b * b}$, $z + \\bar z = {2 * a}$, total ${a * a + b * b + 2 * a}$.")
    nn = rng.choice([3, 4, 6])
    roots = [sp.nsimplify(sp.exp(2 * sp.pi * I * kk / nn).expand(complex=True)) for kk in range(nn)]
    return Problem(
        prompt=f"Find all solutions of $z^{{{nn}}} = 1$ (the {nn}th roots of unity) in the form $a + bi$.",
        answer=[fmt(r) for r in roots], kind="set", answer_hint="comma-separated, e.g. 1, -1/2 + sqrt(3)/2*i, ...",
        rubric=["Write 1 = e^{2πik}", f"z = e^(2πik/{nn}) for k = 0..{nn - 1}", "Convert each with cos + i sin"],
        solution="$z_k = e^{2\\pi i k/" + str(nn) + "}$: " + ", ".join(f"${L(r)}$" for r in roots) + ".")


@gen("found.complex", name="complex_equations")
def complex_equations(rng, level):
    I = sp.I
    if level == 1:
        nn = rng.randint(2, 11)
        val = I ** nn
        return Problem(
            prompt=f"Simplify $i^{{{nn}}}$.",
            answer=fmt(val), kind="expr", answer_hint="one of 1, -1, i, -i",
            rubric=["Powers of i cycle with period 4", f"{nn} mod 4"],
            solution=f"${nn} \\bmod 4 = {nn % 4}$, so $i^{{{nn}}} = {L(val)}$.")
    if level == 2:
        p, q = rng.randint(-4, 4), rng.randint(1, 5)
        b, c = -2 * p, p * p + q * q
        return Problem(
            prompt=f"Solve $x^2 {signed(b, 'x')} {signed(c)} = 0$ over the complex numbers.",
            answer=[fmt(p + q * I), fmt(p - q * I)], kind="set",
            rubric=["Discriminant is negative", "√(negative) = i·√(positive)", "Two conjugate roots"],
            solution=f"$x = {L(p + q * I)},\\ {L(p - q * I)}$.")
    r = rng.choice([1, 2]); ang = rng.choice([sp.pi / 4, sp.pi / 3, sp.pi / 6, 2 * sp.pi / 3])
    w = r * sp.exp(I * ang)
    z = sp.nsimplify(sp.expand_complex(w * sp.exp(I * sp.pi / 2)))
    return Problem(
        prompt=(f"Multiplying by $i$ rotates a point by $90^\\circ$. Let $w = {r}e^{{i{L(ang)}}}$. "
                "Compute $i\\,w$ in the form $a + bi$."),
        answer=fmt(z), kind="expr",
        rubric=["i = e^{iπ/2}", "Add the angles", "Convert to a + bi"],
        solution=f"$i w = {r}e^{{i({L(ang)} + \\pi/2)}} = {L(z)}$.")
