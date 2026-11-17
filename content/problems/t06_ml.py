"""Track 6 — Machine learning (classical supervised & unsupervised learning, by hand).

Conventions (see engine/content.py and t00_foundations.py):
  * Each generator: (rng, level) -> Problem, deterministic given rng.
  * Levels: 1 = mechanical, 2 = multi-step, 3 = derivation / reasoning with a checkable final result.
  * Numbers are chosen so the pen-and-paper work stays exact (rationals, ln r logits, ...);
    transcendental answers use kind="num" with `tol` and say so in `answer_hint`.
"""
from __future__ import annotations

import random
from collections import Counter

import sympy as sp

from engine.checker import fmt
from engine.content import Problem, gen

L = sp.latex
R = sp.Rational
LN2 = sp.log(2)
TOL = 1e-3
DEC_HINT = "exact (e.g. exp(2)/(exp(2)+exp(1)+1), log(7/3)) or a decimal to 4 places"


# --------------------------------------------------------------------------- helpers
def nz(rng: random.Random, lo: int, hi: int) -> int:
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def d4(e) -> str:
    return f"{float(sp.N(e)):.4f}"


def vtex(v) -> str:
    return "\\left(" + ",\\ ".join(L(sp.sympify(t)) for t in v) + "\\right)"


def pts(xs, ys) -> str:
    return ", ".join(f"$({L(sp.sympify(a))}, {L(sp.sympify(b))})$" for a, b in zip(xs, ys))


def sigmoid(z):
    return 1 / (1 + sp.exp(-z))


X = sp.Symbol("x")
UNITS = [(R(3, 5), R(4, 5)), (R(4, 5), R(3, 5)), (R(5, 13), R(12, 13)), (R(12, 13), R(5, 13)),
         (R(3, 5), -R(4, 5)), (R(4, 5), -R(3, 5))]
PROB_HINT = "exact like 1/(1+exp(-3)), or a decimal to 4 places"


def ols_line(xs, ys):
    n = len(xs)
    mx, my = R(sum(xs), n), R(sum(ys), n)
    sxx = sum((a - mx) ** 2 for a in xs)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    w1 = sxy / sxx
    return my - w1 * mx, w1, mx, my, sxx, sxy


def _distinct_xs(rng, n, lo=-3, hi=6):
    while True:
        xs = sorted(rng.sample(range(lo, hi + 1), n))
        if len(set(xs)) == n:
            return xs


def H2(counts) -> sp.Expr:
    n = sum(counts)
    return sp.expand(sp.expand_log(sum(-R(c, n) * sp.log(R(c, n)) for c in counts if c) / LN2, force=True))


# --------------------------------------------------------------------------- linear regression
@gen("ml.linear_regression")
def normal_equations(rng, level):
    if level in (1, 3):
        n = rng.choice([3, 4]) if level == 1 else rng.choice([4, 5])
        xs = _distinct_xs(rng, n, -2, 6)
        while True:
            ys = [rng.randint(-3, 9) for _ in range(n)]
            w0, w1, mx, my, sxx, sxy = ols_line(xs, ys)
            rss = sum((b - w0 - w1 * a) ** 2 for a, b in zip(xs, ys))
            if level == 1 or rss != 0:
                break
        if level == 1:
            return Problem(
                prompt=f"Fit $\\hat y = w_0 + w_1 x$ by least squares to the points {pts(xs, ys)}. Give $w_0, w_1$.",
                answer=[fmt(w0), fmt(w1)], kind="list", answer_hint="w0, w1   e.g. 1/2, 3/4",
                rubric=["Means x̄, ȳ", "w₁ = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²", "w₀ = ȳ - w₁x̄"],
                solution=(f"$\\bar x = {L(mx)}, \\bar y = {L(my)}$, $S_{{xy}} = {L(sxy)}, S_{{xx}} = {L(sxx)}$, "
                          f"$w_1 = {L(w1)}$, $w_0 = {L(w0)}$."),
                misconceptions=[([fmt(w1), fmt(w0)], "swapped-intercept-slope")],
            )
        s2 = rss / n
        return Problem(
            prompt=(f"Model: $y_i = w_0 + w_1 x_i + \\varepsilon_i$ with $\\varepsilon_i \\sim \\mathcal N(0, \\sigma^2)$ i.i.d. "
                    f"Data: {pts(xs, ys)}. Show that maximising the likelihood in $(w_0, w_1)$ is least squares, fit the line, "
                    "then give the **maximum-likelihood** estimate $\\hat\\sigma^2_{\\text{MLE}}$."),
            answer=fmt(s2), kind="num",
            rubric=["Log-likelihood = -n/2 log(2πσ²) - RSS/(2σ²) ⇒ minimise RSS", "Fit w₀, w₁ by least squares",
                    "Compute residuals and RSS", "∂/∂σ² = 0 ⇒ σ̂² = RSS/n"],
            solution=(f"$w_1 = {L(w1)}, w_0 = {L(w0)}$; $\\mathrm{{RSS}} = {L(rss)}$; "
                      f"$\\hat\\sigma^2_{{\\text{{MLE}}}} = \\mathrm{{RSS}}/{n} = {L(s2)}$ (the unbiased version divides by $n-2$)."),
            misconceptions=[(fmt(rss / (n - 2)), "used-unbiased-dof"), (fmt(rss), "forgot-divide-by-n")],
        )
    while True:
        X = sp.Matrix([[rng.randint(-2, 3) for _ in range(2)] for _ in range(4)])
        if (X.T * X).det() != 0:
            break
    y = sp.Matrix([rng.randint(-4, 6) for _ in range(4)])
    XtX, Xty = X.T * X, X.T * y
    w = XtX.inv() * Xty
    return Problem(
        prompt=(f"Least squares without intercept: $X = {L(X)}$, $y = {L(y)}$. Solve the normal equations "
                "$X^\\top X\\,w = X^\\top y$ for $w$."),
        answer=[fmt(v) for v in w], kind="vector", answer_hint="w1, w2   e.g. [3/7, -1/2]",
        rubric=["Compute XᵀX (2×2)", "Compute Xᵀy", "Solve the 2×2 system (Cramer or inverse)", "Optionally check XᵀX w = Xᵀy"],
        solution=f"$X^\\top X = {L(XtX)}$, $X^\\top y = {L(Xty)}$, $w = {L(w)}$.",
        misconceptions=[([fmt(v) for v in Xty], "forgot-inverse")],
    )


@gen("ml.linear_regression")
def regression_properties(rng, level):
    if level == 1:
        a, b = rng.randint(-3, 3), nz(rng, -2, 3)
        xs = _distinct_xs(rng, 4, -2, 5)
        ys = [a + b * t + rng.randint(-2, 2) for t in xs]
        res = [yv - (a + b * t) for t, yv in zip(xs, ys)]
        mse = R(sum(r ** 2 for r in res), 4)
        return Problem(
            prompt=(f"A fitted model predicts $\\hat y = {L(a + b * sp.Symbol('x'))}$. Compute its mean squared error on "
                    f"{pts(xs, ys)}."),
            answer=fmt(mse), kind="num",
            rubric=["Predictions ŷᵢ", "Residuals yᵢ - ŷᵢ", "Average of squared residuals"],
            solution=f"Residuals ${res}$, MSE $= {L(mse)}$.",
            misconceptions=[(fmt(R(sum(abs(r) for r in res), 4)), "used-absolute-error"),
                            (fmt(sum(r ** 2 for r in res)), "forgot-average")],
        )
    if level == 2:
        xs = _distinct_xs(rng, 4, -2, 5)
        while True:
            ys = [rng.randint(-2, 9) for _ in range(4)]
            w0, w1, mx, my, sxx, sxy = ols_line(xs, ys)
            sst = sum((v - my) ** 2 for v in ys)
            if sst and w1:
                break
        rss = sum((v - w0 - w1 * t) ** 2 for t, v in zip(xs, ys))
        r2 = 1 - rss / sst
        return Problem(
            prompt=f"Fit a least-squares line to {pts(xs, ys)} and compute its coefficient of determination $R^2$.",
            answer=fmt(r2), kind="num", tol=1e-4, answer_hint="exact fraction or a decimal to 4 places",
            rubric=["Fit w₀, w₁", "RSS = Σ(y - ŷ)²", "TSS = Σ(y - ȳ)²", "R² = 1 - RSS/TSS"],
            solution=f"$\\hat y = {L(w0)} + {L(w1)}x$, RSS $= {L(rss)}$, TSS $= {L(sst)}$, $R^2 = {L(r2)} \\approx {d4(r2)}$.",
            misconceptions=[(fmt(rss / sst), "reported-unexplained-fraction")],
        )
    xs = [nz(rng, -3, 4) for _ in range(3)]
    ys = [rng.randint(-5, 8) for _ in range(3)]
    cs = [rng.randint(1, 4) for _ in range(3)]
    num = sum(c * a * b for c, a, b in zip(cs, xs, ys))
    den = sum(c * a * a for c, a in zip(cs, xs))
    w = R(num, den)
    w_unw = R(sum(a * b for a, b in zip(xs, ys)), sum(a * a for a in xs))
    return Problem(
        prompt=(f"Weighted least squares through the origin: minimise $J(w) = \\sum_i c_i (y_i - w x_i)^2$. "
                f"Derive the minimiser, then evaluate it for $(x_i, y_i, c_i)$ = " +
                ", ".join(f"$({a}, {b}, {c})$" for a, b, c in zip(xs, ys, cs)) + "."),
        answer=fmt(w), kind="num",
        rubric=["dJ/dw = -2 Σ cᵢ xᵢ (yᵢ - w xᵢ)", "Set to zero", "w* = Σ cᵢxᵢyᵢ / Σ cᵢxᵢ²", "Evaluate"],
        solution=f"$w^* = \\frac{{\\sum c_i x_i y_i}}{{\\sum c_i x_i^2}} = \\frac{{{num}}}{{{den}}} = {L(w)}$.",
        misconceptions=[(fmt(w_unw), "ignored-weights")],
    )


@gen("ml.linear_regression")
def regression_more(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        w0, w1, x0 = rng.randint(-4, 5), nz(rng, -3, 3), rng.randint(-3, 5)
        while True:
            y0 = rng.randint(-8, 12)
            res = y0 - (w0 + w1 * x0)
            if res != 0:
                break
        return Problem(
            prompt=(f"A regression line has already been fitted: $\\hat y = {L(w0 + w1 * X)}$. For the observation "
                    f"$(x, y) = ({x0}, {y0})$, compute the residual $e = y - \\hat y$."),
            answer=str(res), kind="num",
            rubric=["Prediction ŷ = w₀ + w₁x", "Residual = observed - predicted (sign matters)"],
            solution=f"$\\hat y = {L(w0 + w1 * X)}\\big|_{{x={x0}}} = {w0 + w1 * x0}$, $e = {y0} - ({w0 + w1 * x0}) = {res}$.",
            misconceptions=[(str(-res), "residual-sign-reversed")],
        )
    if level == 1:
        n = rng.choice([4, 5, 6, 8])
        while True:
            xs = [rng.randint(0, 9) for _ in range(n)]
            ys = [rng.randint(0, 12) for _ in range(n)]
            w0, w1, mx, my, sxx, sxy = ols_line(xs, ys)
            if sxx and w1 != 0 and w0 != 0:
                break
        Sx, Sy = sum(xs), sum(ys)
        Sxx, Sxy = sum(a * a for a in xs), sum(a * b for a, b in zip(xs, ys))
        raw = R(Sxy, Sxx)
        return Problem(
            prompt=(f"Summary statistics of {n} paired observations: $\\sum x_i = {Sx}$, $\\sum y_i = {Sy}$, "
                    f"$\\sum x_i^2 = {Sxx}$, $\\sum x_i y_i = {Sxy}$. Compute the least-squares intercept $w_0$ and slope $w_1$ "
                    "of $\\hat y = w_0 + w_1 x$."),
            answer=[fmt(w0), fmt(w1)], kind="list", answer_hint="w0, w1   e.g. 1/2, 3/4",
            rubric=["S_xy = Σxy - (Σx)(Σy)/n", "S_xx = Σx² - (Σx)²/n", "w₁ = S_xy / S_xx", "w₀ = ȳ - w₁x̄"],
            solution=(f"$S_{{xy}} = {Sxy} - \\frac{{{Sx}\\cdot{Sy}}}{{{n}}} = {L(sxy)}$, $S_{{xx}} = {Sxx} - \\frac{{{Sx}^2}}{{{n}}} = {L(sxx)}$; "
                      f"$w_1 = {L(w1)}$, $w_0 = {L(my)} - \\left({L(w1)}\\right)\\cdot {L(mx)} = {L(w0)}$."),
            misconceptions=[([fmt(my - raw * mx), fmt(raw)], "forgot-centering"), ([fmt(w1), fmt(w0)], "swapped-intercept-slope")],
        )
    if level == 2 and v == 0:
        while True:
            xs = [nz(rng, -3, 3) for _ in range(3)]
            ys = [rng.randint(-4, 6) for _ in range(3)]
            w = rng.randint(-1, 2)
            grad = R(2, 3) * sum(a * (w * a - b) for a, b in zip(xs, ys))
            if grad != 0:
                break
        eta = rng.choice([R(1, 10), R(1, 20), R(1, 4)])
        new = w - eta * grad
        return Problem(
            prompt=(f"Gradient descent for the model $\\hat y = w x$ (no intercept) on the mean squared error "
                    f"$J(w) = \\frac13\\sum_i (w x_i - y_i)^2$ with data {pts(xs, ys)}. Starting from $w = {w}$ with learning "
                    f"rate $\\eta = {L(eta)}$, take **one** step. Give the new $w$."),
            answer=fmt(new), kind="num",
            rubric=["J'(w) = (2/n) Σ xᵢ(w xᵢ - yᵢ)", f"Evaluate at w = {w}", "w ← w - η J'(w)"],
            solution=(f"$J'({w}) = \\frac23\\sum_i x_i(w x_i - y_i) = {L(grad)}$, so $w \\leftarrow {w} - {L(eta)}\\cdot({L(grad)}) = {L(new)}$."),
            misconceptions=[(fmt(w + eta * grad), "gradient-ascent-sign"), (fmt(w - eta * grad / 2), "forgot-factor-2")],
        )
    if level == 2:
        r = rng.choice([R(1, 2), R(3, 5), R(4, 5), R(2, 5), -R(1, 2), -R(3, 5)])
        sx, sy = rng.choice([2, 3, 4, 5]), rng.choice([2, 3, 5, 6, 10])
        mx, my = rng.randint(0, 20), rng.randint(10, 60)
        x0 = mx + nz(rng, -6, 6)
        w1 = r * sy / sx
        yhat = my + w1 * (x0 - mx)
        return Problem(
            prompt=(f"From a large sample: $\\bar x = {mx}$, $\\bar y = {my}$, $s_x = {sx}$, $s_y = {sy}$ and the correlation "
                    f"is $r = {float(r):g}$. Using the least-squares line of $y$ on $x$, predict $y$ at $x = {x0}$."),
            answer=fmt(yhat), kind="num", answer_hint="exact fraction or decimal, e.g. 47.5",
            rubric=["Slope w₁ = r·s_y/s_x", "Line passes through (x̄, ȳ)", "ŷ = ȳ + w₁(x - x̄)"],
            solution=(f"$w_1 = {L(r)}\\cdot\\frac{{{sy}}}{{{sx}}} = {L(w1)}$, $\\hat y = {my} + \\left({L(w1)}\\right)\\cdot({x0 - mx}) = {L(yhat)}$."),
            misconceptions=[(fmt(my + r * sx / sy * (x0 - mx)), "inverted-sd-ratio"),
                            (fmt(my + R(sy, sx) * (x0 - mx)), "forgot-correlation")],
        )
    if v == 0:
        while True:
            xs = [nz(rng, -3, 3) for _ in range(3)]
            ys = [rng.randint(-5, 9) for _ in range(3)]
            num, den = sum(a * a * b for a, b in zip(xs, ys)), sum(a ** 4 for a in xs)
            if num != 0:
                break
        w = R(num, den)
        lin = sum(a * b for a, b in zip(xs, ys))
        mis = [(fmt(R(lin, sum(a * a for a in xs))), "used-linear-feature")] if lin else []
        return Problem(
            prompt=("Derive the least-squares estimate of $w$ in the model $\\hat y = w x^2$ (a single quadratic feature, no "
                    f"intercept), then evaluate it on the data {pts(xs, ys)}."),
            answer=fmt(w), kind="num",
            rubric=["J(w) = Σ(yᵢ - w xᵢ²)²", "dJ/dw = -2 Σ xᵢ²(yᵢ - w xᵢ²) = 0", "w* = Σ xᵢ² yᵢ / Σ xᵢ⁴", "Evaluate"],
            solution=f"$w^* = \\frac{{\\sum x_i^2 y_i}}{{\\sum x_i^4}} = \\frac{{{num}}}{{{den}}} = {L(w)}$.",
            misconceptions=mis + [(fmt(R(num, sum(a * a for a in xs))), "wrong-denominator")],
        )
    n = rng.choice([5, 7])
    while True:
        ys = sorted(rng.randint(0, 20) for _ in range(n))
        med, mean = ys[n // 2], R(sum(ys), n)
        if mean != med:
            break
    J = sum(abs(t - med) for t in ys)
    return Problem(
        prompt=("Under absolute loss, the best constant prediction $c$ minimises $J(c) = \\sum_i |y_i - c|$. Derive which "
                f"statistic of the data this is (subgradient argument), then give $c^*$ and $J(c^*)$ for $y = {ys}$."),
        answer=[fmt(med), fmt(J)], kind="list", answer_hint="c*, J(c*)   e.g. 7, 23",
        rubric=["Subgradient of |y - c| in c is -sign(y - c)", "0 ∈ ∂J ⇔ as many points above as below c",
                "c* = median", "Sum the absolute deviations"],
        solution=(f"$\\partial J = \\#\\{{y_i < c\\}} - \\#\\{{y_i > c\\}}$ vanishes at the median, $c^* = {med}$; "
                  f"$J = {L(J)}$ (the mean ${L(mean)}$ minimises *squared* loss instead)."),
        misconceptions=[([fmt(mean), fmt(sum(abs(t - mean) for t in ys))], "used-mean-not-median")],
    )


# --------------------------------------------------------------------------- logistic regression
@gen("ml.logistic_regression")
def logreg_gradient(rng, level):
    if level == 1:
        while True:
            w = [nz(rng, -2, 2), nz(rng, -2, 2)]
            x = [rng.randint(-2, 2), rng.randint(-2, 2)]
            b = rng.randint(-2, 2)
            z = w[0] * x[0] + w[1] * x[1] + b
            if z != 0 and abs(z) <= 4:
                break
        p = sigmoid(sp.Integer(z))
        return Problem(
            prompt=(f"Logistic regression with $w = {vtex(w)}$, $b = {b}$. For $x = {vtex(x)}$ compute "
                    "$P(y=1\\mid x) = \\sigma(w^\\top x + b)$."),
            answer=fmt(p), kind="num", tol=TOL, answer_hint="exact like 1/(1+exp(-2)), or a decimal to 4 places",
            rubric=["Logit z = wᵀx + b", "σ(z) = 1/(1 + e^{-z})", "Evaluate"],
            solution=f"$z = {z}$, $\\sigma({z}) = {L(p)} \\approx {d4(p)}$.",
            misconceptions=[(fmt(sigmoid(sp.Integer(-z))), "sigmoid-sign-error"), (fmt(z), "forgot-sigmoid")],
        )
    if level == 2:
        n = rng.choice([2, 3])
        eta = rng.choice([R(1, 2), sp.Integer(1), R(1, 10)])
        while True:
            X = [[rng.randint(-2, 3), rng.randint(-2, 3)] for _ in range(n)]
            Y = [rng.randint(0, 1) for _ in range(n)]
            g = [sum((R(1, 2) - Y[i]) * X[i][j] for i in range(n)) / n for j in range(2)]
            gb = sum(R(1, 2) - Y[i] for i in range(n)) / n
            if any(g) or gb:
                break
        new = [-eta * g[0], -eta * g[1], -eta * gb]
        return Problem(
            prompt=(f"Logistic regression initialised at $w = (0, 0)$, $b = 0$. Training batch (x, y): " +
                    ", ".join(f"$({vtex(X[i])}, {Y[i]})$" for i in range(n)) +
                    f". Take **one** gradient-descent step on the mean binary cross-entropy with learning rate $\\eta = {L(eta)}$. "
                    "Give the new $(w_1, w_2, b)$."),
            answer=[fmt(v) for v in new], kind="vector", answer_hint="w1, w2, b   e.g. [1/4, -1/2, 0]",
            rubric=["At zero init every σ(z) = 1/2", "∇_w = (1/n) Σ (σᵢ - yᵢ) xᵢ, ∂_b = (1/n) Σ (σᵢ - yᵢ)",
                    "Update θ ← θ - η∇", "Collect (w₁, w₂, b)"],
            solution=(f"Gradient $= {vtex(g + [gb])}$; new parameters $= -{L(eta)}\\cdot$ gradient $= {vtex(new)}$."),
            misconceptions=[([fmt(-v) for v in new], "gradient-ascent-sign")],
        )
    r = rng.choice([sp.Integer(2), sp.Integer(3), sp.Integer(4), R(1, 2), R(1, 3)])
    y = rng.randint(0, 1)
    x = [nz(rng, -3, 3), nz(rng, -3, 3)]
    s = r / (1 + r)
    g = [(s - y) * x[0], (s - y) * x[1], s - y]
    return Problem(
        prompt=(f"For one example with label $y = {y}$ and features $x = {vtex(x)}$, the current logit is "
                f"$z = w^\\top x + b = \\ln {L(r)}$. Derive $\\partial L/\\partial z$ for $L = -[y\\ln\\sigma(z) + (1-y)\\ln(1-\\sigma(z))]$, "
                "then give the gradient $(\\partial L/\\partial w_1, \\partial L/\\partial w_2, \\partial L/\\partial b)$."),
        answer=[fmt(v) for v in g], kind="vector", answer_hint="three exact values, e.g. [1/2, -1, 1/4]",
        rubric=["σ'(z) = σ(z)(1 - σ(z))", "∂L/∂z = σ(z) - y", f"σ(ln r) = r/(1+r) = {s}", "Chain rule: ∂L/∂w = (σ - y)x, ∂L/∂b = σ - y"],
        solution=(f"$\\frac{{\\partial L}}{{\\partial z}} = -\\frac{{y}}{{\\sigma}}\\sigma(1-\\sigma) + \\frac{{1-y}}{{1-\\sigma}}\\sigma(1-\\sigma) "
                  f"= \\sigma(z) - y = {L(s)} - {y} = {L(s - y)}$; gradient $= {vtex(g)}$."),
        misconceptions=[([fmt(-v) for v in g], "gradient-sign-flip"),
                        ([fmt(s * x[0]), fmt(s * x[1]), fmt(s)], "forgot-subtract-label")],
    )


@gen("ml.logistic_regression")
def logreg_decision(rng, level):
    if level == 1:
        w1, w2, b = nz(rng, -4, 4), nz(rng, -4, 4), rng.randint(-6, 6)
        a = rng.randint(-3, 3)
        x2 = R(-(b + w1 * a), w2)
        return Problem(
            prompt=(f"A logistic classifier predicts class 1 when $\\sigma({w1}x_1 {'+' if w2 > 0 else '-'} {abs(w2)}x_2 "
                    f"{'+' if b >= 0 else '-'} {abs(b)}) \\ge \\tfrac12$. At $x_1 = {a}$, for which $x_2$ is the point exactly "
                    "on the decision boundary?"),
            answer=fmt(x2), kind="num",
            rubric=["σ(z) = 1/2 ⇔ z = 0", "Set w₁x₁ + w₂x₂ + b = 0", "Solve for x₂"],
            solution=f"${w1}\\cdot({a}) + {w2}x_2 + ({b}) = 0 \\Rightarrow x_2 = {L(x2)}$.",
            misconceptions=[(fmt(-x2), "sign-error")],
        )
    if level == 2:
        p0 = rng.choice([R(1, 5), R(1, 4), R(1, 3), R(2, 5), R(1, 2)])
        m = rng.choice([2, 3, 4])
        dlt = rng.choice([1, 2])
        o = p0 / (1 - p0) * m ** dlt
        p1 = o / (1 + o)
        return Problem(
            prompt=(f"In a logistic model the coefficient of feature $x_j$ is $w_j = \\ln {m}$. For a patient the model gives "
                    f"$P(y=1) = {L(p0)}$. If $x_j$ increases by ${dlt}$ (everything else fixed), what is the new probability?"),
            answer=fmt(p1), kind="num",
            rubric=["Convert probability to odds p/(1-p)", f"Logit increases by {dlt}·ln {m} ⇒ odds × {m}^{dlt}",
                    "Convert odds back to probability"],
            solution=f"Odds ${L(p0 / (1 - p0))} \\to {L(o)}$, so $p = {L(o)}/(1 + {L(o)}) = {L(p1)}$.",
            misconceptions=[(fmt(p0 * m ** dlt), "multiplied-probability-not-odds"),
                            (fmt(p0 + dlt * sp.log(m)), "added-coefficient-to-probability")],
        )
    n1, n0 = rng.randint(1, 12), rng.randint(1, 12)
    if n1 == n0:
        n1 += 1
    b = sp.log(R(n1, n0))
    return Problem(
        prompt=(f"A logistic model with **only a bias**, $P(y=1) = \\sigma(b)$, is fit by maximum likelihood to "
                f"{n1} positive and {n0} negative examples. Derive $b^*$."),
        answer=fmt(b), kind="num", tol=TOL, answer_hint="exact like log(7/3), or a decimal to 4 places",
        rubric=["Log-likelihood n₁ ln σ(b) + n₀ ln(1-σ(b))", "Derivative n₁(1-σ) - n₀σ = 0",
                "σ(b*) = n₁/(n₁+n₀)", "Invert the sigmoid: b* = ln(n₁/n₀)"],
        solution=f"$\\sigma(b^*) = {L(R(n1, n1 + n0))}$, so $b^* = \\ln\\frac{{{n1}}}{{{n0}}} = {L(b)} \\approx {d4(b)}$ (the log-odds).",
        misconceptions=[(fmt(R(n1, n1 + n0)), "gave-probability-not-logit"), (fmt(sp.log(R(n0, n1))), "inverted-odds")],
    )


PROBS = [R(1, 10), R(1, 5), R(1, 4), R(1, 3), R(2, 5), R(3, 5), R(2, 3), R(3, 4), R(4, 5), R(9, 10)]


@gen("ml.logistic_regression")
def logreg_more(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        p = rng.choice(PROBS)
        z = sp.log(p / (1 - p))
        return Problem(
            prompt=(f"A logistic model outputs $P(y=1\\mid x) = {L(p)}$. What logit $z = w^\\top x + b$ (the log-odds) "
                    "produced this output?"),
            answer=fmt(z), kind="num", tol=TOL, answer_hint="exact like log(3), or a decimal to 4 places",
            rubric=["σ(z) = p ⇔ z = ln(p/(1-p))", "Odds p/(1-p)", "Take the natural log"],
            solution=f"Odds $= {L(p)}/{L(1 - p)} = {L(p / (1 - p))}$, so $z = \\ln {L(p / (1 - p))} \\approx {d4(z)}$.",
            misconceptions=[(fmt(sp.log(p)), "logit-is-log-probability"), (fmt(p / (1 - p)), "forgot-log")],
        )
    if level == 1:
        w = R(rng.choice([2, 3, 4, 5, 6, 7, -3, -5]), 10)
        d = rng.choice([1, 2, 3])
        f = sp.exp(w * d)
        return Problem(
            prompt=(f"In a fitted logistic regression the coefficient of feature $x_j$ is $w_j = {float(w):g}$. By what factor "
                    f"are the **odds** of $y = 1$ multiplied when $x_j$ increases by ${d}$ (all else fixed)?"),
            answer=fmt(f), kind="num", tol=TOL, answer_hint="exact like exp(4/5), or a decimal to 4 places",
            rubric=["Log-odds are linear: Δz = w_j·Δx_j", "Odds = e^z, so they multiply by e^{w_j Δx_j}", "Evaluate"],
            solution=f"$\\Delta z = {L(w)}\\cdot{d} = {L(w * d)}$, factor $= e^{{{L(w * d)}}} \\approx {d4(f)}$.",
            misconceptions=[(fmt(w * d), "used-coefficient-not-exp"), (fmt(sp.exp(w)), "ignored-step-size")],
        )
    if level == 2 and v == 0:
        x1s, x2s = sp.symbols("x_1 x_2")
        w1, w2, b = nz(rng, -3, 3), nz(rng, -3, 3), rng.randint(-4, 4)
        x2 = rng.randint(-3, 3)
        p = rng.choice([q for q in PROBS if q != R(1, 2)])
        zt = sp.log(p / (1 - p))
        x1 = (zt - w2 * x2 - b) / w1
        return Problem(
            prompt=(f"A logistic classifier has $P(y=1\\mid x) = \\sigma({L(w1 * x1s + w2 * x2s + b)})$. With $x_2 = {x2}$ fixed, "
                    f"which value of $x_1$ makes $P(y=1\\mid x) = {L(p)}$?"),
            answer=fmt(x1), kind="num", tol=TOL, answer_hint="exact like (log(3) - 2)/3, or a decimal to 4 places",
            rubric=["Invert the sigmoid: z = ln(p/(1-p))", f"Substitute x₂ = {x2}", "Solve the linear equation for x₁"],
            solution=(f"Need $z = \\ln {L(p / (1 - p))}$. ${L(w1 * x1s)} + ({w2})({x2}) + ({b}) = \\ln {L(p / (1 - p))}$ "
                      f"$\\Rightarrow x_1 = {L(x1)} \\approx {d4(x1)}$."),
            misconceptions=[(fmt((p - w2 * x2 - b) / w1), "set-logit-equal-probability")],
        )
    if level == 2:
        rs = [rng.choice([sp.Integer(2), sp.Integer(3), sp.Integer(4), R(1, 2), R(1, 3)]) for _ in range(3)]
        ys = [rng.randint(0, 1) for _ in range(3)]
        terms = [sp.log(r / (1 + r)) if y else sp.log(1 / (1 + r)) for r, y in zip(rs, ys)]
        ll = sp.expand_log(sum(terms), force=True)
        return Problem(
            prompt=("Three training examples have logits and labels $(z_i, y_i)$: " +
                    ", ".join(f"$(\\ln {L(r)},\\ {y})$" for r, y in zip(rs, ys)) +
                    ". Compute the log-likelihood $\\sum_i [y_i\\ln\\sigma(z_i) + (1-y_i)\\ln(1-\\sigma(z_i))]$ in nats."),
            answer=fmt(ll), kind="num", tol=TOL, answer_hint="exact like log(2/9), or a decimal to 4 places (it is negative)",
            rubric=["σ(ln r) = r/(1+r), 1 - σ(ln r) = 1/(1+r)", "Pick the term selected by each label", "Sum the logs"],
            solution=("Terms: " + ", ".join(f"${L(t)}$" for t in terms) + f"; total $\\approx {d4(ll)}$."),
            misconceptions=[(fmt(-ll), "reported-loss-not-likelihood"),
                            (fmt(sum(sp.log(r / (1 + r)) if not y else sp.log(1 / (1 + r)) for r, y in zip(rs, ys))),
                             "used-wrong-term")],
        )
    if v == 0:
        cfn, cfp = rng.sample(range(1, 11), 2)
        pt, zt = R(cfp, cfp + cfn), sp.log(R(cfp, cfn))
        return Problem(
            prompt=(f"A false negative costs ${cfn}$ and a false positive costs ${cfp}$ (correct decisions cost 0). Given a "
                    "calibrated probability $p = P(y=1\\mid x)$, derive the decision rule that minimises expected cost: give "
                    "the probability threshold $p^*$ and the corresponding logit threshold $z^*$."),
            answer=[fmt(pt), fmt(zt)], kind="list", tol=TOL, answer_hint="p*, z*   e.g. 1/4, log(1/3)",
            rubric=["Expected cost of predicting 1: (1-p)·c_FP; of predicting 0: p·c_FN", "Predict 1 iff p·c_FN > (1-p)·c_FP",
                    "p* = c_FP/(c_FP + c_FN)", "z* = ln(p*/(1-p*)) = ln(c_FP/c_FN)"],
            solution=(f"$p\\,{cfn} > (1-p)\\,{cfp} \\iff p > \\frac{{{cfp}}}{{{cfp + cfn}}}$; "
                      f"$z^* = \\ln\\frac{{{cfp}}}{{{cfn}}} \\approx {d4(zt)}$."),
            misconceptions=[([fmt(R(cfn, cfp + cfn)), fmt(sp.log(R(cfn, cfp)))], "swapped-costs")],
        )
    while True:
        xs = [nz(rng, -3, 3) for _ in range(3)]
        ys = [rng.randint(0, 1) for _ in range(3)]
        g = sum((R(1, 2) - y) * x for x, y in zip(xs, ys)) / 3
        if g != 0:
            break
    Hs = sum(R(x * x, 4) for x in xs) / 3
    w1 = -g / Hs
    return Problem(
        prompt=("Derive the gradient and second derivative of the mean BCE for a 1-D logistic model "
                "$P(y=1\\mid x) = \\sigma(wx)$ (no bias), then take **one Newton step** from $w = 0$ on the data (x, y): " +
                ", ".join(f"$({x}, {y})$" for x, y in zip(xs, ys)) + "."),
        answer=fmt(w1), kind="num",
        rubric=["J'(w) = (1/n) Σ (σ(wxᵢ) - yᵢ) xᵢ", "J''(w) = (1/n) Σ σ(1-σ) xᵢ²", "At w = 0: σ = 1/2, σ(1-σ) = 1/4",
                "Newton: w ← w - J'/J''"],
        solution=(f"$J'(0) = {L(g)}$, $J''(0) = \\frac{{1}}{{3}}\\sum \\tfrac14 x_i^2 = {L(Hs)}$, "
                  f"$w_1 = 0 - \\left({L(g)}\\right)/{L(Hs)} = {L(w1)}$."),
        misconceptions=[(fmt(-g), "used-gradient-step-not-newton"), (fmt(g / Hs), "newton-sign-error")],
    )


# --------------------------------------------------------------------------- softmax
@gen("ml.softmax")
def softmax(rng, level):
    if level == 1:
        while True:
            z = [rng.randint(-1, 3) for _ in range(3)]
            if len(set(z)) > 1:
                break
        k = rng.randrange(3)
        den = sum(sp.exp(t) for t in z)
        p = sp.exp(z[k]) / den
        mis = [(fmt(R(z[k], sum(z))), "normalized-logits-linearly")] if sum(z) != 0 else []
        return Problem(
            prompt=f"Logits $z = {vtex(z)}$ for classes 1, 2, 3. Compute the softmax probability of class {k + 1}.",
            answer=fmt(p), kind="num", tol=TOL, answer_hint=DEC_HINT,
            rubric=["Exponentiate every logit", "Normalise by the sum", f"Read off class {k + 1}"],
            solution=f"$p_{{{k + 1}}} = \\frac{{e^{{{z[k]}}}}}{{{' + '.join(f'e^{{{t}}}' for t in z)}}} \\approx {d4(p)}$.",
            misconceptions=mis + [(fmt(sp.exp(z[k])), "forgot-normalisation")],
        )
    while True:
        a = [rng.randint(1, 6) for _ in range(3)]
        if len(set(a)) > 1:
            break
    S = sum(a)
    p = [R(v, S) for v in a]
    if level == 2:
        y = rng.randrange(3)
        g = [p[i] - (1 if i == y else 0) for i in range(3)]
        return Problem(
            prompt=(f"A 3-class softmax classifier has logits $z = (\\ln {a[0]},\\ \\ln {a[1]},\\ \\ln {a[2]})$ and the true class "
                    f"is {y + 1}. Give the gradient of the cross-entropy loss with respect to the logits, $\\partial L/\\partial z$."),
            answer=[fmt(v) for v in g], kind="vector", answer_hint="three exact values, e.g. [1/6, -1/2, 1/3]",
            rubric=["softmax(ln a) = a / Σa", "∂L/∂z = p - onehot(y)", "Entries sum to 0"],
            solution=f"$p = {vtex(p)}$, $\\nabla_z L = p - e_{{{y + 1}}} = {vtex(g)}$.",
            misconceptions=[([fmt(-v) for v in g], "gradient-sign-flip"), ([fmt(v) for v in p], "forgot-subtract-onehot")],
        )
    J = sp.Matrix(3, 3, lambda i, j: p[i] * ((1 if i == j else 0) - p[j]))
    return Problem(
        prompt=(f"Derive $\\partial p_i/\\partial z_j$ for $p = \\mathrm{{softmax}}(z)$, then evaluate the $3\\times3$ Jacobian at "
                f"$z = (\\ln {a[0]},\\ \\ln {a[1]},\\ \\ln {a[2]})$. (On paper: why does adding a constant to all logits not change $p$?)"),
        answer=fmt(J), kind="matrix", answer_hint="rows, e.g. [[a,b,c],[d,e,f],[g,h,k]] with J[i][j] = dp_i/dz_j",
        rubric=["Quotient rule on e^{z_i}/Σe^{z_k}", "∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ - pⱼ)", f"p = {', '.join(map(str, p))}",
                "J = diag(p) - ppᵀ (symmetric, rows sum to 0)"],
        solution=f"$p = {vtex(p)}$, $J = \\mathrm{{diag}}(p) - pp^\\top = {L(J)}$.",
        misconceptions=[(fmt(sp.diag(*p)), "forgot-off-diagonal")],
    )


def _ln_logits(a) -> str:
    return "(" + ",\\ ".join(f"\\ln {t}" for t in a) + ")"


def _distinct_counts(rng, k=3, lo=1, hi=5):
    while True:
        a = [rng.randint(lo, hi) for _ in range(k)]
        if len(set(a)) > 1:
            return a


@gen("ml.softmax")
def softmax_more(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        a = _distinct_counts(rng, 3, 1, 4)
        T = rng.choice([R(1, 2), R(1, 3)])
        m = int(1 / T)
        k = rng.randrange(3)
        p = R(a[k] ** m, sum(t ** m for t in a))
        return Problem(
            prompt=(f"Softmax with temperature: $p_i = e^{{z_i/T}}/\\sum_j e^{{z_j/T}}$ with $T = {L(T)}$ and logits "
                    f"$z = {_ln_logits(a)}$. Compute $p_{k + 1}$ exactly."),
            answer=fmt(p), kind="num",
            rubric=[f"z/T = {m}·ln a = ln(a^{m})", f"So p ∝ a^{m}", "Normalise and read off the class"],
            solution=(f"$e^{{z_i/T}} = a_i^{m}$: ${', '.join(str(t ** m) for t in a)}$; $p_{k + 1} = {a[k] ** m}/"
                      f"{sum(t ** m for t in a)} = {L(p)}$ (lower $T$ sharpens the distribution)."),
            misconceptions=[(fmt(R(a[k], sum(a))), "ignored-temperature")],
        )
    if level == 1:
        a = _distinct_counts(rng, 3, 1, 6)
        y = rng.randrange(3)
        loss = sp.log(R(sum(a), a[y]))
        return Problem(
            prompt=(f"A 3-class network outputs logits $z = {_ln_logits(a)}$; the true class is {y + 1}. Compute the "
                    "cross-entropy loss $-\\ln p_y$ in nats."),
            answer=fmt(loss), kind="num", tol=TOL, answer_hint="exact like log(7/2), or a decimal to 4 places",
            rubric=["softmax(ln a) = a/Σa", f"p_y = {a[y]}/{sum(a)}", "Loss = -ln p_y"],
            solution=f"$p_{y + 1} = {L(R(a[y], sum(a)))}$, loss $= \\ln {L(R(sum(a), a[y]))} \\approx {d4(loss)}$ nats.",
            misconceptions=[(fmt(-sp.log(a[y])), "forgot-softmax-normalisation"), (fmt(loss / LN2), "used-log2")],
        )
    if level == 2 and v == 0:
        while True:
            a2, a3 = rng.randint(1, 5), rng.randint(1, 5)
            p = rng.choice([R(1, 2), R(1, 3), R(2, 3), R(3, 4), R(1, 4)])
            S = a2 + a3
            r = p * S / (1 - p)
            if r != 1:
                break
        z1 = sp.log(r)
        return Problem(
            prompt=(f"Classes 2 and 3 have fixed logits $z_2 = \\ln {a2}$ and $z_3 = \\ln {a3}$. What logit $z_1$ makes the "
                    f"softmax probability of class 1 equal to ${L(p)}$?"),
            answer=fmt(z1), kind="num", tol=TOL, answer_hint="exact like log(5/2), or a decimal to 4 places",
            rubric=["p₁ = e^{z₁}/(e^{z₁} + S) with S = e^{z₂} + e^{z₃}", f"S = {S}", "Solve: e^{z₁} = p S/(1 - p)", "Take ln"],
            solution=f"$e^{{z_1}} = \\frac{{{L(p)}\\cdot {S}}}{{{L(1 - p)}}} = {L(r)}$, so $z_1 = \\ln {L(r)} \\approx {d4(z1)}$.",
            misconceptions=[(fmt(sp.log(p)), "logit-equals-log-probability"), (fmt(sp.log(p * S)), "forgot-own-term-in-denominator")],
        )
    if level == 2:
        y = rng.randrange(3)
        eta = rng.choice([sp.Integer(1), sp.Integer(3), R(3, 2), sp.Integer(2)])
        py = 1 / (1 + 2 * sp.exp(-eta))
        znew = [eta * (R(2, 3) if i == y else -R(1, 3)) for i in range(3)]
        return Problem(
            prompt=(f"Gradient descent directly on logits: start at $z = (0, 0, 0)$, true class {y + 1}, cross-entropy loss, "
                    f"learning rate $\\eta = {L(eta)}$. After **one** step, what is the softmax probability of the true class?"),
            answer=fmt(py), kind="num", tol=TOL, answer_hint=PROB_HINT,
            rubric=["At z = 0, p = (1/3, 1/3, 1/3)", "∇_z L = p - onehot(y)", "z ← z - η∇", "Recompute the softmax"],
            solution=(f"$\\nabla = p - e_{{{y + 1}}}$, new $z = {vtex(znew)}$; $p_{{{y + 1}}} = \\frac{{e^{{{L(znew[y])}}}}}"
                      f"{{e^{{{L(znew[y])}}} + 2e^{{{L(-eta / 3)}}}}} = \\frac{{1}}{{1 + 2e^{{{L(-eta)}}}}} \\approx {d4(py)}$."),
            misconceptions=[(fmt(1 / (1 + 2 * sp.exp(eta))), "gradient-ascent-sign")],
        )
    if v == 0:
        x = [nz(rng, -3, 3), nz(rng, -3, 3)]
        y = rng.randrange(3)
        g = sp.Matrix([R(1, 3) - (1 if i == y else 0) for i in range(3)])
        G = g * sp.Matrix([x])
        return Problem(
            prompt=("For a linear softmax classifier $z = Wx$ ($W$ is $3\\times 2$, no bias) with cross-entropy loss, derive "
                    f"$\\partial L/\\partial W$. Evaluate it at $W = 0$ for $x = {vtex(x)}$ with true class {y + 1}."),
            answer=fmt(G), kind="matrix", answer_hint="3 rows of 2, e.g. [[1/3, 1], [-2/3, -2], [1/3, 1]]",
            rubric=["∂L/∂z = p - onehot(y)", "z_i = Σ_j W_ij x_j ⇒ ∂z_i/∂W_ij = x_j", "∂L/∂W = (p - e_y) xᵀ",
                    "At W = 0, p is uniform (1/3 each)"],
            solution=f"$p - e_{{{y + 1}}} = {vtex(list(g))}$, $\\partial L/\\partial W = (p - e_y)x^\\top = {L(G)}$.",
            misconceptions=[(fmt(-G), "gradient-sign-flip")],
        )
    a = _distinct_counts(rng, 3, 1, 6)
    S = sum(a)
    y = rng.randrange(3)
    eps = rng.choice([R(1, 10), R(1, 5), R(3, 10)])
    q = [(1 - eps) * (1 if i == y else 0) + eps / 3 for i in range(3)]
    g = [R(a[i], S) - q[i] for i in range(3)]
    return Problem(
        prompt=(f"Label smoothing: the target is $q = (1-\\varepsilon)\\,e_y + \\varepsilon/K$ with $K = 3$, "
                f"$\\varepsilon = {L(eps)}$, true class $y = {y + 1}$. For $L = -\\sum_i q_i \\ln p_i$ with $p = \\mathrm{{softmax}}(z)$, "
                f"derive $\\partial L/\\partial z$ and evaluate it at $z = {_ln_logits(a)}$."),
        answer=[fmt(t) for t in g], kind="vector", answer_hint="three exact values, e.g. [1/5, -3/10, 1/10]",
        rubric=["∂(-ln p_i)/∂z_j = p_j - δ_ij", "Weight by q_i and use Σq_i = 1", "∂L/∂z = p - q",
                f"p = a/{S}, q = {', '.join(str(t) for t in q)}"],
        solution=f"$p = {vtex([R(t, S) for t in a])}$, $q = {vtex(q)}$, $\\nabla_z L = p - q = {vtex(g)}$.",
        misconceptions=[([fmt(R(a[i], S) - (1 if i == y else 0)) for i in range(3)], "ignored-label-smoothing")],
    )


# --------------------------------------------------------------------------- bias / variance
@gen("ml.bias_variance")
def estimator_bias_variance(rng, level):
    mu = nz(rng, -4, 5)
    s2 = rng.choice([1, 2, 4, 9])
    n = rng.choice([4, 5, 8, 10])
    c = rng.choice([R(1, 2), R(2, 3), R(3, 4), R(4, 5)])
    bias, var = (c - 1) * mu, c ** 2 * R(s2, n)
    if level == 1:
        return Problem(
            prompt=(f"$X_1..X_{{{n}}}$ are i.i.d. with mean $\\mu = {mu}$ and variance $\\sigma^2 = {s2}$. The shrinkage estimator "
                    f"$\\hat\\mu = {L(c)}\\,\\bar X$ is used. Give its bias and its variance."),
            answer=[fmt(bias), fmt(var)], kind="list", answer_hint="bias, variance   e.g. -1/2, 3/10",
            rubric=["E[cX̄] = cμ ⇒ bias = (c-1)μ", "Var(X̄) = σ²/n", "Var(cX̄) = c²σ²/n"],
            solution=f"bias $= ({L(c)}-1)\\cdot({mu}) ={L(bias)}$; variance $= {L(c ** 2)}\\cdot {s2}/{n} = {L(var)}$.",
            misconceptions=[([fmt(bias), fmt(c * R(s2, n))], "variance-scaled-by-c-not-c-squared"),
                            ([fmt(-bias), fmt(var)], "bias-sign")],
        )
    if level == 2:
        mse = bias ** 2 + var
        return Problem(
            prompt=(f"$X_1..X_{{{n}}}$ i.i.d., mean $\\mu = {mu}$, variance $\\sigma^2 = {s2}$. Compute the MSE of "
                    f"$\\hat\\mu = {L(c)}\\bar X$ and (on paper) compare it with the MSE of $\\bar X$, which is ${L(R(s2, n))}$."),
            answer=fmt(mse), kind="num",
            rubric=["MSE = bias² + variance", "bias = (c-1)μ", "variance = c²σ²/n", "Compare with σ²/n"],
            solution=(f"MSE $= \\left({L(bias)}\\right)^2 +{L(var)} = {L(mse)}$ — "
                      f"{'better' if mse < R(s2, n) else 'worse'} than $\\bar X$'s ${L(R(s2, n))}$."),
            misconceptions=[(fmt(var), "forgot-bias-term"), (fmt(bias + var), "added-bias-not-squared")],
        )
    cstar = R(mu ** 2, 1) / (mu ** 2 + R(s2, n))
    return Problem(
        prompt=(f"For $\\hat\\mu_c = c\\bar X$ with $n = {n}$ i.i.d. samples of mean $\\mu = {mu}$ and variance $\\sigma^2 = {s2}$, "
                "derive the $c$ that minimises the MSE and evaluate it. (On paper: why can't we use it in practice?)"),
        answer=fmt(cstar), kind="num",
        rubric=["MSE(c) = (c-1)²μ² + c²σ²/n", "d/dc = 2(c-1)μ² + 2cσ²/n = 0", "c* = μ²/(μ² + σ²/n)",
                "Depends on unknown μ — the bias-variance trade-off"],
        solution=f"$c^* = \\frac{{\\mu^2}}{{\\mu^2 + \\sigma^2/n}} = \\frac{{{mu ** 2}}}{{{mu ** 2} + {L(R(s2, n))}}} = {L(cstar)}$.",
        misconceptions=[(fmt(R(mu ** 2, mu ** 2 + s2)), "forgot-n")],
    )


@gen("ml.bias_variance")
def bias_variance_decomp(rng, level):
    if level == 1:
        f0 = rng.randint(-2, 6)
        preds = [f0 + rng.randint(-2, 4) for _ in range(4)]
        s2 = rng.choice([R(1, 2), sp.Integer(1), sp.Integer(2)])
        m = R(sum(preds), 4)
        bias2 = (m - f0) ** 2
        var = sum((p - m) ** 2 for p in preds) / 4
        err = bias2 + var + s2
        return Problem(
            prompt=(f"Four models trained on independent datasets predict ${preds}$ at a test input $x_0$, where the true "
                    f"$f(x_0) = {f0}$ and the label noise has variance $\\sigma^2 = {L(s2)}$. Treating the four as the whole "
                    "distribution of fits, compute the expected squared error $\\mathbb E[(y - \\hat f(x_0))^2]$."),
            answer=fmt(err), kind="num",
            rubric=["Average prediction f̄", "bias² = (f̄ - f(x₀))²", "variance = mean of (f̂ - f̄)² (divide by 4)",
                    "Add bias² + variance + σ²"],
            solution=f"$\\bar f = {L(m)}$, bias$^2 = {L(bias2)}$, variance $= {L(var)}$, total $= {L(err)}$.",
            misconceptions=[(fmt(bias2 + var), "forgot-irreducible-noise"),
                            (fmt(bias2 + sum((p - m) ** 2 for p in preds) / 3 + s2), "used-n-minus-1")],
        )
    if level == 2:
        k = rng.choice([3, 4, 5])
        f0 = rng.randint(0, 6)
        fn = [f0 + rng.randint(-2, 2) for _ in range(k)]
        s2 = rng.choice([1, 2, 4])
        bias = R(sum(fn), k) - f0
        err = bias ** 2 + R(s2, k) + s2
        return Problem(
            prompt=(f"$k$-NN regression with $k = {k}$ at $x_0$: the true function there is $f(x_0) = {f0}$; the {k} nearest "
                    f"training inputs have true values $f = {fn}$, and every label has independent noise of variance "
                    f"$\\sigma^2 = {s2}$. Compute the expected test error $\\mathbb E[(y_0 - \\hat f(x_0))^2]$ over training noise "
                    "and test noise (neighbours fixed)."),
            answer=fmt(err), kind="num",
            rubric=["f̂(x₀) = average of k noisy labels", "bias = mean(f of neighbours) - f(x₀)", "variance = σ²/k",
                    "Add irreducible σ²"],
            solution=f"bias $= {L(bias)}$, variance $= {s2}/{k}$, error $= {L(bias ** 2)} + {L(R(s2, k))} + {s2} = {L(err)}$.",
            misconceptions=[(fmt(bias ** 2 + R(s2, k)), "forgot-irreducible-noise"),
                            (fmt(bias ** 2 + 2 * s2), "variance-not-divided-by-k")],
        )
    n = rng.choice([4, 5, 9, 10, 19])
    s2 = rng.choice([1, 2, 3, 4])
    b = rng.choice([0, 1, R(1, 2)])
    err = s2 * (1 + R(1, n)) + b ** 2
    return Problem(
        prompt=(f"We predict a fresh observation $Y \\sim \\mathcal N(\\mu, {s2})$ with $\\hat Y = \\bar X + {L(b)}$, where "
                f"$\\bar X$ is the mean of $n = {n}$ independent training draws from the same distribution. Derive "
                "$\\mathbb E[(Y - \\hat Y)^2]$ and evaluate it."),
        answer=fmt(err), kind="num",
        rubric=["Y - Ŷ = (Y - μ) - (X̄ - μ) - b", "Independence: cross terms vanish", "= σ² + σ²/n + b²", "Evaluate"],
        solution=f"$\\sigma^2(1 + 1/n) + b^2 = {s2}(1 + 1/{n}) + {L(b ** 2)} = {L(err)}$.",
        misconceptions=[(fmt(R(s2, n) + b ** 2), "forgot-irreducible-noise"), (fmt(s2 + b ** 2), "forgot-estimation-variance")],
    )


@gen("ml.bias_variance")
def bias_variance_more(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        B = rng.choice([4, 5, 8, 10])
        s2 = rng.choice([2, 4, 8, 10, 20])
        b = rng.choice([sp.Integer(1), sp.Integer(2), R(1, 2), sp.Integer(-1)])
        noise = rng.choice([1, 2])
        err = b ** 2 + R(s2, B) + noise
        return Problem(
            prompt=(f"Bagging: an ensemble averages $B = {B}$ models whose errors are independent. Each model has bias ${L(b)}$ "
                    f"and variance ${s2}$ at a test input, and the label noise variance is ${noise}$. Compute the ensemble's "
                    "expected squared error there."),
            answer=fmt(err), kind="num",
            rubric=["Averaging keeps the bias", f"Variance of an average of {B} independent models = σ²/B",
                    "Error = bias² + variance + noise"],
            solution=f"${L(b ** 2)} + {s2}/{B} + {noise} = {L(err)}$.",
            misconceptions=[(fmt(b ** 2 + s2 + noise), "no-variance-reduction"),
                            (fmt(b ** 2 / B + R(s2, B) + noise), "bias-also-averaged")],
        )
    if level == 1:
        n, s2 = rng.choice([4, 5, 8, 10]), rng.choice([2, 3, 4, 6, 10])
        bias = -R(s2, n)
        return Problem(
            prompt=(f"The plug-in variance estimator $\\hat\\sigma^2 = \\frac1n\\sum_i (X_i - \\bar X)^2$ is used on $n = {n}$ i.i.d. "
                    f"samples with true variance $\\sigma^2 = {s2}$. Using $\\mathbb E\\left[\\sum_i (X_i-\\bar X)^2\\right] = (n-1)\\sigma^2$, "
                    "give the bias $\\mathbb E[\\hat\\sigma^2] - \\sigma^2$."),
            answer=fmt(bias), kind="num",
            rubric=["E[σ̂²] = (n-1)σ²/n", "Bias = E[σ̂²] - σ²", "= -σ²/n"],
            solution=f"$\\mathbb E[\\hat\\sigma^2] = \\frac{{{n - 1}}}{{{n}}}\\cdot {s2} = {L(R((n - 1) * s2, n))}$, bias $= {L(bias)}$.",
            misconceptions=[(fmt(-bias), "bias-sign"), (fmt(R((n - 1) * s2, n)), "reported-expectation-not-bias")],
        )
    if level == 2 and v == 0:
        n = rng.choice([4, 6, 8, 10])
        p = rng.choice([R(1, 4), R(1, 5), R(1, 3), R(3, 4), R(1, 10)])
        bias = R(1, n + 2) * (1 - 2 * p)
        var = n * p * (1 - p) / (n + 2) ** 2
        mse = bias ** 2 + var
        return Problem(
            prompt=(f"Estimating a coin's $P(\\text{{head}}) = p$ from $n = {n}$ tosses with $k$ heads, the Laplace estimator is "
                    f"$\\hat p = (k+1)/(n+2)$. For true $p = {L(p)}$, compute its bias, variance and MSE."),
            answer=[fmt(bias), fmt(var), fmt(mse)], kind="list", answer_hint="bias, variance, MSE   e.g. 1/12, 3/64, 31/576",
            rubric=["E[k] = np, Var(k) = np(1-p)", "Bias = (np+1)/(n+2) - p = (1-2p)/(n+2)", "Variance = np(1-p)/(n+2)²",
                    "MSE = bias² + variance"],
            solution=(f"bias $= {L(bias)}$, variance $= {L(var)}$, MSE $= {L(mse)}$ (the MLE $k/n$ has MSE "
                      f"${L(p * (1 - p) / n)}$)."),
            misconceptions=[([fmt(bias), fmt(var), fmt(var)], "forgot-bias-term"),
                            ([fmt(bias), fmt(p * (1 - p) / n), fmt(bias ** 2 + p * (1 - p) / n)], "used-mle-variance")],
        )
    if level == 2:
        B = rng.choice([2, 4, 5, 10])
        s2 = rng.choice([4, 8, 10, 20])
        rho = rng.choice([R(1, 4), R(1, 2), R(1, 5), R(3, 10)])
        var = rho * s2 + (1 - rho) * s2 / B
        return Problem(
            prompt=(f"Random-forest style averaging: $B = {B}$ trees each have prediction variance $\\sigma^2 = {s2}$ and every "
                    f"pair of trees has correlation $\\rho = {L(rho)}$. Expand $\\mathrm{{Var}}\\left(\\frac1B\\sum_b T_b\\right)$ "
                    "and evaluate it."),
            answer=fmt(var), kind="num",
            rubric=["Var(ΣT) = Σ Var(T_b) + Σ_{b≠c} Cov(T_b, T_c)", "= Bσ² + B(B-1)ρσ²", "Divide by B²",
                    "= ρσ² + (1-ρ)σ²/B"],
            solution=(f"$\\frac{{1}}{{B^2}}\\left(B\\sigma^2 + B(B-1)\\rho\\sigma^2\\right) = {L(rho)}\\cdot{s2} + "
                      f"{L(1 - rho)}\\cdot\\frac{{{s2}}}{{{B}}} = {L(var)}$."),
            misconceptions=[(fmt(R(s2, B)), "assumed-independent"), (fmt(rho * s2), "only-correlated-part")],
        )
    if v == 0:
        while True:
            s2 = rng.choice([4, 8, 10, 20])
            rho = rng.choice([R(1, 10), R(1, 5), R(1, 4)])
            T = R(rng.randint(int(10 * rho * s2) + 1, 5 * s2), 10)
            Bx = (1 - rho) * s2 / (T - rho * s2)
            B = sp.ceiling(Bx)
            if 2 <= B <= 40:
                break
        return Problem(
            prompt=(f"Trees in an ensemble each have variance $\\sigma^2 = {s2}$ with pairwise correlation $\\rho = {L(rho)}$. "
                    f"Derive the variance of the average of $B$ trees, its limit as $B\\to\\infty$, and the **smallest** $B$ "
                    f"for which the ensemble variance is at most ${float(T):g}$."),
            answer=str(B), kind="num", answer_hint="an integer",
            rubric=["Var = ρσ² + (1-ρ)σ²/B", "Limit ρσ²: correlation caps the gain", "Solve (1-ρ)σ²/B ≤ T - ρσ²",
                    "Round B up"],
            solution=(f"Limit $\\rho\\sigma^2 = {L(rho * s2)}$. Need $B \\ge \\frac{{(1-\\rho)\\sigma^2}}{{T - \\rho\\sigma^2}} = "
                      f"{L(Bx)}$, so $B = {B}$."),
            misconceptions=[(str(sp.ceiling(R(s2, 1) / T)), "ignored-correlation"), (str(sp.floor(Bx)), "rounded-down")],
        )
    n = rng.choice([4, 5, 9, 10, 20])
    c = R(1, n + 1)
    return Problem(
        prompt=(f"For $n = {n}$ i.i.d. Gaussian samples let $S = \\sum_i (X_i - \\bar X)^2$, with $\\mathbb E[S] = (n-1)\\sigma^2$ "
                "and $\\mathrm{Var}(S) = 2(n-1)\\sigma^4$. Among estimators $cS$ of $\\sigma^2$, derive the $c$ minimising the MSE."),
        answer=fmt(c), kind="num",
        rubric=["MSE(c) = c²·2(n-1)σ⁴ + (c(n-1) - 1)²σ⁴", "d/dc: 4c(n-1) + 2(n-1)(c(n-1) - 1) = 0",
                "Divide by 2(n-1): c(n+1) = 1", "c* = 1/(n+1)"],
        solution=(f"$\\frac{{d}}{{dc}}$: $4c(n-1) + 2(n-1)(c(n-1)-1) = 0 \\Rightarrow c^* = \\frac{{1}}{{n+1}} = {L(c)}$ — "
                  "smaller than both the unbiased $1/(n-1)$ and the MLE $1/n$."),
        misconceptions=[(fmt(R(1, n - 1)), "unbiased-is-mse-optimal"), (fmt(R(1, n)), "used-mle")],
    )


# --------------------------------------------------------------------------- regularisation & CV
@gen("ml.regularization_cv")
def regularization_cv(rng, level):
    if level == 1:
        xs = [nz(rng, -3, 3) for _ in range(3)]
        ys = [rng.randint(-4, 6) for _ in range(3)]
        lam = rng.randint(1, 5)
        sxy, sxx = sum(a * b for a, b in zip(xs, ys)), sum(a * a for a in xs)
        w = R(sxy, sxx + lam)
        return Problem(
            prompt=(f"One-feature ridge regression without intercept: minimise $\\sum_i (y_i - w x_i)^2 + {lam} w^2$ on "
                    f"{pts(xs, ys)}. Give $w$."),
            answer=fmt(w), kind="num",
            rubric=["Derivative: -2Σxᵢ(yᵢ - wxᵢ) + 2λw = 0", "w = Σxy / (Σx² + λ)", "Evaluate"],
            solution=f"$w = \\frac{{\\sum xy}}{{\\sum x^2 + \\lambda}} = \\frac{{{sxy}}}{{{sxx} + {lam}}} = {L(w)}$.",
            misconceptions=[(fmt(R(sxy, sxx)), "forgot-penalty"), (fmt(R(sxy, sxx + 2 * lam)), "doubled-lambda")],
        )
    if level == 2:
        while True:
            X = sp.Matrix([[rng.randint(-2, 2) for _ in range(2)] for _ in range(3)])
            if X.rank() == 2 and X[:, 0] != X[:, 1]:
                break
        y = sp.Matrix([rng.randint(-3, 4) for _ in range(3)])
        lam = rng.choice([1, 2])
        A = X.T * X + lam * sp.eye(2)
        w = A.inv() * (X.T * y)
        w_ols = (X.T * X).inv() * (X.T * y)
        return Problem(
            prompt=(f"Ridge regression (no intercept, penalty $\\lambda\\|w\\|^2$ with $\\lambda = {lam}$): $X = {L(X)}$, "
                    f"$y = {L(y)}$. Compute $w = (X^\\top X + \\lambda I)^{{-1}} X^\\top y$."),
            answer=[fmt(v) for v in w], kind="vector", answer_hint="w1, w2   e.g. [2/9, -1/3]",
            rubric=["XᵀX", "Add λ to the diagonal", "Xᵀy", "Solve the 2×2 system"],
            solution=f"$X^\\top X + {lam}I = {L(A)}$, $X^\\top y = {L(X.T * y)}$, $w = {L(w)}$ (OLS would give ${L(w_ols)}$).",
            misconceptions=[([fmt(v) for v in w_ols], "forgot-penalty")],
        )
    n = rng.choice([4, 5])
    while True:
        ys = [rng.randint(0, 10) for _ in range(n)]
        if len(set(ys)) > 1:
            break
    m = R(sum(ys), n)
    ss = sum((v - m) ** 2 for v in ys)
    cv = R(n, n - 1) ** 2 * ss / n
    return Problem(
        prompt=(f"The 'model' predicts the mean of its training labels. Compute its leave-one-out cross-validation MSE on "
                f"$y = {ys}$. (On paper: show that the held-out residual is $\\frac{{n}}{{n-1}}(y_i - \\bar y)$.)"),
        answer=fmt(cv), kind="num",
        rubric=["For each i, the mean of the other n-1 labels", "Residual yᵢ - ȳ₋ᵢ = n/(n-1)(yᵢ - ȳ)", "Square and average",
                "Compare with the (smaller) training MSE"],
        solution=(f"$\\bar y = {L(m)}$, $\\sum(y_i-\\bar y)^2 = {L(ss)}$; LOOCV $= \\left(\\frac{{{n}}}{{{n - 1}}}\\right)^2\\frac{{{L(ss)}}}{{{n}}} "
                  f"= {L(cv)}$ vs training MSE ${L(ss / n)}$."),
        misconceptions=[(fmt(ss / n), "used-training-error"), (fmt(ss / (n - 1)), "reported-sample-variance")],
    )


def _soft(z, lam):
    return sp.sign(z) * (abs(z) - lam)


@gen("ml.regularization_cv")
def regularization_more(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        lam = rng.choice([sp.Integer(1), sp.Integer(2), sp.Integer(3), R(1, 2)])
        while True:
            z = R(nz(rng, -18, 18), 2)
            if abs(z) > lam:
                break
        w = _soft(z, lam)
        return Problem(
            prompt=(f"Lasso proximal step: minimise $\\tfrac12 (w - z)^2 + \\lambda |w|$ over $w$ for $z = {L(z)}$ and "
                    f"$\\lambda = {L(lam)}$."),
            answer=fmt(w), kind="num",
            rubric=["For w ≠ 0: w - z + λ·sign(w) = 0", "Soft-thresholding: w = sign(z)·max(|z| - λ, 0)", "Evaluate"],
            solution=f"$|z| = {L(abs(z))} > \\lambda$, so $w = \\operatorname{{sign}}(z)(|z| - \\lambda) = {L(w)}$.",
            misconceptions=[(fmt(z / (1 + lam)), "used-ridge-shrinkage"), (fmt(z - lam), "ignored-sign")],
        )
    if level == 1:
        lams = sorted(rng.sample([R(1, 10), sp.Integer(1), sp.Integer(10), R(1, 100), sp.Integer(100)], 3))
        while True:
            errs = [[rng.randint(8, 30) for _ in range(4)] for _ in lams]
            means = [R(sum(e), 4) for e in errs]
            best = min(range(3), key=lambda i: means[i])
            lone = min(range(3), key=lambda i: min(errs[i]))
            if means.count(means[best]) == 1 and lone != best:
                break
        rows = " \\\\ ".join(f"{L(l)} & " + " & ".join(map(str, e)) for l, e in zip(lams, errs))
        tab = ("$$\\begin{array}{c|cccc}\\lambda & \\text{fold 1} & \\text{fold 2} & \\text{fold 3} & \\text{fold 4} \\\\ \\hline "
               + rows + "\\end{array}$$")
        return Problem(
            prompt=("Four-fold cross-validation errors (validation MSE per fold) for three ridge penalties:\n" + tab +
                    "\nWhich $\\lambda$ does CV select, and what is its CV error (mean over folds)?"),
            answer=[fmt(lams[best]), fmt(means[best])], kind="list", answer_hint="lambda, CV error   e.g. 1/10, 63/4",
            rubric=["CV error = average of the fold errors", "Compute it for each λ", "Pick the smallest mean"],
            solution=("Means: " + ", ".join(f"$\\lambda={L(l)}: {L(m)}$" for l, m in zip(lams, means)) +
                      f" → $\\lambda = {L(lams[best])}$."),
            misconceptions=[([fmt(lams[lone]), fmt(means[lone])], "picked-best-single-fold")],
        )
    if level == 2 and v == 0:
        ds = rng.sample([1, 4, 9, 16], rng.choice([2, 3]))
        lam = rng.choice([1, 2, 4])
        df = sum(R(d, d + lam) for d in ds)
        D = sp.diag(*ds)
        return Problem(
            prompt=(f"Ridge regression has hat matrix $H = X(X^\\top X + \\lambda I)^{{-1}}X^\\top$, and its effective degrees of "
                    f"freedom is $\\operatorname{{tr}} H$. If $X^\\top X = {L(D)}$ and $\\lambda = {lam}$, compute $\\operatorname{{tr}} H$."),
            answer=fmt(df), kind="num",
            rubric=["Cyclic trace: tr H = tr((XᵀX + λI)⁻¹ XᵀX)", "Diagonal ⇒ invert entrywise", "Sum d/(d + λ)",
                    "Check: between 0 and the number of features"],
            solution=(f"$\\operatorname{{tr}} H = \\sum_j \\frac{{d_j}}{{d_j + {lam}}} = " +
                      " + ".join(f"\\frac{{{d}}}{{{d + lam}}}" for d in ds) + f" = {L(df)}$."),
            misconceptions=[(str(len(ds)), "ignored-penalty"), (fmt(sum(R(lam, d + lam) for d in ds)), "inverted-shrinkage-factor")],
        )
    if level == 2:
        lam = rng.choice([1, 2, 3, 4])
        while True:
            xs = [nz(rng, -3, 3) for _ in range(3)]
            ys = [rng.randint(-5, 7) for _ in range(3)]
            sxy, sxx = sum(a * b for a, b in zip(xs, ys)), sum(a * a for a in xs)
            if abs(sxy) > lam:
                break
        w = _soft(sp.Integer(sxy), lam) / sxx
        return Problem(
            prompt=(f"One-feature lasso without intercept: minimise $\\tfrac12\\sum_i (y_i - w x_i)^2 + \\lambda|w|$ with "
                    f"$\\lambda = {lam}$ on {pts(xs, ys)}. Give $w$."),
            answer=fmt(w), kind="num",
            rubric=["Σxy and Σx²", "Stationarity for w ≠ 0: -Σx(y - wx) + λ·sign(w) = 0", "w = sign(Σxy)(|Σxy| - λ)/Σx²",
                    "Check |Σxy| > λ (else w = 0)"],
            solution=f"$\\sum xy = {sxy}$, $\\sum x^2 = {sxx}$, $w = \\frac{{{L(_soft(sp.Integer(sxy), lam))}}}{{{sxx}}} = {L(w)}$.",
            misconceptions=[(fmt(R(sxy, sxx + lam)), "used-ridge-formula"), (fmt(R(sxy, sxx)), "forgot-penalty")],
        )
    if v == 0:
        s2, t2 = rng.choice([1, 2, 4]), rng.choice([sp.Integer(1), sp.Integer(2), R(1, 2)])
        while True:
            xs = [nz(rng, -3, 3) for _ in range(3)]
            ys = [rng.randint(-5, 7) for _ in range(3)]
            sxy, sxx = sum(a * b for a, b in zip(xs, ys)), sum(a * a for a in xs)
            if sxy:
                break
        lam = s2 / t2
        w = sxy / (sxx + lam)
        return Problem(
            prompt=(f"Bayesian view of ridge: $y_i = w x_i + \\varepsilon_i$, $\\varepsilon_i \\sim \\mathcal N(0, {s2})$, prior "
                    f"$w \\sim \\mathcal N(0, {L(t2)})$. Derive the MAP estimate (which $\\lambda$ does it correspond to?) and "
                    f"evaluate it on {pts(xs, ys)}."),
            answer=fmt(w), kind="num",
            rubric=["-log posterior = Σ(y - wx)²/(2σ²) + w²/(2τ²) + const", "Multiply by 2σ²: ridge with λ = σ²/τ²",
                    "w_MAP = Σxy / (Σx² + σ²/τ²)", "Evaluate"],
            solution=(f"$\\lambda = \\sigma^2/\\tau^2 = {L(lam)}$, $w_{{\\text{{MAP}}}} = \\frac{{{sxy}}}{{{sxx} + {L(lam)}}} = {L(w)}$."),
            misconceptions=[(fmt(sxy / (sxx + t2 / s2)), "inverted-variance-ratio"), (fmt(R(sxy, sxx)), "used-mle")],
        )
    w = rng.choice([sp.Integer(1), sp.Integer(2), sp.Integer(-1), sp.Integer(3), R(1, 2)])
    s2 = rng.choice([1, 2, 4, 9])
    s = rng.choice([4, 5, 10])
    lam = s2 / w ** 2
    return Problem(
        prompt=(f"One-feature ridge: $\\hat w_\\lambda = \\frac{{\\sum x_i y_i}}{{s + \\lambda}}$ with fixed $s = \\sum x_i^2 = {s}$. "
                f"The data follow $y_i = w x_i + \\varepsilon_i$ with true $w = {L(w)}$ and noise variance $\\sigma^2 = {s2}$. "
                "Derive the bias and variance of $\\hat w_\\lambda$ and the $\\lambda$ that minimises its MSE."),
        answer=fmt(lam), kind="num",
        rubric=["E[ŵ] = s w/(s + λ) ⇒ bias = -λw/(s + λ)", "Var(ŵ) = σ² s/(s + λ)²", "MSE = (λ²w² + σ²s)/(s + λ)²",
                "d/dλ = 0 ⇒ λ* = σ²/w² (independent of s)"],
        solution=(f"$\\mathrm{{MSE}}(\\lambda) = \\frac{{\\lambda^2 w^2 + \\sigma^2 s}}{{(s+\\lambda)^2}}$; setting the derivative to zero "
                  f"gives $\\lambda w^2 s = \\sigma^2 s$, so $\\lambda^* = \\sigma^2/w^2 = {L(lam)}$."),
        misconceptions=[(fmt(w ** 2 / s2), "inverted-ratio"), (fmt(R(s2, s)), "divided-by-s")],
    )


# --------------------------------------------------------------------------- metrics
@gen("ml.metrics")
def confusion_metrics(rng, level):
    tp, fp, fn, tn = rng.randint(5, 60), rng.randint(1, 30), rng.randint(1, 30), rng.randint(20, 200)
    prec, rec = R(tp, tp + fp), R(tp, tp + fn)
    table = ("$$\\begin{array}{c|cc} & \\text{pred }+ & \\text{pred }- \\\\ \\hline \\text{actual }+ & "
             f"{tp} & {fn} \\\\ \\text{{actual }}- & {fp} & {tn} \\end{{array}}$$")
    if level == 1:
        return Problem(
            prompt="Confusion matrix of a classifier:\n" + table + "\nGive precision and recall.",
            answer=[fmt(prec), fmt(rec)], kind="list", answer_hint="precision, recall   e.g. 3/4, 2/3",
            rubric=["TP, FP, FN from the table", "Precision = TP/(TP+FP)", "Recall = TP/(TP+FN)"],
            solution=f"Precision $= {tp}/{tp + fp} = {L(prec)}$, recall $= {tp}/{tp + fn} = {L(rec)}$.",
            misconceptions=[([fmt(rec), fmt(prec)], "swapped-precision-recall")],
        )
    if level == 2:
        f1 = 2 * prec * rec / (prec + rec)
        return Problem(
            prompt="Confusion matrix:\n" + table + "\nCompute the $F_1$ score exactly. (On paper: also the accuracy.)",
            answer=fmt(f1), kind="num",
            rubric=["Precision and recall", "F₁ = harmonic mean = 2PR/(P+R) = 2TP/(2TP+FP+FN)", "Simplify"],
            solution=(f"$F_1 = \\frac{{2\\cdot{tp}}}{{2\\cdot{tp} + {fp} + {fn}}} = {L(f1)}$; accuracy "
                      f"$= {L(R(tp + tn, tp + fp + fn + tn))}$."),
            misconceptions=[(fmt((prec + rec) / 2), "arithmetic-mean-not-harmonic"),
                            (fmt(R(tp + tn, tp + fp + fn + tn)), "reported-accuracy")],
        )
    sens = rng.choice([R(9, 10), R(95, 100), R(99, 100)])
    spec = rng.choice([R(9, 10), R(95, 100), R(99, 100)])
    prev = rng.choice([R(1, 1000), R(1, 100), R(1, 50), R(1, 10)])
    ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    return Problem(
        prompt=(f"A screening model has recall (sensitivity) ${float(sens):g}$ and specificity ${float(spec):g}$. "
                f"The condition has prevalence ${L(prev)}$ in the deployed population. What precision (PPV) will be observed?"),
        answer=fmt(ppv), kind="num", tol=1e-4, answer_hint="exact fraction or a decimal to 4 places",
        rubric=["P(+|pos) = sensitivity, P(+|neg) = 1 - specificity", "Bayes: PPV = sens·π / (sens·π + (1-spec)(1-π))",
                "Evaluate — note how low prevalence crushes precision"],
        solution=f"PPV $= \\frac{{{L(sens)}\\cdot{L(prev)}}}{{{L(sens)}\\cdot{L(prev)} + {L(1 - spec)}\\cdot{L(1 - prev)}}} = {L(ppv)} \\approx {d4(ppv)}$.",
        misconceptions=[(fmt(sens), "confused-ppv-with-sensitivity"), (fmt(spec), "confused-ppv-with-specificity")],
    )


@gen("ml.metrics")
def roc_auc(rng, level):
    if level in (1, 2):
        n_pos, n_neg = rng.choice([3, 4]), rng.choice([3, 4])
        scores = rng.sample(range(5, 96, 5), n_pos + n_neg)
        pos, neg = sorted(scores[:n_pos], reverse=True), sorted(scores[n_pos:], reverse=True)
        items = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], reverse=True)
        listing = ", ".join(f"$({s / 100:.2f}, {y})$" for s, y in items)
        if level == 1:
            t = rng.choice([s for s, _ in items[1:-1]])
            tpr = R(sum(1 for s in pos if s >= t), n_pos)
            fpr = R(sum(1 for s in neg if s >= t), n_neg)
            return Problem(
                prompt=(f"Classifier scores with true labels (score, y): {listing}. Predict positive when score $\\ge {t / 100:.2f}$. "
                        "Give the true-positive rate and false-positive rate."),
                answer=[fmt(tpr), fmt(fpr)], kind="list", answer_hint="TPR, FPR   e.g. 2/3, 1/4",
                rubric=["Mark predictions at the threshold", "TPR = TP / #positives", "FPR = FP / #negatives"],
                solution=f"TPR $= {L(tpr)}$, FPR $= {L(fpr)}$ — one point of the ROC curve.",
                misconceptions=[([fmt(fpr), fmt(tpr)], "swapped-tpr-fpr")],
            )
        good = sum(1 for p in pos for q in neg if p > q)
        auc = R(good, n_pos * n_neg)
        return Problem(
            prompt=f"Scores with labels (score, y): {listing}. Compute the ROC AUC exactly.",
            answer=fmt(auc), kind="num",
            rubric=["AUC = P(random positive scores above random negative)", f"Count correctly ordered pairs out of {n_pos * n_neg}",
                    "(Equivalently the area under the step ROC curve)"],
            solution=f"{good} of {n_pos * n_neg} positive–negative pairs are ordered correctly: AUC $= {L(auc)}$.",
            misconceptions=[(fmt(1 - auc), "reversed-ranking")],
        )
    nb = rng.choice([2, 3])
    bins = []
    for _ in range(nb):
        nbin = rng.choice([10, 20, 25, 40, 50])
        conf = R(rng.randint(5, 19), 20)
        acc = R(rng.randint(0, nbin), nbin)
        bins.append((nbin, conf, acc))
    N = sum(b[0] for b in bins)
    ece = sum(R(nbin, N) * abs(acc - conf) for nbin, conf, acc in bins)
    unw = sum(abs(acc - conf) for _, conf, acc in bins) / nb
    rows = " \\\\ ".join(f"{i + 1} & {nbin} & {L(conf)} & {L(acc * nbin)}" for i, (nbin, conf, acc) in enumerate(bins))
    tab = ("$$\\begin{array}{c|ccc}\\text{bin} & \\text{count} & \\text{mean confidence} & \\text{\\# correct} \\\\ \\hline "
           + rows + "\\end{array}$$")
    return Problem(
        prompt="Reliability-diagram bins of a classifier:\n" + tab + "\nCompute the expected calibration error "
               "$\\mathrm{ECE} = \\sum_b \\frac{n_b}{N}\\,|\\mathrm{acc}_b - \\mathrm{conf}_b|$.",
        answer=fmt(ece), kind="num", tol=1e-4, answer_hint="exact fraction or a decimal to 4 places",
        rubric=["Accuracy per bin = #correct / count", "Gap |acc - conf| per bin", "Weight by count/N", "Sum"],
        solution=("Gaps: " + ", ".join(f"${L(abs(acc - conf))}$" for _, conf, acc in bins) + f"; ECE $= {L(ece)} \\approx {d4(ece)}$."),
        misconceptions=[(fmt(unw), "unweighted-bins")],
    )


@gen("ml.metrics")
def metrics_more(rng, level):
    if level == 1:
        tp, fp, fn, tn = rng.randint(5, 60), rng.randint(1, 30), rng.randint(1, 30), rng.randint(20, 200)
        acc, spec = R(tp + tn, tp + fp + fn + tn), R(tn, tn + fp)
        table = ("$$\\begin{array}{c|cc} & \\text{pred }+ & \\text{pred }- \\\\ \\hline \\text{actual }+ & "
                 f"{tp} & {fn} \\\\ \\text{{actual }}- & {fp} & {tn} \\end{{array}}$$")
        return Problem(
            prompt="On a held-out test set a classifier produced\n" + table + "\nGive its accuracy and its specificity (true-negative rate).",
            answer=[fmt(acc), fmt(spec)], kind="list", answer_hint="accuracy, specificity   e.g. 9/10, 4/5",
            rubric=["Accuracy = (TP + TN)/total", "Specificity = TN/(TN + FP) — over actual negatives"],
            solution=f"Accuracy $= {tp + tn}/{tp + fp + fn + tn} = {L(acc)}$, specificity $= {tn}/{tn + fp} = {L(spec)}$.",
            misconceptions=[([fmt(acc), fmt(R(tp, tp + fn))], "confused-specificity-with-recall"),
                            ([fmt(acc), fmt(R(tn, tn + fn))], "specificity-wrong-denominator")],
        )
    if level == 2:
        while True:
            M = [[rng.randint(5, 15) if i == j else rng.randint(0, 5) for j in range(3)] for i in range(3)]
            rec = [R(M[i][i], sum(M[i])) for i in range(3)]
            prc = [R(M[i][i], sum(M[k][i] for k in range(3))) for i in range(3)]
            if sum(rec) != sum(prc):
                break
        macro = sum(rec) / 3
        acc = R(sum(M[i][i] for i in range(3)), sum(map(sum, M)))
        rows = " \\\\ ".join(f"\\text{{actual {'ABC'[i]}}} & " + " & ".join(map(str, M[i])) for i in range(3))
        tab = ("$$\\begin{array}{c|ccc} & \\text{pred A} & \\text{pred B} & \\text{pred C} \\\\ \\hline " + rows + "\\end{array}$$")
        return Problem(
            prompt="Three-class confusion matrix (rows = actual, columns = predicted):\n" + tab +
                   "\nCompute the macro-averaged recall and the overall accuracy.",
            answer=[fmt(macro), fmt(acc)], kind="list", tol=1e-4,
            answer_hint="macro recall, accuracy (fractions or decimals to 4 places)   e.g. 7/10, 0.7333",
            rubric=["Recall of class c = diagonal / row sum", "Macro average = plain mean of the 3 recalls",
                    "Accuracy = trace / total (equals micro-averaged recall)"],
            solution=("Recalls: " + ", ".join(f"${L(r)}$" for r in rec) + f"; macro $= {L(macro)}$; accuracy $= {L(acc)}$."),
            misconceptions=[([fmt(sum(prc) / 3), fmt(acc)], "macro-precision-not-recall")],
        )
    while True:
        pi = rng.choice([R(1, 10), R(1, 5), R(1, 4), R(1, 20)])
        cfn, cfp = rng.choice([5, 10, 20]), rng.choice([1, 2])
        roc = sorted({(R(rng.randint(6, 19), 20), R(rng.randint(1, 12), 20)) for _ in range(3)})
        if len(roc) < 3 or any(t <= f for t, f in roc):
            continue
        costs = [pi * (1 - t) * cfn + (1 - pi) * f * cfp for t, f in roc]
        best = min(range(3), key=lambda i: costs[i])
        you = max(range(3), key=lambda i: roc[i][0] - roc[i][1])
        if costs.count(costs[best]) == 1 and you != best:
            break
    return Problem(
        prompt=(f"Choosing an operating point: three thresholds give (TPR, FPR) = " +
                ", ".join(f"$({L(t)},\\ {L(f)})$" for t, f in roc) +
                f". Prevalence is ${L(pi)}$, a missed positive costs ${cfn}$ and a false alarm costs ${cfp}$. Derive the expected "
                "cost per case of each point and give the minimum."),
        answer=fmt(costs[best]), kind="num", answer_hint="exact fraction or a decimal to 4 places", tol=1e-4,
        rubric=["FN rate = π(1 - TPR), FP rate = (1 - π)·FPR", "Cost = π(1-TPR)c_FN + (1-π)FPR·c_FP",
                "Evaluate for each point", "Pick the minimum (not necessarily the max of TPR - FPR)"],
        solution=("Costs: " + ", ".join(f"${L(c)}$" for c in costs) + f"; minimum $= {L(costs[best])} \\approx {d4(costs[best])}$."),
        misconceptions=[(fmt(costs[you]), "used-youden-not-cost")],
    )


# --------------------------------------------------------------------------- naive Bayes
@gen("ml.naive_bayes")
def naive_bayes(rng, level):
    if level == 1:
        pi = rng.choice([R(1, 4), R(1, 3), R(2, 5), R(1, 2)])
        p1 = [R(rng.randint(1, 9), 10) for _ in range(2)]
        p0 = [R(rng.randint(1, 9), 10) for _ in range(2)]
        x = [rng.randint(0, 1), rng.randint(0, 1)]
        lik = lambda ps: sp.Mul(*[(p if xi else 1 - p) for p, xi in zip(ps, x)])  # noqa: E731
        a, b = pi * lik(p1), (1 - pi) * lik(p0)
        post = a / (a + b)
        noprior = lik(p1) / (lik(p1) + lik(p0))
        return Problem(
            prompt=(f"Spam filter with two binary word features. Prior $P(\\text{{spam}}) = {L(pi)}$. "
                    f"$P(x_1=1\\mid\\text{{spam}}) = {L(p1[0])},\\ P(x_2=1\\mid\\text{{spam}}) = {L(p1[1])}$; "
                    f"$P(x_1=1\\mid\\text{{ham}}) = {L(p0[0])},\\ P(x_2=1\\mid\\text{{ham}}) = {L(p0[1])}$. "
                    f"Under the naive Bayes assumption, compute $P(\\text{{spam}}\\mid x_1 = {x[0]}, x_2 = {x[1]})$."),
            answer=fmt(post), kind="num",
            rubric=["Class-conditional likelihood = product of per-feature terms", "Use 1-p for features equal to 0",
                    "Multiply by priors", "Normalise"],
            solution=f"spam: ${L(pi)}\\cdot{L(lik(p1))} = {L(a)}$; ham: ${L(1 - pi)}\\cdot{L(lik(p0))} = {L(b)}$; posterior $= {L(post)}$.",
            misconceptions=[(fmt(noprior), "ignored-prior")],
        )
    if level == 2:
        while True:
            data = [(rng.randint(0, 1), rng.randint(0, 1), rng.randint(0, 1)) for _ in range(8)]
            n1 = sum(d[2] for d in data)
            if 2 <= n1 <= 6:
                break
        q = (rng.randint(0, 1), rng.randint(0, 1))

        def score(c, alpha):
            nc = sum(1 for d in data if d[2] == c)
            s = R(nc, 8)
            for j in range(2):
                cnt = sum(1 for d in data if d[2] == c and d[j] == q[j])
                s *= R(cnt + alpha, nc + 2 * alpha)
            return s

        a, b = score(1, 1), score(0, 1)
        post = a / (a + b)
        ua, ub = score(1, 0), score(0, 0)
        mis = [(fmt(ua / (ua + ub)), "no-laplace-smoothing")] if ua + ub != 0 else []
        rows = " \\\\ ".join(f"{d[0]} & {d[1]} & {d[2]}" for d in data)
        return Problem(
            prompt=("Training data (binary features $x_1, x_2$ and class $y$):\n"
                    "$$\\begin{array}{ccc} x_1 & x_2 & y \\\\ \\hline " + rows + "\\end{array}$$\n"
                    "Fit Bernoulli naive Bayes with **Laplace (add-one) smoothing** for $P(x_j\\mid y)$ and the plain MLE for "
                    f"$P(y)$. Compute $P(y = 1 \\mid x_1 = {q[0]}, x_2 = {q[1]})$."),
            answer=fmt(post), kind="num",
            rubric=["Priors from class counts", "P(x_j = v | y) = (count + 1)/(n_y + 2)", "Multiply per class", "Normalise"],
            solution=f"Unnormalised: $y=1$: ${L(a)}$, $y=0$: ${L(b)}$; posterior $= {L(post)}$.",
            misconceptions=mis,
        )
    mu0, mu1 = rng.randint(-2, 2), rng.randint(3, 7)
    s2 = rng.choice([1, 2, 4])
    pi1 = rng.choice([R(1, 4), R(1, 3), R(2, 3), R(1, 5)])
    xstar = R(mu0 + mu1, 2) + s2 * sp.log((1 - pi1) / pi1) / (mu1 - mu0)
    return Problem(
        prompt=(f"Gaussian generative classifier in 1-D: $x\\mid y=0 \\sim \\mathcal N({mu0}, {s2})$, "
                f"$x\\mid y=1 \\sim \\mathcal N({mu1}, {s2})$, prior $P(y=1) = {L(pi1)}$. Derive the decision threshold $x^*$ "
                "where both posteriors equal $1/2$."),
        answer=fmt(xstar), kind="num", tol=TOL, answer_hint="exact like 5/2 + log(3)/2, or a decimal to 4 places",
        rubric=["Set π₁ N(x; μ₁, σ²) = π₀ N(x; μ₀, σ²)", "Take logs: the x² terms cancel (equal variances)",
                "Linear equation in x", "x* = (μ₀+μ₁)/2 + σ² ln(π₀/π₁)/(μ₁-μ₀)"],
        solution=f"$x^* = \\frac{{{mu0}+{mu1}}}{{2}} + \\frac{{{s2}\\ln({L((1 - pi1) / pi1)})}}{{{mu1 - mu0}}} = {L(xstar)} \\approx {d4(xstar)}$.",
        misconceptions=[(fmt(R(mu0 + mu1, 2)), "ignored-prior"),
                        (fmt(R(mu0 + mu1, 2) - s2 * sp.log((1 - pi1) / pi1) / (mu1 - mu0)), "prior-shift-wrong-direction")],
    )


@gen("ml.naive_bayes")
def naive_bayes_more(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        V, N, c = rng.choice([4, 5, 6, 10]), rng.randint(20, 60), rng.randint(0, 9)
        p = R(c + 1, N + V)
        return Problem(
            prompt=(f"Multinomial naive Bayes for text: the training documents of class *sports* contain ${N}$ word tokens in "
                    f"total, ${c}$ of which are 'goal'. The vocabulary has ${V}$ words. With Laplace (add-one) smoothing, "
                    "compute $P(\\text{goal}\\mid\\text{sports})$."),
            answer=fmt(p), kind="num",
            rubric=["Add 1 to the word count", "Add |V| to the class's token total", "Divide"],
            solution=f"$P = \\frac{{{c} + 1}}{{{N} + {V}}} = {L(p)}$.",
            misconceptions=[(fmt(R(c, N)), "no-laplace-smoothing"), (fmt(R(c + 1, N + 1)), "smoothing-wrong-denominator")],
        )
    if level == 1:
        pi = rng.choice([R(1, 5), R(1, 4), R(1, 3), R(1, 2)])
        r1, r2 = rng.choice([2, 3, 4, R(1, 2)]), rng.choice([2, 3, 5, R(1, 3)])
        o = pi / (1 - pi) * r1 * r2
        p = o / (1 + o)
        return Problem(
            prompt=(f"Naive Bayes in odds form: prior $P(\\text{{spam}}) = {L(pi)}$. Two observed words have likelihood ratios "
                    f"$\\frac{{P(w\\mid\\text{{spam}})}}{{P(w\\mid\\text{{ham}})}} = {L(sp.sympify(r1))}$ and ${L(sp.sympify(r2))}$. "
                    "Compute the posterior $P(\\text{spam}\\mid\\text{both words})$."),
            answer=fmt(p), kind="num",
            rubric=["Prior odds π/(1-π)", "Posterior odds = prior odds × product of likelihood ratios", "p = odds/(1 + odds)"],
            solution=f"Odds $= {L(pi / (1 - pi))}\\cdot {L(sp.sympify(r1))}\\cdot {L(sp.sympify(r2))} = {L(o)}$, $p = {L(p)}$.",
            misconceptions=[(fmt(r1 * r2 / (1 + r1 * r2)), "ignored-prior"), (fmt(o), "reported-odds-not-probability")],
        )
    if level == 2 and v == 0:
        while True:
            A = [rng.randint(0, 6) for _ in range(3)]
            B = [rng.randint(0, 6) for _ in range(3)]
            d = [rng.randint(0, 1) for _ in range(3)]
            d[rng.randrange(3)] += 1
            if 2 <= sum(d) <= 3 and A != B:
                pa = [R(a + 1, sum(A) + 3) for a in A]
                pb = [R(b + 1, sum(B) + 3) for b in B]
                sa = R(1, 2) * sp.Mul(*[q ** k for q, k in zip(pa, d)])
                sb = R(1, 2) * sp.Mul(*[q ** k for q, k in zip(pb, d)])
                if sa != sb:
                    break
        post = sa / (sa + sb)
        ua = sp.Mul(*[R(a, sum(A)) ** k for a, k in zip(A, d)]) if sum(A) else 0
        ub = sp.Mul(*[R(b, sum(B)) ** k for b, k in zip(B, d)]) if sum(B) else 0
        mis = [(fmt(ua / (ua + ub)), "no-laplace-smoothing")] if ua + ub != 0 else []
        doc = " ".join(w for w, k in zip(["red", "blue", "green"], d) for _ in range(k))
        return Problem(
            prompt=("Classify a short document with multinomial naive Bayes (equal priors, Laplace smoothing, vocabulary "
                    "{red, blue, green}). Training word counts — class A: " +
                    ", ".join(f"{w} {a}" for w, a in zip(["red", "blue", "green"], A)) + "; class B: " +
                    ", ".join(f"{w} {b}" for w, b in zip(["red", "blue", "green"], B)) +
                    f". Document: \"{doc}\". Compute $P(A\\mid\\text{{doc}})$."),
            answer=fmt(post), kind="num", tol=1e-4, answer_hint="exact fraction (e.g. 25/41) or a decimal to 4 places",
            rubric=["P(w|c) = (count + 1)/(total_c + 3)", "Likelihood = product over document tokens (repeats count twice)",
                    "Multiply by prior 1/2", "Normalise"],
            solution=(f"$P(\\cdot\\mid A) = {vtex(pa)}$, $P(\\cdot\\mid B) = {vtex(pb)}$; scores ${L(sa)}$ vs ${L(sb)}$, "
                      f"posterior $= {L(post)}$."),
            misconceptions=mis,
        )
    if level == 2:
        while True:
            m0 = [rng.randint(-2, 2), rng.randint(-2, 2)]
            m1 = [rng.randint(0, 4), rng.randint(0, 4)]
            x = [rng.randint(-1, 3), rng.randint(-1, 3)]
            llr = sum(R((x[j] - m0[j]) ** 2 - (x[j] - m1[j]) ** 2, 2) for j in range(2))
            if m0 != m1 and llr != 0 and abs(llr) <= 6:
                break
        p = 1 / (1 + sp.exp(-llr))
        return Problem(
            prompt=(f"Gaussian naive Bayes with two features, unit variances and equal priors. Class 0 means $({m0[0]}, {m0[1]})$, "
                    f"class 1 means $({m1[0]}, {m1[1]})$. Compute $P(y = 1\\mid x = ({x[0]}, {x[1]}))$."),
            answer=fmt(p), kind="num", tol=TOL, answer_hint=PROB_HINT,
            rubric=["Log-likelihood ratio = Σ_j [(x_j - μ0_j)² - (x_j - μ1_j)²]/2", "Normalising constants cancel",
                    "Posterior = σ(LLR) with equal priors"],
            solution=f"LLR $= {L(llr)}$, so $P = \\sigma({L(llr)}) = {L(p)} \\approx {d4(p)}$.",
            misconceptions=[(fmt(1 / (1 + sp.exp(-2 * llr))), "forgot-half-in-gaussian-exponent"),
                            (fmt(1 / (1 + sp.exp(llr))), "llr-sign-error")],
        )
    if v == 0:
        while True:
            pi = rng.choice([R(1, 4), R(1, 3), R(1, 2), R(2, 3)])
            p1 = [R(rng.randint(1, 9), 10) for _ in range(2)]
            p0 = [R(rng.randint(1, 9), 10) for _ in range(2)]
            ws = [sp.log(p1[j] * (1 - p0[j]) / (p0[j] * (1 - p1[j]))) for j in range(2)]
            w0 = sp.log(pi / (1 - pi)) + sum(sp.log((1 - p1[j]) / (1 - p0[j])) for j in range(2))
            if all(p1[j] != p0[j] for j in range(2)) and abs(float(w0)) > 0.05:
                break
        w0 = sp.expand_log(w0, force=True)
        return Problem(
            prompt=(f"Show that Bernoulli naive Bayes is a linear classifier: $\\ln\\frac{{P(y=1\\mid x)}}{{P(y=0\\mid x)}} = w_0 + "
                    f"w_1x_1 + w_2x_2$. With $P(y=1) = {L(pi)}$, $P(x_j=1\\mid y=1) = ({L(p1[0])}, {L(p1[1])})$ and "
                    f"$P(x_j=1\\mid y=0) = ({L(p0[0])}, {L(p0[1])})$, give $w_0, w_1, w_2$."),
            answer=[fmt(w0), fmt(ws[0]), fmt(ws[1])], kind="list", tol=TOL,
            answer_hint="w0, w1, w2 exact like log(7/3) or decimals to 4 places",
            rubric=["P(x_j|y) = θ^{x_j}(1-θ)^{1-x_j}", "Log-ratio of each factor is linear in x_j",
                    "w_j = ln[θ₁ⱼ(1-θ₀ⱼ)/(θ₀ⱼ(1-θ₁ⱼ))]", "w₀ = ln(π/(1-π)) + Σ ln[(1-θ₁ⱼ)/(1-θ₀ⱼ)]"],
            solution=(f"$w_0 \\approx {d4(w0)}$, $w_1 = \\ln {L(sp.exp(ws[0]))} \\approx {d4(ws[0])}$, "
                      f"$w_2 = \\ln {L(sp.exp(ws[1]))} \\approx {d4(ws[1])}$."),
            misconceptions=[([fmt(sp.log(pi / (1 - pi))), fmt(sp.log(p1[0] / p0[0])), fmt(sp.log(p1[1] / p0[1]))],
                             "ignored-absent-feature-term")],
        )
    while True:
        a = [rng.randint(-3, 3) for _ in range(3)]
        b = [rng.randint(3, 9) for _ in range(3)]
        ma, mb = R(sum(a), 3), R(sum(b), 3)
        ss = sum((t - ma) ** 2 for t in a) + sum((t - mb) ** 2 for t in b)
        if ss:
            break
    s2 = ss / 6
    allx = a + b
    m = R(sum(allx), 6)
    return Problem(
        prompt=(f"1-D LDA: each class has its own mean but both share one variance $\\sigma^2$. Derive the MLE of $\\sigma^2$ and "
                f"evaluate it for class 0: ${a}$, class 1: ${b}$."),
        answer=fmt(s2), kind="num",
        rubric=["Log-likelihood: -n/2 ln σ² - Σ_c Σ_i (x_i - μ_c)²/(2σ²)", "μ̂_c = class means",
                "∂/∂σ² = 0 ⇒ σ̂² = (1/n) Σ_c Σ_i (x_i - μ̂_c)²", "Evaluate (pooled within-class scatter / n)"],
        solution=f"$\\hat\\mu_0 = {L(ma)}$, $\\hat\\mu_1 = {L(mb)}$, within-class scatter $= {L(ss)}$, $\\hat\\sigma^2 = {L(ss)}/6 = {L(s2)}$.",
        misconceptions=[(fmt(sum((t - m) ** 2 for t in allx) / 6), "ignored-class-means"), (fmt(ss / 4), "used-unbiased-dof")],
    )




# --------------------------------------------------------------------------- trees & k-NN
def _ig(parent, children):
    n = sum(parent)
    return sp.expand(H2(parent) - sum(R(sum(c), n) * H2(c) for c in children))


@gen("ml.knn_trees")
def information_gain(rng, level):
    if level == 1:
        a, b = rng.randint(1, 12), rng.randint(1, 12)
        if a == b:
            b += 1
        h = H2([a, b])
        gini = 1 - R(a, a + b) ** 2 - R(b, a + b) ** 2
        return Problem(
            prompt=(f"A decision-tree node contains {a} positive and {b} negative training examples. Compute its entropy "
                    "(bits). (On paper: also its Gini impurity.)"),
            answer=fmt(h), kind="num", tol=TOL, answer_hint="bits: exact (log(x,2) allowed) or a decimal to 4 places",
            rubric=["Class proportions", "H = -Σ p log₂ p", f"Gini = 1 - Σ p² = {gini}"],
            solution=f"$H = H_b({L(R(a, a + b))}) = {L(h)} \\approx {d4(h)}$ bits; Gini $= {L(gini)}$.",
            misconceptions=[(fmt(gini), "reported-gini"), (fmt(h * LN2), "used-natural-log")],
        )

    def split(parent):
        P, N = parent
        while True:
            lp, ln = rng.randint(0, P), rng.randint(0, N)
            if 0 < lp + ln < P + N and (lp + ln) and (P - lp + N - ln):
                return [lp, ln], [P - lp, N - ln]

    P, N = rng.randint(4, 10), rng.randint(4, 10)
    if level == 2:
        while True:
            l, r = split((P, N))
            ig = _ig([P, N], [l, r])
            if float(ig) > 0.01:
                break
        unw = sp.expand(H2([P, N]) - (H2(l) + H2(r)) / 2)
        return Problem(
            prompt=(f"Parent node: {P} positive, {N} negative. A split sends ({l[0]}+, {l[1]}−) left and ({r[0]}+, {r[1]}−) "
                    "right. Compute the information gain (bits)."),
            answer=fmt(ig), kind="num", tol=TOL, answer_hint="bits: exact or a decimal to 4 places",
            rubric=["Parent entropy", "Entropy of each child", "Weight children by size/n", "IG = H(parent) - weighted child entropy"],
            solution=(f"$H(\\text{{parent}}) \\approx {d4(H2([P, N]))}$, children $\\approx {d4(H2(l))}, {d4(H2(r))}$; "
                      f"IG $= {L(ig)} \\approx {d4(ig)}$ bits."),
            misconceptions=[(fmt(unw), "unweighted-child-entropy")],
        )
    while True:
        la, ra = split((P, N))
        lb, rb = split((P, N))
        ga, gb = _ig([P, N], [la, ra]), _ig([P, N], [lb, rb])
        if abs(float(ga - gb)) > 0.02:
            break
    best, worst = (ga, gb) if ga > gb else (gb, ga)
    return Problem(
        prompt=(f"Root node: {P} positive, {N} negative. Feature A splits it into ({la[0]}+, {la[1]}−) / ({ra[0]}+, {ra[1]}−); "
                f"feature B into ({lb[0]}+, {lb[1]}−) / ({rb[0]}+, {rb[1]}−). The ID3 algorithm picks the split with the larger "
                "information gain. Give that (larger) information gain in bits."),
        answer=fmt(best), kind="num", tol=TOL, answer_hint="bits: exact or a decimal to 4 places",
        rubric=["Parent entropy (shared)", "Weighted child entropy for A", "Weighted child entropy for B", "Pick the larger IG"],
        solution=f"IG(A) $\\approx {d4(ga)}$, IG(B) $\\approx {d4(gb)}$ → choose {'A' if ga > gb else 'B'}, IG $\\approx {d4(best)}$ bits.",
        misconceptions=[(fmt(worst), "picked-worse-split")],
    )


@gen("ml.knn_trees")
def knn(rng, level):
    if level == 1:
        while True:
            xs = rng.sample(range(0, 21), 7)
            ys = [rng.choice([-1, 1]) for _ in xs]
            q = R(2 * rng.randint(0, 19) + 1, 2)  # half-integer query avoids exact distance ties with integer points
            d = sorted((abs(x - q), x, y) for x, y in zip(xs, ys))
            if d[2][0] != d[3][0] and d[0][0] != d[1][0]:
                lab3 = 1 if sum(t[2] for t in d[:3]) > 0 else -1
                break
        return Problem(
            prompt=(f"1-D training points (x, label): " + ", ".join(f"$({x}, {'+1' if y > 0 else '-1'})$" for x, y in zip(xs, ys)) +
                    f". Classify the query $x = {L(q)}$ with 3-nearest-neighbours (majority vote)."),
            answer=str(lab3), kind="num", answer_hint="+1 or -1",
            rubric=["Distances |x - q|", "Three smallest", "Majority label"],
            solution=f"Nearest: " + ", ".join(f"$x={t[1]}$ ({'+' if t[2] > 0 else '−'})" for t in d[:3]) + f" → ${lab3}$.",
            misconceptions=[(str(d[0][2]), "used-1-nn")] if d[0][2] != lab3 else [],
        )
    if level == 2:
        while True:
            P = [(rng.randint(-3, 3), rng.randint(-3, 3)) for _ in range(6)]
            if len(set(P)) < 6:
                continue
            ys = [rng.randint(0, 9) for _ in P]
            q = (rng.randint(-2, 2), rng.randint(-2, 2))
            d2 = sorted(((a - q[0]) ** 2 + (b - q[1]) ** 2, i) for i, (a, b) in enumerate(P))
            dm = sorted((abs(a - q[0]) + abs(b - q[1]), i) for i, (a, b) in enumerate(P))
            if d2[2][0] != d2[3][0] and d2[0][0] > 0:
                break
        pred = R(sum(ys[i] for _, i in d2[:3]), 3)
        mis = []
        if dm[2][0] != dm[3][0]:
            pm = R(sum(ys[i] for _, i in dm[:3]), 3)
            mis = [(fmt(pm), "used-manhattan")]
        return Problem(
            prompt=("Training data (point → target): " + ", ".join(f"$({a},{b}) \\to {y}$" for (a, b), y in zip(P, ys)) +
                    f". Predict the target at $q = ({q[0]}, {q[1]})$ by 3-NN regression (Euclidean distance, plain average)."),
            answer=fmt(pred), kind="num",
            rubric=["Squared Euclidean distances (no need for sqrt)", "Pick the 3 smallest", "Average their targets"],
            solution=("Squared distances: " + ", ".join(f"$({P[i][0]},{P[i][1]})$: {dd}" for dd, i in d2) +
                      f". Average of the 3 nearest targets $= {L(pred)}$."),
            misconceptions=mis,
        )
    while True:
        xs = sorted(rng.sample(range(0, 25), 7))
        ys = [rng.choice([-1, 1]) for _ in xs]
        ok, errs = True, 0
        for i, x in enumerate(xs):
            ds = sorted((abs(x - z), j) for j, z in enumerate(xs) if j != i)
            if ds[0][0] == ds[1][0]:
                ok = False
                break
            errs += ys[ds[0][1]] != ys[i]
        if ok:
            break
    return Problem(
        prompt=("Compute the leave-one-out cross-validation error **count** of a 1-nearest-neighbour classifier on the 1-D data "
                "(x, label): " + ", ".join(f"$({x}, {'+1' if y > 0 else '-1'})$" for x, y in zip(xs, ys)) +
                ". (On paper: why is the 1-NN *training* error always 0?)"),
        answer=str(errs), kind="num", answer_hint="an integer count, e.g. 2",
        rubric=["Hold out each point in turn", "Find its nearest *other* point", "Compare labels", "Count mistakes"],
        solution=f"LOOCV misclassifies {errs} of {len(xs)} points (error rate ${L(R(errs, len(xs)))}$).",
    )


def _gini(counts):
    n = sum(counts)
    return 1 - sum(R(c, n) ** 2 for c in counts)


def _sse(v):
    m = R(sum(v), len(v))
    return sum((t - m) ** 2 for t in v)


@gen("ml.knn_trees")
def knn_trees_more(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        while True:
            c = [rng.randint(0, 9) for _ in range(3)]
            if sum(c) >= 4 and max(c) < sum(c):
                break
        g = _gini(c)
        return Problem(
            prompt=(f"A decision-tree leaf holds {c[0]} examples of class A, {c[1]} of class B and {c[2]} of class C. "
                    "Compute its Gini impurity."),
            answer=fmt(g), kind="num",
            rubric=["Class proportions p_k", "Gini = 1 - Σ p_k²", "Exact fraction"],
            solution=f"$p = {vtex([R(t, sum(c)) for t in c])}$, Gini $= 1 - \\sum p_k^2 = {L(g)}$.",
            misconceptions=[(fmt(1 - g), "reported-purity"), (fmt(1 - R(max(c), sum(c))), "reported-misclassification-rate")],
        )
    if level == 1:
        while True:
            P = [(rng.randint(-4, 4), rng.randint(-4, 4)) for _ in range(4)]
            labs = [rng.choice([-1, 1]) for _ in P]
            q = (rng.randint(-2, 2), rng.randint(-2, 2))
            d1 = sorted((abs(a - q[0]) + abs(b - q[1]), i) for i, (a, b) in enumerate(P))
            d2 = sorted(((a - q[0]) ** 2 + (b - q[1]) ** 2, i) for i, (a, b) in enumerate(P))
            if len(set(P)) == 4 and d1[0][0] > 0 and d1[0][0] != d1[1][0] and d2[0][0] != d2[1][0]:
                break
        ans = labs[d1[0][1]]
        mis = [(str(labs[d2[0][1]]), "used-euclidean")] if labs[d2[0][1]] != ans else []
        return Problem(
            prompt=("Using Manhattan ($L_1$) distance, classify the query $q = " + f"({q[0]}, {q[1]})$ by 1-nearest-neighbour. "
                    "Training points (point, label): " +
                    ", ".join(f"$(({a},{b}),\\ {'+1' if y > 0 else '-1'})$" for (a, b), y in zip(P, labs)) + "."),
            answer=str(ans), kind="num", answer_hint="+1 or -1",
            rubric=["L1 distance |Δx| + |Δy| to each point", "Smallest distance", "Copy its label"],
            solution=("$L_1$ distances: " + ", ".join(f"$({P[i][0]},{P[i][1]})$: {dd}" for dd, i in d1) + f" → label ${ans}$."),
            misconceptions=mis,
        )
    if level == 2 and v == 0:
        while True:
            Pp, Nn = rng.randint(4, 10), rng.randint(4, 10)
            lp, ln = rng.randint(0, Pp), rng.randint(0, Nn)
            if 0 < lp + ln < Pp + Nn:
                l, r = [lp, ln], [Pp - lp, Nn - ln]
                n = Pp + Nn
                gain = _gini([Pp, Nn]) - R(sum(l), n) * _gini(l) - R(sum(r), n) * _gini(r)
                if gain > R(1, 100):
                    break
        unw = _gini([Pp, Nn]) - (_gini(l) + _gini(r)) / 2
        return Problem(
            prompt=(f"CART split evaluation: a node has {Pp} positive and {Nn} negative examples; a split sends "
                    f"({l[0]}+, {l[1]}−) left and ({r[0]}+, {r[1]}−) right. Compute the decrease in Gini impurity."),
            answer=fmt(gain), kind="num", answer_hint="exact fraction, e.g. 8/63",
            rubric=["Gini of the parent", "Gini of each child", "Weight children by their size", "Decrease = parent - weighted"],
            solution=(f"$G_{{\\text{{parent}}}} = {L(_gini([Pp, Nn]))}$, $G_L = {L(_gini(l))}$, $G_R = {L(_gini(r))}$; "
                      f"decrease $= {L(gain)}$."),
            misconceptions=[(fmt(unw), "unweighted-child-impurity")],
        )
    if level == 2:
        while True:
            xs = rng.sample(range(0, 16), 5)
            ys = [rng.randint(0, 9) for _ in xs]
            q = rng.randint(0, 15)
            d = sorted((abs(x - q), i) for i, x in enumerate(xs))
            if d[0][0] > 0 and d[2][0] != d[3][0]:
                near = d[:3]
                wsum = sum(R(1, dd) for dd, _ in near)
                pred = sum(R(ys[i], dd) for dd, i in near) / wsum
                plain = R(sum(ys[i] for _, i in near), 3)
                if pred != plain:
                    break
        return Problem(
            prompt=("Distance-weighted 3-NN regression (weights $1/d$) on 1-D data (x → y): " +
                    ", ".join(f"${x} \\to {y}$" for x, y in zip(xs, ys)) + f". Predict $y$ at $x = {q}$."),
            answer=fmt(pred), kind="num", answer_hint="exact fraction, e.g. 37/7",
            rubric=["Distances |x - q|", "Three nearest", "Weights 1/d", "Weighted average Σ(y/d)/Σ(1/d)"],
            solution=("Nearest: " + ", ".join(f"$x={xs[i]}$ ($d={dd}$, $y={ys[i]}$)" for dd, i in near) +
                      f"; prediction $= {L(pred)}$."),
            misconceptions=[(fmt(plain), "ignored-weights")],
        )
    if v == 0:
        while True:
            xs = sorted(rng.sample(range(0, 20), 7))
            ys = [rng.choice([-1, 1]) for _ in xs]
            best_pos, best_any = 99, 99
            for k in range(8):  # threshold before point k
                errp = sum(1 for i, y in enumerate(ys) if (1 if i >= k else -1) != y)
                best_pos = min(best_pos, errp)
                best_any = min(best_any, errp, 7 - errp)
            if 1 <= best_any < best_pos:
                break
        return Problem(
            prompt=("Train a decision stump $h(x) = s\\cdot\\operatorname{sign}(x - t)$ (threshold $t$, polarity $s = \\pm1$) on the "
                    "1-D data (x, label): " + ", ".join(f"$({x}, {'+1' if y > 0 else '-1'})$" for x, y in zip(xs, ys)) +
                    ". What is the smallest achievable number of training errors?"),
            answer=str(best_any), kind="num", answer_hint="an integer count, e.g. 2",
            rubric=["Only thresholds between consecutive points matter", "For each, count errors with s = +1",
                    "Errors with s = -1 are 7 minus that", "Take the minimum over both"],
            solution=f"Scanning all 8 thresholds and both polarities, the best stump makes {best_any} error(s).",
            misconceptions=[(str(best_pos), "ignored-polarity")],
        )
    while True:
        xs = sorted(rng.sample(range(0, 12), 5))
        ys = [rng.randint(0, 12) for _ in xs]
        sses = [_sse(ys[:k]) + _sse(ys[k:]) for k in range(1, 5)]
        best = min(sses)
        if sses.count(best) == 1 and best < _sse(ys) and sses[1] != best:
            break
    k = sses.index(best) + 1
    return Problem(
        prompt=("Regression tree, one split on 1-D data (x → y): " + ", ".join(f"${x} \\to {y}$" for x, y in zip(xs, ys)) +
                ". Each side predicts its mean. Find the split minimising the total squared error and give that minimum SSE."),
        answer=fmt(best), kind="num", answer_hint="exact fraction, e.g. 26/3",
        rubric=["Candidate thresholds between consecutive x", "For each: SSE(left) + SSE(right) around the side means",
                "SSE of a group = Σ(y - ȳ)²", "Take the minimum"],
        solution=("SSE for splits after the 1st..4th point: " + ", ".join(f"${L(s)}$" for s in sses) +
                  f"; best splits between $x = {xs[k - 1]}$ and $x = {xs[k]}$ with SSE $= {L(best)}$."),
        misconceptions=[(fmt(_sse(ys)), "no-split"), (fmt(sses[1]), "split-at-median")],
    )


# --------------------------------------------------------------------------- SVMs & kernels
@gen("ml.svm_kernels")
def svm_kernels(rng, level):
    if level == 1:
        w = [nz(rng, -4, 4), rng.randint(-4, 4)]
        nrm = sp.sqrt(w[0] ** 2 + w[1] ** 2)
        return Problem(
            prompt=(f"A hard-margin linear SVM has $w = {vtex(w)}$ (canonical scaling: support vectors satisfy "
                    "$y_i(w^\\top x_i + b) = 1$). What is the width of the margin between the two supporting hyperplanes?"),
            answer=fmt(2 / nrm), kind="num",
            rubric=["Distance of a point to the hyperplane = |wᵀx + b| / ||w||", "Support vectors: |wᵀx+b| = 1",
                    "Width = 2/||w||"],
            solution=f"$\\|w\\| = {L(nrm)}$, width $= 2/\\|w\\| = {L(2 / nrm)}$.",
            misconceptions=[(fmt(1 / nrm), "half-margin"), (fmt(R(2, w[0] ** 2 + w[1] ** 2)), "used-squared-norm")],
        )
    if level == 2:
        x = [rng.randint(-2, 3), rng.randint(-2, 3)]
        z = [rng.randint(-2, 3), rng.randint(-2, 3)]
        if rng.random() < 0.5:
            c, dgr = rng.choice([0, 1]), rng.choice([2, 3])
            k = (x[0] * z[0] + x[1] * z[1] + c) ** dgr
            return Problem(
                prompt=(f"Compute the polynomial kernel $k(x, z) = (x^\\top z + {c})^{dgr}$ for $x = {vtex(x)}$, $z = {vtex(z)}$. "
                        "(On paper, for degree 2: write the explicit feature map $\\phi$ and check $\\phi(x)^\\top\\phi(z)$ agrees.)"),
                answer=str(k), kind="num",
                rubric=["Dot product xᵀz", f"Add {c}", f"Raise to the power {dgr}"],
                solution=f"$x^\\top z = {x[0] * z[0] + x[1] * z[1]}$, $k = ({x[0] * z[0] + x[1] * z[1]} + {c})^{dgr} = {k}$.",
                misconceptions=[(str(x[0] * z[0] + x[1] * z[1] + c), "forgot-power")],
            )
        g = rng.choice([R(1, 2), R(1, 4), sp.Integer(1)])
        while True:
            d2 = (x[0] - z[0]) ** 2 + (x[1] - z[1]) ** 2
            if d2:
                break
            z = [z[0] + 1, z[1]]
        k = sp.exp(-g * d2)
        return Problem(
            prompt=(f"Compute the RBF kernel $k(x, z) = \\exp(-\\gamma\\|x - z\\|^2)$ with $\\gamma = {L(g)}$ for "
                    f"$x = {vtex(x)}$, $z = {vtex(z)}$."),
            answer=fmt(k), kind="num", tol=TOL, answer_hint="exact like exp(-5/2), or a decimal to 4 places",
            rubric=["Difference x - z", "Squared norm", "Multiply by -γ and exponentiate"],
            solution=f"$\\|x-z\\|^2 = {d2}$, $k = e^{{-{L(g * d2)}}} \\approx {d4(k)}$.",
            misconceptions=[(fmt(sp.exp(-g * sp.sqrt(d2))), "norm-not-squared")],
        )
    while True:
        xp = [rng.randint(-2, 4), rng.randint(-2, 4)]
        xm = [rng.randint(-4, 2), rng.randint(-4, 2)]
        dv = [xp[0] - xm[0], xp[1] - xm[1]]
        if dv != [0, 0]:
            break
        # loop again on coincident points
    nn = dv[0] ** 2 + dv[1] ** 2
    w = [R(2 * dv[0], nn), R(2 * dv[1], nn)]
    b = -(w[0] * (xp[0] + xm[0]) + w[1] * (xp[1] + xm[1])) / 2
    return Problem(
        prompt=(f"Hard-margin SVM with just two training points: $x_+ = {vtex(xp)}$ (label $+1$) and $x_- = {vtex(xm)}$ "
                "(label $-1$). Solve $\\min \\tfrac12\\|w\\|^2$ s.t. $y_i(w^\\top x_i + b) \\ge 1$ and give $(w_1, w_2, b)$."),
        answer=[fmt(v) for v in w + [b]], kind="vector", answer_hint="w1, w2, b   e.g. [1/2, 1/2, -1]",
        rubric=["Both points are support vectors (equality constraints)", "Lagrange/geometry: w ∥ x₊ - x₋",
                "Subtracting the constraints: wᵀ(x₊ - x₋) = 2 ⇒ w = 2(x₊ - x₋)/||x₊ - x₋||²",
                "b from wᵀx₊ + b = 1"],
        solution=(f"$x_+ - x_- = {vtex(dv)}$, $\\|\\cdot\\|^2 = {nn}$, $w = {vtex(w)}$, $b = {L(b)}$; the boundary is the "
                  "perpendicular bisector."),
        misconceptions=[([str(dv[0]), str(dv[1]), fmt(-(dv[0] * (xp[0] + xm[0]) + dv[1] * (xp[1] + xm[1])) / R(2))],
                         "unnormalised-w")],
    )


PYTH = [(3, 4), (4, 3), (6, 8), (8, 6), (5, 12), (12, 5)]


@gen("ml.svm_kernels")
def svm_more(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        while True:
            w = [nz(rng, -3, 3), rng.randint(-3, 3)]
            b = rng.randint(-3, 3)
            x = [rng.randint(-3, 3), rng.randint(-3, 3)]
            y = rng.choice([-1, 1])
            m = y * (w[0] * x[0] + w[1] * x[1] + b)
            if m < 1:
                break
        loss = 1 - m
        return Problem(
            prompt=(f"Hinge loss of a linear classifier $f(x) = w^\\top x + b$ with $w = {vtex(w)}$, $b = {b}$ on the example "
                    f"$x = {vtex(x)}$, $y = {y:+d}$: compute $\\max(0,\\ 1 - y f(x))$."),
            answer=str(loss), kind="num",
            rubric=["f(x) = wᵀx + b", "Margin y·f(x)", "Hinge = max(0, 1 - margin)"],
            solution=f"$f(x) = {w[0] * x[0] + w[1] * x[1] + b}$, $y f(x) = {m}$, loss $= \\max(0, 1 - ({m})) = {loss}$.",
            misconceptions=[(str(1 + m), "label-sign-error"), (str(max(0, -m)), "perceptron-loss")],
        )
    if level == 1:
        a, c = rng.choice(PYTH)
        w = [a * rng.choice([-1, 1]), c * rng.choice([-1, 1])]
        nrm = sp.sqrt(a * a + c * c)
        while True:
            b = rng.randint(-10, 10)
            x = [rng.randint(-3, 3), rng.randint(-3, 3)]
            f = w[0] * x[0] + w[1] * x[1] + b
            if f != 0:
                break
        dist = abs(f) / nrm
        return Problem(
            prompt=(f"Distance from a point to a separating hyperplane: compute the distance from $x = {vtex(x)}$ to the "
                    f"hyperplane $w^\\top x + b = 0$ with $w = {vtex(w)}$, $b = {b}$."),
            answer=fmt(dist), kind="num",
            rubric=["Evaluate wᵀx + b", "‖w‖", "Distance = |wᵀx + b| / ‖w‖"],
            solution=f"$w^\\top x + b = {f}$, $\\|w\\| = {L(nrm)}$, distance $= {abs(f)}/{L(nrm)} = {L(dist)}$.",
            misconceptions=[(str(abs(f)), "forgot-divide-by-norm"), (fmt(abs(f) / nrm ** 2), "divided-by-squared-norm")],
        )
    if level == 2 and v == 0:
        C = rng.choice([sp.Integer(1), sp.Integer(2), R(1, 2)])
        while True:
            w = [rng.randint(-2, 2), rng.randint(-2, 2)]
            b = rng.randint(-2, 2)
            Xs = [[rng.randint(-2, 2), rng.randint(-2, 2)] for _ in range(3)]
            Y = [rng.choice([-1, 1]) for _ in range(3)]
            hinges = [max(0, 1 - y * (w[0] * x[0] + w[1] * x[1] + b)) for x, y in zip(Xs, Y)]
            if w != [0, 0] and sum(hinges) > 0:
                break
        J = R(w[0] ** 2 + w[1] ** 2, 2) + C * sum(hinges)
        return Problem(
            prompt=(f"Soft-margin SVM objective $J = \\tfrac12\\|w\\|^2 + C\\sum_i \\max(0, 1 - y_i(w^\\top x_i + b))$ with "
                    f"$C = {L(C)}$, $w = {vtex(w)}$, $b = {b}$. Data $(x_i, y_i)$: " +
                    ", ".join(f"$({vtex(x)}, {y:+d})$" for x, y in zip(Xs, Y)) + ". Evaluate $J$."),
            answer=fmt(J), kind="num",
            rubric=["Regulariser ½‖w‖²", "Margin yᵢ(wᵀxᵢ + b) for each point", "Hinge per point", "J = ½‖w‖² + C·Σ hinge"],
            solution=(f"$\\tfrac12\\|w\\|^2 = {L(R(w[0] ** 2 + w[1] ** 2, 2))}$, hinges ${hinges}$, "
                      f"$J = {L(R(w[0] ** 2 + w[1] ** 2, 2))} + {L(C)}\\cdot{sum(hinges)} = {L(J)}$."),
            misconceptions=[(fmt(w[0] ** 2 + w[1] ** 2 + C * sum(hinges)), "forgot-half"),
                            (fmt(R(w[0] ** 2 + w[1] ** 2, 2) + sum(hinges)), "forgot-C")],
        )
    if level == 2:
        while True:
            Xs = [[rng.randint(-2, 2), rng.randint(-2, 2)] for _ in range(3)]
            if len({tuple(x) for x in Xs}) == 3:
                break
        Gl = sp.Matrix(3, 3, lambda i, j: Xs[i][0] * Xs[j][0] + Xs[i][1] * Xs[j][1])
        K = Gl.applyfunc(lambda t: (t + 1) ** 2)
        return Problem(
            prompt=("Build the Gram (kernel) matrix $K_{ij} = (x_i^\\top x_j + 1)^2$ for the three points " +
                    ", ".join(f"$x_{i + 1} = {vtex(x)}$" for i, x in enumerate(Xs)) + "."),
            answer=fmt(K), kind="matrix", answer_hint="3×3 rows, e.g. [[9,1,4],[1,4,0],[4,0,9]]",
            rubric=["All pairwise dot products xᵢᵀxⱼ (a symmetric 3×3)", "Add 1 and square entrywise",
                    "Check: symmetric with positive diagonal"],
            solution=f"Dot products ${L(Gl)}$, so $K = {L(K)}$.",
            misconceptions=[(fmt(Gl), "forgot-kernel"), (fmt(Gl.applyfunc(lambda t: t ** 2 + 1)), "squared-before-adding")],
        )
    if v == 0:
        while True:
            Xs = [[rng.randint(-2, 3), rng.randint(-2, 3)] for _ in range(3)]
            Y = [1, 1, -1]
            a1, a2 = rng.choice([R(1, 4), R(1, 2), sp.Integer(1)]), rng.choice([R(1, 4), R(1, 2), sp.Integer(1)])
            al = [a1, a2, a1 + a2]
            w = [sum(al[i] * Y[i] * Xs[i][d] for i in range(3)) for d in range(2)]
            b = rng.randint(-2, 2)
            xq = [rng.randint(-2, 2), rng.randint(-2, 2)]
            f = w[0] * xq[0] + w[1] * xq[1] + b
            if w != [0, 0] and f != 0:
                break
        wu = [sum(al[i] * Xs[i][d] for i in range(3)) for d in range(2)]
        return Problem(
            prompt=("From the SVM dual to the primal: a linear-kernel SVM has multipliers $\\alpha = " + vtex(al) +
                    "$ for the points " + ", ".join(f"$x_{i + 1} = {vtex(x)}$ ($y = {y:+d}$)" for i, (x, y) in enumerate(zip(Xs, Y))) +
                    f", and $b = {b}$. Derive $w$ from the stationarity condition, then give $w_1, w_2$ and the decision value "
                    f"$f(x_q)$ at $x_q = {vtex(xq)}$."),
            answer=[fmt(w[0]), fmt(w[1]), fmt(f)], kind="list", answer_hint="w1, w2, f(xq)   e.g. 1/2, -1, 3/2",
            rubric=["∂L/∂w = 0 ⇒ w = Σ αᵢ yᵢ xᵢ", "Check Σ αᵢ yᵢ = 0", "f(x_q) = wᵀx_q + b"],
            solution=f"$w = \\sum_i \\alpha_i y_i x_i = {vtex(w)}$, $f(x_q) = {L(f)}$ (class ${'+1' if f > 0 else '-1'}$).",
            misconceptions=[([fmt(u) for u in wu] + [fmt(wu[0] * xq[0] + wu[1] * xq[1] + b)], "forgot-labels")],
        )
    while True:
        x = [rng.randint(-2, 3), rng.randint(-2, 3)]
        z = [rng.randint(-2, 3), rng.randint(-2, 3)]
        if x != z and x != [-z[0], -z[1]] and x != [0, 0] and z != [0, 0]:
            break
    k = lambda u, t: (u[0] * t[0] + u[1] * t[1]) ** 2  # noqa: E731
    d2 = k(x, x) + k(z, z) - 2 * k(x, z)
    return Problem(
        prompt=("Kernel trick for distances: $k(x, z) = (x^\\top z)^2$ corresponds to $\\phi(x) = (x_1^2,\\ \\sqrt2 x_1x_2,\\ x_2^2)$. "
                f"Show $\\|\\phi(x) - \\phi(z)\\|^2 = k(x,x) + k(z,z) - 2k(x,z)$ and evaluate it for $x = {vtex(x)}$, $z = {vtex(z)}$."),
        answer=str(d2), kind="num",
        rubric=["Expand ‖φ(x) - φ(z)‖² = φ(x)ᵀφ(x) + φ(z)ᵀφ(z) - 2φ(x)ᵀφ(z)", "Replace each inner product by k",
                "Evaluate k(x,x), k(z,z), k(x,z)", "Combine"],
        solution=f"$k(x,x) = {k(x, x)}$, $k(z,z) = {k(z, z)}$, $k(x,z) = {k(x, z)}$; distance² $= {d2}$.",
        misconceptions=[(str((x[0] - z[0]) ** 2 + (x[1] - z[1]) ** 2), "used-input-space-distance"),
                        (str(k(x, x) + k(z, z) - k(x, z)), "forgot-factor-2")],
    )


# --------------------------------------------------------------------------- clustering
def _assign(points, cents, dist):
    out = []
    for p in points:
        ds = [dist(p, c) for c in cents]
        m = min(ds)
        if ds.count(m) > 1:
            return None
        out.append(ds.index(m))
    return out


@gen("ml.clustering")
def kmeans(rng, level):
    if level == 1:
        while True:
            xs = sorted(rng.sample(range(0, 21), 6))
            c = sorted(rng.sample(range(0, 21), 2))
            a = _assign(xs, c, lambda p, q: abs(p - q))
            if a and len(set(a)) == 2:
                break
        new = [R(sum(x for x, k in zip(xs, a) if k == j), a.count(j)) for j in range(2)]
        return Problem(
            prompt=(f"1-D $k$-means with $k=2$: data ${xs}$, initial centroids $c_1 = {c[0]}$, $c_2 = {c[1]}$. "
                    "Do **one** iteration (assign, then update). Give the new $c_1, c_2$."),
            answer=[fmt(v) for v in new], kind="list", answer_hint="c1, c2   e.g. 7/3, 15",
            rubric=["Assign each point to the nearest centroid", "Recompute each centroid as the mean of its points"],
            solution=(f"Cluster 1: ${[x for x, k in zip(xs, a) if k == 0]}$, cluster 2: ${[x for x, k in zip(xs, a) if k == 1]}$; "
                      f"new centroids ${L(new[0])}, {L(new[1])}$."),
            misconceptions=[([fmt(new[1]), fmt(new[0])], "swapped-cluster-order")],
        )
    if level == 2:
        while True:
            P = [(rng.randint(0, 8), rng.randint(0, 8)) for _ in range(6)]
            C = [(rng.randint(0, 8), rng.randint(0, 8)) for _ in range(2)]
            if len(set(P)) < 6 or C[0] == C[1]:
                continue
            a = _assign(P, C, lambda p, q: (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
            if a and len(set(a)) == 2:
                break
        new = [[R(sum(p[d] for p, k in zip(P, a) if k == j), a.count(j)) for d in range(2)] for j in range(2)]
        M = sp.Matrix(new)
        return Problem(
            prompt=("2-D $k$-means, $k = 2$. Points: " + ", ".join(f"$({p[0]},{p[1]})$" for p in P) +
                    f". Initial centroids $c_1 = ({C[0][0]},{C[0][1]})$, $c_2 = ({C[1][0]},{C[1][1]})$. After one assignment + "
                    "update step, give the new centroids as a matrix whose rows are $c_1$ and $c_2$."),
            answer=fmt(M), kind="matrix", answer_hint="[[c1x, c1y], [c2x, c2y]]",
            rubric=["Squared Euclidean distance to each centroid", "Assign to the nearest", "Mean of each cluster (per coordinate)"],
            solution=(f"Assignments ${[k + 1 for k in a]}$; new centroids $c_1 = {vtex(new[0])}$, $c_2 = {vtex(new[1])}$."),
        )
    mu1, mu2 = rng.randint(-2, 1), rng.randint(2, 5)
    pi1 = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(2, 3)])
    x = rng.choice([t for t in range(mu1, mu2 + 1) if 2 * t != mu1 + mu2])  # avoid the trivial midpoint
    e = R((x - mu1) ** 2 - (x - mu2) ** 2, 2)
    r = 1 / (1 + (1 - pi1) / pi1 * sp.exp(e))
    mis = [(fmt(1 / (1 + sp.exp(e))), "ignored-mixing-weights")] if pi1 != R(1, 2) else []
    return Problem(
        prompt=(f"EM for a 1-D mixture of two unit-variance Gaussians: current parameters $\\mu_1 = {mu1}$, $\\mu_2 = {mu2}$, "
                f"$\\pi_1 = {L(pi1)}$, $\\pi_2 = {L(1 - pi1)}$, $\\sigma_1 = \\sigma_2 = 1$. Compute the E-step responsibility "
                f"$r_1 = P(z = 1\\mid x = {x})$."),
        answer=fmt(r), kind="num", tol=TOL, answer_hint=DEC_HINT,
        rubric=["rₖ ∝ πₖ N(x; μₖ, 1)", "The 1/√(2π) constants cancel", "r₁ = 1/(1 + (π₂/π₁) exp(((x-μ₁)² - (x-μ₂)²)/2))",
                "Evaluate"],
        solution=(f"$r_1 = \\frac{{1}}{{1 + \\frac{{{L(1 - pi1)}}}{{{L(pi1)}}} e^{{{L(e)}}}}} \\approx {d4(r)}$."),
        misconceptions=mis + [(fmt(1 - r), "computed-other-component")],
    )


@gen("ml.clustering")
def em_mstep(rng, level):
    xs = [rng.randint(-3, 8) for _ in range(4)]
    rs = [R(rng.randint(0, 4), 4) for _ in range(4)]
    if sum(rs) == 0:
        rs[0] = R(1, 2)
    Nk = sum(rs)
    mu = sum(r * x for r, x in zip(rs, xs)) / Nk
    data = ", ".join(f"$x={x}$ ($r={L(r)}$)" for x, r in zip(xs, rs))
    if level == 1:
        return Problem(
            prompt=f"EM M-step for component 1 of a Gaussian mixture. Points with their responsibilities $r_{{i1}}$: {data}. Compute the updated mean $\\mu_1$.",
            answer=fmt(mu), kind="num",
            rubric=["N₁ = Σ rᵢ", "μ₁ = Σ rᵢ xᵢ / N₁"],
            solution=f"$N_1 = {L(Nk)}$, $\\mu_1 = {L(sum(r * x for r, x in zip(rs, xs)))}/{L(Nk)} = {L(mu)}$.",
            misconceptions=[(fmt(R(sum(xs), 4)), "ignored-responsibilities"),
                            (fmt(sum(r * x for r, x in zip(rs, xs)) / 4), "divided-by-n-not-Nk")],
        )
    var = sum(r * (x - mu) ** 2 for r, x in zip(rs, xs)) / Nk
    if level == 2:
        return Problem(
            prompt=(f"EM M-step for component 1 of a 1-D Gaussian mixture. Points with responsibilities $r_{{i1}}$: {data}. "
                    "Compute the updated mixing weight $\\pi_1$ (4 points total), mean $\\mu_1$ and variance $\\sigma_1^2$."),
            answer=[fmt(Nk / 4), fmt(mu), fmt(var)], kind="list", answer_hint="pi1, mu1, var1   e.g. 1/2, 3, 5/4",
            rubric=["N₁ = Σ rᵢ, π₁ = N₁/n", "μ₁ = Σ rᵢxᵢ / N₁", "σ₁² = Σ rᵢ(xᵢ - μ₁)² / N₁ (use the *new* μ₁)"],
            solution=f"$\\pi_1 = {L(Nk / 4)}$, $\\mu_1 = {L(mu)}$, $\\sigma_1^2 = {L(var)}$.",
            misconceptions=[([fmt(Nk), fmt(mu), fmt(var)], "unnormalised-mixing-weight")],
        )
    while True:
        P = sorted(rng.sample(range(0, 20), 6))
        c = sorted(rng.sample(range(0, 20), 2))
        a = _assign(P, c, lambda p, q: abs(p - q))
        if a and len(set(a)) == 2:
            break
    J0 = sum((p - c[k]) ** 2 for p, k in zip(P, a))
    new = [R(sum(p for p, k in zip(P, a) if k == j), a.count(j)) for j in range(2)]
    J1 = sum((p - new[k]) ** 2 for p, k in zip(P, a))
    return Problem(
        prompt=(f"$k$-means is 'hard EM'. Data ${P}$, centroids $c = ({c[0]}, {c[1]})$. Compute the $k$-means objective "
                "$J = \\sum_i \\|x_i - c_{z_i}\\|^2$ after the assignment step, then after the update step. Give the "
                "**decrease** in $J$ due to the update step. (On paper: argue why neither step can increase $J$.)"),
        answer=fmt(J0 - J1), kind="num",
        rubric=["Assign points (J with old centroids)", "Update centroids to cluster means", "Recompute J",
                "Decrease = Σ_k n_k (c_k - mean_k)²"],
        solution=f"$J$ before update $= {J0}$, new centroids ${L(new[0])}, {L(new[1])}$, $J$ after $= {L(J1)}$; decrease $= {L(J0 - J1)}$.",
        misconceptions=[(fmt(J1), "reported-new-objective")],
    )


@gen("ml.clustering")
def clustering_more(rng, level):
    v = rng.randrange(2)
    if level == 1 and v == 0:
        while True:
            xs = sorted(rng.sample(range(0, 21), 6))
            c = sorted(rng.sample(range(0, 21), 2))
            a = _assign(xs, c, lambda p, q: abs(p - q))
            if a and len(set(a)) == 2:
                break
        J = sum((x - c[k]) ** 2 for x, k in zip(xs, a))
        return Problem(
            prompt=(f"Inertia of a clustering: 1-D data ${xs}$, centroids $c_1 = {c[0]}$, $c_2 = {c[1]}$. Assign each point to its "
                    "nearest centroid and compute the $k$-means objective $J = \\sum_i (x_i - c_{z_i})^2$."),
            answer=str(J), kind="num",
            rubric=["Nearest centroid for each point", "Squared distance to it", "Sum"],
            solution=("Squared distances: " + ", ".join(str((x - c[k]) ** 2) for x, k in zip(xs, a)) + f"; $J = {J}$."),
            misconceptions=[(str(sum(abs(x - c[k]) for x, k in zip(xs, a))), "used-absolute-distance")],
        )
    if level == 1:
        while True:
            A = sorted(rng.sample(range(0, 10), 3))
            B = sorted(rng.sample(range(8, 20), 3))
            if max(A) < min(B):
                break
        single, complete = min(B) - max(A), max(B) - min(A)
        avg = R(sum(b - a for a in A for b in B), 9)
        return Problem(
            prompt=(f"Linkage between two clusters of 1-D points, $A = {A}$ and $B = {B}$: give the single-linkage and the "
                    "complete-linkage distance."),
            answer=[str(single), str(complete)], kind="list", answer_hint="single, complete   e.g. 2, 15",
            rubric=["Single linkage = smallest pairwise distance", "Complete linkage = largest pairwise distance"],
            solution=(f"Single $= {min(B)} - {max(A)} = {single}$, complete $= {max(B)} - {min(A)} = {complete}$ "
                      f"(average linkage would give ${L(avg)}$)."),
            misconceptions=[([str(complete), str(single)], "swapped-single-complete")],
        )
    if level == 2 and v == 0:
        while True:
            xs = sorted(rng.sample(range(0, 30), 5))
            gaps = [b - a for a, b in zip(xs, xs[1:])]
            if len(set(gaps)) == 4 and gaps != sorted(gaps):
                break
        return Problem(
            prompt=(f"Agglomerative clustering with single linkage on the 1-D points ${xs}$: list the merge distances in the "
                    "order the merges happen (until one cluster remains)."),
            answer=[str(g) for g in sorted(gaps)], kind="list", answer_hint="4 distances in merge order, e.g. 1, 3, 4, 9",
            rubric=["Single-linkage distance between neighbouring clusters = gap between their closest points",
                    "In 1-D only consecutive gaps matter", "Merge the smallest gap first, then repeat"],
            solution=f"Consecutive gaps ${gaps}$; merges happen at ${sorted(gaps)}$ (the dendrogram heights).",
            misconceptions=[([str(g) for g in gaps], "listed-gaps-in-position-order")],
        )
    if level == 2:
        while True:
            A = sorted(rng.sample(range(0, 10), 3))
            B = sorted(rng.sample(range(6, 20), 3))
            x = A[rng.randrange(3)]
            others = [t for t in A if t != x]
            a = R(sum(abs(x - t) for t in others), 2)
            b = R(sum(abs(x - t) for t in B), 3)
            if a != b and set(A).isdisjoint(B):
                break
        s = (b - a) / max(a, b)
        a_bad = R(sum(abs(x - t) for t in A), 3)
        return Problem(
            prompt=(f"Silhouette of a single point: clusters $A = {A}$ and $B = {B}$ (1-D). Compute the silhouette "
                    f"$s = (b - a)/\\max(a, b)$ of the point $x = {x}$ in $A$."),
            answer=fmt(s), kind="num",
            rubric=["a = mean distance to the *other* points of its own cluster", "b = mean distance to the points of B",
                    "s = (b - a)/max(a, b)"],
            solution=f"$a = {L(a)}$, $b = {L(b)}$, $s = {L(s)}$.",
            misconceptions=[(fmt((b - a_bad) / max(a_bad, b)), "included-point-itself"), (fmt(-s), "silhouette-sign")],
        )
    if v == 0:
        while True:
            xs = sorted(rng.sample(range(0, 25), 6))
            costs = [_sse(xs[:k]) + _sse(xs[k:]) for k in range(1, 6)]
            best = min(costs)
            gaps = [b - a for a, b in zip(xs, xs[1:])]
            kg = gaps.index(max(gaps)) + 1
            if costs.count(best) == 1 and gaps.count(max(gaps)) == 1 and costs[kg - 1] != best:
                break
        k = costs.index(best) + 1
        return Problem(
            prompt=(f"Globally optimal 2-means in 1-D: for the data ${xs}$, argue that an optimal clustering splits the sorted data "
                    "into a left and a right block, then find the minimum possible objective $J = \\sum_i (x_i - c_{z_i})^2$."),
            answer=fmt(best), kind="num", answer_hint="exact fraction, e.g. 98/3",
            rubric=["Optimal clusters in 1-D are contiguous (each point is nearer its own centroid)",
                    "Try the 5 split positions", "SSE of each block around its mean", "Take the minimum"],
            solution=("Split after the 1st..5th point: $J = " + ",\\ ".join(L(c) for c in costs) +
                      f"$; best is $\\{{{', '.join(map(str, xs[:k]))}\\}}\\ |\\ \\{{{', '.join(map(str, xs[k:]))}\\}}$ with $J = {L(best)}$."),
            misconceptions=[(fmt(costs[kg - 1]), "split-at-largest-gap")],
        )
    while True:
        A = sorted(rng.sample(range(0, 12), 3))
        B = sorted(rng.sample(range(6, 20), 2))
        x = max(A)
        if x in B:
            continue
        cA, cB = R(sum(A), 3), R(sum(B), 2)
        dJ = R(2, 3) * (x - cB) ** 2 - R(3, 2) * (x - cA) ** 2
        naive = (x - cB) ** 2 - (x - cA) ** 2
        if dJ != 0 and dJ != naive:
            break
    return Problem(
        prompt=(f"Moving one point between $k$-means clusters: $A = {A}$, $B = {B}$ (1-D), centroids are the means. Derive the "
                f"change $\\Delta J$ in the objective when $x = {x}$ moves from $A$ to $B$ and both centroids are recomputed."),
        answer=fmt(dJ), kind="num", answer_hint="exact fraction (negative means J decreases), e.g. -7/6",
        rubric=["Adding x to B (size n_B) raises its SSE by n_B/(n_B+1)·(x - c_B)²",
                "Removing x from A (size n_A) lowers its SSE by n_A/(n_A-1)·(x - c_A)²",
                "ΔJ = n_B/(n_B+1)(x - c_B)² - n_A/(n_A-1)(x - c_A)²", "Evaluate"],
        solution=(f"$c_A = {L(cA)}$, $c_B = {L(cB)}$; $\\Delta J = \\tfrac23({L(x - cB)})^2 - \\tfrac32({L(x - cA)})^2 = {L(dJ)}$ "
                  f"{'(the move helps)' if dJ < 0 else '(the move hurts)'}."),
        misconceptions=[(fmt(naive), "ignored-centroid-shift")],
    )


# --------------------------------------------------------------------------- PCA in practice
@gen("ml.pca_applied")
def pca_applied(rng, level):
    if level == 1:
        while True:
            lam = sorted(rng.sample(range(1, 30), 5), reverse=True)
            thr = rng.choice([R(8, 10), R(9, 10)])
            tot = sum(lam)
            cum = [R(sum(lam[:k]), tot) for k in range(1, 6)]
            k = next(i + 1 for i, v in enumerate(cum) if v >= thr)
            if cum[k - 1] != thr:
                break
        return Problem(
            prompt=(f"The covariance matrix of a 5-feature dataset has eigenvalues ${lam}$. What is the smallest number $k$ of "
                    f"principal components that explains at least ${int(thr * 100)}\\%$ of the variance, and what fraction do they explain?"),
            answer=[str(k), fmt(cum[k - 1])], kind="list", answer_hint="k, fraction   e.g. 2, 41/50",
            rubric=["Total variance = Σλ (trace)", "Cumulative sums of sorted eigenvalues", "First k reaching the threshold"],
            solution=f"Total $= {tot}$; cumulative fractions $" + ", ".join(L(v) for v in cum) + f"$ → $k = {k}$, fraction ${L(cum[k - 1])}$.",
            misconceptions=[([str(k - 1), fmt(cum[k - 2]) if k > 1 else "0"], "stopped-before-threshold")],
        )
    if level == 2:
        while True:
            P = [(rng.randint(-3, 5), rng.randint(-3, 5)) for _ in range(4)]
            M = sp.Matrix(P)
            mean = sp.Matrix([[R(sum(p[0] for p in P), 4), R(sum(p[1] for p in P), 4)]])
            Xc = M - sp.ones(4, 1) * mean
            S = Xc.T * Xc / 4
            if S.det() != 0:
                break
        return Problem(
            prompt=("Before running PCA, compute the covariance matrix (divide by $n = 4$) of the points " +
                    ", ".join(f"$({p[0]},{p[1]})$" for p in P) + "."),
            answer=fmt(S), kind="matrix", answer_hint="[[s11, s12], [s21, s22]]",
            rubric=["Subtract the mean from every point", "Form XcᵀXc", "Divide by n"],
            solution=f"Mean ${L(mean)}$, $\\Sigma = \\frac14 X_c^\\top X_c = {L(S)}$.",
            misconceptions=[(fmt(Xc.T * Xc / 3), "divided-by-n-minus-1"), (fmt(M.T * M / 4), "forgot-centering")],
        )
    while True:
        a, c, b = rng.randint(1, 9), rng.randint(1, 9), nz(rng, -4, 4)
        if a * c - b * b > 0:
            break
    lam1 = R(a + c, 2) + sp.sqrt(R((a - c) ** 2, 4) + b * b)
    frac = sp.radsimp(lam1 / (a + c))
    return Problem(
        prompt=(f"2-D data has covariance $\\Sigma = \\begin{{pmatrix}} {a} & {b} \\\\ {b} & {c} \\end{{pmatrix}}$. Derive the "
                "eigenvalues and give the fraction of total variance explained by the first principal component. "
                "(On paper: what is the mean squared reconstruction error if you keep only PC1?)"),
        answer=fmt(frac), kind="num", tol=TOL, answer_hint="exact (with sqrt) or a decimal to 4 places",
        rubric=["Characteristic polynomial λ² - (a+c)λ + (ac - b²) = 0", "λ₁ = (a+c)/2 + sqrt(((a-c)/2)² + b²)",
                "Fraction = λ₁/(λ₁+λ₂) = λ₁/trace", "Reconstruction error = λ₂"],
        solution=(f"$\\lambda_1 = {L(lam1)} \\approx {d4(lam1)}$, $\\lambda_2 \\approx {d4(a + c - lam1)}$; fraction $= {L(frac)} "
                  f"\\approx {d4(frac)}$."),
        misconceptions=[(fmt(R(max(a, c), a + c)), "used-largest-diagonal")],
    )


@gen("ml.pca_applied")
def pca_more(rng, level):
    v = rng.randrange(2)
    if level in (1, 2) and v == 0:
        u = list(rng.choice(UNITS))
        while True:
            m = [rng.randint(-3, 5), rng.randint(-3, 5)]
            x = [rng.randint(-4, 8), rng.randint(-4, 8)]
            dx = [x[0] - m[0], x[1] - m[1]]
            sc = u[0] * dx[0] + u[1] * dx[1]
            if sc != 0 and u[0] * x[0] + u[1] * x[1] != sc:
                break
        if level == 1:
            return Problem(
                prompt=(f"Projecting onto a principal component: PCA found the unit first direction $u_1 = {vtex(u)}$ and the data "
                        f"mean is $\\mu = {vtex(m)}$. Compute the PC1 score of $x = {vtex(x)}$."),
                answer=fmt(sc), kind="num",
                rubric=["Centre: x - μ", "Score = u₁ᵀ(x - μ)"],
                solution=f"$x - \\mu = {vtex(dx)}$, score $= \\left({L(u[0])}\\right)({dx[0]}) + \\left({L(u[1])}\\right)({dx[1]}) = {L(sc)}$.",
                misconceptions=[(fmt(u[0] * x[0] + u[1] * x[1]), "forgot-centering")],
            )
        xh = [m[0] + sc * u[0], m[1] + sc * u[1]]
        return Problem(
            prompt=(f"Rank-1 PCA reconstruction: unit first direction $u_1 = {vtex(u)}$, data mean $\\mu = {vtex(m)}$. "
                    f"Reconstruct $x = {vtex(x)}$ from its PC1 score alone, $\\hat x = \\mu + (u_1^\\top(x-\\mu))\\,u_1$, and give "
                    "$\\hat x$."),
            answer=[fmt(t) for t in xh], kind="vector", answer_hint="two exact values, e.g. [62/25, 41/25]",
            rubric=["Centre: x - μ", "Score s = u₁ᵀ(x - μ)", "x̂ = μ + s u₁", "(On paper: residual x - x̂ is ⟂ u₁)"],
            solution=f"$s = {L(sc)}$, $\\hat x = {vtex(m)} + {L(sc)}\\,{vtex(u)} = {vtex(xh)}$.",
            misconceptions=[([fmt(sc * u[0]), fmt(sc * u[1])], "forgot-add-mean")],
        )
    if level == 1:
        while True:
            lam = sorted(rng.sample(range(1, 30), 5), reverse=True)
            k = rng.choice([1, 2, 3])
            if sum(lam[k:]) != sum(lam[:k]):
                break
        err = sum(lam[k:])
        return Problem(
            prompt=(f"Reconstruction error of truncated PCA: the covariance eigenvalues are ${lam}$. Keeping only the top "
                    f"${k}$ component(s), what is the mean squared reconstruction error $\\frac1n\\sum_i\\|x_i - \\hat x_i\\|^2$?"),
            answer=str(err), kind="num",
            rubric=["Error = variance in the discarded directions", "Sum the eigenvalues that were dropped"],
            solution=f"Discarded: ${lam[k:]}$, error $= {err}$ (out of total variance ${sum(lam)}$).",
            misconceptions=[(str(sum(lam[:k])), "reported-kept-variance"), (fmt(R(err, sum(lam))), "reported-fraction")],
        )
    if level == 2:
        s1, s2 = rng.sample([1, 2, 3, 4, 5], 2)
        r = rng.choice([R(1, 2), R(1, 4), R(3, 5), R(4, 5), -R(1, 2), -R(3, 5), -R(1, 4)])
        c = r * s1 * s2
        S = sp.Matrix([[s1 ** 2, c], [c, s2 ** 2]])
        ev = [1 + abs(r), 1 - abs(r)]
        return Problem(
            prompt=(f"PCA on standardised features: the raw covariance is $\\Sigma = {L(S)}$. Form the correlation matrix and "
                    "give its eigenvalues (largest first)."),
            answer=[fmt(t) for t in ev], kind="list", answer_hint="two values, largest first, e.g. 7/5, 3/5",
            rubric=["Standardise: divide by the standard deviations", "r = Σ₁₂/(σ₁σ₂)",
                    "Correlation matrix [[1, r], [r, 1]]", "Eigenvalues 1 ± |r|"],
            solution=(f"$\\sigma_1 = {s1}$, $\\sigma_2 = {s2}$, $r = {L(c)}/({s1}\\cdot{s2}) = {L(r)}$; eigenvalues "
                      f"${L(ev[0])}, {L(ev[1])}$ (eigenvectors $(1, \\pm1)/\\sqrt2$)."),
            misconceptions=[([fmt(1 + r), fmt(1 - r)], "sign-of-correlation"),
                            ([str(max(s1, s2) ** 2), str(min(s1, s2) ** 2)], "used-diagonal")],
        )
    if v == 0:
        p, q = rng.choice([(3, 1), (4, 2), (5, 3), (5, 1)])
        a, b = (p * p + q * q) // 2, (p * p - q * q) // 2
        while True:
            mu = [rng.randint(-2, 2), rng.randint(-2, 2)]
            x = [rng.randint(-3, 5), rng.randint(-3, 5)]
            d = [x[0] - mu[0], x[1] - mu[1]]
            if d[0] + d[1] != 0 and d[0] - d[1] != 0:
                break
        r2 = sp.sqrt(2)
        z = [sp.radsimp((d[0] + d[1]) / (r2 * p)), sp.radsimp((d[0] - d[1]) / (r2 * q))]
        rot = [sp.radsimp((d[0] + d[1]) / r2), sp.radsimp((d[0] - d[1]) / r2)]
        return Problem(
            prompt=(f"PCA whitening: data have mean $\\mu = {vtex(mu)}$ and covariance $\\Sigma = "
                    f"\\begin{{pmatrix}} {a} & {b} \\\\ {b} & {a} \\end{{pmatrix}}$. Derive the eigen-decomposition (use "
                    "$u_1 = (1, 1)/\\sqrt2$, $u_2 = (1, -1)/\\sqrt2$) and compute the whitened point "
                    f"$z = \\Lambda^{{-1/2}} U^\\top (x - \\mu)$ for $x = {vtex(x)}$."),
            answer=[fmt(t) for t in z], kind="vector", tol=TOL,
            answer_hint="two values, exact with sqrt(2) or decimals, e.g. [5*sqrt(2)/6, -sqrt(2)/2]",
            rubric=["Σ u₁ = (a+b)u₁, Σ u₂ = (a-b)u₂", f"λ₁ = {p * p}, λ₂ = {q * q}", "Rotate: Uᵀ(x - μ)",
                    "Scale each coordinate by 1/√λ"],
            solution=(f"$\\lambda_1 = {a} + {b} = {p * p}$, $\\lambda_2 = {a} - {b} = {q * q}$; $U^\\top(x-\\mu) = {vtex(rot)}$, "
                      f"$z = ({L(rot[0])}/{p},\\ {L(rot[1])}/{q}) = {vtex(z)}$."),
            misconceptions=[([fmt(t) for t in rot], "rotated-but-not-scaled"),
                            ([fmt(sp.radsimp((d[0] + d[1]) / (r2 * p * p))), fmt(sp.radsimp((d[0] - d[1]) / (r2 * q * q)))],
                             "divided-by-variance-not-sd")],
        )
    while True:
        a, c, b = rng.randint(1, 9), rng.randint(1, 9), nz(rng, -4, 4)
        u = list(rng.choice(UNITS))
        lam1 = R(a + c, 2) + sp.sqrt(R((a - c) ** 2, 4) + b * b)
        cap = a * u[0] ** 2 + 2 * b * u[0] * u[1] + c * u[1] ** 2
        gap = sp.radsimp(lam1 - cap)
        if a * c - b * b > 0 and float(gap) > 0.2:
            break
    return Problem(
        prompt=(f"Variance captured by a non-optimal direction: data covariance $\\Sigma = \\begin{{pmatrix}} {a} & {b} \\\\ {b} & {c} "
                f"\\end{{pmatrix}}$. Derive $\\mathrm{{Var}}(u^\\top x)$ for a unit vector $u$, evaluate it for $u = {vtex(u)}$, and "
                "give how much less variance this captures than PC1, i.e. $\\lambda_1 - \\mathrm{Var}(u^\\top x)$."),
        answer=fmt(gap), kind="num", tol=TOL, answer_hint="exact (with sqrt) or a decimal to 4 places",
        rubric=["Var(uᵀx) = uᵀΣu", f"uᵀΣu = a u₁² + 2b u₁u₂ + c u₂² = {cap}", "λ₁ = (a+c)/2 + sqrt(((a-c)/2)² + b²)",
                "Subtract (≥ 0 by the Rayleigh quotient)"],
        solution=(f"$u^\\top\\Sigma u = {L(cap)}$, $\\lambda_1 = {L(lam1)} \\approx {d4(lam1)}$; gap $= {L(gap)} \\approx {d4(gap)}$."),
        misconceptions=[(fmt(cap), "reported-captured-variance")],
    )
