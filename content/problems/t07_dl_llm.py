"""Track 7 — Deep learning & LLMs.

Levels: 1 = mechanical, 2 = multi-step, 3 = derivation / paper-derived with a checkable result.
All answers are computed with sympy / python, never hand-typed. Problems that need exp/log/sqrt
accept decimals via `tol` (the prompt asks for >= 4 significant figures).
"""
from __future__ import annotations

import math
import random
from collections import Counter

import sympy as sp

from engine.checker import fmt
from engine.content import Problem, gen

L = sp.latex
R = sp.Rational
TOL = 1e-3  # for answers involving exp / log / sqrt


# --------------------------------------------------------------------------- helpers
def nz(rng: random.Random, lo: int, hi: int) -> int:
    """Random non-zero integer in [lo, hi]."""
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def tup(v) -> str:
    """LaTeX tuple (a, b, c) of numbers / sympy values."""
    return "(" + ",\\ ".join(L(sp.sympify(t)) for t in v) + ")"


def mat(m) -> str:
    return L(sp.Matrix(m))


def sig(v, s: int = 4) -> str:
    """Plain decimal with s significant figures (for solutions)."""
    return f"{float(v):.{s}g}"


def sci(v, s: int = 4) -> str:
    """LaTeX scientific notation, e.g. 3.15 \\times 10^{23}."""
    v = float(v)
    if v == 0:
        return "0"
    e = math.floor(math.log10(abs(v)))
    m = round(v / 10 ** e, s - 1)
    if abs(m) >= 10:
        m, e = m / 10, e + 1
    ms = f"{m:.{s}g}"
    return ms if e == 0 else f"{ms} \\times 10^{{{e}}}"


def sigmoid(z):
    return 1 / (1 + sp.exp(-z))


def dec(v) -> str:
    """Show a Rational with one decimal like -1.3 (for prompts)."""
    return f"{float(v):.1f}"


def nfmt(v, digits: int = 15) -> str:
    """Numeric answer string (for answers that are transcendental)."""
    return fmt(sp.N(v, digits))


SIG4 = "Give at least 4 significant figures (exact forms like exp(1)/(1+exp(1)) are also accepted)."


# =========================================================================== dl.mlp
@gen("dl.mlp", levels=(1, 2, 3))
def mlp_forward(rng, level):
    if level == 1:
        nin = rng.choice([2, 3])
        want_neg = rng.random() < 0.5
        while True:
            w = [nz(rng, -4, 4) for _ in range(nin)]
            xv = [rng.randint(-3, 3) for _ in range(nin)]
            b = rng.randint(-5, 5)
            z = sum(a * c for a, c in zip(w, xv)) + b
            if z != 0 and (z < 0) == want_neg:
                break
        h = max(z, 0)
        terms = " + ".join(f"({a})({c})" for a, c in zip(w, xv))
        return Problem(
            prompt=(f"A single neuron computes $h = \\mathrm{{ReLU}}(\\mathbf w^\\top \\mathbf x + b)$ with "
                    f"$\\mathbf w = {tup(w)}$, $\\mathbf x = {tup(xv)}$ and $b = {b}$. Compute $h$."),
            answer=str(h), kind="num",
            rubric=["Compute the pre-activation z = w·x + b", "Apply ReLU: h = max(0, z)"],
            solution=f"$z = {terms} + ({b}) = {z}$, so $h = \\max(0, {z}) = {h}$.",
            misconceptions=[(str(z), "forgot-activation"), (str(max(z - b, 0)), "forgot-bias")],
        )
    if level == 2:
        want_dead = rng.random() < 0.6  # usually make one hidden unit inactive so the ReLU matters
        while True:
            xv = [nz(rng, -2, 2) for _ in range(2)]
            W1 = [[nz(rng, -3, 3) for _ in range(2)] for _ in range(2)]
            b1 = [rng.randint(-2, 2) for _ in range(2)]
            z1 = [W1[i][0] * xv[0] + W1[i][1] * xv[1] + b1[i] for i in range(2)]
            if 0 not in z1 and any(v > 0 for v in z1) and any(v < 0 for v in z1) == want_dead:
                break
        w2 = [nz(rng, -3, 3) for _ in range(2)]
        b2 = rng.randint(-3, 3)
        h = [max(v, 0) for v in z1]
        yv = w2[0] * h[0] + w2[1] * h[1] + b2
        ylin = w2[0] * z1[0] + w2[1] * z1[1] + b2
        return Problem(
            prompt=(f"A 2-2-1 network: $\\mathbf h = \\mathrm{{ReLU}}(W_1 \\mathbf x + \\mathbf b_1)$, "
                    f"$\\hat y = \\mathbf w_2^\\top \\mathbf h + b_2$ with $W_1 = {mat(W1)}$, "
                    f"$\\mathbf b_1 = {tup(b1)}$, $\\mathbf w_2 = {tup(w2)}$, $b_2 = {b2}$ and input "
                    f"$\\mathbf x = {tup(xv)}$. Compute $\\hat y$."),
            answer=str(yv), kind="num",
            rubric=["Hidden pre-activations z1 = W1 x + b1", "Apply ReLU elementwise",
                    "Output y = w2·h + b2"],
            solution=(f"$\\mathbf z_1 = {tup(z1)}$, $\\mathbf h = {tup(h)}$, "
                      f"$\\hat y = ({w2[0]})({h[0]}) + ({w2[1]})({h[1]}) + ({b2}) = {yv}$."),
            misconceptions=[(str(ylin), "forgot-hidden-relu")],
        )
    while True:
        W1 = sp.Matrix(3, 2, lambda i, j: rng.randint(-2, 2))
        W2 = sp.Matrix(2, 3, lambda i, j: rng.randint(-2, 2))
        W = W2 * W1
        if W != sp.zeros(2, 2) and W != W.T:
            break
    b1 = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
    b2 = sp.Matrix([rng.randint(-2, 2) for _ in range(2)])
    c = W2 * b1 + b2
    return Problem(
        prompt=("A 'deep' network **without activations**: $\\mathbf h = W_1\\mathbf x + \\mathbf b_1$, "
                "$\\mathbf y = W_2 \\mathbf h + \\mathbf b_2$ with "
                f"$W_1 = {L(W1)}$ ($3\\times2$), $\\mathbf b_1 = {L(b1)}$, $W_2 = {L(W2)}$ ($2\\times3$), "
                f"$\\mathbf b_2 = {L(b2)}$. Show that it collapses to one linear layer "
                "$\\mathbf y = W\\mathbf x + \\mathbf c$ and give the matrix $W$."),
        answer=fmt(W), kind="matrix",
        rubric=["Substitute h into the second layer", "Factor: y = (W2 W1) x + (W2 b1 + b2)",
                "Multiply W2 W1 in the right order (2x3 times 3x2)", "Note: depth without nonlinearity adds no expressive power"],
        solution=(f"$\\mathbf y = W_2(W_1\\mathbf x + \\mathbf b_1) + \\mathbf b_2 = (W_2W_1)\\mathbf x + (W_2\\mathbf b_1 + \\mathbf b_2)$, "
                  f"so $W = W_2W_1 = {L(W)}$ and $\\mathbf c = {L(c)}$. Without a nonlinearity the composition of affine maps is affine."),
        misconceptions=[(fmt(W1.T * W2.T), "transposed-product"), (fmt(W1 * W2), "wrong-multiplication-order")],
    )


@gen("dl.mlp", levels=(1, 2, 3))
def mlp_params(rng, level):
    if level == 1:
        nin, nout = rng.choice([3, 4, 10, 16, 32, 64, 128, 784]), rng.choice([2, 5, 10, 16, 32, 100, 256])
        ans = nin * nout + nout
        return Problem(
            prompt=f"How many trainable parameters does a fully-connected layer from ${nin}$ inputs to ${nout}$ outputs (with bias) have?",
            answer=str(ans), kind="num",
            rubric=["Weight matrix has n_out x n_in entries", "Add one bias per output unit"],
            solution=f"${nout}\\cdot{nin} + {nout} = {ans}$.",
            misconceptions=[(str(nin * nout), "forgot-bias"), (str(nin * nout + nin), "bias-per-input")],
        )
    if level == 2:
        sizes = [rng.choice([784, 64, 100, 32]), rng.choice([256, 128, 64, 50]), rng.choice([128, 64, 32, 20]), rng.choice([10, 2, 5, 1])]
        ans = sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(3))
        nob = sum(sizes[i] * sizes[i + 1] for i in range(3))
        arrow = " \\to ".join(map(str, sizes))
        parts = " + ".join(f"({sizes[i]}\\cdot{sizes[i+1]} + {sizes[i+1]})" for i in range(3))
        return Problem(
            prompt=f"An MLP has layer widths ${arrow}$ (all layers dense with biases). Count its trainable parameters.",
            answer=str(ans), kind="num",
            rubric=["One weight matrix per consecutive pair of widths", "One bias vector per non-input layer", "Sum the layers"],
            solution=f"${parts} = {ans}$.",
            misconceptions=[(str(nob), "forgot-bias")],
        )
    din, dout = rng.choice([784, 100, 64, 50, 20]), rng.choice([10, 5, 2, 1])
    h = rng.choice([16, 32, 50, 64, 100, 128, 200, 256])
    P = h * (din + 1) + dout * (h + 1)
    return Problem(
        prompt=(f"A one-hidden-layer MLP maps ${din}$ inputs $\\to h \\to {dout}$ outputs (dense, with biases). "
                f"It has exactly ${P}$ parameters. Derive a formula for the parameter count as a function of $h$ and solve for $h$."),
        answer=str(h), kind="num",
        rubric=["Write P(h) = h(d_in + 1) + d_out(h + 1)", "Collect: P = h(d_in + d_out + 1) + d_out", "Solve the linear equation for h"],
        solution=(f"$P(h) = h({din}+1) + {dout}(h+1) = h\\cdot{din + dout + 1} + {dout}$. "
                  f"So $h = ({P} - {dout})/{din + dout + 1} = {h}$."),
    )


def _pn(v) -> str:
    """LaTeX of v, parenthesised when negative (for products like 2·(-3))."""
    v = sp.sympify(v)
    return f"\\left({L(v)}\\right)" if v < 0 else L(v)


def _xm(t: int) -> str:
    """'x - t' with a clean sign."""
    return "x" if t == 0 else (f"x - {t}" if t > 0 else f"x + {-t}")


@gen("dl.mlp", levels=(1, 2, 3))
def mlp_batch(rng, level):
    if level == 1:
        while True:
            B = rng.choice([4, 8, 16, 32, 64, 128])
            nin = rng.choice([3, 10, 64, 128, 512, 784])
            nout = rng.choice([2, 10, 32, 100, 256])
            if len({B, nin, nout}) == 3:
                break
        ans = [nout, nin, B, nout]
        return Problem(
            prompt=(f"Shapes in a dense layer: PyTorch `nn.Linear({nin}, {nout})` stores its weight $W$ with shape "
                    "$(n_{out}, n_{in})$, and a mini-batch $X$ holds one example per **row**. For a batch of "
                    f"${B}$ examples, give the shape of $W$ and the shape of the output $Y = XW^\\top + \\mathbf b$."),
            answer=[str(v) for v in ans], kind="list",
            answer_hint="rows of W, cols of W, rows of Y, cols of Y   e.g. 10, 784, 32, 10",
            rubric=["W maps n_in -> n_out and is stored as (n_out, n_in)", f"X has shape ({B}, {nin})",
                    "X W^T: (B, n_in)(n_in, n_out) -> (B, n_out)", "The bias is broadcast over the batch rows"],
            solution=(f"$W$ is ${nout}\\times{nin}$; $X$ is ${B}\\times{nin}$ and $W^\\top$ is ${nin}\\times{nout}$, so "
                      f"$Y$ is ${B}\\times{nout}$ — one output row per example."),
            misconceptions=[([str(nin), str(nout), str(B), str(nout)], "transposed-weight-shape"),
                            ([str(nout), str(nin), str(nout), str(B)], "transposed-output-shape")],
        )
    if level == 2:
        while True:
            X = sp.Matrix(2, 3, lambda i, j: rng.randint(-2, 2))
            W = sp.Matrix(2, 3, lambda i, j: rng.randint(-2, 2))
            b = [rng.randint(-2, 2) for _ in range(2)]
            Z = X * W.T + sp.Matrix(2, 2, lambda i, j: b[j])
            if 0 not in list(Z) and any(v < 0 for v in Z) and any(v > 0 for v in Z):
                break
        H = Z.applyfunc(lambda v: max(v, 0))
        Hnb = (X * W.T).applyfunc(lambda v: max(v, 0))
        return Problem(
            prompt=(f"Batched forward pass: two examples are stacked as the rows of $X = {L(X)}$. A dense ReLU layer with "
                    f"$W = {L(W)}$ (shape $n_{{out}}\\times n_{{in}}$) and bias $\\mathbf b = {tup(b)}$ computes "
                    "$H = \\mathrm{ReLU}(XW^\\top + \\mathbf b)$, the bias being added to every row. Give $H$ (row $i$ = example $i$)."),
            answer=fmt(H), kind="matrix", answer_hint="a 2x2 matrix, e.g. [[1,0],[3,2]]",
            rubric=["(X W^T)_ij = x_i · w_j, where w_j is row j of W", "Add b_j to column j of every row",
                    "Apply ReLU elementwise"],
            solution=(f"$XW^\\top = {L(X * W.T)}$; adding $\\mathbf b$ to each row gives $Z = {L(Z)}$, so "
                      f"$H = \\mathrm{{ReLU}}(Z) = {L(H)}$. One matrix product handles the whole batch."),
            misconceptions=[(fmt(Z), "forgot-activation"), (fmt(Hnb), "forgot-bias")],
        )
    while True:
        t = sorted(rng.sample(range(-3, 5), 3))
        s = [nz(rng, -3, 3) for _ in range(3)]
        if s[1] != s[0] and s[2] != s[1]:
            break
    c = [s[0], s[1] - s[0], s[2] - s[1]]
    f2 = s[0] * (t[1] - t[0])
    f3 = f2 + s[1] * (t[2] - t[1])
    relus = " + ".join(f"c_{k + 1}\\,\\mathrm{{ReLU}}({_xm(t[k])})" for k in range(3))
    return Problem(
        prompt=(f"Universal approximation, constructively: a continuous piecewise-linear function $f$ has $f(x) = 0$ for "
                f"$x \\le {t[0]}$, is linear on $[{t[0]}, {t[1]}]$ and on $[{t[1]}, {t[2]}]$ with $f({t[1]}) = {f2}$ and "
                f"$f({t[2]}) = {f3}$, and has slope ${s[2]}$ for $x \\ge {t[2]}$. Show that $f$ is exactly a one-hidden-layer "
                f"ReLU network $f(x) = {relus}$ and derive $(c_1, c_2, c_3)$."),
        answer=[str(v) for v in c], kind="list", answer_hint="c1, c2, c3",
        rubric=["Slopes of the pieces: s1 = f(t2)/(t2 - t1), s2 = (f(t3) - f(t2))/(t3 - t2), s3 given",
                "ReLU(x - t_k) is 0 left of t_k and adds slope c_k right of it",
                "So s1 = c1, s2 = c1 + c2, s3 = c1 + c2 + c3", "Each c_k is the change of slope at t_k"],
        solution=(f"Slopes: $s_1 = {f2}/{t[1] - t[0]} = {s[0]}$, $s_2 = ({f3} - {_pn(f2)})/{t[2] - t[1]} = {s[1]}$, $s_3 = {s[2]}$. "
                  f"Right of $t_k$ the slope is $c_1 + \\dots + c_k$, so $c_1 = {c[0]}$, $c_2 = s_2 - s_1 = {c[1]}$, "
                  f"$c_3 = s_3 - s_2 = {c[2]}$. Every continuous piecewise-linear function is a sum of shifted ReLUs — "
                  "the constructive core of the universal approximation theorem."),
        misconceptions=[([str(v) for v in s], "used-slopes-not-slope-changes")],
    )


# =========================================================================== dl.backprop
@gen("dl.backprop", levels=(1, 2, 3))
def backprop_graph(rng, level):
    if level == 1:
        a, b, c, d = (nz(rng, -4, 4) for _ in range(4))
        g = [b * d, a * d, d, a * b + c]
        return Problem(
            prompt=(f"Computational graph: $e = a\\cdot b$, $s = e + c$, $f = s\\cdot d$ with $a={a}, b={b}, c={c}, d={d}$. "
                    "Run backprop by hand and give $\\left(\\frac{\\partial f}{\\partial a}, \\frac{\\partial f}{\\partial b}, "
                    "\\frac{\\partial f}{\\partial c}, \\frac{\\partial f}{\\partial d}\\right)$."),
            answer=[str(v) for v in g], kind="list", answer_hint="df/da, df/db, df/dc, df/dd",
            rubric=["Forward pass: e, s, f", "df/ds = d and df/dd = s", "Through the + node: df/de = df/dc = df/ds",
                    "Through the * node: df/da = df/de · b, df/db = df/de · a"],
            solution=(f"Forward: $e={a*b}$, $s={a*b+c}$, $f={(a*b+c)*d}$. Backward: $\\bar f = 1$, $\\bar s = d = {d}$, "
                      f"$\\bar d = s = {a*b+c}$, $\\bar e = \\bar c = \\bar s = {d}$, $\\bar a = \\bar e\\, b = {b*d}$, $\\bar b = \\bar e\\, a = {a*d}$."),
            misconceptions=[([str(b), str(a), "1", str(a * b + c)], "forgot-upstream-gradient")],
        )
    if level == 2:
        a, b, c = (nz(rng, -4, 4) for _ in range(3))
        g = [b + c, (b + c) + (a + b), (a + b) + 2 * c]
        return Problem(
            prompt=(f"Let $u = a + b$, $v = b + c$, $f = u\\cdot v + c^2$ with $a={a}, b={b}, c={c}$. Note that $b$ and $c$ "
                    "each feed into **two** nodes. Backpropagate and give "
                    "$\\left(\\frac{\\partial f}{\\partial a}, \\frac{\\partial f}{\\partial b}, \\frac{\\partial f}{\\partial c}\\right)$."),
            answer=[str(v) for v in g], kind="list", answer_hint="df/da, df/db, df/dc",
            rubric=["Forward pass: u, v, f", "df/du = v, df/dv = u", "Gradients from multiple paths ADD (+=, not =)",
                    "df/dc collects both the v-path and the c^2 path"],
            solution=(f"$u={a+b}$, $v={b+c}$. $\\bar u = v = {b+c}$, $\\bar v = u = {a+b}$. "
                      f"$\\bar a = \\bar u = {g[0]}$; $\\bar b = \\bar u + \\bar v = {g[1]}$; $\\bar c = \\bar v + 2c = {g[2]}$. "
                      "This is why micrograd uses `grad +=` and must zero gradients before each backward pass."),
            misconceptions=[([str(b + c), str(b + c), str(a + b)], "no-gradient-accumulation"),
                            ([str(b + c), str(a + b), str(2 * c)], "overwrote-gradient")],
        )
    n = 3
    while True:
        a = [rng.randint(1, 6) for _ in range(n)]
        if len(set(a)) > 1:
            break
    y = rng.randrange(n)
    S = sum(a)
    p = [R(ai, S) for ai in a]
    grad = [p[i] - (1 if i == y else 0) for i in range(n)]
    logits = ",\\ ".join(f"\\ln {ai}" if ai != 1 else "0" for ai in a)
    return Problem(
        prompt=(f"Logits $\\mathbf z = ({logits})$ go through softmax $p_i = e^{{z_i}}/\\sum_j e^{{z_j}}$ and the "
                f"cross-entropy loss $\\mathcal L = -\\log p_y$ with true class $y = {y + 1}$ (1-indexed). "
                "Derive $\\partial \\mathcal L/\\partial z_i$ in general, then give the gradient vector $\\nabla_{\\mathbf z}\\mathcal L$ exactly."),
        answer=[fmt(v) for v in grad], kind="vector",
        rubric=["Write L = -z_y + log sum_j exp(z_j)", "Differentiate: dL/dz_i = p_i - 1[i = y]",
                "Compute p from e^{ln a_i} = a_i", "Subtract 1 at the true class"],
        solution=(f"$\\mathcal L = -z_y + \\log\\sum_j e^{{z_j}} \\Rightarrow \\partial\\mathcal L/\\partial z_i = p_i - \\mathbb 1[i=y]$. "
                  f"Here $e^{{z_i}} = {tup(a)}$, so $\\mathbf p = {tup(p)}$ and $\\nabla = {tup(grad)}$. The gradient entries sum to 0."),
        misconceptions=[([fmt(-v) for v in grad], "sign-flip"), ([fmt(v) for v in p], "forgot-onehot")],
    )


def _mlp221(rng):
    while True:
        xv = [nz(rng, -2, 2) for _ in range(2)]
        W1 = [[rng.randint(-2, 2) for _ in range(2)] for _ in range(2)]
        b1 = [rng.randint(-1, 1) for _ in range(2)]
        z1 = [W1[i][0] * xv[0] + W1[i][1] * xv[1] + b1[i] for i in range(2)]
        if 0 in z1 or not any(v > 0 for v in z1):
            continue
        w2 = [nz(rng, -2, 2) for _ in range(2)]
        b2 = rng.randint(-1, 1)
        h = [max(v, 0) for v in z1]
        yhat = w2[0] * h[0] + w2[1] * h[1] + b2
        yt = rng.randint(-3, 3)
        if yhat != yt:
            return xv, W1, b1, z1, h, w2, b2, yhat, yt


@gen("dl.backprop", levels=(1, 2, 3))
def backprop_mlp(rng, level):
    xv, W1, b1, z1, h, w2, b2, yhat, yt = _mlp221(rng)
    delta = yhat - yt
    dh = [delta * w2[0], delta * w2[1]]
    mask = [1 if v > 0 else 0 for v in z1]
    dz = [dh[i] * mask[i] for i in range(2)]
    dW1 = sp.Matrix(2, 2, lambda i, j: dz[i] * xv[j])
    dW1_nomask = sp.Matrix(2, 2, lambda i, j: dh[i] * xv[j])
    dx = sp.Matrix(W1).T * sp.Matrix(dz)
    setup = (f"Network: $\\mathbf z_1 = W_1\\mathbf x + \\mathbf b_1$, $\\mathbf h = \\mathrm{{ReLU}}(\\mathbf z_1)$, "
             f"$\\hat y = \\mathbf w_2^\\top\\mathbf h + b_2$, loss $\\mathcal L = \\tfrac12(\\hat y - y)^2$, with "
             f"$W_1 = {mat(W1)}$, $\\mathbf b_1 = {tup(b1)}$, $\\mathbf w_2 = {tup(w2)}$, $b_2 = {b2}$, "
             f"$\\mathbf x = {tup(xv)}$, target $y = {yt}$. ")
    fwd = (f"Forward: $\\mathbf z_1 = {tup(z1)}$, $\\mathbf h = {tup(h)}$, $\\hat y = {yhat}$, "
           f"$\\delta = \\partial\\mathcal L/\\partial\\hat y = \\hat y - y = {delta}$. ")
    if level == 1:
        ans = [delta * h[0], delta * h[1], delta]
        return Problem(
            prompt=setup + "Give $\\left(\\frac{\\partial\\mathcal L}{\\partial w_{2,1}}, \\frac{\\partial\\mathcal L}{\\partial w_{2,2}}, \\frac{\\partial\\mathcal L}{\\partial b_2}\\right)$.",
            answer=[str(v) for v in ans], kind="list", answer_hint="dL/dw21, dL/dw22, dL/db2",
            rubric=["Forward pass to get h and y_hat", "dL/dy_hat = y_hat - y", "dL/dw2 = delta · h, dL/db2 = delta"],
            solution=fwd + f"$\\partial\\mathcal L/\\partial\\mathbf w_2 = \\delta\\,\\mathbf h = {tup(ans[:2])}$, $\\partial\\mathcal L/\\partial b_2 = {delta}$.",
            misconceptions=[([str(h[0]), str(h[1]), "1"], "forgot-loss-gradient"),
                            ([str(-delta * h[0]), str(-delta * h[1]), str(-delta)], "sign-flip")],
        )
    back = (f"Backward: $\\partial\\mathcal L/\\partial\\mathbf h = \\delta\\,\\mathbf w_2 = {tup(dh)}$; ReLU mask $= {tup(mask)}$ "
            f"so $\\partial\\mathcal L/\\partial\\mathbf z_1 = {tup(dz)}$. ")
    if level == 2:
        return Problem(
            prompt=setup + "Backpropagate all the way to the first layer and give the $2\\times2$ matrix $\\partial\\mathcal L/\\partial W_1$.",
            answer=fmt(dW1), kind="matrix",
            rubric=["Forward pass (keep z1 for the ReLU mask)", "dL/dh = delta · w2",
                    "Multiply by ReLU'(z1) = 1[z1 > 0]", "dL/dW1 = (dL/dz1) x^T (outer product)"],
            solution=fwd + back + f"$\\partial\\mathcal L/\\partial W_1 = (\\partial\\mathcal L/\\partial\\mathbf z_1)\\,\\mathbf x^\\top = {L(dW1)}$.",
            misconceptions=[(fmt(dW1_nomask), "ignored-relu-mask"), (fmt(dW1.T), "transposed-outer-product")],
        )
    wrong = sp.Matrix(W1) * sp.Matrix(dz)
    return Problem(
        prompt=setup + "Derive the gradient with respect to the **input**, $\\nabla_{\\mathbf x}\\mathcal L$ (used e.g. for saliency maps and adversarial examples).",
        answer=[str(v) for v in dx], kind="vector",
        rubric=["Forward pass", "Backprop to dL/dz1 through w2 and the ReLU mask",
                "dL/dx = W1^T (dL/dz1) — transpose, because z1 = W1 x"],
        solution=fwd + back + f"$\\nabla_{{\\mathbf x}}\\mathcal L = W_1^\\top\\,\\partial\\mathcal L/\\partial\\mathbf z_1 = {tup(list(dx))}$.",
        misconceptions=[([str(v) for v in wrong], "forgot-transpose")],
    )


@gen("dl.backprop", levels=(1, 2, 3))
def backprop_gates(rng, level):
    if level == 1:
        while True:
            x, y, z = nz(rng, -5, 5), nz(rng, -5, 5), nz(rng, -4, 4)
            if x != y:
                break
        m = max(x, y)
        g = [z if x > y else 0, z if y > x else 0, m]
        return Problem(
            prompt=(f"Gate routing in backprop: $m = \\max(x, y)$ and $f = m\\cdot z$ with $x = {x}$, $y = {y}$, $z = {z}$. "
                    "Give $\\left(\\frac{\\partial f}{\\partial x}, \\frac{\\partial f}{\\partial y}, \\frac{\\partial f}{\\partial z}\\right)$."),
            answer=[str(v) for v in g], kind="list", answer_hint="df/dx, df/dy, df/dz",
            rubric=["Forward: m = max(x, y), f = m z", "Multiply gate: df/dm = z and df/dz = m (it swaps its inputs)",
                    "Max gate routes the whole upstream gradient to the larger input and 0 to the other"],
            solution=(f"$m = {m}$, $f = {m * z}$. $\\bar m = z = {z}$, $\\bar z = m = {m}$. The max gate sends $\\bar m$ to the "
                      f"winner (${'x' if x > y else 'y'}$) and $0$ to the loser, so the gradient is $({g[0]}, {g[1]}, {g[2]})$."),
            misconceptions=[([fmt(R(z, 2)), fmt(R(z, 2)), str(m)], "max-gate-split-gradient"),
                            ([str(1 if x > y else 0), str(1 if y > x else 0), str(m)], "forgot-upstream-gradient")],
        )
    if level == 2:
        while True:
            w, b = rng.randint(-2, 2), rng.randint(-2, 2)
            xs = [nz(rng, -3, 3) for _ in range(2)]
            ys = [rng.randint(-4, 4) for _ in range(2)]
            r = [w * xs[i] + b - ys[i] for i in range(2)]
            if xs[0] != xs[1] and any(r):
                break
        eta = rng.choice([R(1, 2), R(1, 4), R(1, 10)])
        gw = R(r[0] * xs[0] + r[1] * xs[1], 2)
        gb = R(r[0] + r[1], 2)
        new = [w - eta * gw, b - eta * gb]
        return Problem(
            prompt=(f"One step of gradient descent by hand: the model $\\hat y = wx + b$ starts at $w = {w}$, $b = {b}$. The data are "
                    f"$(x, y) = ({xs[0]}, {ys[0]})$ and $({xs[1]}, {ys[1]})$, and the loss is the **mean** over the two points of "
                    f"$\\tfrac12(\\hat y_i - y_i)^2$. Take one step with learning rate $\\eta = {L(eta)}$ and give the new $(w, b)$."),
            answer=[fmt(v) for v in new], kind="list", answer_hint="w, b   e.g. 3/2, -1/4",
            rubric=["Forward: residuals r_i = y_hat_i - y_i", "dL/dw = mean(r_i x_i), dL/db = mean(r_i)",
                    "Update: w <- w - eta dL/dw, b <- b - eta dL/db"],
            solution=(f"Residuals $r = ({r[0]}, {r[1]})$. $\\partial\\mathcal L/\\partial w = \\tfrac12({_pn(r[0])}\\cdot{_pn(xs[0])} + {_pn(r[1])}\\cdot{_pn(xs[1])}) = {L(gw)}$, "
                      f"$\\partial\\mathcal L/\\partial b = \\tfrac12({r[0]} + {_pn(r[1])}) = {L(gb)}$. New $w = {w} - {L(eta)}\\cdot{_pn(gw)} = {L(new[0])}$, "
                      f"$b = {b} - {L(eta)}\\cdot{_pn(gb)} = {L(new[1])}$."),
            misconceptions=[([fmt(w + eta * gw), fmt(b + eta * gb)], "gradient-ascent"),
                            ([fmt(w - 2 * eta * gw), fmt(b - 2 * eta * gb)], "summed-not-averaged")],
        )
    while True:
        a = [rng.randint(1, 5) for _ in range(3)]
        if len(set(a)) > 1:
            break
    S = sum(a)
    p = sp.Matrix([R(ai, S) for ai in a])
    J = sp.diag(*p) - p * p.T
    logits = ",\\ ".join(f"\\ln {ai}" if ai != 1 else "0" for ai in a)
    return Problem(
        prompt=("Softmax Jacobian: for $\\mathbf p = \\mathrm{softmax}(\\mathbf z)$ derive $\\partial p_i/\\partial z_j$ in general, then "
                f"evaluate the $3\\times3$ Jacobian $J_{{ij}} = \\partial p_i/\\partial z_j$ at $\\mathbf z = ({logits})$ exactly."),
        answer=fmt(J), kind="matrix", answer_hint="a 3x3 matrix of fractions, e.g. [[2/9,-1/9,-1/9],[...],[...]]",
        rubric=["Quotient rule on p_i = e^{z_i} / sum_k e^{z_k}", "dp_i/dz_j = p_i (1[i=j] - p_j), i.e. J = diag(p) - p p^T",
                "Compute p from e^{ln a_i} = a_i", "Check: every row sums to 0 (probabilities stay normalised)"],
        solution=(f"$\\partial p_i/\\partial z_j = p_i(\\delta_{{ij}} - p_j)$, so $J = \\mathrm{{diag}}(\\mathbf p) - \\mathbf p\\mathbf p^\\top$. "
                  f"Here $\\mathbf p = {tup(list(p))}$ and $J = {L(J)}$. Rows sum to zero because $\\sum_i p_i = 1$ for every $\\mathbf z$."),
        misconceptions=[(fmt(sp.diag(*[pi * (1 - pi) for pi in p])), "ignored-cross-terms"),
                        (fmt(-p * p.T), "forgot-diagonal-term")],
    )


# =========================================================================== dl.init_activations
@gen("dl.init_activations", levels=(1, 2, 3))
def activations_init(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            pv = rng.choice([R(1, 4), R(1, 3), R(1, 2), R(2, 3), R(3, 4), R(1, 5), R(4, 5), R(2, 5)])
            ans = pv * (1 - pv)
            return Problem(
                prompt=f"For the sigmoid $\\sigma(z) = 1/(1+e^{{-z}})$, at some point $z_0$ we have $\\sigma(z_0) = {L(pv)}$. Find $\\sigma'(z_0)$ exactly.",
                answer=fmt(ans), kind="num",
                rubric=["Derive sigma'(z) = sigma(z)(1 - sigma(z))", "Substitute the given value"],
                solution=f"$\\sigma' = \\sigma(1-\\sigma) = {L(pv)}\\cdot{L(1 - pv)} = {L(ans)}$. (Max is $1/4$ at $z=0$ — sigmoids shrink gradients.)",
                misconceptions=[(fmt(pv), "derivative-equals-function"), (fmt(1 - pv), "forgot-factor")],
            )
        t = rng.choice([R(1, 2), R(-1, 2), R(1, 3), R(-2, 3), R(3, 5), R(-4, 5), R(1, 4)])
        ans = 1 - t ** 2
        return Problem(
            prompt=f"At some point $z_0$, $\\tanh(z_0) = {L(t)}$. Find $\\frac{{d}}{{dz}}\\tanh(z)$ at $z_0$ exactly.",
            answer=fmt(ans), kind="num",
            rubric=["Derive tanh'(z) = 1 - tanh^2(z)", "Substitute the given value"],
            solution=f"$\\tanh' = 1 - \\tanh^2 = 1 - {L(t**2)} = {L(ans)}$.",
            misconceptions=[(fmt(1 - t), "forgot-square"), (fmt(t * (1 - t)), "used-sigmoid-formula")],
        )
    if level == 2:
        nin = rng.choice([64, 128, 256, 300, 512, 768, 1024])
        nout = rng.choice([32, 64, 100, 128, 256, 512, 1024])
        scheme = rng.choice(["he_normal", "xavier_normal", "xavier_uniform", "he_uniform"])
        if scheme == "he_normal":
            ans, what = sp.sqrt(R(2, nin)), "the **standard deviation** of He (Kaiming) normal initialisation for a ReLU layer"
            mis = [(fmt(sp.sqrt(R(2, nout))), "used-fan-out"), (fmt(R(2, nin)), "variance-not-std")]
            sol = f"He: $\\mathrm{{Var}}(w) = 2/n_{{in}}$, so $\\sigma = \\sqrt{{2/{nin}}} = {L(ans)} \\approx {sig(ans)}$."
        elif scheme == "xavier_normal":
            ans, what = sp.sqrt(R(2, nin + nout)), "the **standard deviation** of Xavier (Glorot) normal initialisation"
            mis = [(fmt(sp.sqrt(R(1, nin))), "used-fan-in-only"), (fmt(R(2, nin + nout)), "variance-not-std")]
            sol = f"Xavier: $\\mathrm{{Var}}(w) = 2/(n_{{in}}+n_{{out}})$, so $\\sigma = \\sqrt{{2/{nin + nout}}} = {L(ans)} \\approx {sig(ans)}$."
        elif scheme == "xavier_uniform":
            ans, what = sp.sqrt(R(6, nin + nout)), "the bound $a$ of Xavier (Glorot) **uniform** initialisation $w \\sim U(-a, a)$"
            mis = [(fmt(sp.sqrt(R(2, nin + nout))), "used-normal-std-as-bound")]
            sol = (f"Want $\\mathrm{{Var}}(w) = 2/(n_{{in}}+n_{{out}})$; for $U(-a,a)$, $\\mathrm{{Var}} = a^2/3$, so "
                   f"$a = \\sqrt{{6/(n_{{in}}+n_{{out}})}} = \\sqrt{{6/{nin + nout}}} = {L(ans)} \\approx {sig(ans)}$.")
        else:
            ans, what = sp.sqrt(R(6, nin)), "the bound $a$ of He (Kaiming) **uniform** initialisation $w\\sim U(-a,a)$ for a ReLU layer"
            mis = [(fmt(sp.sqrt(R(2, nin))), "used-normal-std-as-bound")]
            sol = (f"Want $\\mathrm{{Var}}(w) = 2/n_{{in}}$; $U(-a,a)$ has variance $a^2/3$, so "
                   f"$a = \\sqrt{{6/n_{{in}}}} = \\sqrt{{6/{nin}}} = {L(ans)} \\approx {sig(ans)}$.")
        return Problem(
            prompt=f"A dense layer has fan-in $n_{{in}} = {nin}$ and fan-out $n_{{out}} = {nout}$. Compute {what}. {SIG4}",
            answer=fmt(ans), kind="num", tol=TOL,
            rubric=["Recall the target variance of the scheme", "If uniform: use Var(U(-a,a)) = a^2/3",
                    "Take the square root"],
            solution=sol, misconceptions=mis,
        )
    c = rng.choice([1, 3, 4, R(1, 2), R(3, 2), R(5, 2)])
    Ln = rng.randint(3, 10)
    ans = (sp.sympify(c) / 2) ** Ln
    return Problem(
        prompt=(f"A deep ReLU MLP has $L = {Ln}$ layers of width $n$: $\\mathbf z_\\ell = W_\\ell \\mathbf x_{{\\ell-1}}$, "
                f"$\\mathbf x_\\ell = \\mathrm{{ReLU}}(\\mathbf z_\\ell)$, weights i.i.d. zero-mean with $\\mathrm{{Var}}(w) = {L(c)}/n$ "
                "(no biases). Assume each $z$ is symmetric around 0, so $\\mathbb E[\\mathrm{ReLU}(z)^2] = \\tfrac12\\mathrm{Var}(z)$. "
                "By what factor is the mean squared activation $\\mathbb E[x^2]$ multiplied from the input to the last layer? "
                "(This is the derivation behind He initialisation.) Exact fraction or at least 4 significant figures."),
        answer=fmt(ans), kind="num", tol=TOL,
        rubric=["Var(z_l) = n Var(w) E[x_{l-1}^2] (independence, zero mean)",
                "E[x_l^2] = Var(z_l)/2 for ReLU", f"Per-layer factor n·Var(w)/2 = {L(c)}/2",
                f"Raise to the power L = {Ln}", "Conclude Var(w) = 2/n keeps the factor at 1"],
        solution=(f"$\\mathrm{{Var}}(z_\\ell) = n\\,\\mathrm{{Var}}(w)\\,\\mathbb E[x_{{\\ell-1}}^2] = {L(c)}\\,\\mathbb E[x_{{\\ell-1}}^2]$ and "
                  f"$\\mathbb E[x_\\ell^2] = \\tfrac12\\mathrm{{Var}}(z_\\ell)$, so each layer multiplies by ${L(sp.sympify(c) / 2)}$. "
                  f"After ${Ln}$ layers: $({L(sp.sympify(c) / 2)})^{{{Ln}}} = {L(ans)}$. Choosing $\\mathrm{{Var}}(w) = 2/n$ makes the factor exactly 1 "
                  "(He et al. 2015, arXiv:1502.01852)."),
        misconceptions=[(fmt(sp.sympify(c) ** Ln), "forgot-relu-halving")],
    )


@gen("dl.init_activations", levels=(1, 2, 3))
def layernorm(rng, level):
    while True:
        xv = [rng.randint(-4, 6) for _ in range(3)]
        if len(set(xv)) > 1:
            break
    mu = R(sum(xv), 3)
    var = sum((R(v) - mu) ** 2 for v in xv) / 3
    xhat = [(v - mu) / sp.sqrt(var) for v in xv]
    if level == 1:
        return Problem(
            prompt=f"LayerNorm statistics: for $\\mathbf x = {tup(xv)}$ compute the mean $\\mu$ and the (biased, divide-by-$n$) variance $\\sigma^2$ over the features.",
            answer=[fmt(mu), fmt(var)], kind="list", answer_hint="mu, sigma^2   e.g. 1/3, 14/9",
            rubric=["mu = (1/n) sum x_i", "sigma^2 = (1/n) sum (x_i - mu)^2  (divide by n, not n-1)"],
            solution=f"$\\mu = {L(mu)}$, $\\sigma^2 = \\tfrac13\\sum(x_i-\\mu)^2 = {L(var)}$.",
            misconceptions=[([fmt(mu), fmt(var * R(3, 2))], "unbiased-variance")],
        )
    if level == 2:
        return Problem(
            prompt=(f"Apply LayerNorm with $\\gamma = \\mathbf 1$, $\\beta = \\mathbf 0$, $\\varepsilon = 0$ to $\\mathbf x = {tup(xv)}$: "
                    f"$\\hat x_i = (x_i - \\mu)/\\sqrt{{\\sigma^2}}$ (biased variance). Give $\\hat{{\\mathbf x}}$. {SIG4}"),
            answer=[fmt(v) for v in xhat], kind="vector", tol=TOL,
            rubric=["Compute mu", "Compute the biased variance", "Subtract the mean and divide by the standard deviation",
                    "Sanity check: result has mean 0 and mean-square 1"],
            solution=f"$\\mu = {L(mu)}$, $\\sigma^2 = {L(var)}$, $\\hat{{\\mathbf x}} = {tup(xhat)} \\approx ({', '.join(sig(v) for v in xhat)})$.",
            misconceptions=[([fmt((v - mu) / var) for v in xv], "divided-by-variance"),
                            ([fmt((v - mu) / sp.sqrt(var * R(3, 2))) for v in xv], "unbiased-variance")],
        )
    gam = [nz(rng, -2, 3) for _ in range(3)]
    bet = [rng.randint(-2, 2) for _ in range(3)]
    y = [gam[i] * xhat[i] + bet[i] for i in range(3)]
    return Problem(
        prompt=(f"LayerNorm (Ba, Kiros & Hinton 2016) with learned gain and bias: $y_i = \\gamma_i\\,\\frac{{x_i-\\mu}}{{\\sigma}} + \\beta_i$ "
                f"($\\varepsilon=0$, biased variance). With $\\mathbf x = {tup(xv)}$, $\\boldsymbol\\gamma = {tup(gam)}$, "
                f"$\\boldsymbol\\beta = {tup(bet)}$, compute $\\mathbf y$. {SIG4}"),
        answer=[fmt(v) for v in y], kind="vector", tol=TOL,
        rubric=["Statistics are over the feature dimension of ONE example (not over the batch)", "Normalise",
                "Scale by gamma and shift by beta elementwise"],
        solution=(f"$\\mu = {L(mu)}$, $\\sigma = \\sqrt{{{L(var)}}}$, $\\hat{{\\mathbf x}} = {tup(xhat)}$, "
                  f"$\\mathbf y = \\boldsymbol\\gamma\\odot\\hat{{\\mathbf x}} + \\boldsymbol\\beta = {tup(y)} \\approx ({', '.join(sig(v) for v in y)})$. "
                  "Unlike BatchNorm, the result does not depend on other examples (arXiv:1607.06450)."),
        misconceptions=[([fmt(v) for v in xhat], "forgot-affine"),
                         ([fmt(bet[i] * xhat[i] + gam[i]) for i in range(3)], "swapped-gamma-beta")],
    )


@gen("dl.init_activations", levels=(1, 2, 3))
def activation_identities(rng, level):
    if level == 1:
        n = rng.randint(2, 6)
        w = rng.choice([1, 2, 3])
        ans = R(w, 4) ** n
        mis = [(fmt(R(w ** n, 4)), "applied-bound-once")]
        if w != 1:
            mis.append((fmt(R(1, 4) ** n), "forgot-weight-factor"))
        return Problem(
            prompt=(f"Vanishing gradients through saturating units: a scalar passes through ${n}$ stacked sigmoid units "
                    f"$h_k = \\sigma(w\\,h_{{k-1}})$, all with weight $w = {w}$. Using $\\max_z \\sigma'(z) = \\tfrac14$, give the largest "
                    f"possible value of $|\\partial h_{{{n}}}/\\partial h_0|$ (exact)."),
            answer=fmt(ans), kind="num",
            rubric=["Chain rule: dh_n/dh_0 = prod_k w sigma'(w h_{k-1})", "Bound each factor by |w|/4",
                    "Multiply n factors: (|w|/4)^n"],
            solution=(f"Each layer contributes $w\\,\\sigma'(\\cdot) \\le {w}/4$, so $|\\partial h_{{{n}}}/\\partial h_0| \\le ({L(R(w, 4))})^{{{n}}} = {L(ans)}$ — "
                      "exponentially small in depth, the classic vanishing-gradient problem of sigmoid networks."),
            misconceptions=mis,
        )
    if level == 2:
        while True:
            xv = [rng.randint(-4, 4) for _ in range(4)]
            ms = R(sum(v * v for v in xv), 4)
            rt = sp.sqrt(ms)
            if ms > 0 and rt.is_Rational and len(set(xv)) > 1:
                break
        g = [nz(rng, -2, 2) for _ in range(4)]
        y = [g[i] * xv[i] / rt for i in range(4)]
        mis = [([fmt(g[i] * xv[i] / sp.sqrt(4 * ms)) for i in range(4)], "forgot-mean-in-rms"),
               ([fmt(xv[i] / rt) for i in range(4)], "forgot-gain")]
        return Problem(
            prompt=("RMSNorm (Zhang & Sennrich 2019; used in Llama) drops LayerNorm's mean subtraction: "
                    "$y_i = g_i\\,x_i/\\mathrm{RMS}(\\mathbf x)$ with $\\mathrm{RMS}(\\mathbf x) = \\sqrt{\\tfrac1n\\sum_i x_i^2}$ ($\\varepsilon = 0$). "
                    f"For $\\mathbf x = {tup(xv)}$ and gain $\\mathbf g = {tup(g)}$, compute $\\mathbf y$ exactly."),
            answer=[fmt(v) for v in y], kind="vector",
            rubric=["Mean of squares (1/n) sum x_i^2", "RMS = its square root", "Divide x by RMS (no centring)",
                    "Multiply elementwise by g"],
            solution=(f"$\\tfrac14\\sum x_i^2 = {L(ms)}$, $\\mathrm{{RMS}} = {L(rt)}$, $\\mathbf x/\\mathrm{{RMS}} = {tup([v / rt for v in xv])}$, "
                      f"$\\mathbf y = {tup(y)}$. RMSNorm is cheaper than LayerNorm and works as well in practice (arXiv:1910.07467)."),
            misconceptions=mis,
        )
    pv = rng.choice([R(2, 3), R(3, 4), R(4, 5), R(1, 3), R(1, 4), R(1, 5), R(3, 5), R(2, 5)])
    q = (1 - pv) / pv
    th = (1 - q ** 2) / (1 + q ** 2)
    ans = [1 - pv, 2 * pv - 1, th]
    return Problem(
        prompt=("Derive the identities $\\sigma(-z) = 1 - \\sigma(z)$ and $\\tanh(z) = 2\\sigma(2z) - 1$ from the definitions "
                f"$\\sigma(z) = 1/(1+e^{{-z}})$, $\\tanh z = (e^z - e^{{-z}})/(e^z + e^{{-z}})$. Knowing only that $\\sigma(z_0) = {L(pv)}$, "
                "give $\\sigma(-z_0)$, $\\tanh(z_0/2)$ and $\\tanh(z_0)$ exactly."),
        answer=[fmt(v) for v in ans], kind="list", answer_hint="three fractions: sigma(-z0), tanh(z0/2), tanh(z0)",
        rubric=["sigma(-z) = e^{-z}... = 1 - sigma(z)", "Multiply num/den of tanh by e^{-z}: tanh z = 2 sigma(2z) - 1",
                "So tanh(z0/2) = 2 sigma(z0) - 1", "For tanh(z0): q = e^{-z0} = (1 - p)/p, tanh z0 = (1 - q^2)/(1 + q^2)"],
        solution=(f"$\\sigma(-z_0) = 1 - {L(pv)} = {L(1 - pv)}$. $\\tanh(z_0/2) = 2\\sigma(z_0) - 1 = {L(2 * pv - 1)}$. "
                  f"From $\\sigma(z_0) = 1/(1+e^{{-z_0}})$: $e^{{-z_0}} = (1-p)/p = {L(q)}$, and "
                  f"$\\tanh z_0 = \\frac{{1 - e^{{-2z_0}}}}{{1 + e^{{-2z_0}}}} = \\frac{{1 - {L(q**2)}}}{{1 + {L(q**2)}}} = {L(th)}$. "
                  "So tanh is just a rescaled, recentred sigmoid."),
        misconceptions=[([fmt(-pv), fmt(2 * pv - 1), fmt(th)], "sigmoid-is-odd"),
                        ([fmt(1 - pv), fmt(2 * pv - 1), fmt(2 * pv - 1)], "confused-tanh-argument")],
    )


# =========================================================================== dl.training
@gen("dl.training", levels=(1, 2, 3))
def training_math(rng, level):
    if level == 1:
        while True:
            N = rng.choice([1000, 2500, 5000, 50000, 60000, 12345, 10000])
            B = rng.choice([32, 64, 100, 128, 256])
            if N % B:
                break
        E = rng.randint(2, 20)
        ans = E * math.ceil(N / B)
        return Problem(
            prompt=(f"You train on $N = {N}$ examples with mini-batch size $B = {B}$ for $E = {E}$ epochs. The last, smaller "
                    "batch of each epoch is kept (no `drop_last`). How many optimiser steps are taken in total?"),
            answer=str(ans), kind="num",
            rubric=["Steps per epoch = ceil(N / B)", "Multiply by the number of epochs"],
            solution=f"$\\lceil {N}/{B}\\rceil = {math.ceil(N / B)}$ steps per epoch, times ${E}$ gives ${ans}$.",
            misconceptions=[(str(E * (N // B)), "floor-instead-of-ceil"), (fmt(R(E * N, B)), "fractional-steps")],
        )
    if level == 2:
        lrmax = rng.choice([3, 6, 8, 10, 20])
        lrmin = R(lrmax, 10)
        W = rng.choice([100, 200, 500, 1000, 2000])
        K = rng.choice([1200, 2400, 6000, 12000])
        frac = rng.choice([R(1, 4), R(1, 3), R(1, 2), R(2, 3), R(3, 4), R(1, 6), R(5, 6)])
        T = W + K
        t = W + frac * K
        ans = lrmin + R(1, 2) * (lrmax - lrmin) * (1 + sp.cos(sp.pi * frac))
        wrong = R(1, 2) * lrmax * (1 + sp.cos(sp.pi * frac))
        return Problem(
            prompt=(f"A GPT-style LR schedule: linear warmup $\\eta_t = \\eta_{{max}}\\,t/W$ for $t < W$, then cosine decay "
                    f"$\\eta_t = \\eta_{{min}} + \\tfrac12(\\eta_{{max}}-\\eta_{{min}})\\left(1+\\cos\\left(\\pi\\frac{{t-W}}{{T-W}}\\right)\\right)$ "
                    f"for $W\\le t\\le T$. With $\\eta_{{max}} = {lrmax}\\times10^{{-4}}$, $\\eta_{{min}} = \\eta_{{max}}/10$, $W = {W}$, "
                    f"$T = {T}$: what is $\\eta_t$ at step $t = {t}$? Answer **in units of $10^{{-4}}$** (e.g. 4.5 means $4.5\\times10^{{-4}}$). {SIG4}"),
            answer=fmt(ans), kind="num", tol=TOL,
            rubric=["Decide which phase t is in", f"Progress (t - W)/(T - W) = {frac}", "Evaluate the cosine exactly",
                    "Add eta_min back"],
            solution=(f"$t \\ge W$, progress $= ({t}-{W})/{K} = {L(frac)}$, $\\cos(\\pi\\cdot{L(frac)}) = {L(sp.cos(sp.pi * frac))}$. "
                      f"$\\eta_t = {L(lrmin)} + \\tfrac12({L(lrmax - lrmin)})(1 + {L(sp.cos(sp.pi * frac))}) = {L(sp.nsimplify(ans))} \\approx {sig(ans)}$ (units of $10^{{-4}}$)."),
            misconceptions=[(fmt(wrong), "forgot-eta-min")],
        )
    g1, g2 = nz(rng, -5, 5), nz(rng, -5, 5)
    b1, b2 = R(9, 10), R(999, 1000)
    m1, v1 = (1 - b1) * g1, (1 - b2) * g1 ** 2
    d1 = -(m1 / (1 - b1)) / sp.sqrt(v1 / (1 - b2))
    m2, v2 = b1 * m1 + (1 - b1) * g2, b2 * v1 + (1 - b2) * g2 ** 2
    d2 = -(m2 / (1 - b1 ** 2)) / sp.sqrt(v2 / (1 - b2 ** 2))
    nb1 = -m1 / sp.sqrt(v1)
    nb2 = -m2 / sp.sqrt(v2)
    return Problem(
        prompt=(f"Adam (Kingma & Ba 2014) with $\\beta_1 = 0.9$, $\\beta_2 = 0.999$, $\\varepsilon = 0$, $m_0 = v_0 = 0$ sees gradients "
                f"$g_1 = {g1}$, $g_2 = {g2}$ for a scalar parameter. Using bias-corrected moments "
                "$\\hat m_t = m_t/(1-\\beta_1^t)$, $\\hat v_t = v_t/(1-\\beta_2^t)$ and update $\\Delta\\theta_t = -\\eta\\,\\hat m_t/\\sqrt{\\hat v_t}$, "
                f"give $(\\Delta\\theta_1/\\eta,\\ \\Delta\\theta_2/\\eta)$. {SIG4}"),
        answer=[fmt(d1), fmt(d2)], kind="list", tol=TOL, answer_hint="two numbers, e.g. -1, -0.84",
        rubric=["m_1 = (1-b1) g1, v_1 = (1-b2) g1^2", "Bias-correct: first step is exactly -sign(g1)·eta",
                "m_2 = b1 m_1 + (1-b1) g2, v_2 = b2 v_1 + (1-b2) g2^2", "Correct with 1 - b^2 and divide"],
        solution=(f"$m_1 = {L(m1)}$, $v_1 = {L(v1)}$, $\\hat m_1 = {g1}$, $\\hat v_1 = {g1**2}$, so $\\Delta\\theta_1/\\eta = {L(d1)}$. "
                  f"$m_2 = {L(m2)}$, $v_2 = {L(v2)}$, $\\hat m_2 = {L(m2 / (1 - b1**2))}$, $\\hat v_2 = {L(v2 / (1 - b2**2))}$, "
                  f"$\\Delta\\theta_2/\\eta \\approx {sig(d2)}$. Without bias correction the first step would be "
                  f"$\\approx {sig(nb1)}\\,\\eta$ — far too large, which is why Adam corrects the zero-initialised moments (arXiv:1412.6980, §3)."),
        misconceptions=[([fmt(nb1), fmt(nb2)], "no-bias-correction")],
    )


@gen("dl.training", levels=(1, 2, 3))
def training_steps(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            th = [rng.randint(-5, 5) for _ in range(3)]
            g = [nz(rng, -6, 6) for _ in range(3)]
            eta = rng.choice([R(1, 2), R(1, 4), R(1, 10), R(1, 5)])
            new = [th[i] - eta * g[i] for i in range(3)]
            return Problem(
                prompt=(f"Plain SGD update $\\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta - \\eta\\,\\nabla\\mathcal L$: the parameters are "
                        f"$\\boldsymbol\\theta = {tup(th)}$, the mini-batch gradient is $\\nabla\\mathcal L = {tup(g)}$ and $\\eta = {L(eta)}$. "
                        "Give the updated parameter vector."),
                answer=[fmt(v) for v in new], kind="vector",
                rubric=["Scale the gradient by eta", "Subtract it (descent moves against the gradient)"],
                solution=f"$\\eta\\nabla\\mathcal L = {tup([eta * v for v in g])}$, so $\\boldsymbol\\theta' = {tup(new)}$.",
                misconceptions=[([fmt(th[i] + eta * g[i]) for i in range(3)], "gradient-ascent"),
                                ([fmt(th[i] - g[i]) for i in range(3)], "forgot-learning-rate")],
            )
        m = rng.choice([1, 2, 4, 8, 16])
        n = rng.choice([1, 2, 4, 8])
        k = rng.choice([2, 4, 8, 16, 32])
        G = m * n * k
        return Problem(
            prompt=(f"Gradient accumulation: you want a global batch of ${G}$ sequences per optimiser step, you train data-parallel on "
                    f"${n}$ GPU(s), and each GPU fits only ${m}$ sequence(s) per forward/backward pass. Over how many micro-steps must "
                    "each GPU accumulate gradients before calling `optimizer.step()`?"),
            answer=str(k), kind="num",
            rubric=["Sequences per micro-step across all GPUs = m · n_gpu", "Accumulation steps = global batch / (m · n_gpu)",
                    "Scale the loss by 1/steps so the accumulated gradient is a mean"],
            solution=f"Per micro-step the GPUs together process ${m}\\cdot{n} = {m * n}$ sequences, so ${G}/{m * n} = {k}$ micro-steps.",
            misconceptions=[(str(G // m), "forgot-data-parallel")] if n > 1 else [],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                mu = rng.choice([R(1, 2), R(9, 10)])
                eta = rng.choice([R(1, 10), R(1, 2), sp.Integer(1)])
                th0 = rng.randint(-3, 3)
                gs = [nz(rng, -4, 4) for _ in range(3)]
                v, th, rows = sp.Integer(0), sp.Integer(th0), []
                for gt in gs:
                    v = mu * v + gt
                    th = th - eta * v
                    rows.append((v, th))
                plain = th0 - eta * sum(gs)
                if th != plain:
                    break
            steps = "; ".join(f"$v_{t + 1} = {L(rv)}$, $\\theta_{t + 1} = {L(rt)}$" for t, (rv, rt) in enumerate(rows))
            return Problem(
                prompt=(f"SGD with momentum, $v_t = \\mu v_{{t-1}} + g_t$, $\\theta_t = \\theta_{{t-1}} - \\eta v_t$, $v_0 = 0$: a scalar "
                        f"parameter starts at $\\theta_0 = {th0}$ and receives the gradients $g_1, g_2, g_3 = {gs[0]}, {gs[1]}, {gs[2]}$ "
                        f"(take them as given). With $\\mu = {L(mu)}$ and $\\eta = {L(eta)}$, give $\\theta_3$ exactly."),
                answer=fmt(th), kind="num",
                rubric=["Update the velocity first: v_t = mu v_{t-1} + g_t", "Then the parameter: theta_t = theta_{t-1} - eta v_t",
                        "Repeat three times, carrying v forward"],
                solution=steps + f". Without momentum you would get ${L(plain)}$ — momentum keeps pushing along past gradients.",
                misconceptions=[(fmt(plain), "ignored-momentum")],
            )
        quads = [(1, 2, 2, 0), (2, 3, 6, 0), (1, 4, 8, 0), (4, 4, 7, 0), (1, 1, 1, 1), (1, 1, 3, 5), (2, 2, 4, 5),
                 (1, 2, 2, 4), (2, 4, 4, 0)]
        qv = list(rng.choice(quads))
        rng.shuffle(qv)
        gv = [v * rng.choice([-1, 1]) for v in qv]
        nrm = sp.sqrt(sum(v * v for v in gv))
        c = rng.randint(1, int(nrm) - 1)
        clipped = [R(c) * v / nrm for v in gv]
        per = []
        for part in (gv[:2], gv[2:]):
            pn = sp.sqrt(sum(v * v for v in part))
            per += [c * v / pn if pn > c else sp.Integer(v) for v in part]
        return Problem(
            prompt=(f"Global-norm gradient clipping (`clip_grad_norm_` with `max_norm` $c = {c}$): a model has two parameter tensors "
                    f"with gradients $\\mathbf g_a = {tup(gv[:2])}$ and $\\mathbf g_b = {tup(gv[2:])}$. If the norm of all gradients "
                    "together exceeds $c$, every gradient is multiplied by $c/\\|\\mathbf g\\|$. Give the clipped gradients as one "
                    "vector (components of $\\mathbf g_a$, then $\\mathbf g_b$)."),
            answer=[fmt(v) for v in clipped], kind="vector", answer_hint="4 numbers, e.g. [1/3, 2/3, -2/3, 0]",
            rubric=["Global norm = sqrt of the sum of squares over ALL tensors", f"Compare with c = {c}",
                    "Scale every component by c/||g|| (direction preserved)"],
            solution=(f"$\\|\\mathbf g\\| = \\sqrt{{{sum(v * v for v in gv)}}} = {nrm} > {c}$, scale $= {c}/{nrm}$, clipped "
                      f"$= {tup(clipped)}$. Clipping the global norm keeps the update direction; clipping each tensor separately would not."),
            misconceptions=[([fmt(v) for v in per], "clipped-per-tensor"), ([fmt(R(c) * v / nrm ** 2) for v in gv], "divided-by-squared-norm")],
        )
    if rng.random() < 0.5:
        mu = rng.choice([R(1, 2), R(1, 3), R(2, 3), R(3, 4), R(9, 10)])
        g = nz(rng, -4, 4)
        t = rng.randint(2, 5)
        vt = g * (1 - mu ** t) / (1 - mu)
        vinf = g / (1 - mu)
        return Problem(
            prompt=(f"Heavy-ball momentum $v_t = \\mu v_{{t-1}} + g_t$, $v_0 = 0$. Derive a closed form for $v_t$ when the gradient is "
                    f"constant, $g_t = g$, and its limit as $t\\to\\infty$ (the 'effective step' multiplier). With $\\mu = {L(mu)}$, "
                    f"$g = {g}$, give $v_{{{t}}}$ and $v_\\infty$ exactly."),
            answer=[fmt(vt), fmt(vinf)], kind="list", answer_hint="v_t, v_inf   e.g. 15/4, 4",
            rubric=["Unroll: v_t = g(1 + mu + ... + mu^{t-1})", "Geometric sum: v_t = g(1 - mu^t)/(1 - mu)",
                    "Limit for |mu| < 1: g/(1 - mu)", "Interpretation: momentum multiplies the step size by 1/(1 - mu)"],
            solution=(f"$v_t = g\\sum_{{k=0}}^{{t-1}}\\mu^k = g\\frac{{1-\\mu^t}}{{1-\\mu}}$, so $v_{{{t}}} = {g}\\cdot\\frac{{1 - ({L(mu)})^{{{t}}}}}{{1 - {L(mu)}}} = {L(vt)}$ "
                      f"and $v_\\infty = {L(vinf)}$. With $\\mu = 0.9$ momentum acts like a $10\\times$ larger learning rate on a steady slope."),
            misconceptions=[([str(g * t), fmt(vinf)], "no-decay"),
                            ([fmt(g * (1 - mu ** (t + 1)) / (1 - mu)), fmt(vinf)], "off-by-one-power")],
        )
    while True:
        beta = rng.choice([R(8, 10), R(9, 10), R(95, 100), R(99, 100)])
        f = rng.choice([R(1, 2), R(9, 10), R(99, 100)])
        if not (beta == R(99, 100) and f == R(99, 100)):
            break
    t = 1
    while 1 - beta ** t < f:
        t += 1
    bound = math.log(float(1 - f)) / math.log(float(beta))
    return Problem(
        prompt=(f"Why Adam needs bias correction: an exponential moving average $m_t = \\beta m_{{t-1}} + (1-\\beta)g$ with $m_0 = 0$ "
                f"sees a constant gradient $g$. Derive $m_t$ in closed form, and find the smallest $t$ for which the **uncorrected** "
                f"$m_t$ has reached at least ${L(f)}$ of $g$, for $\\beta = {dec(beta) if beta.q == 10 else float(beta)}$."),
        answer=str(t), kind="num",
        rubric=["Unroll: m_t = (1 - beta) g (1 + beta + ... + beta^{t-1}) = g (1 - beta^t)", "Condition 1 - beta^t >= f",
                "Take logs: t >= ln(1 - f)/ln(beta) (dividing by a negative number flips the inequality)", "Round UP to an integer",
                "Dividing by 1 - beta^t (Adam's correction) recovers g exactly at every t"],
        solution=(f"$m_t = g(1-\\beta^t)$. Need $\\beta^t \\le {L(1 - f)}$, i.e. $t \\ge \\ln({L(1 - f)})/\\ln({float(beta)}) \\approx {bound:.3f}$, "
                  f"so $t = {t}$. Until then $m_t$ underestimates $g$; $\\hat m_t = m_t/(1-\\beta^t) = g$ removes that bias (arXiv:1412.6980, §3)."),
        misconceptions=[(str(t - 1), "rounded-down"), (str(round(1 / (1 - float(beta)))), "used-ema-time-constant")],
    )


# =========================================================================== dl.cnn
@gen("dl.cnn", levels=(1, 2, 3))
def conv(rng, level):
    if level == 1:
        while True:
            n = rng.randint(8, 64)
            k = rng.choice([1, 3, 5, 7])
            p = rng.choice([0, 1, 2, 3])
            s = rng.choice([1, 2, 3])
            if n + 2 * p >= k:
                break
        ans = (n + 2 * p - k) // s + 1
        return Problem(
            prompt=(f"A 2-D convolution with kernel ${k}\\times{k}$, padding ${p}$ and stride ${s}$ is applied to a "
                    f"${n}\\times{n}$ input. What is the output height?"),
            answer=str(ans), kind="num",
            rubric=["Use floor((n + 2p - k)/s) + 1", "Floor before adding 1"],
            solution=f"$\\lfloor({n} + 2\\cdot{p} - {k})/{s}\\rfloor + 1 = \\lfloor {n + 2*p - k}/{s} \\rfloor + 1 = {ans}$.",
            misconceptions=[(str((n + 2 * p - k) // s), "forgot-plus-one"), (str(max((n - k) // s + 1, 0)), "ignored-padding")],
        )
    if level == 2:
        xv = [rng.randint(-2, 3) for _ in range(6)]
        while True:
            kk = [rng.randint(-2, 2) for _ in range(3)]
            if kk != kk[::-1]:
                break
        s = rng.choice([1, 2])
        outs = [sum(xv[i * s + j] * kk[j] for j in range(3)) for i in range((6 - 3) // s + 1)]
        flipped = [sum(xv[i * s + j] * kk[2 - j] for j in range(3)) for i in range((6 - 3) // s + 1)]
        return Problem(
            prompt=(f"Deep-learning 'convolution' (really cross-correlation, as in PyTorch `conv1d`): input $\\mathbf x = {tup(xv)}$, "
                    f"kernel $\\mathbf k = {tup(kk)}$, stride ${s}$, no padding: $y_i = \\sum_j x_{{is + j}}\\,k_j$. Give the output vector."),
            answer=[str(v) for v in outs], kind="vector",
            rubric=["Output length floor((6 - 3)/s) + 1", "Slide the kernel WITHOUT flipping it", "Dot product at each position"],
            solution=f"Output length ${len(outs)}$; $\\mathbf y = {tup(outs)}$.",
            misconceptions=[([str(v) for v in flipped], "flipped-kernel")],
        )
    cin, cout = rng.choice([1, 3, 16]), rng.choice([8, 16, 32, 64])
    k, H = rng.choice([3, 5]), rng.choice([8, 16, 28, 32])
    pc = cout * (cin * k * k + 1)
    pd = cin * H * H * cout * H * H + cout * H * H
    return Problem(
        prompt=(f"A conv layer maps a ${cin}\\times{H}\\times{H}$ input (channels × height × width) to ${cout}$ channels with "
                f"${k}\\times{k}$ kernels, stride 1 and 'same' padding (bias per output channel). (a) How many parameters? "
                f"(b) How many parameters would a dense layer need to map the flattened input to the flattened "
                f"${cout}\\times{H}\\times{H}$ output (with bias)? Give (a), (b)."),
        answer=[str(pc), str(pd)], kind="list", answer_hint="conv params, dense params",
        rubric=["Conv: each output channel has C_in·k·k weights + 1 bias", "Same padding keeps H x W",
                "Dense: (C_in H W)·(C_out H W) weights + C_out H W biases", "Weight sharing + locality explain the gap"],
        solution=(f"(a) ${cout}\\cdot({cin}\\cdot{k}^2 + 1) = {pc}$. (b) $({cin}\\cdot{H}^2)({cout}\\cdot{H}^2) + {cout}\\cdot{H}^2 = {pd}$ — "
                  f"about ${sci(pd / pc, 3)}$ times more."),
        misconceptions=[([str(cout * cin * k * k), str(pd)], "forgot-bias")],
    )


@gen("dl.cnn", levels=(1, 2, 3))
def conv_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            cin, cout = rng.choice([1, 3, 16, 32, 64]), rng.choice([8, 16, 32, 64, 128])
            k = rng.choice([1, 3, 5, 7])
            ans = cout * (cin * k * k + 1)
            mis = [(str(cout * cin * k * k), "forgot-bias")]
            if k > 1:
                mis.append((str(cout * (cin * k + 1)), "forgot-kernel-square"))
            return Problem(
                prompt=(f"Parameter count of a convolution: a `Conv2d` layer maps ${cin}$ input channels to ${cout}$ output channels with "
                        f"${k}\\times{k}$ kernels and one bias per output channel. How many trainable parameters does it have? "
                        "(Does the image size matter?)"),
                answer=str(ans), kind="num",
                rubric=["Each output channel has one k x k kernel per input channel", "Plus one bias per output channel",
                        "The count is independent of the image size (weight sharing)"],
                solution=f"${cout}\\cdot({cin}\\cdot{k}^2 + 1) = {ans}$, regardless of the image size.",
                misconceptions=mis,
            )
        X = sp.Matrix(4, 4, lambda i, j: rng.randint(0, 9))
        op = rng.choice(["max", "average"])
        blocks = [[[X[2 * i + a, 2 * j + b] for a in range(2) for b in range(2)] for j in range(2)] for i in range(2)]
        mx = sp.Matrix(2, 2, lambda i, j: max(blocks[i][j]))
        av = sp.Matrix(2, 2, lambda i, j: R(sum(blocks[i][j]), 4))
        ans, other = (mx, av) if op == "max" else (av, mx)
        return Problem(
            prompt=(f"Pooling: apply $2\\times2$ **{op}** pooling with stride 2 to the $4\\times4$ feature map $X = {L(X)}$. "
                    "Give the $2\\times2$ output."),
            answer=fmt(ans), kind="matrix", answer_hint="a 2x2 matrix, e.g. [[7,5],[9,3]]",
            rubric=["Split X into four non-overlapping 2x2 blocks", f"Take the {op} of each block",
                    "Output (i, j) comes from block (i, j)"],
            solution=f"Blocks top-left, top-right, bottom-left, bottom-right give {op}s ${L(ans)}$.",
            misconceptions=[(fmt(other), "used-average" if op == "max" else "used-max")],
        )
    if level == 2:
        if rng.random() < 0.5:
            X = sp.Matrix(4, 4, lambda i, j: rng.randint(-2, 2))
            while True:
                K = sp.Matrix(2, 2, lambda i, j: rng.randint(-2, 2))
                Kf = sp.Matrix(2, 2, lambda a, b: K[1 - a, 1 - b])
                if K != Kf:
                    break
            s = rng.choice([1, 2])
            n = (4 - 2) // s + 1
            Y = sp.Matrix(n, n, lambda i, j: sum(X[i * s + a, j * s + b] * K[a, b] for a in range(2) for b in range(2)))
            Yf = sp.Matrix(n, n, lambda i, j: sum(X[i * s + a, j * s + b] * Kf[a, b] for a in range(2) for b in range(2)))
            return Problem(
                prompt=(f"2-D cross-correlation (what `Conv2d` computes): slide $K = {L(K)}$ over $X = {L(X)}$ with stride ${s}$ and "
                        "no padding, $Y_{ij} = \\sum_{a,b} X_{is+a,\\,js+b}\\,K_{ab}$ (0-indexed). Give $Y$."),
                answer=fmt(Y), kind="matrix", answer_hint=f"a {n}x{n} matrix",
                rubric=[f"Output size (4 - 2)/{s} + 1 = {n}", "Place the kernel WITHOUT flipping it",
                        "Each output = sum of elementwise products over the 2x2 window", f"Move the window by {s}"],
                solution=(f"Output is ${n}\\times{n}$. E.g. $Y_{{00}} = "
                          + " + ".join(f"{_pn(X[a, b])}\\cdot{_pn(K[a, b])}" for a in range(2) for b in range(2))
                          + f" = {Y[0, 0]}$; all entries: $Y = {L(Y)}$."),
                misconceptions=[(fmt(Yf), "flipped-kernel")],
            )
        while True:
            layers = [(rng.choice([3, 5, 7]), rng.choice([1, 2]))] + [(rng.choice([3, 3, 5]), rng.choice([1, 2])) for _ in range(2)]
            r, jump, terms = 1, 1, []
            for k, s in layers:
                terms.append(f"({k}-1)\\cdot{jump}")
                r += (k - 1) * jump
                jump *= s
            naive = 1 + sum(k - 1 for k, _ in layers)
            if naive != r:
                break
        desc = ", ".join(f"layer {i + 1}: ${k}\\times{k}$, stride ${s}$" for i, (k, s) in enumerate(layers))
        return Problem(
            prompt=(f"Receptive field: three conv layers are stacked (no dilation, no pooling) — {desc}. How many input pixels wide "
                    "is the receptive field of one unit in layer 3?"),
            answer=str(r), kind="num",
            rubric=["Start from r = 1 and a 'jump' (input pixels per step) of 1",
                    "Each layer adds (k - 1) · jump, where jump = product of the strides of EARLIER layers",
                    "Then multiply the jump by this layer's stride"],
            solution=(f"$r = 1 + {' + '.join(terms)} = {r}$. Ignoring the strides would give only ${naive}$ — strides make deep "
                      "units see large regions quickly."),
            misconceptions=[(str(naive), "ignored-strides")],
        )
    if rng.random() < 0.5:
        cin, cout = rng.choice([16, 32, 64, 128]), rng.choice([32, 64, 128, 256])
        k = rng.choice([3, 5])
        std, sep = cin * k * k * cout, cin * k * k + cin * cout
        ratio = R(sep, std)
        return Problem(
            prompt=(f"Depthwise-separable convolution (MobileNet, Howard et al. 2017) replaces a ${k}\\times{k}$ conv from $C_{{in}} = {cin}$ "
                    f"to $C_{{out}} = {cout}$ channels by a ${k}\\times{k}$ depthwise conv (one filter per input channel) followed by a "
                    "$1\\times1$ pointwise conv $C_{in}\\to C_{out}$. Ignoring biases, derive the cost ratio separable/standard "
                    "$= 1/C_{out} + 1/k^2$ and give the standard count, the separable count and the ratio (exact)."),
            answer=[str(std), str(sep), fmt(ratio)], kind="list", answer_hint="standard, separable, ratio   e.g. 18432, 2336, 73/576",
            rubric=["Standard: C_in k^2 C_out", "Depthwise: C_in k^2; pointwise: C_in C_out",
                    "Ratio (C_in k^2 + C_in C_out)/(C_in k^2 C_out) = 1/C_out + 1/k^2"],
            solution=(f"Standard ${cin}\\cdot{k}^2\\cdot{cout} = {std}$; separable ${cin}\\cdot{k}^2 + {cin}\\cdot{cout} = {sep}$; "
                      f"ratio $= \\frac{{1}}{{{cout}}} + \\frac{{1}}{{{k * k}}} = {L(ratio)}$ — roughly a ${k * k}\\times$ saving (arXiv:1704.04861)."),
            misconceptions=[([str(std), str(sep), fmt(1 / ratio)], "inverted-ratio")],
        )
    n = rng.choice([2, 3, 4])
    k = 3 if n > 2 else rng.choice([3, 5])
    C = rng.choice([32, 64, 128, 256])
    rf = n * (k - 1) + 1
    stacked, single = n * k * k * C * C, rf * rf * C * C
    return Problem(
        prompt=(f"Why VGG stacks small kernels (Simonyan & Zisserman 2014): ${n}$ stacked ${k}\\times{k}$ convolutions with ${C}$ channels in "
                "and out, stride 1, no biases. Derive (a) the receptive field of the stack, (b) its parameter count, and (c) the parameter "
                "count of a single convolution with the same receptive field. Give (a), (b), (c)."),
        answer=[str(rf), str(stacked), str(single)], kind="list", answer_hint="RF, stacked params, single params",
        rubric=["Each stride-1 k x k layer widens the receptive field by k - 1", f"RF = n(k - 1) + 1 = {rf}",
                "Stack: n k^2 C^2 weights", "Single conv: RF^2 C^2 weights"],
        solution=(f"(a) $1 + {n}\\cdot{k - 1} = {rf}$. (b) ${n}\\cdot{k}^2\\cdot{C}^2 = {stacked}$. (c) ${rf}^2\\cdot{C}^2 = {single}$ — the stack is "
                  f"${L(R(single, stacked))}\\times$ cheaper and has {n - 1} extra nonlinearit{'y' if n == 2 else 'ies'} (arXiv:1409.1556, §2.3)."),
        misconceptions=[([str(n * k), str(stacked), str(n * n * k * k * C * C)], "receptive-field-adds-kernel-sizes")],
    )


# =========================================================================== dl.embeddings
def _cos(u, v):
    return sp.Matrix(u).dot(sp.Matrix(v)) / (sp.sqrt(sp.Matrix(u).dot(sp.Matrix(u))) * sp.sqrt(sp.Matrix(v).dot(sp.Matrix(v))))


@gen("dl.embeddings", levels=(1, 2, 3))
def embeddings(rng, level):
    if level == 1:
        while True:
            u = [rng.randint(-3, 3) for _ in range(3)]
            v = [rng.randint(-3, 3) for _ in range(3)]
            if any(u) and any(v) and sp.Matrix(u).dot(sp.Matrix(v)) != 0:
                break
        c = sp.nsimplify(_cos(u, v))
        dot = sp.Matrix(u).dot(sp.Matrix(v))
        return Problem(
            prompt=f"Two token embeddings are $\\mathbf u = {tup(u)}$ and $\\mathbf v = {tup(v)}$. Compute their cosine similarity. {SIG4}",
            answer=fmt(c), kind="num", tol=TOL,
            rubric=["Dot product u·v", "Norms |u| and |v|", "cos = u·v / (|u||v|)"],
            solution=f"$\\mathbf u\\cdot\\mathbf v = {dot}$, $\\|\\mathbf u\\| = {L(sp.sqrt(sum(a*a for a in u)))}$, $\\|\\mathbf v\\| = {L(sp.sqrt(sum(a*a for a in v)))}$, cos $= {L(c)} \\approx {sig(c)}$.",
            misconceptions=[(str(dot), "forgot-normalization")],
        )
    if level == 2:
        while True:
            king, man, woman = ([rng.randint(-3, 3) for _ in range(3)] for _ in range(3))
            t = [king[i] - man[i] + woman[i] for i in range(3)]
            if not any(t):
                continue
            cands = [[t[i] + (rng.randint(-1, 1) if i == rng.randrange(3) else 0) for i in range(3)]]
            cands += [[rng.randint(-3, 3) for _ in range(3)] for _ in range(3)]
            if any(not any(c) for c in cands):
                continue
            sims = [float(_cos(t, c)) for c in cands]
            order = sorted(range(4), key=lambda i: -sims[i])
            if sims[order[0]] - sims[order[1]] < 0.05:
                continue
            perm = list(range(4))
            rng.shuffle(perm)
            cands = [cands[i] for i in perm]
            sims = [sims[i] for i in perm]
            best = max(range(4), key=lambda i: sims[i])
            break
        letters = "ABCD"
        cl = ", ".join(f"({letters[i]}) ${tup(cands[i])}$" for i in range(4))
        return Problem(
            prompt=(f"Word-analogy arithmetic: with $\\mathbf e_{{king}} = {tup(king)}$, $\\mathbf e_{{man}} = {tup(man)}$, "
                    f"$\\mathbf e_{{woman}} = {tup(woman)}$, form $\\mathbf t = \\mathbf e_{{king}} - \\mathbf e_{{man}} + \\mathbf e_{{woman}}$. "
                    f"Which candidate embedding has the highest **cosine similarity** with $\\mathbf t$? {cl}"),
            answer=letters[best], kind="choice",
            rubric=["Compute t componentwise", "Cosine similarity of t with each candidate", "Pick the argmax"],
            solution=(f"$\\mathbf t = {tup(t)}$. Cosines: " + ", ".join(f"{letters[i]}: {sims[i]:.3f}" for i in range(4))
                      + f". Best: **{letters[best]}**."),
        )
    while True:
        ids = [rng.randrange(4) for _ in range(5)]
        cnt = Counter(ids)
        reps = [k for k, c in cnt.items() if c >= 2]
        if reps:
            break
    j = rng.choice(reps)
    G = [[rng.randint(-3, 3) for _ in range(2)] for _ in range(5)]
    pos = [t for t in range(5) if ids[t] == j]
    grad = [sum(G[t][c] for t in pos) for c in range(2)]
    last = G[pos[-1]]
    mean = [R(grad[c], len(pos)) for c in range(2)]
    gl = ", ".join(f"$\\mathbf g_{t + 1} = {tup(G[t])}$" for t in range(5))
    return Problem(
        prompt=(f"An embedding table $E\\in\\mathbb R^{{4\\times 2}}$ (vocab 4, dim 2) is used on the token sequence "
                f"$({', '.join(map(str, ids))})$ (token ids). Backprop delivers upstream gradients "
                f"$\\mathbf g_t = \\partial\\mathcal L/\\partial\\mathbf e_t$ for the looked-up vectors at positions $t=1..5$: {gl}. "
                f"Writing the lookup as $\\mathbf e_t = \\mathrm{{onehot}}(x_t)^\\top E$, derive $\\partial\\mathcal L/\\partial E$ and give "
                f"its row for token id ${j}$."),
        answer=[str(v) for v in grad], kind="vector",
        rubric=["Lookup = one-hot row vector times E", "dL/dE = sum_t onehot(x_t) g_t^T",
                f"Row {j} sums g_t over every position where token {j} occurs", "Rows of unused tokens get zero gradient"],
        solution=(f"Token ${j}$ occurs at positions ${', '.join(str(t + 1) for t in pos)}$, so its row gradient is "
                  f"${' + '.join(tup(G[t]) for t in pos)} = {tup(grad)}$ (this is what `index_add`/scatter-add does in `nn.Embedding`)."),
        misconceptions=[([str(v) for v in last], "overwrote-gradient"), ([fmt(v) for v in mean], "averaged-gradient")],
    )


_UNIT3 = [(1, 2, 2), (2, 3, 6), (1, 4, 8), (4, 4, 7), (2, 6, 9), (3, 4, 12), (2, 10, 11)]
_PTRIPLES = [(R(1, 2), R(1, 3), R(1, 6)), (R(1, 2), R(1, 4), R(1, 4)), (R(2, 5), R(2, 5), R(1, 5)),
             (R(3, 5), R(1, 5), R(1, 5)), (R(1, 4), R(1, 4), R(1, 2)), (R(2, 3), R(1, 6), R(1, 6))]


@gen("dl.embeddings", levels=(1, 2, 3))
def embeddings_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            while True:
                E = sp.Matrix(4, 2, lambda i, j: rng.randint(-3, 3))
                if len({tuple(E.row(i)) for i in range(4)}) == 4:
                    break
            ids = [rng.randrange(4) for _ in range(3)]
            out = sp.Matrix([list(E.row(i)) for i in ids])
            mis = []
            if max(ids) < 3:
                mis.append((fmt(sp.Matrix([list(E.row(i + 1)) for i in ids])), "one-indexed-ids"))
            return Problem(
                prompt=(f"Embedding lookup as a matrix product: the table $E = {L(E)}$ has one row per token id (ids start at 0). "
                        f"For the token ids $({', '.join(map(str, ids))})$, let $X$ be the $3\\times4$ matrix whose row $t$ is "
                        "$\\mathrm{onehot}(x_t)$. Compute $XE$."),
                answer=fmt(out), kind="matrix", answer_hint="a 3x2 matrix, e.g. [[1,2],[0,-1],[1,2]]",
                rubric=["Row t of X has a single 1 in column x_t", "So row t of XE is row x_t of E",
                        "This is why nn.Embedding is implemented as indexing, not a matmul"],
                solution=f"Row $t$ of $XE$ selects row $x_t$ of $E$: $XE = {L(out)}$.",
                misconceptions=mis,
            )
        tr = list(rng.choice(_UNIT3))
        rng.shuffle(tr)
        v = [a * rng.choice([-1, 1]) for a in tr]
        n = sp.sqrt(sum(a * a for a in v))
        return Problem(
            prompt=(f"Retrieval systems often L2-normalise embeddings before comparing them, so that the dot product equals the "
                    f"cosine similarity. Normalise $\\mathbf v = {tup(v)}$ to unit length (exact)."),
            answer=[fmt(R(a) / n) for a in v], kind="vector",
            rubric=["||v|| = sqrt(sum v_i^2)", "Divide every component by ||v||", "Check: the result has length 1"],
            solution=f"$\\|\\mathbf v\\| = \\sqrt{{{sum(a * a for a in v)}}} = {n}$, so $\\hat{{\\mathbf v}} = {tup([R(a) / n for a in v])}$.",
            misconceptions=[([fmt(R(a) / n ** 2) for a in v], "divided-by-squared-norm"),
                            ([fmt(R(a, sum(abs(b) for b in v))) for a in v], "l1-normalised")],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                u = [rng.randint(-2, 2) for _ in range(2)]
                vs = [[rng.randint(-2, 2) for _ in range(2)] for _ in range(3)]
                if any(u):
                    break
            dots = [u[0] * w[0] + u[1] * w[1] for w in vs]
            loss = -sp.log(sigmoid(dots[0])) - sp.log(sigmoid(-dots[1])) - sp.log(sigmoid(-dots[2]))
            wrong = -sp.log(sigmoid(dots[0])) - sp.log(sigmoid(dots[1])) - sp.log(sigmoid(dots[2]))
            return Problem(
                prompt=(f"Skip-gram with negative sampling (word2vec, Mikolov et al. 2013): the centre word has vector $\\mathbf u = {tup(u)}$, "
                        f"the true context word $\\mathbf v_o = {tup(vs[0])}$, and two sampled negatives $\\mathbf v_1 = {tup(vs[1])}$, "
                        f"$\\mathbf v_2 = {tup(vs[2])}$. The loss is $-\\log\\sigma(\\mathbf u\\cdot\\mathbf v_o) - \\sum_k\\log\\sigma(-\\mathbf u\\cdot\\mathbf v_k)$ "
                        f"(natural log). Compute it. {SIG4}"),
                answer=nfmt(loss), kind="num", tol=TOL,
                rubric=["Dot products u·v_o, u·v_1, u·v_2", "Positive term: -log sigma(u·v_o) = log(1 + e^{-u·v_o})",
                        "Negative terms use the NEGATED score: log(1 + e^{u·v_k})", "Add the three terms"],
                solution=(f"Scores: $\\mathbf u\\cdot\\mathbf v_o = {dots[0]}$, $\\mathbf u\\cdot\\mathbf v_1 = {dots[1]}$, $\\mathbf u\\cdot\\mathbf v_2 = {dots[2]}$. "
                          f"$\\mathcal L = \\log(1+e^{{{-dots[0]}}}) + \\log(1+e^{{{dots[1]}}}) + \\log(1+e^{{{dots[2]}}}) \\approx {sig(loss)}$. "
                          "The loss pulls the true context closer and pushes the negatives away (arXiv:1310.4546)."),
                misconceptions=[(nfmt(wrong), "negatives-not-negated")],
            )
        while True:
            T = [[rng.randint(1, 12) for _ in range(2)] for _ in range(2)]
            N = sum(map(sum, T))
            i, j = rng.randrange(2), rng.randrange(2)
            ratio = R(T[i][j] * N, sum(T[i]) * (T[0][j] + T[1][j]))
            if ratio in (R(1, 4), R(1, 2), 2, 4):
                break
        ws, cs = ("ice", "steam"), ("solid", "gas")
        ans = sp.log(ratio, 2)
        table = ", ".join(f"(`{ws[a]}`, `{cs[b]}`): {T[a][b]}" for a in range(2) for b in range(2))
        return Problem(
            prompt=(f"Pointwise mutual information from co-occurrence counts: (word, context) pairs in a corpus were counted as {table}. "
                    f"Compute $\\mathrm{{PMI}}(\\texttt{{{ws[i]}}}, \\texttt{{{cs[j]}}}) = \\log_2\\frac{{P(w,c)}}{{P(w)P(c)}}$ in bits."),
            answer=fmt(ans), kind="num",
            rubric=["N = total count", "P(w, c) = count/N; P(w) = row sum/N; P(c) = column sum/N",
                    "Ratio = count · N / (row sum · column sum)", "Take log base 2"],
            solution=(f"$N = {N}$, $P(w,c) = {T[i][j]}/{N}$, $P(w) = {sum(T[i])}/{N}$, $P(c) = {T[0][j] + T[1][j]}/{N}$. Ratio "
                      f"$= \\frac{{{T[i][j]}\\cdot{N}}}{{{sum(T[i])}\\cdot{T[0][j] + T[1][j]}}} = {L(ratio)}$, PMI $= {L(ans)}$ bits. "
                      "Word2vec's SGNS implicitly factorises a shifted PMI matrix (Levy & Goldberg 2014)."),
            misconceptions=[(fmt(ratio), "forgot-log")],
        )
    if rng.random() < 0.5:
        while True:
            U = sp.Matrix(3, 2, lambda i, j: rng.randint(-2, 2))
            if len({tuple(U.row(i)) for i in range(3)}) == 3:
                break
        pt = list(rng.choice(_PTRIPLES))
        rng.shuffle(pt)
        p = sp.Matrix(pt)
        o = rng.randrange(3)
        uo = U.row(o).T
        grad = U.T * p - uo
        return Problem(
            prompt=("Word2vec's full-softmax gradient: context word $w$ gets score $s_w = \\mathbf u_w\\cdot\\mathbf v_c$ with output vectors "
                    f"$\\mathbf u_1, \\mathbf u_2, \\mathbf u_3$ = rows of $U = {L(U)}$, and $\\mathcal L = -\\log\\mathrm{{softmax}}(\\mathbf s)_o$. "
                    "Derive $\\partial\\mathcal L/\\partial\\mathbf v_c = \\sum_w p_w\\mathbf u_w - \\mathbf u_o$. At the current parameters "
                    f"the softmax probabilities are $\\mathbf p = {tup(pt)}$ and the observed context word is $o = {o + 1}$; give the gradient."),
            answer=[fmt(v) for v in grad], kind="vector",
            rubric=["L = -s_o + log sum_w exp(s_w)", "ds_w/dv_c = u_w", "dL/dv_c = sum_w p_w u_w - u_o ('expected minus observed')",
                    "Evaluate with the given p"],
            solution=(f"$\\partial\\mathcal L/\\partial\\mathbf v_c = -\\mathbf u_o + \\sum_w p_w\\mathbf u_w = -{tup(list(uo))} + {tup(list(U.T * p))} "
                      f"= {tup(list(grad))}$. Computing $\\sum_w$ over a real vocabulary is what negative sampling avoids."),
            misconceptions=[([fmt(-v) for v in uo], "forgot-expected-term"), ([fmt(v) for v in U.T * p], "forgot-observed-term"),
                            ([fmt(-v) for v in grad], "sign-flip")],
        )
    m = rng.choice([3, 4, 6])
    while True:
        k = rng.randint(1, 2 * m - 1)
        if k % m:
            break
    w = sp.pi / m
    M = sp.Matrix([[sp.cos(k * w), sp.sin(k * w)], [-sp.sin(k * w), sp.cos(k * w)]])
    return Problem(
        prompt=("Sinusoidal position encodings (Vaswani et al. 2017, §3.5) let attention use relative offsets, because "
                "$PE(p+k)$ is a fixed linear function of $PE(p)$. For one frequency pair with "
                f"$\\omega = \\pi/{m}$, $PE(p) = (\\sin p\\omega,\\ \\cos p\\omega)^\\top$. Derive the $2\\times2$ matrix $M$ with "
                f"$PE(p + {k}) = M\\,PE(p)$ for every $p$, and give it exactly."),
        answer=fmt(M), kind="matrix", answer_hint="a 2x2 matrix, e.g. [[1/2, sqrt(3)/2], [-sqrt(3)/2, 1/2]]",
        rubric=["sin((p+k)w) = sin(pw)cos(kw) + cos(pw)sin(kw)", "cos((p+k)w) = cos(pw)cos(kw) - sin(pw)sin(kw)",
                "Read off M = [[cos kw, sin kw], [-sin kw, cos kw]] — independent of p", f"Evaluate at kw = {k}π/{m}"],
        solution=(f"By the addition formulas $M = \\begin{{pmatrix}}\\cos k\\omega & \\sin k\\omega\\\\ -\\sin k\\omega & \\cos k\\omega\\end{{pmatrix}}$, "
                  f"a rotation that does not depend on $p$. With $k\\omega = {L(k * w)}$: $M = {L(M)}$."),
        misconceptions=[(fmt(M.T), "rotated-wrong-direction")],
    )


# =========================================================================== dl.ngram_lm
@gen("dl.ngram_lm", levels=(1, 2, 3))
def bigram(rng, level):
    if level == 1:
        alpha = "abc"[: rng.choice([2, 3])]
        while True:
            words = ["".join(rng.choice(alpha) for _ in range(rng.randint(2, 4))) for _ in range(4)]
            pairs = Counter()
            for w in words:
                s = "." + w + "."
                for a, b in zip(s, s[1:]):
                    pairs[(a, b)] += 1
            prev_tot = Counter()
            for (a, b), c in pairs.items():
                prev_tot[a] += c
            cand = [(a, b) for (a, b) in pairs if prev_tot[a] > pairs[(a, b)]]
            if cand:
                break
        a, b = rng.choice(sorted(cand))
        ans = R(pairs[(a, b)], prev_tot[a])
        total = sum(pairs.values())
        return Problem(
            prompt=(f"Character bigram model. Training words: {', '.join('`' + w + '`' for w in words)}. Wrap each word in a "
                    f"boundary token `.` (so `{words[0]}` becomes `.{words[0]}.`). Using maximum-likelihood counts, "
                    f"estimate $P(\\text{{next}} = \\texttt{{{b}}} \\mid \\text{{prev}} = \\texttt{{{a}}})$."),
            answer=fmt(ans), kind="num",
            rubric=["List all bigrams including the boundary ones", f"Count c({a}{b})",
                    f"Count all bigrams that START with {a}", "Divide: MLE = c(prev, next)/c(prev, ·)"],
            solution=(f"$c(\\texttt{{{a}}}\\texttt{{{b}}}) = {pairs[(a, b)]}$, bigrams starting with `{a}`: ${prev_tot[a]}$, so "
                      f"$P = {L(ans)}$."),
            misconceptions=[(fmt(R(pairs[(a, b)], total)), "divided-by-total-bigrams")],
        )
    if level == 2:
        N = rng.randint(3, 5)
        ks = [rng.randint(1, 4) for _ in range(N)]
        S = sum(ks)
        pp = sp.Integer(2) ** R(S, N)
        probs = ", ".join(L(R(1, 2 ** k)) for k in ks)
        return Problem(
            prompt=(f"A language model assigns the next-token probabilities ${probs}$ to the ${N}$ tokens of a held-out text. "
                    f"Compute the per-token perplexity $\\mathrm{{PPL}} = \\left(\\prod_t p_t\\right)^{{-1/N}}$. {SIG4}"),
            answer=fmt(pp), kind="num", tol=TOL,
            rubric=["Average NLL per token (in bits: -log2 p)", "PPL = 2^(average bits) = exp(average nats)",
                    "Interpret: an 'effective branching factor'"],
            solution=(f"$-\\log_2 p_t = {', '.join(map(str, ks))}$, mean $= {L(R(S, N))}$ bits, so $\\mathrm{{PPL}} = 2^{{{L(R(S, N))}}} \\approx {sig(pp)}$."),
            misconceptions=[(fmt(R(S, N)), "reported-nll-not-perplexity"),
                            (fmt(sum(2 ** k for k in ks) / sp.Integer(N)), "mean-inverse-probability"),
                            (fmt(sp.exp(R(S, N))), "mixed-bits-and-nats")],
        )
    V = rng.randint(4, 6)
    vocab = ["the", "cat", "dog", "sat", "ran", "big"][:V]
    while True:
        counts = [rng.choice([0, 0, 1, 2, 3, 5]) for _ in range(V)]
        if 0 in counts and sum(counts) >= 3:
            break
    C = sum(counts)
    seen = max(range(V), key=lambda i: counts[i])
    unseen = counts.index(0)
    ans = [R(counts[seen] + 1, C + V), R(1, C + V)]
    table = ", ".join(f"`{vocab[i]}`: {counts[i]}" for i in range(V))
    return Problem(
        prompt=(f"Counts of the word following `a` in a corpus (vocabulary of size $V = {V}$): {table}. "
                f"Under **add-one (Laplace) smoothing** $P(w\\mid a) = \\frac{{c(a,w)+1}}{{c(a)+V}}$, give "
                f"$P(\\texttt{{{vocab[seen]}}}\\mid a)$ and $P(\\texttt{{{vocab[unseen]}}}\\mid a)$. Why does an unsmoothed MLE model "
                "have infinite perplexity on a test set containing an unseen bigram?"),
        answer=[fmt(v) for v in ans], kind="list", answer_hint="two fractions, e.g. 4/13, 1/13",
        rubric=["c(a) = sum of the row", "Add 1 to every count, add V to the denominator",
                "Unseen bigram gets 1/(c(a)+V) > 0", "MLE gives p = 0 -> -log 0 = infinity -> infinite perplexity"],
        solution=(f"$c(a) = {C}$, so $P(\\texttt{{{vocab[seen]}}}) = ({counts[seen]}+1)/({C}+{V}) = {L(ans[0])}$, "
                  f"$P(\\texttt{{{vocab[unseen]}}}) = 1/{C + V} = {L(ans[1])}$. Under MLE the unseen word has probability 0, its NLL is "
                  "$\\infty$, so the average NLL and the perplexity blow up."),
        misconceptions=[([fmt(R(counts[seen] + 1, C)), fmt(R(1, C))], "forgot-vocab-in-denominator")],
    )


_LM_WORDS = ["the", "cat", "dog", "sat", "ran", "big"]


@gen("dl.ngram_lm", levels=(1, 2, 3))
def ngram_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            w = rng.sample(["I", "like", "green", "eggs", "ham", "Sam"], 3)
            ps = [rng.choice([R(1, 2), R(1, 3), R(2, 3), R(1, 4), R(3, 4), R(2, 5), R(1, 5)]) for _ in range(4)]
            ans = ps[0] * ps[1] * ps[2] * ps[3]
            ctx = ["<s>", w[0], w[1], w[2]]
            nxt = [w[0], w[1], w[2], "</s>"]
            lst = ", ".join(f"$P(\\texttt{{{b}}}\\mid\\texttt{{{a}}}) = {L(p)}$" for a, b, p in zip(ctx, nxt, ps))
            return Problem(
                prompt=(f"Chain rule under a bigram model: the model gives {lst}. What probability does it assign to the sentence "
                        f"`<s> {' '.join(w)} </s>`? (exact)"),
                answer=fmt(ans), kind="num",
                rubric=["Chain rule: P(sentence) = prod_t P(w_t | w_{t-1})", "Bigram assumption: condition on one previous word",
                        "Include the end-of-sentence factor"],
                solution="$P = " + " \\cdot ".join(L(p) for p in ps) + f" = {L(ans)}$.",
                misconceptions=[(fmt(ps[0] * ps[1] * ps[2]), "forgot-end-token")],
            )
        V = rng.choice([10, 26, 27, 50, 100, 1000])
        n = rng.choice([2, 3, 4]) if V <= 100 else rng.choice([2, 3])
        ans = V ** (n - 1) * (V - 1)
        return Problem(
            prompt=(f"Counting n-gram parameters: an unsmoothed ${n}$-gram model over a vocabulary of $V = {V}$ symbols stores one "
                    "distribution $P(\\cdot\\mid\\text{context})$ per possible context. How many **free** parameters does it have?"),
            answer=str(ans), kind="num",
            rubric=[f"Number of contexts: V^(n-1) = {V}^{n - 1}", "Each distribution has V entries but they sum to 1: V - 1 free",
                    "Multiply"],
            solution=f"${V}^{{{n - 1}}}\\cdot({V}-1) = {ans}$ — exponential in $n$, which is why n-gram models cannot use long contexts.",
            misconceptions=[(str(V ** n), "forgot-normalisation-constraint"), (str(V ** (n - 1)), "counted-contexts-only")],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                vocab = rng.sample(_LM_WORDS, 4)
                corpus = [[rng.choice(vocab) for _ in range(rng.randint(2, 3))] for _ in range(3)]
                big, ctx = Counter(), Counter()
                for snt in corpus:
                    s = ["<s>"] + snt + ["</s>"]
                    for a, b in zip(s, s[1:]):
                        big[(a, b)] += 1
                        ctx[a] += 1
                cur, test, prob, factors = "<s>", [], sp.Integer(1), []
                for _ in range(5):
                    opts = sorted(b for (a, b) in big if a == cur)
                    nxt = rng.choice(opts)
                    f = R(big[(cur, nxt)], ctx[cur])
                    prob *= f
                    factors.append((cur, nxt, f))
                    if nxt == "</s>":
                        break
                    test.append(nxt)
                    cur = nxt
                if factors[-1][1] == "</s>" and len(test) >= 2 and prob != 1 and test not in corpus:
                    break
            lines = " / ".join("`" + " ".join(s) + "`" for s in corpus)
            fl = " \\cdot ".join(L(f) for _, _, f in factors)
            return Problem(
                prompt=(f"Train a word-bigram model by maximum likelihood on the three-sentence corpus {lines} (each sentence is "
                        f"wrapped as `<s> … </s>`). What probability does it assign to the new sentence `<s> {' '.join(test)} </s>`? (exact)"),
                answer=fmt(prob), kind="num",
                rubric=["Add <s> and </s> to every training sentence", "Count bigrams c(a, b) and contexts c(a, ·)",
                        "P(b | a) = c(a, b)/c(a, ·) for every bigram of the test sentence", "Multiply, including the </s> factor"],
                solution=("Factors: " + ", ".join(f"$P(\\texttt{{{b}}}\\mid\\texttt{{{a}}}) = {big[(a, b)]}/{ctx[a]}$" for a, b, _ in factors)
                          + f"; product $= {fl} = {L(prob)}$."),
                misconceptions=[(fmt(prob / factors[-1][2]), "forgot-end-token")] if factors[-1][2] != 1 else [],
            )
        while True:
            Ntot = rng.choice([100, 200, 500, 1000])
            cprev = rng.randint(5, 40)
            cw = rng.randint(5, 60)
            cbi = rng.randint(0, min(cprev, cw) // 2)
            lam = rng.choice([R(1, 2), R(2, 3), R(3, 4), R(1, 3)])
            ans = lam * R(cbi, cprev) + (1 - lam) * R(cw, Ntot)
            swp = (1 - lam) * R(cbi, cprev) + lam * R(cw, Ntot)
            if swp != ans:
                break
        a, b = rng.sample(_LM_WORDS, 2)
        return Problem(
            prompt=(f"Interpolated (Jelinek–Mercer) smoothing: $P(w\\mid v) = \\lambda\\,P_{{MLE}}(w\\mid v) + (1-\\lambda)\\,P_{{MLE}}(w)$. In a corpus "
                    f"of ${Ntot}$ tokens, `{a}` occurs ${cprev}$ times, `{b}` occurs ${cw}$ times and the bigram `{a} {b}` occurs ${cbi}$ "
                    f"times. With $\\lambda = {L(lam)}$, estimate $P(\\texttt{{{b}}}\\mid\\texttt{{{a}}})$ (exact; treat every occurrence of `{a}` as a context)."),
            answer=fmt(ans), kind="num",
            rubric=["Bigram MLE c(v, w)/c(v)", "Unigram MLE c(w)/N", "Mix with weights lambda and 1 - lambda",
                    "Unseen bigrams still get (1 - lambda) P(w) > 0"],
            solution=(f"$P_{{MLE}}(w\\mid v) = {cbi}/{cprev}$, $P_{{MLE}}(w) = {cw}/{Ntot}$, so "
                      f"$P = {L(lam)}\\cdot{L(R(cbi, cprev))} + {L(1 - lam)}\\cdot{L(R(cw, Ntot))} = {L(ans)}$."),
            misconceptions=[(fmt(swp), "swapped-lambda"), (fmt(R(cbi, cprev)), "no-interpolation")],
        )
    if rng.random() < 0.5:
        while True:
            N = rng.choice([8, 16])
            counts = [N]
            for _ in range(rng.randint(2, 4)):
                big = [c for c in counts if c > 1]
                c = rng.choice(big)
                counts.remove(c)
                counts += [c // 2, c // 2]
            if 3 <= len(counts) <= 5:
                break
        rng.shuffle(counts)
        names = _LM_WORDS[: len(counts)]
        tgt = rng.randrange(len(counts))
        LL = sp.Integer(sum(c * round(math.log2(c / N)) for c in counts))
        p = R(counts[tgt], N)
        table = ", ".join(f"`{names[i]}`: {counts[i]}" for i in range(len(counts)))
        return Problem(
            prompt=("Derive the maximum-likelihood unigram model with a Lagrange multiplier: maximise $\\sum_w c_w\\log p_w$ subject to "
                    f"$\\sum_w p_w = 1$. For the counts {table}, give $\\hat p(\\texttt{{{names[tgt]}}})$ and the maximised log-likelihood "
                    "$\\sum_w c_w\\log_2\\hat p_w$ in **bits**."),
            answer=[fmt(p), fmt(LL)], kind="list", answer_hint="p, log-likelihood   e.g. 1/4, -12",
            rubric=["Lagrangian sum c_w log p_w - lambda (sum p_w - 1)", "Stationarity: c_w/p_w = lambda", "Constraint gives lambda = N, so p_w = c_w/N",
                    "Plug in: sum c_w log2(c_w/N) (negative)"],
            solution=(f"$\\partial/\\partial p_w$: $c_w/p_w = \\lambda \\Rightarrow p_w = c_w/\\lambda$, and $\\sum p_w = 1$ gives $\\lambda = N = {N}$. "
                      f"So $\\hat p(\\texttt{{{names[tgt]}}}) = {L(p)}$, and the log-likelihood is "
                      + " + ".join(f"{c}\\log_2\\tfrac{{{c}}}{{{N}}}" for c in counts).join(["$", "$"]) + f" $= {L(LL)}$ bits."),
            misconceptions=[([fmt(p), fmt(-LL)], "sign-error"), ([fmt(R(1, len(counts))), fmt(LL)], "uniform-estimate")],
        )
    V = rng.choice([4, 5, 10, 20])
    N = rng.choice([10, 20, 30, 50, 100])
    alpha = rng.choice([R(1, 2), sp.Integer(1), sp.Integer(2), R(1, 10)])
    q = alpha / (N + V * alpha)
    return Problem(
        prompt=("Add-α smoothing as Bayesian estimation: with a symmetric Dirichlet(α) prior on a unigram distribution over $V$ words, "
                "the posterior mean after observing counts $c_w$ (total $N$) is $P(w) = \\frac{c_w + \\alpha}{N + V\\alpha}$. "
                f"A corpus has $N = {N}$ tokens and $V = {V}$. Derive the value of $\\alpha$ for which every **unseen** word gets "
                f"probability exactly ${L(q)}$."),
        answer=fmt(alpha), kind="num",
        rubric=["Unseen word: P = alpha/(N + V alpha)", "Set equal to q and multiply out: alpha = q N + q V alpha",
                "Solve: alpha = q N/(1 - q V)", "alpha = 1 is Laplace; smaller alpha trusts the data more"],
        solution=(f"$\\frac{{\\alpha}}{{{N} + {V}\\alpha}} = {L(q)} \\Rightarrow \\alpha = \\frac{{qN}}{{1 - qV}} = "
                  f"\\frac{{{L(q * N)}}}{{{L(1 - q * V)}}} = {L(alpha)}$."),
        misconceptions=[(fmt(q * N), "forgot-v-in-denominator")],
    )


# =========================================================================== dl.attention
def _softmax_rows(S, causal=False):
    n, m = S.shape
    A = sp.zeros(n, m)
    for i in range(n):
        cols = range(i + 1) if causal else range(m)
        den = sum(sp.exp(S[i, j]) for j in cols)
        for j in cols:
            A[i, j] = sp.exp(S[i, j]) / den
    return A


@gen("dl.attention", levels=(1, 2, 3))
def attention_compute(rng, level):
    if level == 1:
        Q = sp.Matrix(2, 4, lambda i, j: rng.randint(-2, 2))
        K = sp.Matrix(2, 4, lambda i, j: rng.randint(-2, 2))
        S = Q * K.T / 2
        return Problem(
            prompt=(f"Two tokens with queries $Q = {L(Q)}$ and keys $K = {L(K)}$ (rows are tokens, $d_k = 4$). "
                    "Compute the scaled score matrix $S = QK^\\top/\\sqrt{d_k}$."),
            answer=fmt(S), kind="matrix",
            rubric=["S_ij = q_i · k_j", "Divide every entry by sqrt(d_k) = 2", "Row i = query i, column j = key j"],
            solution=f"$QK^\\top = {L(Q * K.T)}$, so $S = {L(S)}$.",
            misconceptions=[(fmt(Q * K.T), "forgot-scaling"), (fmt(Q * K.T / 4), "divided-by-dk")],
        )
    if level == 2:
        while True:
            q = [rng.randint(-2, 2) for _ in range(2)]
            k1, k2 = [rng.randint(-2, 2) for _ in range(2)], [rng.randint(-2, 2) for _ in range(2)]
            v1, v2 = [rng.randint(-3, 3) for _ in range(2)], [rng.randint(-3, 3) for _ in range(2)]
            d1, d2 = q[0] * k1[0] + q[1] * k1[1], q[0] * k2[0] + q[1] * k2[1]
            if d1 != d2 and v1 != v2:
                break
        s1, s2 = d1 / sp.sqrt(2), d2 / sp.sqrt(2)
        a1 = 1 / (1 + sp.exp(s2 - s1))
        out = [sp.simplify(a1 * v1[i] + (1 - a1) * v2[i]) for i in range(2)]
        a1u = 1 / (1 + sp.exp(d2 - d1))
        outu = [a1u * v1[i] + (1 - a1u) * v2[i] for i in range(2)]
        return Problem(
            prompt=(f"Scaled dot-product attention, one query attending to two positions, $d_k = 2$: "
                    f"$\\mathbf q = {tup(q)}$, keys $\\mathbf k_1 = {tup(k1)}$, $\\mathbf k_2 = {tup(k2)}$, values "
                    f"$\\mathbf v_1 = {tup(v1)}$, $\\mathbf v_2 = {tup(v2)}$. Compute the output "
                    f"$\\sum_j \\mathrm{{softmax}}_j(\\mathbf q\\cdot\\mathbf k_j/\\sqrt{{d_k}})\\,\\mathbf v_j$. {SIG4}"),
            answer=[fmt(v) for v in out], kind="vector", tol=TOL,
            rubric=["Scores q·k_j", "Scale by 1/sqrt(2)", "Softmax over the two scores (weights sum to 1)",
                    "Weighted average of the value vectors"],
            solution=(f"Scores $({d1}, {d2})/\\sqrt2 \\approx ({sig(s1)}, {sig(s2)})$; weights "
                      f"$a_1 = 1/(1+e^{{s_2-s_1}}) \\approx {sig(a1)}$, $a_2 \\approx {sig(1 - a1)}$. "
                      f"Output $= a_1\\mathbf v_1 + a_2\\mathbf v_2 \\approx ({sig(out[0])}, {sig(out[1])})$."),
            misconceptions=[([fmt(v) for v in outu], "forgot-scaling")],
        )
    Q = sp.Matrix(3, 4, lambda i, j: rng.randint(-1, 1))
    K = sp.Matrix(3, 4, lambda i, j: 2 * rng.randint(-1, 1))
    S = Q * K.T / 2
    A = _softmax_rows(S, causal=True)
    Afull = _softmax_rows(S, causal=False)
    Apost = sp.Matrix(3, 3, lambda i, j: Afull[i, j] if j <= i else 0)
    return Problem(
        prompt=(f"Causal self-attention over 3 tokens, $d_k = 4$: $Q = {L(Q)}$, $K = {L(K)}$ (rows are tokens). "
                "Apply the causal mask (token $i$ may only attend to $j\\le i$: set $S_{ij} = -\\infty$ for $j>i$ **before** the softmax) "
                f"and give the $3\\times3$ attention-weight matrix $A = \\mathrm{{softmax}}(\\mathrm{{mask}}(QK^\\top/\\sqrt{{d_k}}))$. {SIG4}"),
        answer=fmt(A), kind="matrix", tol=TOL, answer_hint="a 3x3 matrix, e.g. [[1,0,0],[0.27,0.73,0],[...]]",
        rubric=["Compute S = QK^T / 2", "Mask the upper triangle with -inf", "Row-wise softmax over the unmasked entries",
                "Check: row 1 is (1, 0, 0) and each row sums to 1"],
        solution=(f"$S = {L(S)}$. After masking and a row-wise softmax: $A \\approx "
                  + L(sp.Matrix(3, 3, lambda i, j: sp.Float(float(A[i, j]), 4))) +
                  "$. Masking after the softmax (zeroing) would leave rows that don't sum to 1 — the mask must come first "
                  "(Vaswani et al. 2017, §3.2.3, arXiv:1706.03762)."),
        misconceptions=[(fmt(Afull), "no-causal-mask"), (fmt(Apost), "masked-after-softmax")],
    )


@gen("dl.attention", levels=(1, 2, 3))
def attention_theory(rng, level):
    if level == 1:
        d = rng.choice([256, 512, 768, 1024, 2048])
        h = rng.choice([hh for hh in (4, 8, 12, 16, 32) if d % hh == 0])
        P = 4 * d * d + 4 * d
        return Problem(
            prompt=(f"A multi-head self-attention layer has $d_{{model}} = {d}$ and $h = {h}$ heads with $d_{{head}} = d_{{model}}/h$. "
                    "It has projections $W_Q, W_K, W_V, W_O$ (each with a bias). Give $d_{head}$ and the total parameter count."),
            answer=[str(d // h), str(P)], kind="list", answer_hint="d_head, params",
            rubric=["d_head = d_model / h", "Heads are slices of the same d x d projections — h does not change the count",
                    "4 matrices of d^2 plus 4 bias vectors of d"],
            solution=f"$d_{{head}} = {d}/{h} = {d // h}$; params $= 4d^2 + 4d = {P}$, independent of $h$.",
            misconceptions=[([str(d // h), str(4 * d * d * h + 4 * d * h)], "multiplied-by-heads"),
                            ([str(d // h), str(4 * d * d)], "forgot-bias")],
        )
    if level == 2:
        n = rng.choice([512, 1024, 2048, 4096, 8192])
        d = rng.choice([512, 768, 1024, 2048])
        ans = [4 * n * n * d, 2 * d]
        return Problem(
            prompt=(f"For one self-attention layer on a sequence of $n = {n}$ tokens with $d_{{model}} = {d}$ (count a multiply-add as 2 FLOPs): "
                    "(a) how many FLOPs do the two attention-core matmuls $QK^\\top$ and $AV$ take, summed over all heads? "
                    "(b) The four projections cost $8nd^2$ FLOPs. At what sequence length $n$ do (a) and the projections cost the same? Give (a), (b)."),
            answer=[str(v) for v in ans], kind="list", answer_hint="FLOPs, n",
            rubric=["Per head QK^T: (n x d_head)(d_head x n) -> 2 n^2 d_head", "Sum over heads: 2 n^2 d; AV costs the same",
                    "Total 4 n^2 d", "Solve 4 n^2 d = 8 n d^2 -> n = 2d"],
            solution=f"(a) $4n^2d = 4\\cdot{n}^2\\cdot{d} = {ans[0]}$. (b) $4n^2d = 8nd^2 \\Rightarrow n = 2d = {2 * d}$; beyond that the quadratic term dominates.",
            misconceptions=[([str(2 * n * n * d), str(4 * d)], "counted-only-qk")],
        )
    dk = rng.choice([16, 32, 64, 128, 256])
    s2 = rng.choice([1, 2, 3, R(1, 2), R(1, 4)])
    ans = [dk * s2 ** 2, s2 ** 2]
    return Problem(
        prompt=(f"Why divide by $\\sqrt{{d_k}}$? Let $\\mathbf q, \\mathbf k\\in\\mathbb R^{{{dk}}}$ have i.i.d. zero-mean components, all with "
                f"variance $\\sigma^2 = {L(s2)}$, and $\\mathbf q \\perp \\mathbf k$. Compute (a) $\\mathrm{{Var}}(\\mathbf q\\cdot\\mathbf k)$ and "
                "(b) $\\mathrm{Var}(\\mathbf q\\cdot\\mathbf k/\\sqrt{d_k})$. Give (a), (b)."),
        answer=[fmt(v) for v in ans], kind="list", answer_hint="two numbers",
        rubric=["Var(q_i k_i) = E[q_i^2] E[k_i^2] = sigma^4 (independent, zero mean)",
                "Sum of d_k independent terms: variance d_k sigma^4", "Scaling by 1/sqrt(d_k) divides the variance by d_k",
                "Large-variance logits saturate softmax -> tiny gradients"],
        solution=(f"$\\mathrm{{Var}}(q_ik_i) = \\mathbb E[q_i^2]\\mathbb E[k_i^2] = \\sigma^4 = {L(s2**2)}$; summing ${dk}$ independent terms gives "
                  f"${L(ans[0])}$. Dividing by $\\sqrt{{{dk}}}$ gives ${L(ans[1])}$, independent of $d_k$. Vaswani et al. (2017, footnote 4, "
                  "arXiv:1706.03762) motivate the scaling exactly this way: otherwise softmax saturates."),
        misconceptions=[([fmt(dk * s2), fmt(s2)], "variance-not-squared"),
                        ([fmt(ans[0]), fmt(ans[0] / dk ** 2)], "scaled-variance-by-dk-squared")],
    )


@gen("dl.attention", levels=(1, 2, 3))
def attention_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            n = rng.choice([4, 5, 6, 8, 10, 16, 128, 1024])
            kept = rng.random() < 0.5
            masked, unm = n * (n - 1) // 2, n * (n + 1) // 2
            ans, other = (unm, masked) if kept else (masked, unm)
            word = "kept (finite)" if kept else "masked (set to $-\\infty$)"
            return Problem(
                prompt=(f"Causal masking: a decoder processes a sequence of $n = {n}$ tokens, and the $n\\times n$ score matrix $S$ has "
                        "$S_{ij} = -\\infty$ wherever key $j$ comes after query $i$ ($j > i$). How many entries of $S$ are "
                        f"{word}?"),
                answer=str(ans), kind="num",
                rubric=["Masked = strictly upper triangle (j > i)", "Count: n(n - 1)/2 masked, n(n + 1)/2 kept (diagonal is kept)",
                        "Check: they add up to n^2"],
                solution=(f"Masked: $n(n-1)/2 = {masked}$; kept (lower triangle including the diagonal): $n(n+1)/2 = {unm}$; "
                          f"total ${n * n}$. So {'kept' if kept else 'masked'} $= {ans}$."),
                misconceptions=[(str(other), "diagonal-misassigned"), (fmt(R(n * n, 2)), "half-of-matrix")],
            )
        while True:
            a = [rng.randint(1, 6) for _ in range(3)]
            if len(set(a)) > 1:
                break
        j = rng.randrange(3)
        ans = R(a[j], sum(a))
        sc = ",\\ ".join(f"\\ln {ai}" if ai != 1 else "0" for ai in a)
        mis = [(fmt(R(a[j], 1)), "forgot-normalisation")]
        if any(ai != 1 for ai in a):
            mis.append((nfmt(sp.log(a[j]) / sum(sp.log(ai) for ai in a)), "normalised-scores-not-exponentials"))
        return Problem(
            prompt=(f"One attention weight: a query's **scaled** scores against keys 1, 2, 3 are $({sc})$. What attention weight "
                    f"$\\alpha_{j + 1}$ does it put on key {j + 1}? (exact fraction)"),
            answer=fmt(ans), kind="num",
            rubric=["Exponentiate: e^{ln a} = a", "Softmax: divide by the sum of exponentials"],
            solution=f"$e^{{s}} = {tup(a)}$, sum ${sum(a)}$, so $\\alpha_{j + 1} = {a[j]}/{sum(a)} = {L(ans)}$.",
            misconceptions=mis,
        )
    if level == 2:
        while True:
            a = [rng.randint(1, 5) for _ in range(3)]
            Vv = [[rng.randint(-3, 3) for _ in range(2)] for _ in range(3)]
            if len(set(a)) > 1 and len({tuple(v) for v in Vv}) == 3:
                break
        S = sum(a)
        out = [R(sum(a[k] * Vv[k][c] for k in range(3)), S) for c in range(2)]
        unnorm = [sum(a[k] * Vv[k][c] for k in range(3)) for c in range(2)]
        mean = [R(sum(Vv[k][c] for k in range(3)), 3) for c in range(2)]
        sc = ",\\ ".join(f"\\ln {ai}" if ai != 1 else "0" for ai in a)
        return Problem(
            prompt=(f"Attention output with exact weights: a query's scaled scores against three positions are $({sc})$ and the value "
                    f"vectors are $\\mathbf v_1 = {tup(Vv[0])}$, $\\mathbf v_2 = {tup(Vv[1])}$, $\\mathbf v_3 = {tup(Vv[2])}$. "
                    "Compute the output $\\sum_j \\alpha_j\\mathbf v_j$ exactly."),
            answer=[fmt(v) for v in out], kind="vector",
            rubric=["Softmax weights alpha_j = a_j / sum a", "Weighted sum of the value vectors", "Check: output lies in the convex hull of the values"],
            solution=(f"$\\boldsymbol\\alpha = {tup([R(ai, S) for ai in a])}$; output $= \\tfrac{{1}}{{{S}}}({a[0]}\\mathbf v_1 + {a[1]}\\mathbf v_2 + {a[2]}\\mathbf v_3) = {tup(out)}$."),
            misconceptions=[([str(v) for v in unnorm], "forgot-normalisation"), ([fmt(v) for v in mean], "ignored-scores")],
        )
    while True:
        O = sp.Matrix(3, 2, lambda i, j: R(rng.randint(-9, 9), rng.choice([1, 2, 4])))
        if len({tuple(O.row(i)) for i in range(3)}) == 3:
            break
    order = rng.choice([[2, 0, 1], [1, 2, 0]])
    inv = [order.index(i) for i in range(3)]
    new = sp.Matrix([list(O.row(i)) for i in order])
    wrong = sp.Matrix([list(O.row(i)) for i in inv])
    names = ", ".join(f"\\mathbf x_{i + 1}" for i in order)
    return Problem(
        prompt=("Permutation equivariance: a self-attention layer **without positional encodings** maps the tokens "
                f"$(\\mathbf x_1, \\mathbf x_2, \\mathbf x_3)$ to outputs $O = {L(O)}$ (row $t$ = output at position $t$). Show that "
                "permuting the input rows by $P$ turns the scores into $PSP^\\top$ and the output into $PO$. Then give the output "
                f"matrix when the same tokens are fed in the order $({names})$."),
        answer=fmt(new), kind="matrix", answer_hint="a 3x2 matrix",
        rubric=["Q, K, V are row-wise maps: (PX)W = P(XW)", "Scores: (PQ)(PK)^T = P S P^T",
                "Row-wise softmax commutes with permuting rows and columns together", "Output: P A P^T P V = P (AV) = P O",
                "Position t of the new sequence gets the output of the token now at position t"],
        solution=(f"$\\mathrm{{softmax}}(PSP^\\top)PV = P\\,\\mathrm{{softmax}}(S)P^\\top P V = PO$, so the outputs move with their tokens: "
                  f"new output $= {L(new)}$. This is why transformers need positional information."),
        misconceptions=[(fmt(O), "assumed-order-invariant"), (fmt(wrong), "applied-inverse-permutation")],
    )


# =========================================================================== dl.transformer
@gen("dl.transformer", levels=(1, 2, 3))
def transformer_params(rng, level):
    if level == 1:
        Ln = rng.choice([6, 12, 24, 32, 48])
        d = rng.choice([256, 512, 768, 1024, 1600, 2048])
        ans = 12 * Ln * d * d
        return Problem(
            prompt=(f"Using the approximation that each transformer block (attention with 4 $d\\times d$ projections + MLP with hidden size $4d$) "
                    f"has $\\approx 12d^2$ parameters, estimate the non-embedding parameters of a ${Ln}$-layer model with $d = {d}$."),
            answer=str(ans), kind="num", tol=1e-3, answer_hint="a number, e.g. 85e6 or 84934656",
            rubric=["Attention: 4 d^2", "MLP: d·4d + 4d·d = 8 d^2", "Per block 12 d^2, times the number of layers"],
            solution=f"$12\\cdot{Ln}\\cdot{d}^2 = {ans} \\approx {sci(ans)}$.",
            misconceptions=[(str(4 * Ln * d * d), "attention-only"), (str(8 * Ln * d * d), "mlp-only")],
        )
    if level == 2:
        d = rng.choice([128, 256, 384, 512, 768, 1024])
        ans = 12 * d * d + 13 * d
        return Problem(
            prompt=(f"Count the parameters of ONE GPT-2-style transformer block **exactly**, with $d = {d}$: fused $W_{{QKV}}$ ($d\\times3d$) and "
                    "$W_O$ ($d\\times d$), MLP $d\\to4d\\to d$, every linear layer has a bias, and two LayerNorms each with "
                    "a gain and a bias vector."),
            answer=str(ans), kind="num",
            rubric=["Attention: 3d^2 + 3d (QKV) + d^2 + d (output)", "MLP: 4d^2 + 4d + 4d^2 + d",
                    "LayerNorms: 2 x 2d", "Total 12 d^2 + 13 d"],
            solution=(f"Attention $4d^2 + 4d$, MLP $8d^2 + 5d$, LayerNorms $4d$: total $12d^2 + 13d = {ans}$. "
                      "(For GPT-2 small, $d = 768$: 7,087,872 per block.)"),
            misconceptions=[(str(12 * d * d), "forgot-biases-and-ln"), (str(12 * d * d + 9 * d), "forgot-layernorm")],
        )
    d_head = rng.choice([64, 128])
    nh = rng.choice([8, 16, 32])
    d = nh * d_head
    nkv = rng.choice([v for v in (nh, nh // 4, nh // 8, 1) if v >= 1])
    hid = rng.choice([256 * round(8 * d / 3 / 256), 256 * round(8 * d / 3 / 256) + 256])
    ans = 2 * d * d + 2 * d * nkv * d_head + 3 * d * hid + 2 * d
    full = 4 * d * d + 3 * d * hid + 2 * d
    return Problem(
        prompt=(f"A Llama-style block: no biases anywhere; two RMSNorms (one gain vector of size $d$ each); grouped-query attention with "
                f"$n_{{heads}} = {nh}$, $d_{{head}} = {d_head}$ (so $d = {d}$) and $n_{{kv}} = {nkv}$ key/value heads "
                f"($W_K, W_V$ map $d \\to n_{{kv}} d_{{head}}$; $W_Q, W_O$ are $d\\times d$); SwiGLU MLP with three matrices "
                f"$W_1, W_3: d\\to {hid}$ and $W_2: {hid}\\to d$. Count the block's parameters."),
        answer=str(ans), kind="num",
        rubric=["W_Q and W_O: 2 d^2", "W_K and W_V: 2 d·n_kv·d_head (GQA shrinks these)", "SwiGLU: 3·d·hidden",
                "Norm gains: 2d", "Sum"],
        solution=(f"$2d^2 = {2 * d * d}$; $2\\cdot{d}\\cdot{nkv * d_head} = {2 * d * nkv * d_head}$; "
                  f"$3\\cdot{d}\\cdot{hid} = {3 * d * hid}$; norms ${2 * d}$. Total ${ans}$. "
                  "(Llama-2-7B: $d=4096$, hidden 11008, no GQA: 202,383,360 per block.)"),
        misconceptions=[(str(full), "ignored-gqa"), (str(ans - 2 * d), "forgot-norm-gains")],
    )


@gen("dl.transformer", levels=(1, 2, 3))
def transformer_block(rng, level):
    if level == 1:
        B, T = rng.choice([2, 4, 8, 16, 32]), rng.choice([64, 128, 256, 512, 1024])
        d = rng.choice([256, 512, 768])
        h = rng.choice([hh for hh in (4, 8, 12, 16) if d % hh == 0])
        return Problem(
            prompt=(f"In a multi-head attention layer the input has shape $(B, T, d) = ({B}, {T}, {d})$ and there are $h = {h}$ heads. "
                    "After splitting heads, $Q$ has shape $(B, h, T, d/h)$. What is the shape of the attention-weight tensor "
                    "$\\mathrm{softmax}(QK^\\top/\\sqrt{d_k})$? Give it as a 4-tuple."),
            answer=[str(B), str(h), str(T), str(T)], kind="vector", answer_hint="e.g. [8, 12, 256, 256]",
            rubric=["K^T swaps the last two axes: (B, h, d/h, T)", "Batched matmul over (B, h): (T, d/h) x (d/h, T) -> (T, T)"],
            solution=f"$({B}, {h}, {T}, {d // h}) \\times ({B}, {h}, {d // h}, {T}) \\to ({B}, {h}, {T}, {T})$ — one $T\\times T$ map per head per example.",
            misconceptions=[([str(B), str(T), str(T)], "forgot-head-axis"), ([str(B), str(h), str(T), str(d // h)], "confused-with-output")],
        )
    if level == 2:
        while True:
            a, b = rng.randint(-3, 3), rng.randint(-3, 3)
            if a != b:
                break
        WV = sp.Matrix(2, 2, lambda i, j: rng.randint(-2, 2))
        WO = sp.Matrix(2, 2, lambda i, j: rng.randint(-2, 2))
        x = sp.Matrix([a, b])
        s = 1 if a > b else -1
        nrm = sp.Matrix([s, -s])
        att = WO * WV * nrm
        out = x + att
        return Problem(
            prompt=(f"A pre-LN transformer sublayer $\\mathbf x' = \\mathbf x + \\mathrm{{Attn}}(\\mathrm{{LN}}(\\mathbf x))$ processes a "
                    f"sequence of **one** token $\\mathbf x = {tup([a, b])}$ ($d = 2$). LN has $\\gamma = 1, \\beta = 0, \\varepsilon = 0$ "
                    f"(biased variance). Attention uses $W_V = {L(WV)}$ and output projection $W_O = {L(WO)}$ (no biases; $W_Q, W_K$ "
                    "are irrelevant here — why?). Compute $\\mathbf x'$."),
            answer=[str(v) for v in out], kind="vector",
            rubric=["LN of a 2-vector with distinct entries is (±1, ∓1)", "With one token the softmax weight is exactly 1",
                    "Attention output = W_O W_V LN(x)", "Add the residual x"],
            solution=(f"$\\mu = {L(R(a + b, 2))}$, $\\sigma = {L(R(abs(a - b), 2))}$, so $\\mathrm{{LN}}(\\mathbf x) = {tup([s, -s])}$. "
                      f"Softmax over a single score is 1, so $\\mathrm{{Attn}} = W_OW_V\\,\\mathrm{{LN}}(\\mathbf x) = {tup(list(att))}$ "
                      f"(queries/keys don't matter). $\\mathbf x' = {tup(list(out))}$."),
            misconceptions=[([str(v) for v in att], "forgot-residual"), ([str(v) for v in x + WV * WO * nrm], "wrong-projection-order")],
        )
    d = rng.choice([512, 768, 1024, 1536, 2048, 4096])
    ctx = rng.choice([1024, 2048, 4096, 8192])
    Ln = rng.choice([12, 24, 32, 48])
    N = 12 * Ln * d * d
    C = 2 * N + 2 * Ln * ctx * d
    ratio = R(ctx, 12 * d)
    return Problem(
        prompt=(f"Kaplan et al. (2020, Table 1) estimate forward-pass FLOPs per token as $C_{{fwd}} \\approx 2N + 2\\,n_{{layer}}\\,n_{{ctx}}\\,d$ "
                f"with $N \\approx 12\\,n_{{layer}}d^2$. For $n_{{layer}} = {Ln}$, $d = {d}$, $n_{{ctx}} = {ctx}$: give $C_{{fwd}}$ and "
                "the ratio of the context-dependent attention term to the $2N$ term. Give $C_{fwd}$, ratio (exact)."),
        answer=[str(C), fmt(ratio)], kind="list", answer_hint="FLOPs, ratio  e.g. 2.5e9, 1/3",
        rubric=["N = 12 n_layer d^2", "2N: each parameter does one multiply-add per token",
                "Attention term 2 n_layer n_ctx d (QK^T and AV against the context)", "Ratio simplifies to n_ctx / (12 d)"],
        solution=(f"$N = {N}$, $C_{{fwd}} = {2 * N} + {2 * Ln * ctx * d} = {C}$. Ratio $= \\frac{{2n_{{layer}}n_{{ctx}}d}}{{24 n_{{layer}}d^2}} = "
                  f"\\frac{{n_{{ctx}}}}{{12d}} = {L(ratio)}$ — the attention term matters only when $n_{{ctx}} \\gtrsim 12d$ (arXiv:2001.08361)."),
        misconceptions=[([str(2 * N), fmt(ratio)], "forgot-attention-term")],
    )


@gen("dl.transformer", levels=(1, 2, 3))
def transformer_ffn(rng, level):
    if level == 1:
        d = rng.choice([128, 256, 512, 768, 1024])
        m = rng.choice([2, 4, 4, 8])
        ans = 2 * m * d * d + m * d + d
        return Problem(
            prompt=(f"Feed-forward sublayer size: a transformer's position-wise MLP maps $d \\to {m}d \\to d$ with a bias on both "
                    f"linear layers. For $d = {d}$, how many parameters does it have?"),
            answer=str(ans), kind="num",
            rubric=[f"First layer: d·{m}d weights + {m}d biases", f"Second layer: {m}d·d weights + d biases", "Add"],
            solution=f"$2\\cdot{m}d^2 + {m}d + d = {2 * m * d * d} + {m * d} + {d} = {ans}$.",
            misconceptions=[(str(2 * m * d * d), "forgot-bias"), (str(m * d * d + m * d), "counted-one-layer")],
        )
    if level == 2:
        while True:
            x = [rng.randint(-2, 2) for _ in range(2)]
            W1 = sp.Matrix(3, 2, lambda i, j: rng.randint(-2, 2))
            b1 = sp.Matrix([rng.randint(-1, 1) for _ in range(3)])
            z = W1 * sp.Matrix(x) + b1
            if any(v < 0 for v in z) and any(v > 0 for v in z):
                break
        W2 = sp.Matrix(2, 3, lambda i, j: rng.randint(-2, 2))
        b2 = sp.Matrix([rng.randint(-1, 1) for _ in range(2)])
        h = z.applyfunc(lambda v: max(v, 0))
        f = W2 * h + b2
        out = sp.Matrix(x) + f
        return Problem(
            prompt=("The MLP sublayer with its residual (LayerNorm omitted): $\\mathbf x' = \\mathbf x + W_2\\,\\mathrm{ReLU}(W_1\\mathbf x + \\mathbf b_1) + \\mathbf b_2$ "
                    f"with $\\mathbf x = {tup(x)}$, $W_1 = {L(W1)}$, $\\mathbf b_1 = {tup(list(b1))}$, $W_2 = {L(W2)}$, $\\mathbf b_2 = {tup(list(b2))}$. "
                    "Compute $\\mathbf x'$."),
            answer=[str(v) for v in out], kind="vector",
            rubric=["Hidden pre-activation W1 x + b1 (expands d = 2 to 3)", "ReLU", "Project back: W2 h + b2", "Add the residual x"],
            solution=(f"$W_1\\mathbf x + \\mathbf b_1 = {tup(list(z))}$, ReLU $\\to {tup(list(h))}$, $W_2\\mathbf h + \\mathbf b_2 = {tup(list(f))}$, "
                      f"$\\mathbf x' = {tup(list(out))}$."),
            misconceptions=[([str(v) for v in f], "forgot-residual"),
                            ([str(v) for v in sp.Matrix(x) + W2 * z + b2], "forgot-activation")],
        )
    Ln = rng.choice([6, 12, 24, 48])
    s2 = rng.choice([R(1, 10), R(1, 4), R(1, 2), sp.Integer(1)])
    Nr = 2 * Ln
    ans = [1 + Nr * s2, 1 + s2]
    return Problem(
        prompt=(f"Residual-stream growth and GPT-2's initialisation: each of the $n_{{layer}} = {Ln}$ blocks has two residual branches "
                "(attention and MLP). Each branch adds to the stream an output that is zero-mean, independent of the stream, with "
                f"variance $\\sigma^2 = {L(s2)}$; the stream starts with variance 1. GPT-2 (Radford et al. 2019) scales the weights of each "
                "branch's output projection by $1/\\sqrt N$, $N = 2n_{layer}$. Derive the final residual variance (a) without and (b) with that scaling."),
        answer=[fmt(v) for v in ans], kind="list", answer_hint="two numbers: without, with",
        rubric=["Variances of independent terms add", f"N = 2 n_layer = {Nr} branch outputs are added",
                "Without scaling: 1 + N sigma^2", "Scaling weights by 1/sqrt(N) scales each branch variance by 1/N: 1 + sigma^2"],
        solution=(f"(a) $1 + {Nr}\\cdot{L(s2)} = {L(ans[0])}$ — grows linearly with depth. (b) Each branch now has variance "
                  f"$\\sigma^2/{Nr}$, so $1 + \\sigma^2 = {L(ans[1])}$, independent of depth."),
        misconceptions=[([fmt(1 + Ln * s2), fmt(1 + s2)], "one-branch-per-block"),
                        ([fmt(ans[0]), nfmt(1 + Nr * s2 / sp.sqrt(Nr))], "scaled-variance-by-sqrt")],
    )


# =========================================================================== dl.tokenization
def _bpe_stats(seq):
    return Counter(zip(seq, seq[1:]))


def _bpe_merge(seq, pair):
    out, i = [], 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            out.append(seq[i] + seq[i + 1])
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _bpe_run(rng, nmerges, alpha_choices=("ab", "abc")):
    """Random string whose first `nmerges` BPE steps all have a unique most-frequent pair."""
    while True:
        alpha = rng.choice(alpha_choices)
        units = [alpha[: rng.randint(2, len(alpha))], rng.choice(alpha) * 2, rng.choice(alpha)]
        s = "".join(rng.choice(units) for _ in range(rng.randint(5, 8)))
        if not 8 <= len(s) <= 14 or len(set(s)) < 2:
            continue
        seq, steps, ok = list(s), [], True
        for _ in range(nmerges):
            st = _bpe_stats(seq)
            if not st:
                ok = False
                break
            top = st.most_common()
            if top[0][1] < 2 or (len(top) > 1 and top[1][1] == top[0][1]):
                ok = False
                break
            pair = top[0][0]
            new = _bpe_merge(seq, pair)
            steps.append((pair, top[0][1], new))
            seq = new
        if ok:
            return s, steps


BPE_RULES = ("Pair counting counts every adjacent pair (overlaps included, so `aaa` contains `aa` twice); merging replaces "
             "occurrences left to right without overlap (so `aaa` becomes `[aa][a]`).")


@gen("dl.tokenization", levels=(1, 2, 3))
def bpe(rng, level):
    nm = {1: 1, 2: 1, 3: 3}[level]
    s, steps = _bpe_run(rng, nm)
    if level == 1:
        pair, cnt, _ = steps[0]
        st = _bpe_stats(list(s)).most_common()
        return Problem(
            prompt=(f"BPE training starts with the string `{s}` split into single characters. {BPE_RULES} Which pair is merged first? "
                    "Type the two symbols concatenated (e.g. `ab`)."),
            answer=pair[0] + pair[1], kind="text", answer_hint="two characters, e.g. ab",
            rubric=["List all adjacent pairs", "Count each pair (overlaps count)", "Pick the most frequent"],
            solution="Pair counts: " + ", ".join(f"`{a}{b}`: {c}" for (a, b), c in st) + f". Merge `{pair[0]}{pair[1]}` ({cnt} times).",
            misconceptions=[(st[1][0][0] + st[1][0][1], "picked-wrong-pair")] if len(st) > 1 else [],
        )
    if level == 2:
        pair, cnt, new = steps[0]
        naive = len(s) - cnt
        return Problem(
            prompt=(f"Run ONE BPE merge step on `{s}` (starting from single characters). {BPE_RULES} "
                    "How many tokens does the sequence have after the merge?"),
            answer=str(len(new)), kind="num",
            rubric=["Count adjacent pairs", "Choose the most frequent pair", "Replace occurrences left to right, no overlap",
                    "Count the resulting tokens"],
            solution=(f"Most frequent pair `{pair[0]}{pair[1]}` ({cnt}×). After merging: "
                      + " ".join(f"`{t}`" for t in new) + f" → {len(new)} tokens."),
            misconceptions=[(str(naive), "counted-overlapping-merges")] if naive != len(new) else [],
        )
    lines = []
    for i, (pair, cnt, new) in enumerate(steps):
        lines.append(f"merge {i + 1}: `{pair[0]}`+`{pair[1]}` ({cnt}×) → " + " ".join(f"`{t}`" for t in new))
    final = len(steps[-1][2])
    return Problem(
        prompt=(f"Train byte-level-style BPE for **three** merges on `{s}` (start from single characters; merged tokens become new "
                f"symbols that can be merged again). {BPE_RULES} Give the final number of tokens. "
                "(On paper: also write the merge list — this is what minbpe's `train` returns.)"),
        answer=str(final), kind="num",
        rubric=["Merge 1: count pairs, merge the most frequent", "Re-count pairs on the NEW sequence before each merge",
                "Merged tokens can take part in later merges", "Count tokens after merge 3"],
        solution="<br>".join(lines) + f"<br>Final: {final} tokens (from {len(s)} characters; compression {len(s)}/{final}).",
    )


UTF8_CHARS = {"é": 2, "ü": 2, "ñ": 2, "ß": 2, "ø": 2, "中": 3, "文": 3, "日": 3, "€": 3, "😀": 4, "🚀": 4}


@gen("dl.tokenization", levels=(1, 2))
def byte_level(rng, level):
    if level == 1:
        parts = [rng.choice(["cat", "hi", "a", "GPT", "ok"]) for _ in range(rng.randint(1, 2))]
        extra = rng.sample(list(UTF8_CHARS), rng.randint(2, 3))
        s = "".join(parts) + "".join(extra)
        nb = len(s.encode("utf-8"))
        return Problem(
            prompt=(f"A byte-level BPE tokenizer (GPT-2 style) first encodes text as UTF-8 bytes, one initial token per byte. "
                    f"How many initial tokens does the string `{s}` produce?"),
            answer=str(nb), kind="num",
            rubric=["ASCII characters: 1 byte", "Latin-1 accented letters: 2 bytes", "CJK and € : 3 bytes", "Emoji: 4 bytes"],
            solution=" + ".join(f"`{c}`:{len(c.encode('utf-8'))}" for c in s) + f" = {nb} bytes.",
            misconceptions=[(str(len(s)), "counted-characters")],
        )
    m = rng.choice([1000, 8000, 32000, 50000, 100000])
    sp_tok = rng.choice([1, 2, 3])
    d = rng.choice([512, 768, 1024, 4096])
    V = 256 + m + sp_tok
    return Problem(
        prompt=(f"A byte-level BPE tokenizer is trained with ${m}$ merges and ${sp_tok}$ special token(s). (a) What is its vocabulary size? "
                f"(b) How many parameters does the token-embedding table of a model with $d = {d}$ have? "
                "(GPT-2: 50,000 merges + 1 special token.) Give (a), (b)."),
        answer=[str(V), str(V * d)], kind="list", answer_hint="vocab size, params",
        rubric=["Base vocabulary: 256 byte values", "Each merge adds exactly one token", "Add special tokens", "Embedding: V x d"],
        solution=f"(a) $256 + {m} + {sp_tok} = {V}$. (b) ${V}\\cdot{d} = {V * d}$.",
        misconceptions=[([str(m + sp_tok), str((m + sp_tok) * d)], "forgot-base-bytes")],
    )


@gen("dl.tokenization", levels=(1, 2, 3))
def tokenize_more(rng, level):
    if level == 1:
        while True:
            s = "".join(rng.choice("abc") for _ in range(rng.randint(7, 10)))
            seq = list(s)
            m1 = rng.choice(sorted(set(zip(seq, seq[1:]))))
            seq1 = _bpe_merge(seq, m1)
            pairs1 = sorted(set(zip(seq1, seq1[1:])))
            with_new = [p for p in pairs1 if m1[0] + m1[1] in p]
            if not with_new:
                continue
            m2 = rng.choice(with_new)
            seq2 = _bpe_merge(seq1, m2)
            rev = _bpe_merge(_bpe_merge(seq, m2), m1)
            if len(seq2) <= len(s) - 3:
                break
        mis = [(str(len(s)), "no-merges-applied"), (str(len(seq1)), "stopped-after-first-merge")]
        if len(rev) != len(seq2):
            mis.append((str(len(rev)), "applied-merges-out-of-order"))
        return Problem(
            prompt=(f"Encoding with a learned merge list: a BPE tokenizer learned, in this order, (1) `{m1[0]}` + `{m1[1]}` → "
                    f"`{m1[0] + m1[1]}`, (2) `{m2[0]}` + `{m2[1]}` → `{m2[0] + m2[1]}`. To encode new text, start from single characters "
                    "and apply the merges in order (each to the whole sequence, left to right, no overlap). How many tokens does "
                    f"`{s}` become?"),
            answer=str(len(seq2)), kind="num",
            rubric=["Start from single characters", "Apply merge 1 everywhere (left to right)",
                    "Then merge 2 on the NEW sequence", "Count the tokens"],
            solution=("After merge 1: " + " ".join(f"`{t}`" for t in seq1) + ". After merge 2: "
                      + " ".join(f"`{t}`" for t in seq2) + f" → {len(seq2)} tokens."),
            misconceptions=mis,
        )
    if level == 2:
        c = rng.choice([3, 4, 5])
        nbytes = rng.choice([6000, 12000, 30000, 60000, 120000])
        n = nbytes // c
        return Problem(
            prompt=(f"Why tokenize at all? A document is ${nbytes}$ UTF-8 bytes long. A byte-level model uses one position per byte; "
                    f"a BPE tokenizer averages ${c}$ bytes per token. Give (a) the BPE sequence length and (b) the factor by which the "
                    "byte-level model's attention-core cost (which grows as $n^2$) exceeds the BPE model's."),
            answer=[str(n), str(c * c)], kind="list", answer_hint="tokens, factor",
            rubric=["Tokens = bytes / (bytes per token)", "Attention core ~ n^2", "Ratio (n_bytes / n_tokens)^2 = c^2"],
            solution=(f"(a) ${nbytes}/{c} = {n}$ tokens. (b) $({nbytes}/{n})^2 = {c}^2 = {c * c}$ — and a fixed context window "
                      f"holds ${c}\\times$ more text with BPE."),
            misconceptions=[([str(n), str(c)], "linear-not-quadratic")],
        )
    if rng.random() < 0.5:
        while True:
            s_train, steps = _bpe_run(rng, 2, alpha_choices=("abc",))
            merges = [st[0] for st in steps]
            test = "".join(rng.choice("abc") for _ in range(rng.randint(6, 9)))
            seq = list(test)
            used = 0
            for pr in merges:
                new = _bpe_merge(seq, pr)
                used += new != seq
                seq = new
            if used == 2:
                break
        vid = {"a": 97, "b": 98, "c": 99}
        for k, pr in enumerate(merges):
            vid[pr[0] + pr[1]] = 256 + k
        ids = [vid[t] for t in seq]
        raw = [vid[ch] for ch in test]
        mtxt = ", ".join(f"$({vid[p[0]]}, {vid[p[1]]}) \\to {256 + k}$" for k, p in enumerate(merges))
        return Problem(
            prompt=(f"minbpe-style encoding to token ids: the raw bytes are `a` = 97, `b` = 98, `c` = 99, and training produced the merges "
                    f"{mtxt}. `encode` starts from the bytes and repeatedly merges the adjacent pair with the **lowest** new id until no "
                    f"merge applies. Encode `{test}` and give the list of token ids."),
            answer=[str(v) for v in ids], kind="list", answer_hint="token ids in order, e.g. 256, 99, 257",
            rubric=["Write the string as byte ids", "Merges are applied in order of their new id (rank)",
                    "Replace every occurrence left to right, then look for the next merge", "Read off the final ids"],
            solution=(f"Bytes: ${', '.join(map(str, raw))}$. After applying the merges in rank order: "
                      + " ".join(f"`{t}`" for t in seq) + f" → ids ${', '.join(map(str, ids))}$."),
            misconceptions=[([str(v) for v in raw], "skipped-merges")],
        )
    while True:
        nbytes = rng.choice([1200, 2400, 6000])
        cA, cB = rng.sample([2, 3, 4, 5, 6], 2)
        kA, kB = rng.randint(2, 7), rng.randint(2, 7)
        bA, bB = R(kA, cA), R(kB, cB)
        if bA != bB and kA != kB:
            break
    nA, nB = nbytes // cA, nbytes // cB
    return Problem(
        prompt=(f"Comparing language models across tokenizers: on the same test text of ${nbytes}$ bytes, model A's tokenizer produces "
                f"${nA}$ tokens and A has per-token perplexity $2^{{{kA}}}$; model B's tokenizer produces ${nB}$ tokens and B has per-token "
                f"perplexity $2^{{{kB}}}$. Derive bits-per-byte $=$ (total bits)/(bytes) from the perplexity and give it for A and B (exact). "
                "Which model actually compresses the text better?"),
        answer=[fmt(bA), fmt(bB)], kind="list", answer_hint="bpb_A, bpb_B   e.g. 3/2, 7/5",
        rubric=["log2(PPL) = average bits per token", "Total bits = tokens · bits per token", "Divide by the number of bytes",
                "Lower bpb is better; per-token perplexity alone is not comparable across tokenizers"],
        solution=(f"A: ${kA}\\cdot{nA}/{nbytes} = {L(bA)}$ bits/byte; B: ${kB}\\cdot{nB}/{nbytes} = {L(bB)}$ bits/byte. "
                  f"Model {'A' if bA < bB else 'B'} is better per byte"
                  + (" even though its per-token perplexity is higher." if (bA < bB) == (kA > kB) else ".")),
        misconceptions=[([str(kA), str(kB)], "reported-bits-per-token")],
    )


# =========================================================================== dl.gpt
GPT_CFGS = [("GPT-2 small", 50257, 1024, 768, 12), ("GPT-2 medium", 50257, 1024, 1024, 24),
            ("GPT-2 large", 50257, 1024, 1280, 36), ("char-level baby GPT", 65, 256, 384, 6),
            ("tiny GPT", 50304, 512, 512, 8), ("small GPT", 32000, 2048, 1024, 16)]

GPT3_CFGS = [("GPT-3 Small (125M)", 12, 768), ("GPT-3 Medium (350M)", 24, 1024), ("GPT-3 Large (760M)", 24, 1536),
             ("GPT-3 XL (1.3B)", 24, 2048), ("GPT-3 2.7B", 32, 2560), ("GPT-3 6.7B", 32, 4096),
             ("GPT-3 13B", 40, 5140), ("GPT-3 175B", 96, 12288)]


@gen("dl.gpt", levels=(1, 2, 3))
def gpt_params(rng, level):
    if level == 1:
        name, V, T, d, Ln = rng.choice(GPT_CFGS)
        ans = V * d + T * d
        return Problem(
            prompt=(f"A GPT ({name}) has vocabulary $V = {V}$, context length $T = {T}$, width $d = {d}$ and learned absolute "
                    "position embeddings. How many parameters are in the token + position embedding tables?"),
            answer=str(ans), kind="num",
            rubric=["Token embedding V x d", "Position embedding T x d", "Add"],
            solution=f"${V}\\cdot{d} + {T}\\cdot{d} = {ans}$.",
            misconceptions=[(str(V * d), "forgot-position-embedding")],
        )
    if level == 2:
        name, V, T, d, Ln = rng.choice(GPT_CFGS)
        blk = 12 * d * d + 13 * d
        ans = V * d + T * d + Ln * blk + 2 * d
        return Problem(
            prompt=(f"Count **exactly** the parameters of a GPT-2-architecture model ({name}): $V = {V}$, $T = {T}$, $d = {d}$, "
                    f"$n_{{layer}} = {Ln}$. Each block has $12d^2 + 13d$ parameters (biases + 2 LayerNorms), there is a final LayerNorm "
                    "(gain + bias), and the output head is **tied** to the token embedding."),
            answer=str(ans), kind="num",
            rubric=["Embeddings: V d + T d", "Blocks: n_layer (12 d^2 + 13 d)", "Final LayerNorm: 2d",
                    "Tied head adds nothing"],
            solution=(f"${V * d} + {T * d} + {Ln}\\cdot{blk} + {2 * d} = {ans}$. (GPT-2 small gives 124,439,808 — the '124M' "
                      "reproduced in build-nanogpt.)"),
            misconceptions=[(str(ans + V * d), "counted-tied-head-twice"), (str(ans - 2 * d), "forgot-final-ln"),
                            (str(ans - T * d), "forgot-position-embedding")],
        )
    name, Ln, d = rng.choice(GPT3_CFGS)
    n_approx = 12 * Ln * d * d
    tot = n_approx + 50257 * d
    return Problem(
        prompt=(f"GPT-3 (Brown et al. 2020, Table 2.1) lists {name} with $n_{{layers}} = {Ln}$ and $d_{{model}} = {d}$. "
                "Estimate (a) the non-embedding parameters $N \\approx 12\\,n_{layers}\\,d^2$ and (b) $N$ plus the token "
                "embedding ($V = 50257$). Compare with the nominal size. Give (a), (b)."),
        answer=[str(n_approx), str(tot)], kind="list", tol=1e-3, answer_hint="two numbers, e.g. 1.2e8, 1.6e8",
        rubric=["Per block 12 d^2 (4d^2 attention + 8d^2 MLP)", "Multiply by n_layers", "Embedding V·d",
                "Small models: embedding is a big fraction; large models: negligible"],
        solution=(f"(a) $12\\cdot{Ln}\\cdot{d}^2 \\approx {sci(n_approx)}$; (b) $+\\,50257\\cdot{d} \\approx {sci(tot)}$. "
                  f"The embedding is ${100 * 50257 * d / tot:.1f}\\%$ of the total (arXiv:2005.14165)."),
        misconceptions=[([str(4 * Ln * d * d), str(4 * Ln * d * d + 50257 * d)], "attention-only")],
    )


@gen("dl.gpt", levels=(1, 2, 3))
def gpt_flops(rng, level):
    N = rng.choice([124e6, 350e6, 1.3e9, 7e9, 13e9, 70e9])
    D = rng.choice([10e9, 100e9, 300e9, 1e12, 2e12])
    Ni, Di = int(N), int(D)
    C = 6 * Ni * Di
    if level == 1:
        return Problem(
            prompt=f"Estimate the training compute of a transformer with $N = {sci(N)}$ parameters trained on $D = {sci(D)}$ tokens, using $C \\approx 6ND$. Answer in FLOPs (e.g. 3.15e23).",
            answer=str(C), kind="num", tol=1e-3,
            rubric=["Forward: 2 FLOPs per parameter per token", "Backward: about twice the forward", "C = 6 N D"],
            solution=f"$6\\cdot{sci(N)}\\cdot{sci(D)} = {sci(C)}$ FLOPs.",
            misconceptions=[(str(2 * Ni * Di), "forward-only"), (str(Ni * Di), "forgot-factor-6")],
        )
    if level == 2:
        gpu, peak = rng.choice([("A100", 312e12), ("H100", 989e12), ("RTX 5070 Ti", 176e12)])
        ng = rng.choice([1, 8, 64, 256, 1024])
        mfu = rng.choice([R(3, 10), R(2, 5), R(1, 2)])
        days = sp.Integer(C) / (ng * sp.Integer(int(peak)) * mfu) / 86400
        ideal = sp.Integer(C) / (ng * sp.Integer(int(peak))) / 86400
        return Problem(
            prompt=(f"Training a model with $N = {sci(N)}$ on $D = {sci(D)}$ tokens ($C \\approx 6ND$) on ${ng}$ × {gpu} GPUs with "
                    f"peak ${sci(peak)}$ FLOP/s each, at model-FLOPs utilisation (MFU) {float(mfu):.0%}. How many **days** "
                    "does it take? Give at least 3 significant figures."),
            answer=fmt(days), kind="num", tol=5e-3,
            rubric=["C = 6ND", "Achieved throughput = n_gpu · peak · MFU", "Seconds = C / throughput", "Divide by 86400"],
            solution=(f"$C = {sci(C)}$, throughput $= {ng}\\cdot{sci(peak)}\\cdot{float(mfu)} = {sci(ng * peak * float(mfu))}$ FLOP/s, "
                      f"time $= {sci(float(days) * 86400)}$ s $\\approx {sig(days)}$ days."),
            misconceptions=[(fmt(ideal), "ignored-mfu")],
        )
    return Problem(
        prompt=(f"Derive the per-token training cost per parameter: a forward pass does one multiply-add (2 FLOPs) per parameter per token, "
                "and the backward pass computes gradients w.r.t. both activations and weights, each costing as much as the forward. "
                f"With **full activation recomputation** the forward is run a second time during backward. For $N = {sci(N)}$ and "
                f"$D = {sci(D)}$ give the training FLOPs (a) without and (b) with recomputation."),
        answer=[str(C), str(8 * Ni * Di)], kind="list", tol=1e-3, answer_hint="two numbers, e.g. 6e21, 8e21",
        rubric=["Forward 2N per token", "Backward 4N per token (activation grads + weight grads)", "Total 6ND",
                "Recomputation adds one more forward: 8ND"],
        solution=(f"(a) $(2 + 4)ND = {sci(C)}$. (b) $(2 + 2 + 4)ND = {sci(8 * Ni * Di)}$. Kaplan et al. (arXiv:2001.08361) and "
                  "Chinchilla (arXiv:2203.15556) both use $C \\approx 6ND$; MFU is reported against 6ND, not the 8ND actually executed."),
        misconceptions=[([str(2 * Ni * Di), str(4 * Ni * Di)], "forward-only")],
    )


@gen("dl.gpt", levels=(1, 2, 3))
def gpt_memory(rng, level):
    if level == 1:
        B, T = rng.choice([1, 4, 8, 16, 32]), rng.choice([256, 512, 1024, 2048])
        V = rng.choice([50257, 50304, 32000, 65])
        byt = B * T * V * 4
        ans = R(byt, 2 ** 20)
        return Problem(
            prompt=(f"Memory for the output logits: a GPT with vocabulary $V = {V}$ processes a batch of $B = {B}$ sequences of "
                    f"$T = {T}$ tokens. The logits tensor of shape $(B, T, V)$ is stored in fp32 (4 bytes). How many MiB ($2^{{20}}$ bytes) "
                    "does it take? Give at least 4 significant figures."),
            answer=fmt(ans), kind="num", tol=TOL,
            rubric=["Elements = B · T · V", "Bytes = 4 per fp32 element", "Divide by 2^20"],
            solution=f"${B}\\cdot{T}\\cdot{V}\\cdot4 = {byt}$ bytes $= {sig(ans, 6)}$ MiB. Large vocabularies make the logits a memory hot spot.",
            misconceptions=[(fmt(ans / 2), "used-fp16-size"), (fmt(ans / 4), "counted-elements-not-bytes")],
        )
    if level == 2:
        N = rng.choice([124e6, 350e6, 1.3e9, 7e9])
        gpu, peak = rng.choice([("A100", 312e12), ("H100", 989e12)])
        ng = rng.choice([1, 8, 64])
        target = rng.choice([0.25, 0.35, 0.45, 0.55])
        tps = float(f"{target * ng * peak / (6 * N):.2g}")
        mfu = 6 * N * tps / (ng * peak)
        return Problem(
            prompt=(f"Model FLOPs utilisation (MFU, from the PaLM paper): training a model with $N = {sci(N)}$ parameters on ${ng}$ × {gpu} "
                    f"(peak ${sci(peak)}$ FLOP/s each) processes ${sci(tps, 2)}$ tokens per second. Using $6N$ FLOPs per token, what "
                    "is the MFU? Give a fraction between 0 and 1 to at least 3 significant figures."),
            answer=repr(mfu), kind="num", tol=5e-3,
            rubric=["Achieved FLOP/s = 6 N · tokens/s", "Available FLOP/s = n_gpu · peak", "MFU = achieved / available"],
            solution=(f"Achieved $6\\cdot{sci(N)}\\cdot{sci(tps, 2)} = {sci(6 * N * tps)}$ FLOP/s; available ${ng}\\cdot{sci(peak)} = {sci(ng * peak)}$; "
                      f"MFU $\\approx {mfu:.3f}$."),
            misconceptions=[(repr(2 * N * tps / (ng * peak)), "forward-only"), (repr(6 * N * tps / peak), "forgot-gpu-count")],
        )
    N = rng.choice([1.3e9, 7e9, 13e9, 70e9])
    k = rng.choice([8, 16, 64, 256])
    tot = 16 * N / 1e9
    z1 = (4 * N + 12 * N / k) / 1e9
    z3 = tot / k
    return Problem(
        prompt=("Training memory with mixed-precision Adam (ZeRO paper, Rajbhandari et al. 2019): per parameter you keep fp16 weights "
                "(2 B), fp16 gradients (2 B) and fp32 optimizer state — master weights, momentum, variance (4 + 4 + 4 B). For "
                f"$N = {sci(N)}$ parameters trained on ${k}$ GPUs (ignore activations), derive (a) the total model-state memory, "
                "(b) the per-GPU memory with ZeRO stage 1 (only the optimizer state is partitioned) and (c) per-GPU with stage 3 "
                "(everything partitioned). Answer in GB ($10^9$ bytes)."),
        answer=[repr(tot), repr(z1), repr(z3)], kind="list", tol=1e-3, answer_hint="three numbers in GB, e.g. 112, 17.3, 7",
        rubric=["2 + 2 + 12 = 16 bytes per parameter", "Total 16 N", "Stage 1: weights + grads replicated (4N), optimizer 12N/k",
                "Stage 3: 16N/k"],
        solution=(f"(a) $16N = {sig(tot)}$ GB. (b) $4N + 12N/{k} = {sig(z1)}$ GB. (c) $16N/{k} = {sig(z3)}$ GB per GPU (arXiv:1910.02054)."),
        misconceptions=[([repr(2 * N / 1e9), repr(2 * N / 1e9), repr(2 * N / 1e9 / k)], "weights-only")],
    )


# =========================================================================== dl.decoding
@gen("dl.decoding", levels=(1, 2, 3))
def sampling(rng, level):
    if level == 1:
        T = rng.choice([R(1, 2), sp.Integer(2)])
        n = rng.choice([3, 4])
        while True:
            a = [rng.choice([1, 4, 9, 16]) if T == 2 else rng.randint(1, 5) for _ in range(n)]
            if len(set(a)) > 1:
                break
        j = rng.randrange(n)
        w = [sp.Integer(ai) ** (1 / T) for ai in a]
        ans = w[j] / sum(w)
        logits = ",\\ ".join(f"\\ln {ai}" if ai != 1 else "0" for ai in a)
        return Problem(
            prompt=(f"Next-token logits are $\\mathbf z = ({logits})$. With temperature $T = {L(T)}$, "
                    f"$p_i \\propto e^{{z_i/T}}$. What is the probability of token {j + 1}? Give an exact fraction."),
            answer=fmt(ans), kind="num",
            rubric=["e^{z_i/T} = a_i^{1/T}", "Normalise by the sum", "T < 1 sharpens, T > 1 flattens"],
            solution=(f"$e^{{z_i/T}} = a_i^{{{L(1 / T)}}} = {tup(w)}$, sum $= {sum(w)}$, so $p_{{{j + 1}}} = {L(ans)}$."),
            misconceptions=[(fmt(R(a[j], sum(a))), "ignored-temperature")],
        )
    if level == 2:
        while True:
            c = rng.sample(range(1, 12), 5)
            S = sum(c)
            p = sorted([R(ci, S) for ci in c], reverse=True)
            mode = rng.choice(["k", "p"])
            if mode == "k":
                k = rng.choice([2, 3])
                keep = p[:k]
                thr = None
                break
            thr = rng.choice([R(1, 2), R(3, 5), R(7, 10), R(4, 5), R(9, 10)])
            cum, keep = 0, []
            for pi in p:
                keep.append(pi)
                cum += pi
                if cum >= thr:
                    break
            if cum != thr and len(keep) >= 2:
                break
        ans = p[0] / sum(keep)
        plist = ", ".join(L(v) for v in p)
        rule = (f"**top-$k$** sampling with $k = {k}$" if mode == "k" else
                f"**nucleus (top-$p$)** sampling with $p = {L(thr)}$ (keep the smallest set of most-likely tokens whose cumulative probability is $\\ge p$)")
        return Problem(
            prompt=(f"The model's next-token distribution, sorted, is $({plist})$. Using {rule}, what is the probability "
                    "of sampling the most likely token after renormalisation? Give an exact fraction."),
            answer=fmt(ans), kind="num",
            rubric=["Sort probabilities descending", "Determine the kept set", "Renormalise: divide by the kept mass"],
            solution=(f"Kept set: $({', '.join(L(v) for v in keep)})$ with mass ${L(sum(keep))}$; "
                      f"$p'_1 = {L(p[0])}/{L(sum(keep))} = {L(ans)}$."),
            misconceptions=[(fmt(p[0]), "forgot-renormalize")],
        )
    while True:
        a = rng.sample(range(1, 7), 4)
        w = sorted([sp.Integer(ai) ** 2 for ai in a], reverse=True)
        q = [wi / sum(w) for wi in w]
        thr = rng.choice([R(1, 2), R(7, 10), R(4, 5), R(9, 10)])
        cum, keep = 0, []
        for qi in q:
            keep.append(qi)
            cum += qi
            if cum >= thr:
                break
        # nucleus before temperature (for the misconception) must also be well-defined
        p0 = sorted([R(ai, sum(a)) for ai in a], reverse=True)
        cum0, keep0 = 0, []
        for pi in p0:
            keep0.append(pi)
            cum0 += pi
            if cum0 >= thr:
                break
        if cum != thr and cum0 != thr:
            break
    ans = [len(keep), q[0] / sum(keep)]
    logits = ",\\ ".join(f"\\ln {ai}" if ai != 1 else "0" for ai in a)
    return Problem(
        prompt=(f"Logits $\\mathbf z = ({logits})$. Apply temperature $T = 1/2$ **first**, then nucleus sampling with $p = {L(thr)}$ "
                "(smallest set of top tokens with cumulative probability $\\ge p$), then renormalise. Give (a) the size of the nucleus and "
                "(b) the final probability of the most likely token (exact). (Holtzman et al. 2019 introduced nucleus sampling.)"),
        answer=[str(ans[0]), fmt(ans[1])], kind="list", answer_hint="size, probability   e.g. 2, 25/34",
        rubric=["T = 1/2 squares the e^{z_i}: weights a_i^2", "Normalise and sort", "Accumulate until >= p",
                "Renormalise over the nucleus"],
        solution=(f"Weights $a_i^2$ sorted: ${tup(w)}$, probabilities ${tup(q)}$. Cumulative mass reaches $\\ge {L(thr)}$ after "
                  f"${len(keep)}$ token(s) (mass ${L(sum(keep))}$). Top token: ${L(q[0])}/{L(sum(keep))} = {L(ans[1])}$."),
        misconceptions=[([str(len(keep0)), fmt(p0[0] / sum(keep0))], "ignored-temperature")],
    )


@gen("dl.decoding", levels=(1, 2, 3))
def decoding_more(rng, level):
    def tree():
        while True:
            c = rng.sample(range(1, 10), 3)
            first = sorted([R(ci, sum(c)) for ci in c], reverse=True)
            qs = [rng.choice([R(1, 10), R(1, 5), R(1, 4), R(1, 3), R(2, 5), R(3, 5), R(2, 3), R(3, 4), R(4, 5), R(9, 10)]) for _ in range(3)]
            cond = [(q, 1 - q) for q in qs]
            joint = {(a, b): first[a] * cond[a][b] for a in range(3) for b in range(2)}
            greedy = first[0] * max(cond[0])
            beam = max(joint[(a, b)] for a in range(2) for b in range(2))
            if beam != greedy and len(set(joint.values())) == 6:
                return first, cond, joint, greedy, beam
    names = ("A", "B", "C")

    def tree_txt(first, cond):
        s1 = ", ".join(f"$P(\\texttt{{{names[a]}}}) = {L(first[a])}$" for a in range(3))
        s2 = "; ".join(f"after `{names[a]}`: $P(\\texttt{{x}}) = {L(cond[a][0])}$, $P(\\texttt{{y}}) = {L(cond[a][1])}$" for a in range(3))
        return s1, s2

    if level == 1:
        if rng.random() < 0.5:
            first, cond, joint, greedy, beam = tree()
            s1, s2 = tree_txt(first, cond)
            best2 = "x" if cond[0][0] > cond[0][1] else "y"
            return Problem(
                prompt=(f"Greedy decoding: at step 1 the model gives {s1}; at step 2 it gives {s2}. Greedy decoding takes the argmax "
                        "at every step. What is the probability of the two-token sequence it outputs? (exact)"),
                answer=fmt(greedy), kind="num",
                rubric=["Step 1: pick the most likely first token", "Step 2: pick the most likely continuation of THAT token",
                        "Sequence probability = product of the two"],
                solution=f"Greedy picks `A` then `{best2}`: ${L(first[0])}\\cdot{L(max(cond[0]))} = {L(greedy)}$.",
                misconceptions=[(fmt(beam), "searched-beyond-greedy"), (fmt(first[0]), "forgot-second-step")],
            )
        dists = [(R(1, 2), R(1, 4), R(1, 4)), (R(1, 2), R(1, 4), R(1, 8), R(1, 8)), (R(1, 4),) * 4, (R(1, 2), R(1, 2)),
                 (R(1, 2), R(1, 8), R(1, 8), R(1, 8), R(1, 8)), (R(1, 4), R(1, 4), R(1, 4), R(1, 8), R(1, 8)),
                 (R(1, 2), R(1, 4), R(1, 8), R(1, 16), R(1, 16))]
        p = list(rng.choice(dists))
        rng.shuffle(p)
        H = sum(pi * round(math.log2(1 / pi)) for pi in p)
        Hn = sum(pi * sp.log(1 / pi) for pi in p)
        return Problem(
            prompt=(f"Entropy of a next-token distribution: the model predicts $\\mathbf p = {tup(p)}$ over {len(p)} tokens. "
                    "Compute $H(\\mathbf p) = -\\sum_i p_i\\log_2 p_i$ in bits (exact). Low entropy = confident model."),
            answer=fmt(H), kind="num",
            rubric=["-log2 p_i for each token (powers of 2 give integers)", "Weight by p_i", "Sum"],
            solution=(f"$H = " + " + ".join(f"{L(pi)}\\cdot{round(math.log2(1 / pi))}" for pi in p) + f" = {L(H)}$ bits "
                      f"(perplexity $2^H \\approx {sig(2 ** H)}$)."),
            misconceptions=[(nfmt(Hn), "used-natural-log"), (fmt(-H), "sign-error")],
        )
    if level == 2:
        if rng.random() < 0.5:
            first, cond, joint, greedy, beam = tree()
            s1, s2 = tree_txt(first, cond)
            return Problem(
                prompt=(f"Greedy vs. beam search: at step 1 the model gives {s1}; at step 2 it gives {s2}. Give the probability of the "
                        "two-token sequence found by (a) greedy decoding and (b) beam search with beam width 2 (keep the 2 best "
                        "prefixes after step 1, expand both, return the best full sequence). Exact fractions."),
                answer=[fmt(greedy), fmt(beam)], kind="list", answer_hint="greedy, beam   e.g. 3/10, 9/25",
                rubric=["Greedy: argmax at each step", "Beam: keep A and B after step 1", "Score all 4 two-token sequences by their joint probability",
                        "Beam returns the best of those (it may beat greedy, but is not guaranteed optimal)"],
                solution=("Joint probabilities: " + ", ".join(f"`{names[a]}{'xy'[b]}`: ${L(joint[(a, b)])}$" for a in range(2) for b in range(2))
                          + f". Greedy $= {L(greedy)}$, beam $= {L(beam)}$."),
                misconceptions=[([fmt(greedy), fmt(greedy)], "beam-equals-greedy")],
            )
        while True:
            c = sorted(rng.sample(range(1, 13), 5), reverse=True)
            S = sum(c)
            p = [R(ci, S) for ci in c]
            pb = rng.choice([R(1, 10), R(1, 5), R(1, 4), R(1, 3), R(1, 2)])
            thr = pb * p[0]
            keep = [pi for pi in p if pi >= thr]
            if 2 <= len(keep) < 5 and thr not in p:
                break
        ans = p[1] / sum(keep)
        abs_keep = [pi for pi in p if pi >= pb]
        mis = [(fmt(p[1]), "forgot-renormalize")]
        if len(abs_keep) >= 2:
            mis.append((fmt(p[1] / sum(abs_keep)), "absolute-threshold"))
        return Problem(
            prompt=(f"Min-p sampling (Nguyen et al. 2024) keeps every token whose probability is at least $p_{{base}}\\cdot p_{{max}}$, "
                    f"then renormalises. The sorted distribution is $({', '.join(L(v) for v in p)})$ and $p_{{base}} = {L(pb)}$. What is the "
                    "sampling probability of the second most likely token? (exact)"),
            answer=fmt(ans), kind="num",
            rubric=["Threshold = p_base · p_max (scales with the model's confidence)", "Keep tokens with p >= threshold",
                    "Renormalise over the kept set"],
            solution=(f"Threshold $= {L(pb)}\\cdot{L(p[0])} = {L(thr)}$; kept $({', '.join(L(v) for v in keep)})$ with mass ${L(sum(keep))}$; "
                      f"$p_2' = {L(p[1])}/{L(sum(keep))} = {L(ans)}$."),
            misconceptions=mis,
        )
    if rng.random() < 0.5:
        while True:
            k1, k2 = rng.randint(1, 9), rng.randint(0, 8)
            m = rng.choice([1, 2, 3])
            if k1 > k2 and R(k1 - k2, m) != 1:
                break
        T = R(k1 - k2, m)
        z2 = "0" if k2 == 0 else f"{k2}\\ln 2"
        return Problem(
            prompt=(f"Choosing a temperature: two candidate tokens have logits $z_1 = {k1}\\ln 2$ and $z_2 = {z2}$. With "
                    "$p_i \\propto e^{z_i/T}$, derive $p_1/p_2$ as a function of $T$ and find the temperature at which token 1 is "
                    f"exactly ${2 ** m}$ times as likely as token 2. (exact)"),
            answer=fmt(T), kind="num",
            rubric=["The normaliser cancels: p1/p2 = exp((z1 - z2)/T)", f"Set exp((z1 - z2)/T) = 2^{m}",
                    "Take logs: (z1 - z2)/T = m ln 2", "Solve for T"],
            solution=(f"$p_1/p_2 = e^{{(z_1-z_2)/T}} = 2^{{{k1 - k2}/T}}$. Setting it to $2^{{{m}}}$: ${k1 - k2}/T = {m}$, so $T = {L(T)}$. "
                      "Larger $T$ flattens the ratio towards 1."),
            misconceptions=[(fmt(1 / T), "inverted-temperature"), (str((k1 - k2) * m), "multiplied-instead-of-divided")],
        )
    while True:
        p = list(rng.choice(_PTRIPLES))
        q = list(rng.choice(_PTRIPLES))
        rng.shuffle(p)
        rng.shuffle(q)
        if p != q:
            break
    alpha = sum(min(a, b) for a, b in zip(p, q))
    g = rng.choice([2, 3, 4])
    E = (1 - alpha ** (g + 1)) / (1 - alpha)
    return Problem(
        prompt=("Speculative decoding (Leviathan et al. 2023): a draft model proposes tokens from $q$, the target model has distribution $p$, "
                "and a draft token $x$ is accepted with probability $\\min(1, p(x)/q(x))$. Over the vocabulary {A, B, C}: "
                f"$p = {tup(p)}$, $q = {tup(q)}$. (a) Derive that the acceptance rate is $\\alpha = \\sum_x \\min(p(x), q(x))$ and "
                f"compute it. (b) With $\\gamma = {g}$ draft tokens per round, the expected number of tokens produced per target-model call "
                "is $(1-\\alpha^{\\gamma+1})/(1-\\alpha)$; compute it. Exact fractions."),
        answer=[fmt(alpha), fmt(E)], kind="list", answer_hint="alpha, expected tokens   e.g. 5/6, 2671/1296",
        rubric=["P(accept) = sum_x q(x) min(1, p(x)/q(x))", "= sum_x min(q(x), p(x))", "Geometric series for the expected tokens",
                "Rejected tokens are resampled from norm(max(0, p - q)), so the output is exactly distributed as p"],
        solution=(f"$\\alpha = " + " + ".join(L(min(a, b)) for a, b in zip(p, q)) + f" = {L(alpha)}$. "
                  f"Expected tokens $= (1 - ({L(alpha)})^{{{g + 1}}})/(1 - {L(alpha)}) = {L(E)} \\approx {sig(E)}$ (arXiv:2211.17192)."),
        misconceptions=[([fmt(sum(a * b for a, b in zip(p, q))), fmt(E)], "multiplied-probabilities"),
                        ([fmt(alpha), fmt(1 + g * alpha)], "linear-acceptance")],
    )


# =========================================================================== dl.efficient_inference
KV_CFGS = [("Llama-2-7B", 32, 32, 128, 6.74e9), ("Llama-2-13B", 40, 40, 128, 13.0e9), ("Llama-3-8B (GQA)", 32, 8, 128, 8.03e9),
           ("Mistral-7B (GQA)", 32, 8, 128, 7.24e9), ("GPT-2 small", 12, 12, 64, 124e6), ("GPT-2 XL", 48, 25, 64, 1.56e9)]


@gen("dl.efficient_inference", levels=(1, 2, 3))
def kv_cache(rng, level):
    name, Ln, H, dh, N = rng.choice(KV_CFGS)
    if level == 1:
        B, T = rng.choice([1, 4, 8, 16]), rng.choice([1024, 2048, 4096, 8192])
        byt = 2 * Ln * H * dh * T * B * 2
        gib = R(byt, 2 ** 30)
        return Problem(
            prompt=(f"{name}: $n_{{layers}} = {Ln}$, $n_{{kv\\,heads}} = {H}$, $d_{{head}} = {dh}$, fp16 (2 bytes). How much memory does the "
                    f"KV cache take for batch ${B}$ at sequence length ${T}$? Answer in GiB ($2^{{30}}$ bytes)."),
            answer=fmt(gib), kind="num", tol=TOL,
            rubric=["Per token per layer: K and V, each n_kv·d_head numbers", "Bytes = 2 · layers · n_kv · d_head · T · B · 2",
                    "Divide by 2^30"],
            solution=f"$2\\cdot{Ln}\\cdot{H}\\cdot{dh}\\cdot{T}\\cdot{B}\\cdot2 = {byt}$ bytes $= {sig(gib)}$ GiB.",
            misconceptions=[(fmt(gib / 2), "forgot-k-and-v"), (fmt(R(byt, 10 ** 9)), "used-gb-not-gib")],
        )
    if level == 2:
        while True:
            M = rng.choice([24, 40, 48, 80])
            T = rng.choice([2048, 4096, 8192])
            per = 2 * Ln * H * dh * T * 2
            free = M * 2 ** 30 - 2 * int(N)
            if free >= per:
                break
        ans = free // per
        return Problem(
            prompt=(f"Serving {name} ($N \\approx {sci(N, 3)}$ parameters in fp16; $n_{{layers}} = {Ln}$, $n_{{kv}} = {H}$, $d_{{head}} = {dh}$) on a "
                    f"${M}$ GiB GPU. Ignoring activations and overhead, what is the largest batch of sequences of length ${T}$ whose "
                    f"fp16 KV cache fits next to the weights? (Use $N = {int(N)}$ exactly.)"),
            answer=str(ans), kind="num",
            rubric=["Weights: 2N bytes", "Free memory = M · 2^30 - 2N", "KV per sequence: 2·layers·n_kv·d_head·T·2 bytes",
                    "Floor of the ratio"],
            solution=(f"Free $= {M}\\cdot2^{{30}} - 2\\cdot{int(N)} = {free}$ bytes; per sequence ${per}$ bytes; "
                      f"$\\lfloor {free}/{per}\\rfloor = {ans}$."),
            misconceptions=[(str((M * 2 ** 30) // per), "forgot-weights")] if (M * 2 ** 30) // per != ans else [],
        )
    gpu, bw = rng.choice([("RTX 5070 Ti", 896), ("RTX 4090", 1008), ("A100 80GB", 2039), ("H100 SXM", 3350)])
    bpp, prec = rng.choice([(2, "fp16"), (1, "int8"), (R(1, 2), "4-bit")])
    T = rng.choice([1024, 4096])
    kv = 2 * Ln * H * dh * T * 2
    byt = int(N) * bpp + kv
    ans = sp.Integer(bw) * 10 ** 9 / byt
    return Problem(
        prompt=(f"Batch-1 decoding is memory-bandwidth bound: every new token must stream all weights plus the KV cache from GPU memory. "
                f"For {name} ($N = {int(N)}$, weights in {prec}, fp16 KV cache with $n_{{layers}} = {Ln}$, $n_{{kv}} = {H}$, $d_{{head}} = {dh}$, "
                f"current length ${T}$) on a {gpu} with ${bw}$ GB/s ($10^9$ B/s), what is the upper bound on tokens per second? "
                "Give at least 3 significant figures."),
        answer=fmt(ans), kind="num", tol=5e-3,
        rubric=["Weight bytes = N · bytes/param", "KV bytes = 2·layers·n_kv·d_head·T·2", "tokens/s <= bandwidth / bytes per token",
                "Batching amortises the weight reads — why serving uses large batches"],
        solution=(f"Bytes per token $= {int(N)}\\cdot{L(sp.sympify(bpp))} + {kv} = {sci(byt)}$; "
                  f"${bw}\\times10^9 / {sci(byt)} \\approx {sig(ans)}$ tokens/s. The same memory-traffic view motivates FlashAttention "
                  "(arXiv:2205.14135), which avoids writing the $n\\times n$ score matrix to HBM."),
        misconceptions=[(fmt(sp.Integer(bw) * 10 ** 9 / (int(N) * bpp)), "ignored-kv-cache")],
    )


@gen("dl.efficient_inference", levels=(1, 2, 3))
def inference_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            N = rng.choice([7e9, 8e9, 13e9, 70e9, 405e9])
            prec, bpp = rng.choice([("fp32", 4), ("bf16", 2), ("int8", 1), ("4-bit", R(1, 2))])
            gb = sp.Integer(int(N)) * bpp / 10 ** 9
            return Problem(
                prompt=(f"Weight memory: how many GB ($10^9$ bytes) do the weights of an $N = {sci(N)}$-parameter model take in {prec}?"),
                answer=fmt(gb), kind="num", tol=TOL,
                rubric=["Bytes per parameter: fp32 4, bf16 2, int8 1, 4-bit 0.5", "Multiply by N", "Divide by 10^9"],
                solution=f"${sci(N)}\\cdot{L(sp.sympify(bpp))} = {sci(float(gb) * 1e9)}$ bytes $= {sig(gb)}$ GB.",
                misconceptions=[(fmt(gb * 2 ** 3), "counted-bits"), (fmt(gb * 10 ** 9 / 2 ** 30), "used-gib-not-gb")],
            )
        n = rng.choice([2048, 4096, 8192, 16384, 32768])
        h = rng.choice([12, 16, 32, 40])
        byt = 2 * h * n * n
        ans = R(byt, 2 ** 30)
        return Problem(
            prompt=(f"The score matrix FlashAttention never materialises: standard attention writes $S = QK^\\top$ ($n\\times n$) for every "
                    f"head to GPU memory. For one layer with $h = {h}$ heads, sequence length $n = {n}$, batch 1, fp16 (2 bytes): how many "
                    "GiB ($2^{30}$ bytes) is that? Give at least 4 significant figures."),
            answer=fmt(ans), kind="num", tol=TOL,
            rubric=["Per head n^2 entries", "Times h heads", "2 bytes each", "Divide by 2^30"],
            solution=(f"${h}\\cdot{n}^2\\cdot2 = {byt}$ bytes $= {sig(ans)}$ GiB — quadratic in $n$, which is why FlashAttention "
                      "tiles the computation in SRAM instead (arXiv:2205.14135)."),
            misconceptions=[(fmt(ans / h), "forgot-heads"), (fmt(ans * 2), "used-fp32")],
        )
    if level == 2:
        if rng.random() < 0.5:
            Ln, nh, dh = rng.choice([(32, 32, 128), (40, 40, 128), (80, 64, 128), (32, 32, 64)])
            nkv = rng.choice([v for v in (8, 4, 1) if v < nh])
            B, T = rng.choice([1, 4, 8, 16]), rng.choice([2048, 4096, 8192])
            mha = R(2 * Ln * nh * dh * T * B * 2, 2 ** 30)
            gqa = mha * R(nkv, nh)
            return Problem(
                prompt=(f"Grouped-query attention (Ainslie et al. 2023) shares each K/V head among a group of query heads. A model has "
                        f"${Ln}$ layers and ${nh}$ query heads of size $d_{{head}} = {dh}$. For batch ${B}$, sequence length ${T}$ and an fp16 "
                        f"KV cache, give the cache size in GiB (a) with standard multi-head attention and (b) with GQA using ${nkv}$ K/V head(s)."),
                answer=[fmt(mha), fmt(gqa)], kind="list", tol=TOL, answer_hint="MHA GiB, GQA GiB",
                rubric=["Per token per layer: K and V, each n_kv · d_head numbers", "Bytes = 2 · layers · n_kv · d_head · T · B · 2",
                        "MHA uses n_kv = n_heads", f"GQA shrinks the cache by n_heads/n_kv = {nh // nkv}"],
                solution=(f"(a) $2\\cdot{Ln}\\cdot{nh}\\cdot{dh}\\cdot{T}\\cdot{B}\\cdot2$ bytes $= {sig(mha)}$ GiB. (b) $\\times{nkv}/{nh}$: "
                          f"${sig(gqa)}$ GiB (arXiv:2305.13245)."),
                misconceptions=[([fmt(mha / 2), fmt(gqa / 2)], "forgot-k-and-v")],
            )
        gpu, peak, bw = rng.choice([("A100 80GB", 312e12, 2.039e12), ("H100 SXM", 989e12, 3.35e12),
                                    ("RTX 4090", 165e12, 1.008e12), ("RTX 5070 Ti", 176e12, 0.896e12)])
        ratio = peak / bw
        ans = math.ceil(ratio)
        return Problem(
            prompt=("Roofline for batched decoding: in fp16 each weight (2 bytes) is read once per decoding step and used for one "
                    "multiply-add (2 FLOPs) per sequence in the batch, so the arithmetic intensity is $\\approx B$ FLOP/byte. A GPU is "
                    f"compute-bound when the intensity exceeds peak FLOP/s ÷ bandwidth. For an {gpu} (peak ${sci(peak)}$ FLOP/s, "
                    f"bandwidth ${sci(bw)}$ B/s), what is the smallest integer batch size $B$ at which decoding becomes compute-bound?"),
            answer=str(ans), kind="num",
            rubric=["FLOPs per step = 2 · N · B; bytes per step ~ 2N", "Intensity ~ B FLOP/byte", "Ridge point = peak / bandwidth",
                    "Smallest integer B above the ridge (round up)"],
            solution=(f"Ridge point $= {sci(peak)}/{sci(bw)} \\approx {ratio:.1f}$ FLOP/byte, so $B = {ans}$. Below that, decoding is "
                      "memory-bound and extra sequences in the batch are almost free."),
            misconceptions=[(str(math.floor(ratio)), "rounded-down"), (str(math.ceil(ratio / 2)), "halved-intensity")],
        )
    if rng.random() < 0.5:
        n = rng.choice([1024, 2048, 4096, 8192])
        d = rng.choice([64, 128])
        M = 2 ** rng.choice([16, 17, 18])
        ratio = R((n * d + n * n) * M, n * n * d * d)
        return Problem(
            prompt=("FlashAttention's IO complexity (Dao et al. 2022, Theorem 2): standard attention needs $\\Theta(Nd + N^2)$ HBM accesses, "
                    "FlashAttention $\\Theta(N^2d^2/M)$, where $M$ is the SRAM size in elements. Treating $\\Theta$ as equality, for "
                    f"$N = {n}$, $d = {d}$, $M = 2^{{{int(math.log2(M))}}}$, give the ratio (standard)/(Flash) and show it is $\\approx M/d^2$ "
                    "when $N \\gg d$. Exact or 4 significant figures."),
            answer=fmt(ratio), kind="num", tol=TOL,
            rubric=["Ratio = (N d + N^2) M / (N^2 d^2)", "= M (N + d)/(N d^2)", "For N >> d: ~ M/d^2",
                    "So FlashAttention wins whenever d^2 < M (typical: d = 64-128, M ~ 100K)"],
            solution=(f"$\\frac{{(Nd + N^2)M}}{{N^2d^2}} = \\frac{{M(N+d)}}{{Nd^2}} = {L(ratio)} \\approx {sig(ratio)}$; "
                      f"$M/d^2 = {L(R(M, d * d))}$ (arXiv:2205.14135)."),
            misconceptions=[(fmt(1 / ratio), "inverted-ratio")],
        )
    nh, dh, dc, dr, Ln = rng.choice([(128, 128, 512, 64, 60), (32, 128, 512, 64, 32), (16, 128, 256, 64, 24), (64, 128, 512, 64, 40)])
    ratio = R(2 * nh * dh, dc + dr)
    byt = Ln * (dc + dr) * 2
    return Problem(
        prompt=("Multi-head latent attention (DeepSeek-V2, 2024) caches, per token and layer, one compressed latent of size $d_c$ plus a "
                "shared RoPE key of size $d_R$ instead of full keys and values for every head. With "
                f"$n_h = {nh}$ heads of size $d_h = {dh}$, $d_c = {dc}$, $d_R = {dr}$ and ${Ln}$ layers in fp16, derive (a) the KV-cache "
                "reduction factor versus standard multi-head attention (exact) and (b) the MLA cache size per token in bytes."),
        answer=[fmt(ratio), str(byt)], kind="list", answer_hint="factor, bytes per token   e.g. 512/9, 69120",
        rubric=["MHA per token per layer: 2 n_h d_h numbers (K and V)", "MLA: d_c + d_R numbers", "Factor = 2 n_h d_h/(d_c + d_R)",
                "Bytes per token = layers · (d_c + d_R) · 2"],
        solution=(f"(a) $\\frac{{2\\cdot{nh}\\cdot{dh}}}{{{dc} + {dr}}} = {L(ratio)} \\approx {sig(ratio)}$. (b) ${Ln}\\cdot{dc + dr}\\cdot2 = {byt}$ bytes "
                  "(arXiv:2405.04434)."),
        misconceptions=[([fmt(R(nh * dh, dc + dr)), str(byt)], "forgot-k-and-v")],
    )


# =========================================================================== dl.scaling_laws
@gen("dl.scaling_laws", levels=(1, 2, 3))
def chinchilla(rng, level):
    if level == 1:
        N = rng.choice([400e6, 1e9, 7e9, 13e9, 70e9])
        Ni = int(N)
        D = 20 * Ni
        return Problem(
            prompt=(f"By the Chinchilla rule of thumb a compute-optimal model trains on $D \\approx 20N$ tokens. For $N = {sci(N)}$ "
                    "parameters, give $D$ and the training compute $C \\approx 6ND$ (FLOPs)."),
            answer=[str(D), str(6 * Ni * D)], kind="list", tol=1e-3, answer_hint="D, C   e.g. 2e10, 1.2e20",
            rubric=["D = 20 N", "C = 6 N D = 120 N^2"],
            solution=f"$D = 20\\cdot{sci(N)} = {sci(D)}$ tokens, $C = 6ND = {sci(6 * Ni * D)}$ FLOPs.",
            misconceptions=[([str(D), str(2 * Ni * D)], "forward-only")],
        )
    if level == 2:
        N = rng.choice([1e9, 2e9, 5e9, 1e10, 3e10, 7e10])
        Ni = int(N)
        C = 120 * Ni * Ni
        return Problem(
            prompt=(f"You have a compute budget of $C = {sci(C)}$ FLOPs. Using $C = 6ND$ and the Chinchilla ratio $D = 20N$, find the "
                    "compute-optimal parameter count $N$ and token count $D$. Give N, D."),
            answer=[str(Ni), str(20 * Ni)], kind="list", tol=1e-3, answer_hint="N, D   e.g. 7e10, 1.4e12",
            rubric=["Substitute D = 20N: C = 120 N^2", "N = sqrt(C/120)", "D = 20N"],
            solution=f"$N = \\sqrt{{C/120}} = {sci(N)}$, $D = {sci(20 * N)}$. (Chinchilla itself: 70B params, 1.4T tokens.)",
            misconceptions=[([fmt(sp.sqrt(sp.Integer(C) / 6)), fmt(sp.sqrt(sp.Integer(C) / 6))], "dropped-20x")],
        )
    if rng.random() < 0.5:
        N = rng.choice([1e9, 7e9, 70e9, 175e9, 280e9])
        D = rng.choice([300e9, 1.4e12, 2e12, 15e12])
        E, A, B, al, be = sp.Rational(169, 100), sp.Rational(4064, 10), sp.Rational(4107, 10), sp.Rational(34, 100), sp.Rational(28, 100)
        Lv = E + A / sp.Integer(int(N)) ** al + B / sp.Integer(int(D)) ** be
        return Problem(
            prompt=(f"Chinchilla's parametric fit (Hoffmann et al. 2022, Approach 3): $L(N,D) = E + A/N^{{\\alpha}} + B/D^{{\\beta}}$ with "
                    f"$E = 1.69$, $A = 406.4$, $B = 410.7$, $\\alpha = 0.34$, $\\beta = 0.28$. Predict the loss for $N = {sci(N)}$, "
                    f"$D = {sci(D)}$. {SIG4}"),
            answer=nfmt(Lv), kind="num", tol=TOL,
            rubric=["Compute N^alpha and D^beta (use logs: x^a = e^{a ln x})", "A/N^alpha and B/D^beta",
                    "Add the irreducible loss E"],
            solution=(f"$A/N^\\alpha \\approx {sig(A / sp.Integer(int(N)) ** al)}$, $B/D^\\beta \\approx {sig(B / sp.Integer(int(D)) ** be)}$, "
                      f"$L \\approx {sig(Lv)}$ nats/token (arXiv:2203.15556)."),
            misconceptions=[(nfmt(Lv - E), "forgot-irreducible-loss")],
        )
    k = rng.choice([2, 4, 10, 100])
    ans = sp.Integer(k) ** sp.Rational(-76, 1000)
    return Problem(
        prompt=(f"Kaplan et al. (2020) fit $L(N) = (N_c/N)^{{\\alpha_N}}$ with $\\alpha_N = 0.076$ (data not limiting). "
                f"If you make the model ${k}\\times$ larger, by what factor $L(kN)/L(N)$ does the loss change? {SIG4}"),
        answer=nfmt(ans), kind="num", tol=1e-4,
        rubric=["Write the ratio: N_c cancels", "Ratio = k^{-alpha_N}", "Evaluate with logs"],
        solution=(f"$L(kN)/L(N) = k^{{-0.076}} = {k}^{{-0.076}} \\approx {sig(ans, 5)}$ — power laws give the same relative gain for "
                  "every constant factor of scale (arXiv:2001.08361)."),
        misconceptions=[(nfmt(sp.Integer(k) ** sp.Rational(76, 1000)), "sign-of-exponent"), (nfmt(1 - sp.Rational(76, 1000) * k), "linearised")],
    )


_TPP = [("Chinchilla", 70e9, 1.4e12), ("GPT-3", 175e9, 300e9), ("Llama-2-7B", 7e9, 2e12), ("Llama-3-8B", 8e9, 15e12),
        ("Gopher", 280e9, 300e9), ("Llama-2-70B", 70e9, 2e12)]


@gen("dl.scaling_laws", levels=(1, 2, 3))
def scaling_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            name, N, D = rng.choice(_TPP)
            ans = R(int(D), int(N))
            return Problem(
                prompt=(f"Tokens per parameter: {name} has $N = {sci(N)}$ parameters and was trained on $D = {sci(D)}$ tokens. Compute "
                        "$D/N$ and say whether it is under-trained (< 20) or over-trained (> 20) relative to Chinchilla. "
                        "Give $D/N$ (at least 4 significant figures)."),
                answer=fmt(ans), kind="num", tol=TOL,
                rubric=["Divide tokens by parameters", "Compare with Chinchilla's ~20 tokens per parameter"],
                solution=(f"$D/N = {sci(D)}/{sci(N)} \\approx {sig(ans)}$ — "
                          + ("under-trained for its size (Chinchilla-style models use ~20)." if ans < 20 else
                             "over-trained relative to Chinchilla: cheaper to serve, at extra training compute." if ans > 20 else
                             "exactly Chinchilla-optimal.")),
                misconceptions=[(fmt(1 / ans), "inverted-ratio")],
            )
        k = rng.choice([4, 9, 16, 25, 100])
        r = sp.sqrt(k)
        return Problem(
            prompt=(f"Splitting a bigger compute budget: Chinchilla found $N_{{opt}} \\propto C^{{0.5}}$ and $D_{{opt}} \\propto C^{{0.5}}$. If the "
                    f"compute budget is multiplied by ${k}$, by what factors do $N_{{opt}}$ and $D_{{opt}}$ grow?"),
            answer=[fmt(r), fmt(r)], kind="list", answer_hint="N factor, D factor",
            rubric=["Each grows by k^0.5", "Check: N·D grows by k, consistent with C = 6ND"],
            solution=f"$\\sqrt{{{k}}} = {r}$ for both; their product ${k}$ matches $C = 6ND$.",
            misconceptions=[([str(k), "1"], "all-into-parameters"), ([fmt(R(k, 2)), fmt(R(k, 2))], "split-linearly")],
        )
    if level == 2:
        if rng.random() < 0.5:
            N = rng.choice([1e9, 2e9, 7e9, 13e9, 70e9])
            rt = rng.choice([5, 10, 20, 40, 100, 200])
            Ni = int(N)
            D = rt * Ni
            C = 6 * Ni * D
            return Problem(
                prompt=(f"Under- or over-trained? You spend $C = {sci(C)}$ FLOPs ($C \\approx 6ND$) on a model with $N = {sci(N)}$ parameters. "
                        "Give the number of training tokens $D$ and the tokens-per-parameter ratio $D/N$."),
                answer=[str(D), str(rt)], kind="list", tol=1e-3, answer_hint="D, D/N   e.g. 1.4e12, 20",
                rubric=["D = C/(6N)", "Divide by N", "Compare with ~20 (Chinchilla-optimal)"],
                solution=(f"$D = {sci(C)}/(6\\cdot{sci(N)}) = {sci(D)}$ tokens, $D/N = {rt}$ — "
                          + ("compute-optimal." if rt == 20 else "under-trained (too many parameters for the budget)." if rt < 20 else
                             "over-trained (smaller than compute-optimal, cheaper inference).")),
                misconceptions=[([str(3 * D), str(3 * rt)], "forward-only")],
            )
        r = rng.choice([5, 10, 20])
        k = (1 - R(r, 100)) ** (-1 / sp.Rational(95, 1000))
        return Problem(
            prompt=(f"Kaplan et al. (2020) fit the data-limited loss as $L(D) = (D_c/D)^{{0.095}}$. By what factor must the dataset grow "
                    f"to reduce the loss by ${r}\\%$? {SIG4}"),
            answer=nfmt(k), kind="num", tol=TOL,
            rubric=["Ratio L(kD)/L(D) = k^{-0.095}", f"Set it equal to {1 - r / 100:g}", "Solve: k = (1 - r)^{-1/0.095}",
                    "Evaluate with logs"],
            solution=(f"$k^{{-0.095}} = {1 - r / 100:g} \\Rightarrow k = {1 - r / 100:g}^{{-1/0.095}} = e^{{-\\ln({1 - r / 100:g})/0.095}} \\approx {sig(k)}$ — "
                      "small exponents mean steep data requirements (arXiv:2001.08361)."),
            misconceptions=[(nfmt((1 - R(r, 100)) ** (-sp.Rational(95, 1000))), "inverted-exponent")],
        )
    C = rng.choice([1e21, 1e22, 1e23, 1e24, 5.76e23])
    A, B, al, be = sp.Rational(4064, 10), sp.Rational(4107, 10), sp.Rational(34, 100), sp.Rational(28, 100)
    a = be / (al + be)
    G = (al * A / (be * B)) ** (1 / (al + be))
    Nopt = G * (sp.Float(C, 30) / 6) ** a
    return Problem(
        prompt=("Derive Chinchilla's compute-optimal model size from the parametric fit $L(N,D) = E + A/N^\\alpha + B/D^\\beta$ under the "
                "constraint $C = 6ND$ (Hoffmann et al. 2022, eq. 4): substitute $D = C/(6N)$ and minimise over $N$ to show "
                "$N_{opt} = G\\,(C/6)^{\\beta/(\\alpha+\\beta)}$ with $G = (\\alpha A/(\\beta B))^{1/(\\alpha+\\beta)}$. With $A = 406.4$, "
                f"$B = 410.7$, $\\alpha = 0.34$, $\\beta = 0.28$ and $C = {sci(C)}$ FLOPs, give the exponent $\\beta/(\\alpha+\\beta)$ (exact) "
                f"and $N_{{opt}}$. {SIG4}"),
        answer=[fmt(a), nfmt(Nopt)], kind="list", tol=TOL, answer_hint="exponent, N_opt   e.g. 14/31, 3.1e10",
        rubric=["L(N) = E + A N^{-alpha} + B (C/6)^{-beta} N^{beta}", "dL/dN = 0: alpha A N^{-alpha} = beta B (C/6)^{-beta} N^{beta}",
                "Solve: N^{alpha+beta} = (alpha A/(beta B)) (C/6)^{beta}", "Evaluate G and the power with logs"],
        solution=(f"Exponent $= 0.28/0.62 = {L(a)}$; $G = ({sig(al * A / (be * B))})^{{1/0.62}} \\approx {sig(G)}$; "
                  f"$N_{{opt}} \\approx {sig(G)}\\cdot({sci(C / 6)})^{{0.4516}} \\approx {sci(float(Nopt))}$ parameters (arXiv:2203.15556)."),
        misconceptions=[([fmt(al / (al + be)), nfmt(G * (sp.Float(C, 30) / 6) ** (al / (al + be)))], "swapped-exponents")],
    )


# =========================================================================== dl.finetuning_lora
@gen("dl.finetuning_lora", levels=(1, 2, 3))
def lora(rng, level):
    if level == 1:
        din, dout = rng.choice([768, 1024, 2048, 4096]), rng.choice([768, 1024, 3072, 4096, 11008])
        r = rng.choice([1, 4, 8, 16, 64])
        ans = r * (din + dout)
        return Problem(
            prompt=(f"LoRA (Hu et al. 2021) adapts a frozen weight $W_0\\in\\mathbb R^{{{dout}\\times{din}}}$ with $W_0 + BA$, "
                    f"$B\\in\\mathbb R^{{{dout}\\times r}}$, $A\\in\\mathbb R^{{r\\times{din}}}$, rank $r = {r}$. How many trainable parameters does the "
                    "adapter have (vs. full fine-tuning of this matrix)?"),
            answer=str(ans), kind="num",
            rubric=["B has d_out·r entries", "A has r·d_in entries", "Total r(d_in + d_out)", "Compare with d_in·d_out"],
            solution=f"$r(d_{{in}}+d_{{out}}) = {r}({din}+{dout}) = {ans}$ vs ${din * dout}$ — a ${din * dout / ans:.0f}\\times$ reduction.",
            misconceptions=[(str(din * dout), "counted-full-matrix"), (str(r * din * dout), "multiplied-by-rank")],
        )
    if level == 2:
        Ln, d = rng.choice([(12, 768), (24, 1024), (32, 4096), (96, 12288)])
        r = rng.choice([1, 2, 4, 8, 16])
        cnt = Ln * 2 * 2 * d * r
        ratio = R(12 * Ln * d * d, cnt)
        return Problem(
            prompt=(f"A transformer has $n_{{layer}} = {Ln}$, $d = {d}$ ($\\approx 12d^2$ parameters per block). LoRA with rank $r = {r}$ is applied to "
                    "$W_Q$ and $W_V$ (both $d\\times d$) in every layer, as in the LoRA paper's GPT-3 experiments. Give (a) the number "
                    "of trainable parameters and (b) the ratio (block parameters)/(LoRA parameters)."),
            answer=[str(cnt), fmt(ratio)], kind="list", answer_hint="count, ratio",
            rubric=["Per adapted matrix: r(d + d) = 2dr", "Two matrices per layer", "Times n_layer",
                    "Ratio 12 n d^2 / (4 n d r) = 3d/r"],
            solution=f"(a) ${Ln}\\cdot2\\cdot2\\cdot{d}\\cdot{r} = {cnt}$. (b) $\\frac{{12d^2}}{{4dr}} = \\frac{{3d}}{{r}} = {L(ratio)}$.",
            misconceptions=[([str(cnt // 2), fmt(ratio * 2)], "adapted-one-matrix")],
        )
    r = rng.choice([1, 2])
    alpha = rng.choice([2, 4, 8])
    W0 = sp.Matrix(2, 2, lambda i, j: rng.randint(-3, 3))
    while True:
        B = sp.Matrix(2, r, lambda i, j: rng.randint(-2, 2))
        A = sp.Matrix(r, 2, lambda i, j: rng.randint(-2, 2))
        if B * A != sp.zeros(2, 2):
            break
    s = R(alpha, r)
    Wm = W0 + s * B * A
    return Problem(
        prompt=(f"LoRA forward: $\\mathbf h = W_0\\mathbf x + \\frac{{\\alpha}}{{r}}BA\\mathbf x$ with $r = {r}$, $\\alpha = {alpha}$, "
                f"$W_0 = {L(W0)}$, $B = {L(B)}$, $A = {L(A)}$. For deployment, LoRA merges the update into the weights so inference "
                "costs nothing extra. Give the merged matrix $W' = W_0 + \\frac{\\alpha}{r}BA$. (Why is $B$ initialised to zero?)"),
        answer=fmt(Wm), kind="matrix",
        rubric=["BA is a 2x2 matrix of rank <= r", f"Scale by alpha/r = {s}", "Add to W0",
                "B = 0 at init so the adapted model starts exactly at the pretrained one"],
        solution=(f"$BA = {L(B * A)}$ (rank $\\le {r}$), $\\tfrac{{\\alpha}}{{r}}BA = {L(s * B * A)}$, $W' = {L(Wm)}$. With $B = 0$ initially, "
                  "$W' = W_0$ at step 0, so fine-tuning starts from the pretrained function (arXiv:2106.09685, §4.1)."),
        misconceptions=[(fmt(W0 + B * A), "forgot-scaling")],
    )


@gen("dl.finetuning_lora", levels=(1, 2, 3))
def lora_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            r = rng.choice([1, 2])
            alpha = rng.choice([2, 4, 8])
            while True:
                A = sp.Matrix(r, 3, lambda i, j: rng.randint(-2, 2))
                Bm = sp.Matrix(2, r, lambda i, j: rng.randint(-2, 2))
                x = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
                if Bm * A * x != sp.zeros(2, 1):
                    break
            s = R(alpha, r)
            dh = s * Bm * (A * x)
            return Problem(
                prompt=(f"LoRA's side path: besides the frozen $W_0\\mathbf x$, a LoRA layer adds $\\Delta\\mathbf h = \\frac{{\\alpha}}{{r}}B(A\\mathbf x)$. "
                        f"With $r = {r}$, $\\alpha = {alpha}$, $A = {L(A)}$, $B = {L(Bm)}$ and $\\mathbf x = {tup(list(x))}$, compute $\\Delta\\mathbf h$. "
                        "(Why is computing $A\\mathbf x$ first cheaper than forming $BA$?)"),
                answer=[fmt(v) for v in dh], kind="vector",
                rubric=["A x: project down to r dimensions", "B (A x): project back up", f"Scale by alpha/r = {s}",
                        "Cost r(d_in + d_out) instead of d_in d_out"],
                solution=(f"$A\\mathbf x = {tup(list(A * x))}$, $B(A\\mathbf x) = {tup(list(Bm * A * x))}$, times ${L(s)}$: "
                          f"$\\Delta\\mathbf h = {tup(list(dh))}$."),
                misconceptions=[([fmt(v) for v in Bm * A * x], "forgot-scaling")],
            )
        N = rng.choice([7e9, 13e9, 33e9, 65e9, 70e9])
        q, f = N * 0.5 / 1e9, N * 2 / 1e9
        return Problem(
            prompt=(f"QLoRA (Dettmers et al. 2023) stores the frozen base model in 4-bit NF4. Ignoring quantisation constants, how many GB "
                    f"($10^9$ bytes) do the base weights of an $N = {sci(N)}$-parameter model take in 4-bit, and in 16-bit?"),
            answer=[repr(q), repr(f)], kind="list", tol=TOL, answer_hint="4-bit GB, 16-bit GB",
            rubric=["4 bits = 0.5 byte per parameter", "16 bits = 2 bytes per parameter", "Divide by 10^9"],
            solution=f"4-bit: ${sci(N)}\\cdot0.5 = {sig(q)}$ GB; 16-bit: ${sig(f)}$ GB — a $4\\times$ saving (arXiv:2305.14314).",
            misconceptions=[([repr(N * 4 / 1e9), repr(N * 16 / 1e9)], "bits-as-bytes")],
        )
    if level == 2:
        if rng.random() < 0.5:
            Ln, d, N = rng.choice([(32, 4096, 6.74e9), (40, 5120, 13.0e9), (80, 8192, 70e9)])
            r = rng.choice([8, 16, 64])
            P = Ln * 2 * 2 * d * r
            full = 16 * N / 1e9
            lo = (2 * N + 16 * P) / 1e9
            return Problem(
                prompt=("Why LoRA fits on one GPU: full fine-tuning with mixed-precision Adam costs about 16 bytes per parameter (weights, "
                        "gradients, fp32 optimizer state). With LoRA, the frozen fp16 base costs 2 bytes per parameter and only the adapter "
                        f"parameters cost 16 bytes each. For a model with $n_{{layer}} = {Ln}$, $d = {d}$, $N = {sci(N)}$, and LoRA rank $r = {r}$ "
                        "on $W_Q$ and $W_V$ of every layer, give the model-state memory in GB ($10^9$ B) for (a) full fine-tuning, (b) LoRA."),
                answer=[repr(full), repr(lo)], kind="list", tol=TOL, answer_hint="full GB, LoRA GB",
                rubric=["Full: 16 N bytes", "LoRA params: n_layer · 2 matrices · r(d + d)", "LoRA: 2 N + 16 · (LoRA params) bytes",
                        "Activations are ignored here"],
                solution=(f"(a) $16\\cdot{sci(N)} = {sig(full)}$ GB. (b) LoRA has ${Ln}\\cdot2\\cdot2\\cdot{d}\\cdot{r} = {P}$ parameters, so "
                          f"$2N + 16\\cdot{P} \\approx {sig(lo)}$ GB."),
                misconceptions=[([repr(full), repr(16 * P / 1e9)], "forgot-frozen-weights")],
            )
        p = rng.choice([10, 20, 50, 100])
        Ln, d = rng.choice([(12, 768), (24, 1024), (32, 4096), (48, 1600)])
        return Problem(
            prompt=(f"Prompt tuning vs. prefix tuning: prompt tuning (Lester et al. 2021) learns ${p}$ soft-prompt embeddings at the input only; "
                    f"prefix tuning (Li & Liang 2021) learns ${p}$ key **and** ${p}$ value vectors in every layer. For a model with "
                    f"$n_{{layer}} = {Ln}$ and $d = {d}$, count the trainable parameters of each."),
            answer=[str(p * d), str(2 * Ln * p * d)], kind="list", answer_hint="prompt tuning, prefix tuning",
            rubric=["Prompt tuning: p · d", "Prefix tuning: per layer p keys + p values of size d", "Times n_layer"],
            solution=f"Prompt tuning: ${p}\\cdot{d} = {p * d}$. Prefix tuning: $2\\cdot{Ln}\\cdot{p}\\cdot{d} = {2 * Ln * p * d}$.",
            misconceptions=[([str(p * d), str(Ln * p * d)], "forgot-keys-and-values")],
        )
    if rng.random() < 0.5:
        r = rng.choice([1, 2])
        while True:
            Bm = sp.Matrix(3, r, lambda i, j: rng.randint(-2, 2))
            A = sp.Matrix(r, 3, lambda i, j: rng.randint(-2, 2))
            dW = Bm * A
            if dW.rank() == r:
                break
        return Problem(
            prompt=(f"Intrinsic rank of an update: after full fine-tuning, a $3\\times3$ weight changed by $\\Delta W = {L(dW)}$. What is the "
                    "smallest LoRA rank $r$ for which $\\Delta W = BA$ exactly ($B$ is $3\\times r$, $A$ is $r\\times3$)? Derive it by row "
                    "reduction and, on paper, write down one valid factorisation."),
            answer=str(r), kind="num",
            rubric=["rank(BA) <= r, and every rank-r matrix factors as BA", "Row-reduce Delta W", "Count the non-zero rows (pivots)",
                    "Factorise: B = independent columns, A = coefficients"],
            solution=(f"Row reduction leaves ${r}$ non-zero row(s): $\\mathrm{{rank}}(\\Delta W) = {r}$, so $r = {r}$ suffices, e.g. "
                      f"$B = {L(Bm)}$, $A = {L(A)}$. LoRA's bet is that fine-tuning updates have low intrinsic rank (arXiv:2106.09685, §7)."),
            misconceptions=[("3", "assumed-full-rank")],
        )
    alpha = rng.choice([2, 4, 8, 16])
    r = 2
    while True:
        A = sp.Matrix(r, 3, lambda i, j: rng.randint(-2, 2))
        x = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
        g = sp.Matrix([nz(rng, -2, 2) for _ in range(2)])
        if A * x != sp.zeros(r, 1):
            break
    s = R(alpha, r)
    dB = s * g * (A * x).T
    return Problem(
        prompt=(f"LoRA's first gradient step: $\\mathbf h = W_0\\mathbf x + \\frac{{\\alpha}}{{r}}BA\\mathbf x$ with $B = 0$ at initialisation, "
                f"$r = {r}$, $\\alpha = {alpha}$, $A = {L(A)}$, $\\mathbf x = {tup(list(x))}$ and upstream gradient "
                f"$\\mathbf g = \\partial\\mathcal L/\\partial\\mathbf h = {tup(list(g))}$. Derive $\\partial\\mathcal L/\\partial B$ and "
                "$\\partial\\mathcal L/\\partial A$ at initialisation, and give the $2\\times2$ matrix $\\partial\\mathcal L/\\partial B$."),
        answer=fmt(dB), kind="matrix", answer_hint="a 2x2 matrix",
        rubric=["Let u = A x: h = W0 x + (alpha/r) B u", "dL/dB = (alpha/r) g u^T (outer product)",
                "dL/dA = (alpha/r) B^T g x^T = 0 because B = 0", "So only B moves on the first step; A starts moving afterwards"],
        solution=(f"$\\mathbf u = A\\mathbf x = {tup(list(A * x))}$; $\\partial\\mathcal L/\\partial B = \\tfrac{{\\alpha}}{{r}}\\mathbf g\\mathbf u^\\top = {L(dB)}$; "
                  "$\\partial\\mathcal L/\\partial A = \\tfrac{\\alpha}{r}B^\\top\\mathbf g\\mathbf x^\\top = 0$ since $B = 0$."),
        misconceptions=[(fmt(g * (A * x).T), "forgot-scaling"), (fmt((s * g * (A * x).T).T), "transposed-outer-product")],
    )


# =========================================================================== dl.alignment
@gen("dl.alignment", levels=(1, 2, 3))
def alignment(rng, level):
    if level == 1:
        while True:
            r1, r2 = R(rng.randint(1, 30), 10), R(rng.randint(1, 30), 10)
            if r1 != r2:
                break
        ans = sigmoid(r1 - r2)
        return Problem(
            prompt=(f"A reward model scores two responses $r_1 = {dec(r1)}$ and $r_2 = {dec(r2)}$. Under the Bradley–Terry model "
                    f"$P(y_1 \\succ y_2) = \\sigma(r_1 - r_2)$, what is the probability that a labeller prefers response 1? {SIG4}"),
            answer=nfmt(ans), kind="num", tol=TOL,
            rubric=["Difference r1 - r2", "Apply the logistic sigmoid", "Only reward differences matter (shift invariance)"],
            solution=f"$\\sigma({dec(r1 - r2)}) = 1/(1 + e^{{{dec(r2 - r1)}}}) \\approx {sig(ans)}$.",
            misconceptions=[(fmt(r1 / (r1 + r2)), "ratio-not-logistic"), (nfmt(sigmoid(r2 - r1)), "sign-flip")],
        )
    if level == 2:
        pairs = []
        for _ in range(2):
            while True:
                w, l = R(rng.randint(-20, 20), 10), R(rng.randint(-20, 20), 10)
                if w != l:
                    break
            pairs.append((w, l))
        losses = [-sp.log(sigmoid(w - l)) for w, l in pairs]
        ans = (losses[0] + losses[1]) / 2
        desc = "; ".join(f"pair {i + 1}: $r(y_w) = {dec(w)}$, $r(y_l) = {dec(l)}$" for i, (w, l) in enumerate(pairs))
        return Problem(
            prompt=(f"InstructGPT-style reward-model loss $\\mathcal L = -\\mathbb E\\,[\\log\\sigma(r(y_w) - r(y_l))]$ "
                    f"($y_w$ = preferred). For a batch of two comparisons ({desc}) compute the mean loss (natural log). {SIG4}"),
            answer=nfmt(ans), kind="num", tol=TOL,
            rubric=["Per pair: margin r_w - r_l", "-log sigma(margin) = log(1 + e^{-margin})", "Average over pairs",
                    "A mis-ranked pair (negative margin) dominates the loss"],
            solution=(", ".join(f"pair {i + 1}: $\\log(1+e^{{{dec(l - w)}}}) \\approx {sig(losses[i])}$" for i, (w, l) in enumerate(pairs))
                      + f"; mean $\\approx {sig(ans)}$ (arXiv:2203.02155, eq. 1)."),
            misconceptions=[(nfmt(-ans), "sign-error"), (nfmt(losses[0] + losses[1]), "summed-not-averaged")],
        )
    beta = rng.choice([R(1, 10), R(1, 5), R(1, 2)])
    while True:
        rw, rl = R(-rng.randint(100, 400), 10), R(-rng.randint(100, 400), 10)
        dw, dl = R(rng.randint(-30, 30), 10), R(rng.randint(-30, 30), 10)
        if dw != dl:
            break
    pw, pl = rw + dw, rl + dl
    h = beta * (dw - dl)
    loss = -sp.log(sigmoid(h))
    return Problem(
        prompt=(f"DPO (Rafailov et al. 2023) loss for one preference pair: "
                "$\\mathcal L = -\\log\\sigma\\!\\left(\\beta\\left[\\log\\frac{\\pi_\\theta(y_w)}{\\pi_{ref}(y_w)} - \\log\\frac{\\pi_\\theta(y_l)}{\\pi_{ref}(y_l)}\\right]\\right)$. "
                f"With $\\beta = {dec(beta)}$, $\\log\\pi_\\theta(y_w) = {dec(pw)}$, $\\log\\pi_{{ref}}(y_w) = {dec(rw)}$, "
                f"$\\log\\pi_\\theta(y_l) = {dec(pl)}$, $\\log\\pi_{{ref}}(y_l) = {dec(rl)}$: give (a) the implicit reward margin "
                f"$h$ (the argument of $\\sigma$) and (b) the loss. {SIG4}"),
        answer=[fmt(h), nfmt(loss)], kind="list", tol=TOL, answer_hint="h, loss   e.g. 0.3, 0.5544",
        rubric=["Log-ratio for y_w: log pi_theta - log pi_ref", "Log-ratio for y_l", "h = beta (difference of log-ratios)",
                "Loss = log(1 + e^{-h})"],
        solution=(f"Log-ratios: $y_w$: ${dec(dw)}$, $y_l$: ${dec(dl)}$. $h = {L(beta)}\\cdot({dec(dw - dl)}) = {sig(h)}$, "
                  f"$\\mathcal L = \\log(1 + e^{{-h}}) \\approx {sig(loss)}$. The implicit reward is $\\beta\\log(\\pi_\\theta/\\pi_{{ref}})$ — "
                  "no separate reward model or RL loop is needed (arXiv:2305.18290)."),
        misconceptions=[([fmt(dw - dl), nfmt(-sp.log(sigmoid(dw - dl)))], "forgot-beta"),
                        ([fmt(beta * (pw - pl)), nfmt(-sp.log(sigmoid(beta * (pw - pl))))], "forgot-reference-model")],
    )


@gen("dl.alignment", levels=(1, 2, 3))
def alignment_more(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            rm = R(rng.randint(-20, 30), 10)
            beta = rng.choice([R(1, 10), R(1, 20), R(1, 5)])
            lpr = R(-rng.randint(50, 300), 10)
            lp = lpr + R(rng.randint(-40, 60), 10)
            if lp == lpr:
                lp += 1
            ans = rm - beta * (lp - lpr)
            return Problem(
                prompt=("The KL-penalised RLHF reward: the policy is optimised for $r_{RM}(x,y) - \\beta\\,[\\log\\pi_\\theta(y\\mid x) - \\log\\pi_{ref}(y\\mid x)]$. "
                        f"A sampled response has reward-model score ${dec(rm)}$, $\\log\\pi_\\theta = {dec(lp)}$, $\\log\\pi_{{ref}} = {dec(lpr)}$, "
                        f"and $\\beta = {float(beta):g}$. Compute the penalised reward."),
                answer=fmt(ans), kind="num",
                rubric=["Log-ratio log pi_theta - log pi_ref (a one-sample KL estimate)", "Multiply by beta", "Subtract from the RM score"],
                solution=(f"Log-ratio $= {dec(lp)} - ({dec(lpr)}) = {dec(lp - lpr)}$; penalty $= {float(beta):g}\\cdot({dec(lp - lpr)}) = {float(beta * (lp - lpr)):g}$; "
                          f"reward $= {dec(rm)} - ({float(beta * (lp - lpr)):g}) = {float(ans):g}$. A positive log-ratio (drifting towards "
                          "outputs the reference finds unlikely) costs reward; a negative one adds a bonus."),
                misconceptions=[(fmt(rm + beta * (lp - lpr)), "penalty-sign-flip"), (fmt(rm - (lp - lpr)), "forgot-beta")],
            )
        p = rng.choice([R(2, 3), R(3, 4), R(4, 5), R(1, 3), R(1, 4), R(9, 10), R(1, 5)])
        ans = sp.log(p / (1 - p))
        return Problem(
            prompt=(f"Inverting Bradley–Terry: labellers prefer response 1 over response 2 with probability ${L(p)}$. Under "
                    "$P(y_1\\succ y_2) = \\sigma(r_1 - r_2)$, what reward difference $r_1 - r_2$ does this imply? "
                    "Exact (e.g. log(3)) or at least 4 significant figures."),
            answer=fmt(ans), kind="num", tol=TOL,
            rubric=["Invert the sigmoid: sigma^{-1}(p) = log(p/(1 - p)) (the logit)", "Plug in p"],
            solution=f"$r_1 - r_2 = \\log\\frac{{p}}{{1-p}} = \\log {L(p / (1 - p))} \\approx {sig(ans)}$.",
            misconceptions=[(fmt(sp.log(p)), "forgot-odds"), (fmt(-ans), "sign-flip")],
        )
    if level == 2:
        if rng.random() < 0.5:
            while True:
                rho = [R(rng.randint(5, 15), 10) for _ in range(3)]
                Adv = [nz(rng, -3, 3) for _ in range(3)]
                clip = [min(max(r_, R(8, 10)), R(12, 10)) for r_ in rho]
                terms = [min(r_ * a, c * a) for r_, c, a in zip(rho, clip, Adv)]
                if any(r_ * a != t for r_, a, t in zip(rho, Adv, terms)):
                    break
            ans = sum(terms) / 3
            unclipped = sum(r_ * a for r_, a in zip(rho, Adv)) / 3
            cliponly = sum(c * a for c, a in zip(clip, Adv)) / 3
            samples = ", ".join(f"$({dec(r_)}, {a})$" for r_, a in zip(rho, Adv))
            mis = [(fmt(unclipped), "ignored-clipping")]
            if cliponly != ans:
                mis.append((fmt(cliponly), "forgot-min"))
            return Problem(
                prompt=("PPO's clipped surrogate (Schulman et al. 2017) per sample is $\\min(\\rho A,\\ \\mathrm{clip}(\\rho, 1-\\varepsilon, 1+\\varepsilon)A)$ "
                        f"with $\\rho = \\pi_\\theta/\\pi_{{old}}$. With $\\varepsilon = 0.2$ and three samples $(\\rho, A)$ = {samples}, compute the "
                        "mean objective (exact)."),
                answer=fmt(ans), kind="num",
                rubric=["Clip rho into [0.8, 1.2]", "Per sample take the MIN of the unclipped and clipped terms (pessimistic bound)",
                        "Average over samples"],
                solution=("Per sample: " + ", ".join(f"$\\min({float(r_ * a):g}, {float(c * a):g}) = {float(t):g}$" for r_, a, c, t in zip(rho, Adv, clip, terms))
                          + f"; mean $= {L(ans)}$ (arXiv:1707.06347)."),
                misconceptions=mis,
            )
        q = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(1, 5), R(1, 10)])
        n = rng.choice([2, 3, 4, 5, 8])
        ans = 1 - (1 - q) ** n
        mis = [(fmt(q ** n), "all-must-succeed")]
        if n * q < 1:
            mis.append((fmt(n * q), "added-probabilities"))
        return Problem(
            prompt=(f"Best-of-$n$ sampling: each sample independently passes a reward threshold with probability ${L(q)}$. You draw "
                    f"$n = {n}$ samples and keep the one with the highest reward. What is the probability that the kept sample passes? (exact)"),
            answer=fmt(ans), kind="num",
            rubric=["The kept sample passes iff at least one sample passes", "Complement: all n fail with probability (1 - q)^n",
                    "Answer 1 - (1 - q)^n"],
            solution=f"$1 - (1 - {L(q)})^{{{n}}} = 1 - {L((1 - q) ** n)} = {L(ans)}$.",
            misconceptions=mis,
        )
    if rng.random() < 0.5:
        while True:
            pref = list(rng.choice(_PTRIPLES))
            rng.shuffle(pref)
            a = [rng.randint(1, 4) for _ in range(3)]
            if len(set(a)) > 1:
                break
        w = [pr * ai for pr, ai in zip(pref, a)]
        pi = [v / sum(w) for v in w]
        rw = ",\\ ".join(f"\\beta\\ln {ai}" if ai != 1 else "0" for ai in a)
        return Problem(
            prompt=("The closed-form optimum behind DPO (Rafailov et al. 2023, eq. 4): the KL-regularised objective "
                    "$\\max_\\pi \\mathbb E_{y\\sim\\pi}[r(y)] - \\beta\\,\\mathrm{KL}(\\pi\\,\\|\\,\\pi_{ref})$ is solved by "
                    "$\\pi^*(y) \\propto \\pi_{ref}(y)\\,e^{r(y)/\\beta}$. Derive this, then for three responses with "
                    f"$\\pi_{{ref}} = {tup(pref)}$ and rewards $r = ({rw})$ give $\\pi^*$ exactly."),
            answer=[fmt(v) for v in pi], kind="vector",
            rubric=["Rewrite the objective as -beta KL(pi || pi_ref e^{r/beta}/Z) + beta log Z", "KL is minimised (= 0) at pi* = pi_ref e^{r/beta}/Z",
                    "e^{r/beta} = a_i here", "Normalise"],
            solution=(f"$\\pi^*\\propto\\pi_{{ref}}\\cdot e^{{r/\\beta}} = {tup(w)}$, $Z = {L(sum(w))}$, so $\\pi^* = {tup(pi)}$. Inverting gives "
                      "$r = \\beta\\log(\\pi^*/\\pi_{ref}) + \\beta\\log Z$ — DPO's implicit reward (arXiv:2305.18290)."),
            misconceptions=[([fmt(R(ai, sum(a))) for ai in a], "ignored-reference"), ([fmt(v) for v in pref], "ignored-reward")],
        )
    n = rng.choice([2, 4, 8, 16, 64, 100])
    kl = sp.log(n) - R(n - 1, n)
    return Problem(
        prompt=("The KL cost of best-of-$n$ (Stiennon et al. 2020; Gao et al. 2022): with continuous rewards, the best-of-$n$ policy has "
                "density $n F(r)^{n-1}$ relative to the base policy, where $F$ is the reward CDF under the base policy. Derive "
                "$\\mathrm{KL}(\\pi_{BoN}\\|\\pi) = \\log n - \\frac{n-1}{n}$ (hint: $U = F(r)$ of the best sample has density $nu^{n-1}$ on $[0,1]$) "
                f"and evaluate it for $n = {n}$ in nats. {SIG4}"),
        answer=nfmt(kl), kind="num", tol=TOL,
        rubric=["KL = E_BoN[log(n F^{n-1})] = log n + (n-1) E[log U]", "E[log U] = int_0^1 n u^{n-1} log u du = -1/n",
                "KL = log n - (n-1)/n", "Grows only logarithmically in n"],
        solution=(f"$\\mathbb E[\\log U] = \\int_0^1 nu^{{n-1}}\\log u\\,du = -1/n$, so $\\mathrm{{KL}} = \\log n - (n-1)/n = \\log {n} - {L(R(n - 1, n))} \\approx {sig(kl)}$ nats "
                  "(arXiv:2009.01325, arXiv:2210.10760)."),
        misconceptions=[(nfmt(sp.log(n)), "forgot-correction-term"), (nfmt(sp.log(n, 2) - R(n - 1, n)), "used-log2")],
    )
