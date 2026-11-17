"""Track 5 — Information theory & data.

Conventions (see engine/content.py and t00_foundations.py):
  * Each generator: (rng, level) -> Problem, deterministic given rng.
  * Levels: 1 = mechanical, 2 = multi-step, 3 = derivation / reasoning with a checkable final result.
  * Answers are computed with sympy (exact where possible). Information quantities involving
    non-dyadic probabilities are accepted either exactly (log(3,2), log(3)/log(2), ...) or as
    4-decimal numbers via `tol`.
"""
from __future__ import annotations

import heapq
import itertools
import math
import random

import sympy as sp

from engine.checker import fmt
from engine.content import Problem, gen

L = sp.latex
R = sp.Rational
LN2 = sp.log(2)

TOL = 1e-3
BITS_HINT = ("bits: exact (e.g. 3/2, or 2 - 3*log(3)/(4*log(2)); type log2(x) as log(x,2)) "
             "or a decimal to 4 places")
NATS_HINT = "nats: exact natural-log form (e.g. log(5/2)) or a decimal to 4 places"


# --------------------------------------------------------------------------- helpers
def tidy(e) -> sp.Expr:
    """Expand logs so dyadic quantities collapse to rationals."""
    return sp.expand(sp.expand_log(sp.sympify(e), force=True))


def lg(p) -> sp.Expr:
    """Exact log base 2."""
    return tidy(sp.log(p) / LN2)


def H(ps) -> sp.Expr:
    """Shannon entropy in bits (exact)."""
    return tidy(sum(-p * sp.log(p) for p in ps if p != 0) / LN2)


def Hnat(ps) -> sp.Expr:
    return tidy(sum(-p * sp.log(p) for p in ps if p != 0))


def Hb(p) -> sp.Expr:
    return H([p, 1 - p])


def d4(e) -> str:
    return f"{float(sp.N(e)):.4f}"


def tup(ps) -> str:
    return "\\left(" + ",\\ ".join(L(p) for p in ps) + "\\right)"


def dyadic(rng: random.Random, k: int) -> list:
    """Random dyadic distribution (every p = 2^-l) on k outcomes."""
    ps = [sp.Integer(1)]
    while len(ps) < k:
        p = ps.pop(rng.randrange(len(ps)))
        ps += [p / 2, p / 2]
    rng.shuffle(ps)
    return ps


def count_dist(rng: random.Random, k: int, lo: int = 1, hi: int = 6):
    """Non-uniform distribution from random positive counts."""
    while True:
        c = [rng.randint(lo, hi) for _ in range(k)]
        if len(set(c)) > 1:
            break
    N = sum(c)
    return [R(v, N) for v in c], c, N


def joint_tex(P) -> str:
    cols = len(P[0])
    head = " & " + " & ".join(f"Y={j}" for j in range(cols))
    body = " \\\\ ".join(f"X={i} & " + " & ".join(L(p) for p in row) for i, row in enumerate(P))
    return "$$\\begin{array}{c|" + "c" * cols + "}" + head + " \\\\ \\hline " + body + "\\end{array}$$"


def huffman(ps):
    """Return (expected length, codeword lengths) of a binary Huffman code for ps."""
    cnt = itertools.count()
    heap = [(p, next(cnt), [i]) for i, p in enumerate(ps)]
    heapq.heapify(heap)
    lengths = [0] * len(ps)
    while len(heap) > 1:
        a, _, sa = heapq.heappop(heap)
        b, _, sb = heapq.heappop(heap)
        for i in sa + sb:
            lengths[i] += 1
        heapq.heappush(heap, (a + b, next(cnt), sa + sb))
    return sum(p * l for p, l in zip(ps, lengths)), lengths


SYMS = "abcdefgh"


# --------------------------------------------------------------------------- entropy
@gen("info.entropy")
def entropy_basic(rng, level):
    if level == 1:
        k = rng.randint(3, 5)
        ps = dyadic(rng, k)
        h = H(ps)
        terms = " + ".join(f"{L(p)}\\cdot{L(-lg(p))}" for p in ps)
        return Problem(
            prompt=(f"A source emits symbols ${', '.join(SYMS[:k])}$ with probabilities ${tup(ps)}$. "
                    "Compute its Shannon entropy $H(X)$ in **bits**."),
            answer=fmt(h), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Write H(X) = -Σ p log₂ p", "Use log₂(1/2^l) = -l for each term", "Add the terms"],
            solution=f"$H = \\sum_x p(x)\\log_2\\frac{{1}}{{p(x)}} = {terms} = {L(h)}$ bits.",
            misconceptions=[(fmt(h * LN2), "used-natural-log"), (fmt(-h), "entropy-sign-error"),
                            (fmt(lg(k)), "assumed-uniform")],
        )
    if level == 2:
        k = rng.choice([3, 4])
        ps, c, N = count_dist(rng, k)
        h = H(ps)
        return Problem(
            prompt=(f"A bag holds {N} tokens: " + ", ".join(f"{c[i]} labelled ${SYMS[i]}$" for i in range(k)) +
                    ". One token is drawn uniformly at random; $X$ is its label. Compute $H(X)$ in bits."),
            answer=fmt(h), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Convert counts to probabilities", "Apply H = Σ p log₂(1/p)", "Simplify / evaluate"],
            solution=(f"$p = {tup(ps)}$, so $H = " +
                      " + ".join(f"{L(p)}\\log_2 {L(1 / p)}" for p in ps) + f" = {L(h)} \\approx {d4(h)}$ bits."),
            misconceptions=[(fmt(h * LN2), "used-natural-log"), (fmt(lg(k)), "assumed-uniform")],
        )
    if rng.random() < 0.5:  # geometric source, infinite support
        p = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(2, 3)])
        h = tidy((-p * sp.log(p) - (1 - p) * sp.log(1 - p)) / (p * LN2))
        return Problem(
            prompt=(f"A coin with $P(\\text{{head}}) = p = {L(p)}$ is tossed until the first head; $X$ is the number "
                    f"of tosses (including the head), i.e. $P(X=k) = (1-p)^{{k-1}}p$ for $k \\ge 1$. "
                    "Derive a closed form for $H(X)$ in bits and evaluate it."),
            answer=fmt(h), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Write -log₂P(X=k) = -log₂p - (k-1)log₂(1-p)", "Take expectation: E[X-1] = (1-p)/p",
                    "Simplify to H_b(p)/p", "Evaluate"],
            solution=(f"$H = -\\log_2 p - \\frac{{1-p}}{{p}}\\log_2(1-p) = \\frac{{H_b(p)}}{{p}} = {L(h)} "
                      f"\\approx {d4(h)}$ bits."),
            misconceptions=[(fmt(Hb(p)), "forgot-divide-by-p")],
        )
    s = rng.choice([3, 4])
    counts = [min(t, 2 * s - t) for t in range(1, 2 * s)]
    ps = [R(cc, s * s) for cc in counts]
    h = H(ps)
    return Problem(
        prompt=(f"Two fair {s}-sided dice (faces $1..{s}$) are rolled and $S$ is their sum. "
                "Find the distribution of $S$ and compute $H(S)$ in bits. "
                f"(On paper: compare with $H(\\text{{both dice}}) = 2\\log_2 {s}$.)"),
        answer=fmt(h), kind="num", tol=TOL, answer_hint=BITS_HINT,
        rubric=["Count the ways to get each sum", f"Probabilities are counts/{s*s}",
                "H = log₂(N) - (1/N) Σ c log₂ c", "Evaluate"],
        solution=(f"Counts for $S={2}..{2*s}$: ${counts}$ out of ${s*s}$. "
                  f"$H(S) = {L(h)} \\approx {d4(h)}$ bits $< 2\\log_2 {s}$ because the sum loses information."),
        misconceptions=[(fmt(2 * lg(s)), "entropy-of-pair-not-sum"), (fmt(lg(2 * s - 1)), "assumed-uniform")],
    )


@gen("info.entropy")
def binary_entropy(rng, level):
    if level == 1:
        p = rng.choice([R(1, 3), R(1, 4), R(1, 5), R(1, 8), R(3, 8), R(1, 10), R(2, 5), R(3, 4)])
        h = Hb(p)
        return Problem(
            prompt=f"A biased coin lands heads with probability $p = {L(p)}$. Compute its binary entropy $H_b(p)$ in bits.",
            answer=fmt(h), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["H_b(p) = -p log₂ p - (1-p) log₂(1-p)", "Evaluate both terms", "Add"],
            solution=f"$H_b({L(p)}) = {L(p)}\\log_2{L(1/p)} + {L(1-p)}\\log_2{L(1/(1-p))} = {L(h)} \\approx {d4(h)}$ bits.",
            misconceptions=[(fmt(tidy(-p * lg(p))), "dropped-complement-term"), (fmt(h * LN2), "used-natural-log")],
        )
    if level == 2:
        p = rng.choice([R(1, 4), R(1, 8), R(1, 3), R(1, 5)])
        n = rng.randint(3, 12)
        h = tidy(n * Hb(p))
        return Problem(
            prompt=(f"A sensor emits {n} independent bits, each equal to 1 with probability ${L(p)}$. "
                    f"What is the entropy (in bits) of the whole {n}-bit string? "
                    "(On paper: argue why the *number* of ones has strictly smaller entropy.)"),
            answer=fmt(h), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Independence: joint entropy is a sum", "Compute H_b(p)", f"Multiply by {n}"],
            solution=f"$H(X_1..X_{{{n}}}) = {n}H_b({L(p)}) = {L(h)} \\approx {d4(h)}$ bits.",
            misconceptions=[(fmt(Hb(p)), "forgot-multiply-by-n"), (fmt(n), "assumed-fair-bits")],
        )
    ps, c, N = count_dist(rng, 4)
    i, j = sorted(rng.sample(range(4), 2))
    q = ps[i] + ps[j]
    drop = tidy(q * Hb(ps[i] / q))
    merged = [p for t, p in enumerate(ps) if t not in (i, j)] + [q]
    return Problem(
        prompt=(f"$X$ has distribution ${tup(ps)}$ on $\\{{a,b,c,d\\}}$. We merge outcomes ${SYMS[i]}$ and ${SYMS[j]}$ "
                "into a single outcome. Using the grouping (chain) rule, compute how many bits of entropy are lost, "
                "$H(X) - H(X_{\\text{merged}})$."),
        answer=fmt(drop), kind="num", tol=TOL, answer_hint=BITS_HINT,
        rubric=["Grouping rule: H(X) = H(merged) + q·H(within-group)", f"q = p({SYMS[i]}) + p({SYMS[j]})",
                "Within-group distribution is (p_i/q, p_j/q)", "Loss = q·H_b(p_i/q)"],
        solution=(f"$q = {L(q)}$, loss $= q\\,H_b\\!\\left({L(ps[i]/q)}\\right) = {L(drop)} \\approx {d4(drop)}$ bits "
                  f"(check: $H(X) \\approx {d4(H(ps))}$, $H(X_{{\\text{{merged}}}}) \\approx {d4(H(merged))}$)."),
        misconceptions=[(fmt(H(merged)), "gave-merged-entropy"), (fmt(Hb(ps[i] / q)), "forgot-weight-q")],
    )


# --------------------------------------------------------------------------- joint & conditional
def _table(rng, rows, cols, hi=5):
    while True:
        c = [[rng.randint(1, hi) for _ in range(cols)] for _ in range(rows)]
        N = sum(map(sum, c))
        P = [[R(v, N) for v in row] for row in c]
        px = [sum(row) for row in P]
        py = [sum(P[i][j] for i in range(rows)) for j in range(cols)]
        if any(P[i][j] != px[i] * py[j] for i in range(rows) for j in range(cols)):
            return P, px, py


@gen("info.joint_conditional")
def joint_conditional(rng, level):
    if level == 1:
        P, px, py = _table(rng, 2, 2)
        hxy = H([p for row in P for p in row])
        return Problem(
            prompt="The joint distribution of $(X, Y)$ is\n" + joint_tex(P) + "\nCompute the joint entropy $H(X,Y)$ in bits.",
            answer=fmt(hxy), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Treat the 4 cells as one distribution", "Apply H = Σ p log₂(1/p)", "Evaluate"],
            solution=f"$H(X,Y) = \\sum_{{x,y}} p(x,y)\\log_2\\frac{{1}}{{p(x,y)}} = {L(hxy)} \\approx {d4(hxy)}$ bits.",
            misconceptions=[(fmt(tidy(H(px) + H(py))), "assumed-independence"), (fmt(hxy * LN2), "used-natural-log")],
        )
    if level == 2:
        P, px, py = _table(rng, 2, 3)
        hxy = H([p for row in P for p in row])
        hyx = tidy(hxy - H(px))
        return Problem(
            prompt="The joint distribution of $(X, Y)$ is\n" + joint_tex(P) + "\nCompute the conditional entropy $H(Y\\mid X)$ in bits.",
            answer=fmt(hyx), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Marginal p(x) by summing rows", "Either H(Y|X) = Σ p(x) H(Y|X=x) or chain rule H(X,Y) - H(X)",
                    "Evaluate"],
            solution=(f"$p(x) = {tup(px)}$, $H(X) \\approx {d4(H(px))}$, $H(X,Y) \\approx {d4(hxy)}$; "
                      f"$H(Y\\mid X) = H(X,Y) - H(X) = {L(hyx)} \\approx {d4(hyx)}$ bits."),
            misconceptions=[(fmt(tidy(hxy - H(py))), "conditioned-wrong-way"), (fmt(H(py)), "ignored-conditioning")],
        )
    p = rng.choice([R(1, 8), R(1, 5), R(1, 4), R(1, 3)])
    q = rng.choice([R(1, 4), R(1, 3), R(2, 5), R(3, 4)])
    pq = q * (1 - p) + (1 - q) * p
    hxgy = tidy(Hb(q) + Hb(p) - Hb(pq))
    return Problem(
        prompt=(f"$X \\sim \\text{{Bernoulli}}({L(q)})$ is sent through a noisy wire: $Y = X \\oplus N$ with "
                f"$N \\sim \\text{{Bernoulli}}({L(p)})$ independent of $X$. Compute $H(X\\mid Y)$ in bits "
                "(how uncertain the receiver remains about $X$)."),
        answer=fmt(hxgy), kind="num", tol=TOL, answer_hint=BITS_HINT,
        rubric=["H(Y|X) = H(N) = H_b(p)", "P(Y=1) = q(1-p) + (1-q)p", "Chain rule: H(X|Y) = H(X) + H(Y|X) - H(Y)",
                "Evaluate"],
        solution=(f"$P(Y=1) = {L(pq)}$. $H(X\\mid Y) = H_b({L(q)}) + H_b({L(p)}) - H_b({L(pq)}) "
                  f"= {L(hxgy)} \\approx {d4(hxgy)}$ bits."),
        misconceptions=[(fmt(Hb(p)), "confused-HXgivenY-with-HYgivenX"), (fmt(Hb(q)), "ignored-conditioning")],
    )


# --------------------------------------------------------------------------- KL divergence
def kl(P, Q):
    return tidy(sum(p * (sp.log(p) - sp.log(q)) for p, q in zip(P, Q) if p != 0) / LN2)


@gen("info.kl_divergence")
def kl_divergence(rng, level):
    if level == 1:
        k = rng.choice([3, 4])
        while True:
            P, Q = dyadic(rng, k), dyadic(rng, k)
            if P != Q:
                break
        d = kl(P, Q)
        return Problem(
            prompt=(f"$P = {tup(P)}$ and $Q = {tup(Q)}$ on the same {k} outcomes. "
                    "Compute $D_{\\mathrm{KL}}(P\\,\\|\\,Q)$ in bits."),
            answer=fmt(d), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["D(P||Q) = Σ p log₂(p/q) (weights are P)", "Each log₂(p/q) is an integer here", "Sum"],
            solution="$D = " + " + ".join(f"{L(p)}\\cdot({L(lg(p / q))})" for p, q in zip(P, Q)) + f" = {L(d)}$ bits.",
            misconceptions=[(fmt(kl(Q, P)), "kl-argument-order"), (fmt(-d), "kl-sign-error")],
        )
    if level == 2:
        while True:  # avoid near-zero answers (abs tol would accept 0)
            P, _, _ = count_dist(rng, 3)
            Q, _, _ = count_dist(rng, 3)
            d = kl(P, Q)
            if P != Q and float(d) > 0.02:
                break
        return Problem(
            prompt=(f"A model predicts $Q = {tup(Q)}$ while the data follow $P = {tup(P)}$. "
                    "How many extra bits per symbol does coding with $Q$ cost? Compute $D_{\\mathrm{KL}}(P\\,\\|\\,Q)$ in bits."),
            answer=fmt(d), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Weights come from the true distribution P", "Compute each p log₂(p/q)", "Sum (must be ≥ 0)"],
            solution=f"$D(P\\|Q) = {L(d)} \\approx {d4(d)}$ bits, while $D(Q\\|P) \\approx {d4(kl(Q, P))}$ — KL is not symmetric.",
            misconceptions=[(fmt(kl(Q, P)), "kl-argument-order"), (fmt(d * LN2), "used-natural-log")],
        )
    m1, m2 = rng.randint(-3, 3), rng.randint(-3, 3)
    s1, s2 = rng.choice([1, 2, 3]), rng.choice([1, 2, 3])
    if m1 == m2 and s1 == s2:
        m2 += 1
    d = sp.log(R(s2, s1)) + R(s1 ** 2 + (m1 - m2) ** 2, 2 * s2 ** 2) - R(1, 2)
    rev = sp.log(R(s1, s2)) + R(s2 ** 2 + (m1 - m2) ** 2, 2 * s1 ** 2) - R(1, 2)
    return Problem(
        prompt=(f"Derive $D_{{\\mathrm{{KL}}}}(\\mathcal N(\\mu_1,\\sigma_1^2)\\,\\|\\,\\mathcal N(\\mu_2,\\sigma_2^2))$ "
                f"in nats, then evaluate it for $\\mu_1={m1},\\ \\sigma_1={s1}$ and $\\mu_2={m2},\\ \\sigma_2={s2}$."),
        answer=fmt(d), kind="num", tol=TOL, answer_hint=NATS_HINT,
        rubric=["Write log p/q = log(σ₂/σ₁) - (x-μ₁)²/(2σ₁²) + (x-μ₂)²/(2σ₂²)",
                "Take E under p: E(x-μ₁)² = σ₁², E(x-μ₂)² = σ₁² + (μ₁-μ₂)²",
                "Result: ln(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2", "Substitute numbers"],
        solution=(f"$D = \\ln\\frac{{\\sigma_2}}{{\\sigma_1}} + \\frac{{\\sigma_1^2 + (\\mu_1-\\mu_2)^2}}{{2\\sigma_2^2}} - \\frac12 "
                  f"= {L(d)} \\approx {d4(d)}$ nats."),
        misconceptions=[(fmt(rev), "kl-argument-order"), (fmt(tidy(d / LN2)), "reported-bits-not-nats")],
    )


# --------------------------------------------------------------------------- cross-entropy
@gen("info.cross_entropy")
def cross_entropy_dists(rng, level):
    if level == 1:
        k = rng.choice([3, 4])
        q, _, _ = count_dist(rng, k, 1, 7)
        c = rng.randrange(k)
        loss = sp.log(1 / q[c])
        amax = max(range(k), key=lambda t: q[t])
        return Problem(
            prompt=(f"A {k}-class classifier outputs probabilities $\\hat q = {tup(q)}$ for classes $1..{k}$. "
                    f"The true class is **{c + 1}**. Compute the cross-entropy (log) loss for this example in **nats**."),
            answer=fmt(loss), kind="num", tol=TOL, answer_hint=NATS_HINT,
            rubric=["One-hot target: only the true class term survives", "Loss = -ln q̂(true class)", "Evaluate"],
            solution=f"$\\ell = -\\ln {L(q[c])} = {L(loss)} \\approx {d4(loss)}$ nats.",
            misconceptions=[(fmt(tidy(loss / LN2)), "used-log2"), (fmt(sp.log(1 / q[amax])), "used-predicted-class"),
                            (fmt(-sp.log(1 - q[c])), "used-complement-probability")],
        )
    if level == 2:
        k = rng.choice([3, 4])
        P, _, _ = count_dist(rng, k)
        while True:
            Q = dyadic(rng, k)
            if Q != P:
                break
        ce = tidy(sum(-p * lg(q) for p, q in zip(P, Q)))
        rev = tidy(sum(-q * sp.log(p) for p, q in zip(P, Q)) / LN2)
        return Problem(
            prompt=(f"Data follow $P = {tup(P)}$; a model uses $Q = {tup(Q)}$. Compute the cross-entropy "
                    "$H(P, Q) = -\\sum_x P(x)\\log_2 Q(x)$ in bits. (On paper: verify $H(P,Q) = H(P) + D(P\\|Q)$.)"),
            answer=fmt(ce), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Weights from P, logs of Q", "log₂ of each Q entry is an integer", "Sum",
                    "Check: H(P,Q) ≥ H(P)"],
            solution=(f"$H(P,Q) = " + " + ".join(f"{L(p)}\\cdot{L(-lg(q))}" for p, q in zip(P, Q)) +
                      f" = {L(ce)}$ bits; $H(P) \\approx {d4(H(P))}$, so $D(P\\|Q) \\approx {d4(ce - H(P))}$."),
            misconceptions=[(fmt(rev), "cross-entropy-argument-order"), (fmt(H(P)), "forgot-kl-term")],
        )
    n1, n0 = rng.randint(1, 9), rng.randint(1, 9)
    if n1 == n0:
        n0 += 1
    n = n1 + n0
    qs = R(n1, n)
    lstar = Hnat([qs, 1 - qs])
    return Problem(
        prompt=(f"A dataset has {n1} positive and {n0} negative labels. A *constant* model predicts "
                "$P(y=1) = q$ for every example. Minimise the average binary cross-entropy (nats) over $q$: "
                "give the optimal $q^*$ and the minimum average loss."),
        answer=[fmt(qs), fmt(lstar)], kind="list", tol=TOL, answer_hint="q*, loss   e.g. 3/10, 0.6109",
        rubric=["Average loss = -(n₁/n) ln q - (n₀/n) ln(1-q)", "Set derivative to zero", "q* = n₁/n (empirical rate)",
                "Minimum = H_b(q*) in nats"],
        solution=(f"$\\frac{{d}}{{dq}}: -\\frac{{n_1}}{{nq}} + \\frac{{n_0}}{{n(1-q)}} = 0 \\Rightarrow q^* = {L(qs)}$; "
                  f"loss $= {L(lstar)} \\approx {d4(lstar)}$ nats (the label entropy)."),
        misconceptions=[([fmt(qs), fmt(H([qs, 1 - qs]))], "reported-bits-not-nats"),
                        ([fmt(R(1, 2)), fmt(sp.log(2))], "assumed-q-half")],
    )


@gen("info.cross_entropy")
def bce_logloss(rng, level):
    probs = [R(1, 10), R(1, 5), R(1, 4), R(1, 3), R(2, 5), R(3, 5), R(2, 3), R(3, 4), R(4, 5), R(9, 10)]
    if level == 1:
        p, y = rng.choice(probs), rng.randint(0, 1)
        loss = sp.log(1 / p) if y == 1 else sp.log(1 / (1 - p))
        wrong = sp.log(1 / (1 - p)) if y == 1 else sp.log(1 / p)
        return Problem(
            prompt=(f"A spam filter outputs $\\hat p = P(\\text{{spam}}) = {L(p)}$ for an email whose true label is "
                    f"$y = {y}$ ({'spam' if y else 'not spam'}). Compute the binary cross-entropy loss "
                    "$-[y\\ln\\hat p + (1-y)\\ln(1-\\hat p)]$ in nats."),
            answer=fmt(loss), kind="num", tol=TOL, answer_hint=NATS_HINT,
            rubric=[f"y = {y} selects the {'first' if y else 'second'} term", "Evaluate the log", "Keep the minus sign"],
            solution=f"$\\ell = -\\ln {L(p if y else 1 - p)} = {L(loss)} \\approx {d4(loss)}$ nats.",
            misconceptions=[(fmt(wrong), "used-wrong-term"), (fmt(tidy(loss / LN2)), "used-log2")],
        )
    if level == 2:
        ex = [(rng.choice(probs), rng.randint(0, 1)) for _ in range(3)]
        terms = [sp.log(1 / p) if y else sp.log(1 / (1 - p)) for p, y in ex]
        avg = tidy(sum(terms) / 3)
        return Problem(
            prompt=("Mini-batch of 3 examples with predicted $\\hat p_i = P(y=1)$ and labels $y_i$: " +
                    ", ".join(f"$(\\hat p, y) = ({L(p)}, {y})$" for p, y in ex) +
                    ". Compute the **mean** binary cross-entropy in nats."),
            answer=fmt(avg), kind="num", tol=TOL, answer_hint=NATS_HINT,
            rubric=["Per-example loss -ln(prob assigned to the true label)", "Sum the three losses", "Divide by 3"],
            solution=("Per-example: " + ", ".join(f"${L(t)}$" for t in terms) + f"; mean $= {L(avg)} \\approx {d4(avg)}$ nats."),
            misconceptions=[(fmt(tidy(sum(terms))), "summed-not-averaged"), (fmt(tidy(avg / LN2)), "used-log2")],
        )
    z = rng.choice([-3, -2, -1, 1, 2, 3])
    y = rng.randint(0, 1)
    s = 1 if y == 1 else -1
    loss = sp.log(1 + sp.exp(-s * z))
    return Problem(
        prompt=(f"A logistic model outputs the logit $z = {z}$ and $\\hat p = \\sigma(z) = 1/(1+e^{{-z}})$. "
                f"The label is $y = {y}$. Show that the BCE loss equals $\\ln(1 + e^{{-z}})$ if $y=1$ and "
                "$\\ln(1+e^{z})$ if $y=0$ (the numerically stable 'softplus' form), and give the loss exactly."),
        answer=fmt(loss), kind="num", tol=TOL, answer_hint="exact, e.g. log(1+exp(-2)), or a decimal to 4 places",
        rubric=["-ln σ(z) = ln(1+e^{-z})", "-ln(1-σ(z)) = ln(1+e^{z}) since 1-σ(z) = σ(-z)", f"Substitute z = {z}"],
        solution=f"$\\ell = \\ln(1 + e^{{{-s * z}}}) = {L(loss)} \\approx {d4(loss)}$ nats.",
        misconceptions=[(fmt(sp.log(1 + sp.exp(s * z))), "logit-sign-error")],
    )


# --------------------------------------------------------------------------- mutual information
@gen("info.mutual_information")
def mutual_information(rng, level):
    if level == 1:
        hx = rng.choice([R(1), R(3, 2), R(7, 4), R(2)])
        hy = rng.choice([R(1), R(3, 2), R(7, 4), R(2)])
        i = rng.choice([v for v in (R(1, 4), R(1, 2), R(3, 4)) if v <= min(hx, hy)])
        hxy = hx + hy - i
        return Problem(
            prompt=(f"For two random variables, $H(X) = {L(hx)}$, $H(Y) = {L(hy)}$ and $H(X,Y) = {L(hxy)}$ bits. "
                    "Compute $I(X;Y)$. (On paper: also give $H(X\\mid Y)$ and draw the Venn diagram.)"),
            answer=fmt(i), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["I(X;Y) = H(X) + H(Y) - H(X,Y)", "Substitute", "H(X|Y) = H(X) - I"],
            solution=f"$I = {L(hx)} + {L(hy)} - {L(hxy)} = {L(i)}$ bits; $H(X\\mid Y) = {L(hx - i)}$.",
            misconceptions=[(fmt(-i), "mi-sign-error"), (fmt(hx + hy + hxy), "added-all-entropies")],
        )
    if level == 2:
        while True:  # avoid near-independent tables (answer ~ 0)
            P, px, py = _table(rng, 2, 2, hi=6)
            hxy = H([p for row in P for p in row])
            mi = tidy(H(px) + H(py) - hxy)
            if float(mi) > 0.02:
                break
        return Problem(
            prompt="The joint distribution of $(X, Y)$ is\n" + joint_tex(P) + "\nCompute the mutual information $I(X;Y)$ in bits.",
            answer=fmt(mi), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Marginals p(x), p(y)", "Either Σ p(x,y) log₂[p(x,y)/(p(x)p(y))] or H(X)+H(Y)-H(X,Y)", "Evaluate (≥ 0)"],
            solution=(f"$p(x) = {tup(px)}$, $p(y) = {tup(py)}$; $I = H(X) + H(Y) - H(X,Y) = {L(mi)} \\approx {d4(mi)}$ bits."),
            misconceptions=[(fmt(-mi), "mi-sign-error"), (fmt(hxy), "reported-joint-entropy")],
        )
    n = rng.randint(5, 10)
    m = rng.choice([v for v in (2, 3, 4) if v < n])
    cnt = [sum(1 for v in range(1, n + 1) if v % m == r) for r in range(m)]
    hy = H([R(c, n) for c in cnt])
    return Problem(
        prompt=(f"$X$ is uniform on $\\{{1, 2, \\dots, {n}\\}}$ and $Y = X \\bmod {m}$. "
                "Compute $I(X;Y)$ in bits. (Hint: what is $H(Y\\mid X)$?)"),
        answer=fmt(hy), kind="num", tol=TOL, answer_hint=BITS_HINT,
        rubric=["Y is a function of X, so H(Y|X) = 0", "Hence I(X;Y) = H(Y)", f"Count how many x give each residue mod {m}",
                "Compute H(Y)"],
        solution=(f"Residues $0..{m-1}$ occur ${cnt}$ times out of ${n}$. $I(X;Y) = H(Y) = {L(hy)} \\approx {d4(hy)}$ bits."),
        misconceptions=[(fmt(lg(m)), "assumed-Y-uniform"), (fmt(lg(n)), "mi-equals-HX")],
    )


# --------------------------------------------------------------------------- source coding
@gen("info.source_coding")
def huffman_coding(rng, level):
    if level == 1:
        k = rng.randint(4, 6)
        ls = sorted(rng.randint(1, 5) for _ in range(k))
        s = sum(R(1, 2 ** l) for l in ls)
        return Problem(
            prompt=(f"A binary code has codeword lengths ${ls}$. Compute the Kraft sum $\\sum_i 2^{{-\\ell_i}}$. "
                    "(On paper: can a prefix-free code with these lengths exist? If yes, construct one.)"),
            answer=fmt(s), kind="num",
            rubric=["Kraft sum Σ 2^{-ℓ}", "Exact fraction", "Prefix code exists iff sum ≤ 1"],
            solution=f"$\\sum 2^{{-\\ell_i}} = {L(s)}$ → a prefix code {'exists' if s <= 1 else 'does **not** exist'}.",
            misconceptions=[(fmt(sum(ls)), "summed-lengths")],
        )
    k = rng.choice([4, 5])
    while True:  # keep the redundancy clearly non-zero
        ps, c, N = count_dist(rng, k, 1, 9)
        Lbar, lens = huffman(ps)
        h = H(ps)
        if float(Lbar - h) > 0.01:
            break
    if level == 2:
        return Problem(
            prompt=(f"Symbols ${', '.join(SYMS[:k])}$ have probabilities ${tup(ps)}$. Build a binary Huffman code "
                    "and give its expected codeword length (bits/symbol)."),
            answer=fmt(Lbar), kind="num", answer_hint="exact fraction, e.g. 37/16",
            rubric=["Repeatedly merge the two least probable nodes", "Read codeword lengths off the tree",
                    "Expected length Σ p ℓ", "Check H(X) ≤ L̄ < H(X)+1"],
            solution=(f"Codeword lengths ${lens}$ (ties may give other trees with the same $\\bar L$); "
                      f"$\\bar L = {L(Lbar)} \\approx {float(Lbar):.4f}$ vs $H \\approx {d4(h)}$ bits."),
            misconceptions=[(fmt(math.ceil(math.log2(k))), "used-fixed-length"), (fmt(h), "entropy-not-achievable")],
        )
    red = tidy(Lbar - h)
    return Problem(
        prompt=(f"For the source ${tup(ps)}$ (symbols ${', '.join(SYMS[:k])}$), compute the **redundancy** of a "
                "Huffman code, $\\bar L - H(X)$, in bits/symbol."),
        answer=fmt(red), kind="num", tol=TOL, answer_hint=BITS_HINT,
        rubric=["Build the Huffman tree", "Expected length L̄", "Entropy H(X)", "Subtract (must lie in [0, 1))"],
        solution=f"$\\bar L = {L(Lbar)}$ (lengths ${lens}$), $H \\approx {d4(h)}$, redundancy $\\approx {d4(red)}$ bits.",
        misconceptions=[(fmt(Lbar), "forgot-subtract-entropy"), (fmt(h), "reported-entropy")],
    )


@gen("info.source_coding")
def compression_bounds(rng, level):
    if level == 1:
        k = rng.randint(3, 6)
        ps = dyadic(rng, k)
        N = rng.choice([100, 400, 1000, 2000])
        h = H(ps)
        return Problem(
            prompt=(f"An i.i.d. source has symbol probabilities ${tup(ps)}$. By Shannon's source coding theorem, "
                    f"about how many bits are needed (on average, at best) to losslessly encode a message of {N} symbols?"),
            answer=fmt(N * h), kind="num",
            rubric=["Compute H(X) in bits/symbol", "Total ≈ N·H(X)", "Note: dyadic ⇒ Huffman achieves it exactly"],
            solution=f"$H = {L(h)}$ bits/symbol, so $N H = {N}\\cdot {L(h)} = {L(N * h)}$ bits.",
            misconceptions=[(fmt(N * math.ceil(math.log2(k))), "used-fixed-length")],
        )
    if level == 2:
        p = rng.choice([R(1, 4), R(1, 5), R(1, 10), R(1, 3)])
        pairs = [(1 - p) ** 2, p * (1 - p), p * (1 - p), p ** 2]
        Lbar, lens = huffman(pairs)
        per = Lbar / 2
        return Problem(
            prompt=(f"A binary source emits 1 with probability ${L(p)}$ (i.i.d.). Huffman-coding single symbols needs 1 bit/symbol. "
                    "Instead apply Huffman coding to **pairs** of symbols. What is the expected length per *source symbol*?"),
            answer=fmt(per), kind="num", answer_hint="exact fraction, e.g. 27/32",
            rubric=["List the 4 pair probabilities", "Build the Huffman code on pairs", "Expected length per pair", "Divide by 2"],
            solution=(f"Pairs $00,01,10,11$: ${tup(pairs)}$, lengths ${lens}$, $\\bar L_2 = {L(Lbar)}$, "
                      f"per symbol ${L(per)} \\approx {float(per):.4f}$ (entropy $H_b \\approx {d4(Hb(p))}$)."),
            misconceptions=[(fmt(Lbar), "forgot-divide-by-block-length"), ("1", "no-block-gain")],
        )
    Lmax = rng.randint(4, 6)
    while True:
        ls = sorted(rng.randint(1, Lmax) for _ in range(rng.randint(2, 4)))
        s = sum(R(1, 2 ** l) for l in ls)
        if s < 1:
            break
    extra = (1 - s) * 2 ** Lmax
    return Problem(
        prompt=(f"A prefix-free binary code already has codewords of lengths ${ls}$. Using the Kraft inequality, "
                f"what is the **maximum number of additional codewords of length {Lmax}** that can be added while "
                "keeping the code prefix-free?"),
        answer=fmt(extra), kind="num",
        rubric=["Kraft: Σ 2^{-ℓ} ≤ 1", "Unused Kraft budget 1 - Σ 2^{-ℓ}", f"Each new codeword uses 2^-{Lmax}",
                "Divide (result is an integer)"],
        solution=f"Budget left $1 - {L(s)} = {L(1 - s)}$; each new word costs $2^{{-{Lmax}}}$, so ${L(extra)}$ more.",
        misconceptions=[(fmt(2 ** Lmax), "ignored-existing-codewords")],
    )


# --------------------------------------------------------------------------- channel capacity
@gen("info.channel_capacity")
def channel_capacity(rng, level):
    if level == 1:
        if rng.random() < 0.6:
            p = rng.choice([R(1, 4), R(1, 8), R(1, 3), R(1, 10), R(1, 5), R(3, 4)])
            c = tidy(1 - Hb(p))
            return Problem(
                prompt=(f"A binary symmetric channel flips each bit with probability $p = {L(p)}$. "
                        "Compute its capacity in bits per channel use."),
                answer=fmt(c), kind="num", tol=TOL, answer_hint=BITS_HINT,
                rubric=["C_BSC = 1 - H_b(p)", "Compute H_b(p)", "Subtract"],
                solution=f"$C = 1 - H_b({L(p)}) = {L(c)} \\approx {d4(c)}$ bits/use.",
                misconceptions=[(fmt(Hb(p)), "capacity-is-entropy"), (fmt(1 - p), "used-bec-formula")],
            )
        e = rng.choice([R(1, 10), R(1, 4), R(1, 3), R(1, 2), R(2, 5)])
        return Problem(
            prompt=(f"A binary erasure channel erases each bit (output '?') with probability $\\varepsilon = {L(e)}$; "
                    "unerased bits are always correct. Give its capacity in bits per use."),
            answer=fmt(1 - e), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["I(X;Y) = (1-ε) H(X) for the BEC", "Maximise with uniform input", "C = 1 - ε"],
            solution=f"$C = 1 - \\varepsilon = {L(1 - e)}$ bits/use.",
            misconceptions=[(fmt(tidy(1 - Hb(e))), "used-bsc-formula")],
        )
    if level == 2:
        p1, p2 = rng.choice([R(1, 10), R(1, 8), R(1, 5)]), rng.choice([R(1, 10), R(1, 4), R(1, 5)])
        p = p1 * (1 - p2) + p2 * (1 - p1)
        c = tidy(1 - Hb(p))
        return Problem(
            prompt=(f"Two binary symmetric channels with crossover probabilities ${L(p1)}$ and ${L(p2)}$ are connected in series. "
                    "Find the capacity of the cascade (bits/use)."),
            answer=fmt(c), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["A bit is flipped overall iff exactly one stage flips it", "p = p₁(1-p₂) + p₂(1-p₁)", "C = 1 - H_b(p)"],
            solution=f"$p = {L(p)}$, $C = 1 - H_b({L(p)}) = {L(c)} \\approx {d4(c)}$ bits/use.",
            misconceptions=[(fmt(tidy(1 - Hb(p1 + p2))), "added-crossovers"),
                            (fmt(tidy(1 - Hb(p1) - Hb(p2))), "subtracted-both-entropies")],
        )
    p = rng.choice([R(1, 10), R(1, 8), R(1, 5), R(1, 4)])
    q = rng.choice([R(1, 4), R(1, 3), R(1, 5), R(3, 4)])
    pq = q * (1 - p) + (1 - q) * p
    i = tidy(Hb(pq) - Hb(p))
    return Problem(
        prompt=(f"A BSC with crossover $p = {L(p)}$ is driven by a *non-uniform* input $X \\sim \\text{{Bernoulli}}({L(q)})$. "
                "Derive $I(X;Y)$ for this input and evaluate it (bits/use). On paper: why is it below capacity?"),
        answer=fmt(i), kind="num", tol=TOL, answer_hint=BITS_HINT,
        rubric=["I(X;Y) = H(Y) - H(Y|X)", "H(Y|X) = H_b(p)", "P(Y=1) = q(1-p) + (1-q)p", "Evaluate; capacity needs q = 1/2"],
        solution=(f"$P(Y=1) = {L(pq)}$; $I = H_b({L(pq)}) - H_b({L(p)}) = {L(i)} \\approx {d4(i)}$ bits, "
                  f"less than $C = 1 - H_b(p) \\approx {d4(1 - Hb(p))}$."),
        misconceptions=[(fmt(tidy(1 - Hb(p))), "assumed-uniform-input"), (fmt(tidy(Hb(q) - Hb(p))), "used-HX-not-HY")],
    )


# --------------------------------------------------------------------------- perplexity
@gen("info.perplexity")
def perplexity(rng, level):
    if level == 1:
        n = rng.choice([3, 4, 5])
        ls = [rng.randint(1, 4) for _ in range(n)]
        ps = [R(1, 2 ** l) for l in ls]
        ppl = sp.Integer(2) ** R(sum(ls), n)
        return Problem(
            prompt=(f"A language model assigns probabilities ${tup(ps)}$ to the {n} tokens of a test sentence. "
                    "Compute the perplexity $\\mathrm{PPL} = \\left(\\prod_i p_i\\right)^{-1/n}$."),
            answer=fmt(ppl), kind="num", tol=TOL, answer_hint="exact like 2^(7/4), or a decimal to 3 places",
            rubric=["Average log-loss: (1/n) Σ log₂(1/p_i)", "PPL = 2^{average bits}", "Simplify"],
            solution=(f"Average bits $= ({' + '.join(map(str, ls))})/{n} = {L(R(sum(ls), n))}$, "
                      f"$\\mathrm{{PPL}} = 2^{{{L(R(sum(ls), n))}}} \\approx {float(ppl):.3f}$."),
            misconceptions=[(fmt(sum(1 / p for p in ps) / n), "arithmetic-mean-of-inverse-probs"),
                            (fmt(R(sum(ls), n)), "reported-cross-entropy-not-perplexity")],
        )
    if level == 2:
        loss = R(rng.randint(150, 350), 100)
        cpt = rng.choice([3, 4, 5])
        ppl, bpc = sp.exp(loss), loss / (cpt * LN2)
        return Problem(
            prompt=(f"An LLM reaches a mean cross-entropy of ${float(loss):.2f}$ **nats per token** on a test set; "
                    f"tokens average {cpt} characters. Give (a) the per-token perplexity and (b) the bits per character."),
            answer=[fmt(ppl), fmt(bpc)], kind="list", tol=TOL, answer_hint="ppl, bpc   e.g. 12.18, 0.902",
            rubric=["PPL = e^{loss in nats}", "Convert nats → bits: divide by ln 2", f"Per character: divide by {cpt}"],
            solution=(f"(a) $e^{{{float(loss):.2f}}} \\approx {float(ppl):.3f}$; (b) ${float(loss):.2f}/({cpt}\\ln 2) "
                      f"\\approx {float(bpc):.4f}$ bits/char."),
            misconceptions=[([fmt(2 ** loss), fmt(bpc)], "used-base-2-with-nats"),
                            ([fmt(ppl), fmt(loss / cpt)], "forgot-nats-to-bits")],
        )
    k = rng.choice([3, 4, 5])
    while True:
        P, Q = dyadic(rng, k), dyadic(rng, k)
        if P != Q:
            break
    ce = tidy(sum(-p * lg(q) for p, q in zip(P, Q)))
    ppl = sp.Integer(2) ** ce
    return Problem(
        prompt=(f"Text is generated i.i.d. from the unigram distribution $P = {tup(P)}$ over a {k}-word vocabulary, "
                f"but evaluated with a model $Q = {tup(Q)}$. Derive the long-run perplexity of $Q$ on this text "
                "(law of large numbers) and evaluate it."),
        answer=fmt(ppl), kind="num", tol=TOL, answer_hint="exact like 2^(7/4), or a decimal to 3 places",
        rubric=["(1/n) Σ log₂(1/Q(w_i)) → E_P[log₂ 1/Q] = H(P,Q)", "PPL → 2^{H(P,Q)}", "Compute H(P,Q)",
                "Note 2^{H(P)} is the best achievable (Q = P)"],
        solution=(f"$H(P,Q) = {L(ce)}$ bits, so $\\mathrm{{PPL}} = 2^{{{L(ce)}}} \\approx {float(ppl):.3f}$; "
                  f"the ideal model would give $2^{{H(P)}} = 2^{{{L(H(P))}}} \\approx {float(2 ** H(P)):.3f}$."),
        misconceptions=[(fmt(sp.Integer(2) ** H(P)), "used-entropy-not-cross-entropy"), (fmt(k), "used-vocab-size")],
    )


# --------------------------------------------------------------------------- descriptive stats / EDA
def _median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else R(v[n // 2 - 1] + v[n // 2], 2)


@gen("info.data_eda")
def data_eda(rng, level):
    if level == 1:
        n = rng.randint(5, 7)
        while True:
            data = [rng.randint(1, 15) for _ in range(n)]
            if len(set(data)) > 1:
                break
        m = R(sum(data), n)
        ss = sum((v - m) ** 2 for v in data)
        return Problem(
            prompt=f"Data: ${data}$. Compute the **sample** variance $s^2 = \\frac{{1}}{{n-1}}\\sum (x_i - \\bar x)^2$ exactly.",
            answer=fmt(ss / (n - 1)), kind="num",
            rubric=["Mean x̄", "Squared deviations", "Sum and divide by n-1"],
            solution=f"$\\bar x = {L(m)}$, $\\sum (x_i-\\bar x)^2 = {L(ss)}$, $s^2 = {L(ss / (n - 1))}$.",
            misconceptions=[(fmt(ss / n), "divided-by-n"), (fmt(sp.sqrt(ss / (n - 1))), "reported-std-not-variance")],
        )
    if level == 2:
        n = rng.choice([4, 5])
        while True:
            xs = [rng.randint(0, 8) for _ in range(n)]
            ys = [rng.randint(0, 8) for _ in range(n)]
            mx, my = R(sum(xs), n), R(sum(ys), n)
            sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            sxx = sum((a - mx) ** 2 for a in xs)
            syy = sum((b - my) ** 2 for b in ys)
            if sxx and syy and sxy:
                break
        r = sxy / sp.sqrt(sxx * syy)
        return Problem(
            prompt=("Compute the Pearson correlation $r$ of the paired data $(x, y)$: " +
                    ", ".join(f"$({a},{b})$" for a, b in zip(xs, ys)) + "."),
            answer=fmt(r), kind="num", tol=TOL, answer_hint="exact (with sqrt) or a decimal to 4 places",
            rubric=["Means x̄, ȳ", "S_xy, S_xx, S_yy", "r = S_xy / sqrt(S_xx S_yy)", "Check |r| ≤ 1"],
            solution=(f"$\\bar x = {L(mx)}, \\bar y = {L(my)}$; $S_{{xy}} = {L(sxy)}, S_{{xx}} = {L(sxx)}, S_{{yy}} = {L(syy)}$; "
                      f"$r = {L(r)} \\approx {d4(r)}$."),
            misconceptions=[(fmt(sxy / n), "reported-covariance"), (fmt(sxy / sxx), "reported-slope")],
        )
    n = rng.choice([8, 9])
    data = sorted([rng.randint(10, 30) for _ in range(n - 1)] + [rng.choice([rng.randint(45, 70), rng.randint(-20, -5)])])
    lo_half, hi_half = data[: n // 2], data[(n + 1) // 2:]
    q1, q3 = _median(lo_half), _median(hi_half)
    iqr = q3 - q1
    lf, uf = q1 - R(3, 2) * iqr, q3 + R(3, 2) * iqr
    out = sum(1 for v in data if v < lf or v > uf)
    return Problem(
        prompt=(f"Sorted data: ${data}$. Using Tukey's rule with quartiles taken as medians of the lower/upper halves "
                "(excluding the overall median when $n$ is odd), give the lower fence, the upper fence "
                "$Q_1 - 1.5\\,\\mathrm{IQR},\\ Q_3 + 1.5\\,\\mathrm{IQR}$, and the number of outliers."),
        answer=[fmt(lf), fmt(uf), str(out)], kind="list", answer_hint="lower, upper, count   e.g. 2.5, 38.5, 1",
        rubric=["Split into halves", "Q1, Q3 = medians of the halves", "IQR = Q3 - Q1", "Fences", "Count points outside"],
        solution=(f"$Q_1 = {L(q1)},\\ Q_3 = {L(q3)},\\ \\mathrm{{IQR}} = {L(iqr)}$; fences ${L(lf)}$ and ${L(uf)}$; "
                  f"{out} outlier(s)."),
    )


# --------------------------------------------------------------------------- A/B testing, Simpson
@gen("info.ab_testing")
def ab_testing(rng, level):
    if level == 1:
        n = rng.choice([400, 500, 800, 1000])
        x1 = rng.randint(n // 20, n // 5)
        x2 = x1 + rng.randint(n // 100 + 1, n // 20)
        ph = R(x1 + x2, 2 * n)
        z = (R(x2 - x1, n)) / sp.sqrt(ph * (1 - ph) * R(2, n))
        zu = (R(x2 - x1, n)) / sp.sqrt(R(x1, n) * (1 - R(x1, n)) / n + R(x2, n) * (1 - R(x2, n)) / n)
        return Problem(
            prompt=(f"A/B test: variant A converts {x1} of {n} visitors, variant B converts {x2} of {n}. "
                    "Compute the two-proportion $z$-statistic $(\\hat p_B - \\hat p_A)/\\mathrm{SE}$ using the **pooled** "
                    "proportion. (On paper: is it significant at the two-sided 5% level, $|z| > 1.96$?)"),
            answer=fmt(z), kind="num", tol=0.01, answer_hint="a decimal rounded to 2 places, e.g. 1.87",
            rubric=["Pooled p̂ = (x_A + x_B)/(n_A + n_B)", "SE = sqrt(p̂(1-p̂)(1/n_A + 1/n_B))", "z = difference / SE",
                    "Compare with 1.96"],
            solution=(f"$\\hat p = {L(ph)}$, $z = {L(R(x2 - x1, n))}/\\sqrt{{{L(ph * (1 - ph) * R(2, n))}}} \\approx {float(z):.2f}$ "
                      f"→ {'significant' if abs(float(z)) > 1.96 else 'not significant'} at 5%."),
            misconceptions=[(fmt(zu), "unpooled-se"), (fmt(R(x2 - x1, n) / (ph * (1 - ph) * R(2, n))), "forgot-square-root")],
        )
    if level == 2:
        for _ in range(10000):
            nAs, nAl = rng.randint(10, 40), rng.randint(60, 120)
            nBs, nBl = rng.randint(60, 120), rng.randint(10, 40)
            sAs, sBs = rng.randint(int(0.75 * nAs), nAs), rng.randint(int(0.75 * nBs), nBs)
            sAl, sBl = rng.randint(int(0.35 * nAl), int(0.75 * nAl)), rng.randint(int(0.3 * nBl), int(0.7 * nBl))
            a_s, a_l, b_s, b_l = R(sAs, nAs), R(sAl, nAl), R(sBs, nBs), R(sBl, nBl)
            oa, ob = R(sAs + sAl, nAs + nAl), R(sBs + sBl, nBs + nBl)
            if a_s > b_s and a_l > b_l and oa < ob - R(3, 100):
                break
        diff = oa - ob
        tex = ("$$\\begin{array}{l|cc} & \\text{small cases} & \\text{large cases} \\\\ \\hline "
               f"\\text{{A}} & {sAs}/{nAs} & {sAl}/{nAl} \\\\ \\text{{B}} & {sBs}/{nBs} & {sBl}/{nBl} \\end{{array}}$$")
        return Problem(
            prompt=("Success counts for two treatments, split by case severity:\n" + tex +
                    "\nCompute the **overall** success rate of A minus that of B (exact fraction). "
                    "(On paper: compare with the within-stratum comparisons and explain the paradox.)"),
            answer=fmt(diff), kind="num", tol=1e-4, answer_hint="exact fraction or a decimal to 4 places",
            rubric=["Within each stratum A beats B", "Pool counts: (successes)/(total) per treatment", "Subtract",
                    "Explain: A treats mostly the hard (large) cases — a confounder"],
            solution=(f"Overall A $= {sAs + sAl}/{nAs + nAl}$, B $= {sBs + sBl}/{nBs + nBl}$, difference $= {L(diff)} "
                      f"\\approx {float(diff):.4f} < 0$ although A wins in both strata (Simpson's paradox)."),
            misconceptions=[(fmt((a_s + a_l) / 2 - (b_s + b_l) / 2), "averaged-stratum-rates")],
        )
    p1 = rng.choice([R(1, 10), R(1, 5), R(3, 10), R(1, 2)])
    d = rng.choice([R(1, 20), R(1, 10)])
    p2 = p1 + d
    zsum = R(196, 100) + R(84, 100)
    nexact = zsum ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / d ** 2
    n = sp.ceiling(nexact)
    return Problem(
        prompt=(f"You plan an A/B test to detect a lift from ${float(p1):.2f}$ to ${float(p2):.2f}$ conversion with a two-sided "
                "5% test and 80% power. Using $n = (z_{0.025} + z_{0.2})^2\\,[p_1(1-p_1) + p_2(1-p_2)]/\\delta^2$ with "
                "$z_{0.025} = 1.96,\\ z_{0.2} = 0.84$, how many users per group are needed (round **up**)?"),
        answer=str(n), kind="num",
        rubric=["δ = p₂ - p₁", "(1.96 + 0.84)² = 7.84", "Sum of Bernoulli variances", "Divide by δ² and round up"],
        solution=f"$n = 7.84\\cdot({L(p1 * (1 - p1) + p2 * (1 - p2))})/{L(d ** 2)} = {float(nexact):.2f} \\Rightarrow {n}$ per group.",
        misconceptions=[(str(sp.floor(nexact)), "rounded-down")] if sp.floor(nexact) != n else [],
    )


# =========================================================================== more problem types
# Extra generators (one per skill) so that every skill-level offers several genuinely different
# tasks, not the same sentence with new numbers. Each picks a variant first, then its numbers.

WEATHER = ["sunny", "cloudy", "rainy", "snowy"]
FREQ_WORDS = {R(1, 2): "half of the days", R(1, 4): "a quarter of the days",
              R(1, 8): "one day in eight", R(1, 16): "one day in sixteen"}
MAXENT_SHAPES = [(1, 0, 0, 0), (R(1, 2), R(1, 2), 0, 0), (R(1, 2), R(1, 4), R(1, 4), 0),
                 (R(1, 2), R(1, 4), R(1, 8), R(1, 8)), (R(1, 4),) * 4]


@gen("info.entropy")
def entropy_scenarios(rng, level):
    if level == 1:
        v = rng.randrange(3)
        if v == 0:  # fair n-sided die
            n = rng.choice([4, 6, 8, 10, 12, 16, 20])
            h = lg(n)
            return Problem(
                prompt=(f"A fair die with {n} faces is rolled once. How much information does the outcome carry, "
                        "i.e. what is its entropy $H(X)$ in bits?"),
                answer=fmt(h), kind="num", tol=TOL, answer_hint=BITS_HINT,
                rubric=["Uniform distribution: every face has p = 1/n", "H = Σ (1/n) log₂ n = log₂ n", "Evaluate"],
                solution=f"$H = \\sum_{{i=1}}^{{{n}}} \\tfrac{{1}}{{{n}}}\\log_2 {n} = \\log_2 {n} = {L(h)} \\approx {d4(h)}$ bits.",
                misconceptions=[(str(n), "confused-entropy-with-outcome-count"), (fmt(sp.log(n)), "used-natural-log")],
            )
        if v == 1:  # which distribution has the largest entropy
            shapes = rng.sample(MAXENT_SHAPES, 3)
            dists = []
            for s in shapes:
                s = list(s)
                rng.shuffle(s)
                dists.append([sp.Integer(1) * p for p in s])
            hs = [H(d) for d in dists]
            best, worst = "ABC"[hs.index(max(hs))], "ABC"[hs.index(min(hs))]
            return Problem(
                prompt=("Three sources emit the same four symbols with these distributions:\n\n" +
                        "\n".join(f"- **{c}**: ${tup(d)}$" for c, d in zip("ABC", dists)) +
                        "\n\nWhich source is the *least predictable*, i.e. has the largest entropy?"),
                answer=best, kind="choice", answer_hint="the letter A, B or C",
                rubric=["Compute H = Σ p log₂(1/p) for each source (0·log 0 = 0)",
                        "All logs are integers here", "Pick the largest (spread-out ⇒ larger entropy)"],
                solution=("Entropies: " + ", ".join(f"$H_{c} = {L(h)}$" for c, h in zip("ABC", hs)) +
                          f" bits, so **{best}** is the least predictable (the uniform distribution would give the maximum, 2 bits)."),
                misconceptions=[(worst, "picked-lowest-entropy")],
            )
        k = rng.choice([3, 4])
        ps = dyadic(rng, k)
        names = rng.sample(WEATHER, k)
        h = H(ps)
        desc = ", ".join(f"{w} on {FREQ_WORDS[p]}" for w, p in zip(names, ps))
        return Problem(
            prompt=(f"In a small town the weather is {desc} (days are independent). A weather station broadcasts the "
                    "day's weather each morning. What is the entropy of one broadcast, in bits?"),
            answer=fmt(h), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Translate the words into a probability distribution", "H = Σ p log₂(1/p)",
                    "Each log₂(1/p) is an integer here", "Add"],
            solution=(f"$p = {tup(ps)}$, $H = " + " + ".join(f"{L(p)}\\cdot{L(-lg(p))}" for p in ps) + f" = {L(h)}$ bits."),
            misconceptions=[(fmt(lg(k)), "assumed-uniform"), (fmt(h * LN2), "used-natural-log")],
        )
    if level == 2:  # entropy of a function of X
        k = rng.randint(1, 4)
        n = 2 * k + 1
        ps = [R(1, n)] + [R(2, n)] * k
        h = H(ps)
        return Problem(
            prompt=(f"$X$ is uniform on the integers $\\{{-{k}, \\dots, {k}\\}}$ and $Y = X^2$. Find the distribution of $Y$ "
                    "and compute $H(Y)$ in bits. (On paper: explain why $H(Y) \\le H(X)$.)"),
            answer=fmt(h), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=[f"X has {n} equally likely values", "Y = 0 only for X = 0; every other value of Y comes from ±x",
                    f"P(Y=0) = 1/{n}, P(Y=y) = 2/{n} for the {k} positive squares", "Apply H = Σ p log₂(1/p)"],
            solution=(f"$Y \\in \\{{" + ", ".join(str(t * t) for t in range(k + 1)) + f"\\}}$ with probabilities ${tup(ps)}$; "
                      f"$H(Y) = {L(h)} \\approx {d4(h)}$ bits $< \\log_2 {n} = H(X)$, since a function cannot add information."),
            misconceptions=[(fmt(lg(n)), "entropy-of-X-not-Y"), (fmt(lg(k + 1)), "assumed-Y-uniform")],
        )
    p = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(1, 5), R(2, 5)])
    n = rng.choice([3, 4, 5, 9])
    h = tidy(Hb(p) + (1 - p) * lg(n - 1))
    return Problem(
        prompt=(f"$X$ takes values in $\\{{1, 2, \\dots, {n}\\}}$ and all we know is $P(X=1) = {L(p)}$. "
                "Derive the **largest possible** entropy $H(X)$ (bits) over all distributions with this constraint, "
                "and evaluate it."),
        answer=fmt(h), kind="num", tol=TOL, answer_hint=BITS_HINT,
        rubric=["Grouping rule: H(X) = H_b(p) + (1-p)·H(X | X ≠ 1)", f"H(X | X ≠ 1) ≤ log₂{n - 1}, equality for uniform",
                "Maximum = H_b(p) + (1-p) log₂(n-1)", "Evaluate"],
        solution=(f"Split on the event $X=1$: $H(X) = H_b({L(p)}) + {L(1 - p)}\\,H(X\\mid X\\ne1) \\le H_b({L(p)}) + {L(1 - p)}\\log_2 {n - 1}$, "
                  f"attained when the other {n - 1} values are equally likely: $H_{{\\max}} = {L(h)} \\approx {d4(h)}$ bits."),
        misconceptions=[(fmt(lg(n)), "ignored-constraint"), (fmt(Hb(p)), "forgot-remaining-outcomes")],
    )


@gen("info.joint_conditional")
def joint_conditional_scenarios(rng, level):
    v = rng.randrange(2)
    if level == 1:
        if v == 0:  # marginal entropy from a joint table
            P, px, py = _table(rng, 2, 2)
            hy = H(py)
            return Problem(
                prompt=("Using the joint table below, first find the marginal distribution of $Y$ (column sums), "
                        "then compute its entropy $H(Y)$ in bits.\n" + joint_tex(P)),
                answer=fmt(hy), kind="num", tol=TOL, answer_hint=BITS_HINT,
                rubric=["p(y) = Σ_x p(x, y) (column sums)", "H(Y) = Σ p(y) log₂(1/p(y))", "Evaluate"],
                solution=f"$p(y) = {tup(py)}$, so $H(Y) = {L(hy)} \\approx {d4(hy)}$ bits.",
                misconceptions=[(fmt(H([p for row in P for p in row])), "computed-joint-entropy"),
                                (fmt(H(px)), "used-wrong-marginal")],
            )
        hx = rng.choice([R(1), R(3, 2), R(7, 4), R(2)])
        b = rng.choice([R(1, 2), R(3, 4), R(1), R(5, 4)])
        hy = rng.choice([t for t in (R(1), R(5, 4), R(3, 2), R(7, 4), R(2), R(9, 4)) if b <= t < b + hx])
        hxy = hx + b
        return Problem(
            prompt=(f"For two random variables we know $H(X) = {L(hx)}$, $H(Y\\mid X) = {L(b)}$ and $H(Y) = {L(hy)}$ bits. "
                    "Use the chain rule to find $H(X,Y)$ and then $H(X\\mid Y)$."),
            answer=[fmt(hxy), fmt(hxy - hy)], kind="list", answer_hint="H(X,Y), H(X|Y)   e.g. 5/2, 3/4",
            rubric=["H(X,Y) = H(X) + H(Y|X)", "H(X,Y) = H(Y) + H(X|Y)", "H(X|Y) = H(X,Y) - H(Y)"],
            solution=(f"$H(X,Y) = {L(hx)} + {L(b)} = {L(hxy)}$; $H(X\\mid Y) = {L(hxy)} - {L(hy)} = {L(hxy - hy)}$ bits."),
            misconceptions=[([fmt(hxy), fmt(b)], "assumed-conditional-symmetric"),
                            ([fmt(hx + hy), fmt(hx)], "assumed-independence")],
        )
    if level == 2:
        if v == 0:  # coarse information about a uniform number
            n = rng.choice([6, 8, 10, 12])
            m = rng.randint(2, n - 2)
            hxy = tidy(R(m, n) * lg(m) + R(n - m, n) * lg(n - m))
            return Problem(
                prompt=(f"A number $X$ is drawn uniformly from $\\{{1, 2, \\dots, {n}\\}}$. You are told only $Y$, whether "
                        f"$X \\le {m}$ or not. How much uncertainty about $X$ remains on average, $H(X\\mid Y)$, in bits?"),
                answer=fmt(hxy), kind="num", tol=TOL, answer_hint=BITS_HINT,
                rubric=[f"P(Y = yes) = {m}/{n}", f"Given yes, X is uniform on {m} values: H = log₂{m}",
                        f"Given no, X is uniform on {n - m} values: H = log₂{n - m}", "H(X|Y) = Σ p(y) H(X|Y=y)"],
                solution=(f"$H(X\\mid Y) = \\tfrac{{{m}}}{{{n}}}\\log_2 {m} + \\tfrac{{{n - m}}}{{{n}}}\\log_2 {n - m} = {L(hxy)} "
                          f"\\approx {d4(hxy)}$ bits (check: $= \\log_2 {n} - H(Y)$)."),
                misconceptions=[(fmt(lg(n)), "ignored-conditioning"), (fmt(Hb(R(m, n))), "reported-HY"),
                                (fmt(lg(n) - 1), "assumed-halving")],
            )
        q = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(2, 3), R(3, 4)])
        p = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(1, 5)])
        hxy = tidy(Hb(q) + q * Hb(p))
        return Problem(
            prompt=(f"A faulty switch: the input is $X \\sim \\text{{Bernoulli}}({L(q)})$. When $X = 0$ the output is $Y = 0$; "
                    f"when $X = 1$ the output is $Y = 1$ with probability ${L(1 - p)}$ and $Y = 0$ otherwise. "
                    "Compute the joint entropy $H(X,Y)$ in bits."),
            answer=fmt(hxy), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Chain rule H(X,Y) = H(X) + H(Y|X)", "H(Y|X=0) = 0, H(Y|X=1) = H_b(p)",
                    f"H(Y|X) = {q}·H_b({p})", "Add H_b(q)"],
            solution=(f"$H(X,Y) = H_b({L(q)}) + {L(q)}\\,H_b({L(p)}) = {L(hxy)} \\approx {d4(hxy)}$ bits "
                      "(equivalently: entropy of the three cells $(1-q,\\ qp,\\ q(1-p))$)."),
            misconceptions=[(fmt(tidy(Hb(q) + Hb(p))), "conditional-entropy-not-weighted"), (fmt(Hb(q)), "forgot-HYgivenX")],
        )
    if v == 0:  # only the sum of two bits is seen
        p = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(1, 5), R(2, 5), R(3, 4)])
        ans = 2 * p * (1 - p)
        return Problem(
            prompt=(f"$X$ and $Y$ are independent bits with $P(X=1) = P(Y=1) = {L(p)}$, and only their sum $Z = X + Y$ is "
                    "observed. Derive $H(X\\mid Z)$ in bits. (Think: for which values of $Z$ is $X$ still uncertain?)"),
            answer=fmt(ans), kind="num", tol=TOL, answer_hint="exact fraction, e.g. 4/9",
            rubric=["Z = 0 ⇒ X = 0 and Z = 2 ⇒ X = 1: no uncertainty", "Given Z = 1, X = 1 or 0 equally likely (symmetry): 1 bit",
                    "P(Z = 1) = 2p(1-p)", "H(X|Z) = P(Z=1)·1"],
            solution=(f"$H(X\\mid Z) = P(Z=1)\\cdot H_b(\\tfrac12) = 2\\cdot{L(p)}\\cdot{L(1 - p)} = {L(ans)}$ bits "
                      "(check with the chain rule: $2H_b(p) - H(Z)$)."),
            misconceptions=[(fmt(Hb(p)), "confused-HXgivenZ-with-HZgivenX"), ("1", "ignored-conditioning")],
        )
    s = rng.choice([3, 4, 5])
    counts = [min(t, 2 * s - t) for t in range(1, 2 * s)]
    hs = H([R(c, s * s) for c in counts])
    ans = tidy(2 * lg(s) - hs)
    return Problem(
        prompt=(f"Two fair {s}-sided dice (faces $1..{s}$) are rolled; $X$ is the first die and $S$ is the sum. "
                "Derive $H(X\\mid S)$ in bits: how uncertain is the first die once you know the total?"),
        answer=fmt(ans), kind="num", tol=TOL, answer_hint=BITS_HINT,
        rubric=["(X, S) determines both dice and vice versa ⇒ H(X,S) = 2 log₂ s", "Distribution of S (triangular counts)",
                "H(X|S) = H(X,S) - H(S)", "Evaluate"],
        solution=(f"$H(X,S) = H(X_1,X_2) = 2\\log_2 {s}$. Sum counts ${counts}$ out of ${s * s}$ give $H(S) \\approx {d4(hs)}$; "
                  f"$H(X\\mid S) = {L(ans)} \\approx {d4(ans)}$ bits."),
        misconceptions=[(fmt(lg(s)), "ignored-conditioning"), (fmt(tidy(hs - lg(s))), "reported-mutual-information")],
    )


@gen("info.kl_divergence")
def kl_scenarios(rng, level):
    v = rng.randrange(2)
    coins = [R(1, 2), R(1, 4), R(3, 4), R(1, 8)]
    if level == 1:
        if v == 0:
            p = rng.choice(coins[:3])
            q = rng.choice([c for c in coins if c != p])
            P, Q = [p, 1 - p], [q, 1 - q]
            d = kl(P, Q)
            return Problem(
                prompt=(f"Coin $P$ lands heads with probability ${L(p)}$, coin $Q$ with probability ${L(q)}$. Treating each "
                        "coin as a distribution on {heads, tails}, compute $D_{\\mathrm{KL}}(P\\,\\|\\,Q)$ in bits."),
                answer=fmt(d), kind="num", tol=TOL, answer_hint=BITS_HINT,
                rubric=["D = p log₂(p/q) + (1-p) log₂((1-p)/(1-q))", "Weights come from P", "Evaluate"],
                solution=(f"$D = {L(p)}\\log_2\\frac{{{L(p)}}}{{{L(q)}}} + {L(1 - p)}\\log_2\\frac{{{L(1 - p)}}}{{{L(1 - q)}}} "
                          f"= {L(d)} \\approx {d4(d)}$ bits."),
                misconceptions=[(fmt(kl(Q, P)), "kl-argument-order"), (fmt(d * LN2), "used-natural-log")],
            )
        k = rng.choice([4, 8])
        while True:
            P = dyadic(rng, k)
            if len(set(P)) > 1:
                break
        U = [R(1, k)] * k
        d = kl(P, U)
        return Problem(
            prompt=(f"How far is $P = {tup(P)}$ from the uniform distribution $U$ on the same {k} outcomes? "
                    "Compute $D_{\\mathrm{KL}}(P\\,\\|\\,U)$ in bits."),
            answer=fmt(d), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=[f"D(P||U) = Σ p log₂(p / (1/{k}))", f"= log₂{k} - H(P)", "Evaluate"],
            solution=f"$D(P\\|U) = \\log_2 {k} - H(P) = {L(lg(k))} - {L(H(P))} = {L(d)}$ bits.",
            misconceptions=[(fmt(H(P)), "reported-entropy"), (fmt(kl(U, P)), "kl-argument-order")],
        )
    if level == 2:
        if v == 0:  # both directions
            while True:
                P, Q = dyadic(rng, 4), dyadic(rng, 4)
                d1, d2 = kl(P, Q), kl(Q, P)
                if P != Q and d1 != d2:
                    break
            return Problem(
                prompt=(f"Show that KL divergence is not symmetric: for $P = {tup(P)}$ and $Q = {tup(Q)}$ compute both "
                        "$D_{\\mathrm{KL}}(P\\,\\|\\,Q)$ and $D_{\\mathrm{KL}}(Q\\,\\|\\,P)$ in bits."),
                answer=[fmt(d1), fmt(d2)], kind="list", answer_hint="D(P||Q), D(Q||P)   e.g. 1/4, 3/8",
                rubric=["D(P||Q) = Σ P log₂(P/Q)", "D(Q||P) = Σ Q log₂(Q/P) (weights change too)",
                        "Every log ratio is an integer here", "The two numbers differ"],
                solution=(f"$D(P\\|Q) = " + " + ".join(f"{L(p)}\\cdot({L(lg(p / q))})" for p, q in zip(P, Q)) + f" = {L(d1)}$; "
                          f"$D(Q\\|P) = " + " + ".join(f"{L(q)}\\cdot({L(lg(q / p))})" for p, q in zip(P, Q)) + f" = {L(d2)}$ bits."),
                misconceptions=[([fmt(d2), fmt(d1)], "swapped-directions"), ([fmt(d1), fmt(-d1)], "assumed-antisymmetric")],
            )
        p = rng.choice(coins[:3])
        q = rng.choice([c for c in coins if c != p])
        n = rng.randint(3, 10)
        d1 = kl([p, 1 - p], [q, 1 - q])
        return Problem(
            prompt=(f"A coin is tossed {n} times independently. Under hypothesis $P$ it lands heads with probability ${L(p)}$, "
                    f"under $Q$ with probability ${L(q)}$. Compute the KL divergence (bits) between the two distributions "
                    f"of the whole {n}-toss sequence, $D(P^{{{n}}}\\,\\|\\,Q^{{{n}}})$."),
            answer=fmt(tidy(n * d1)), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["log of a product = sum of logs", "Independence ⇒ KL is additive: D(Pⁿ||Qⁿ) = n·D(P||Q)",
                    "Compute the single-toss KL", f"Multiply by {n}"],
            solution=(f"$D(P\\|Q) = {L(d1)} \\approx {d4(d1)}$ bits per toss, so $D(P^{{{n}}}\\|Q^{{{n}}}) = {n}\\cdot D = "
                      f"{L(tidy(n * d1))} \\approx {d4(n * d1)}$ bits."),
            misconceptions=[(fmt(d1), "forgot-multiply-by-n"), (fmt(tidy(n * kl([q, 1 - q], [p, 1 - p]))), "kl-argument-order")],
        )
    rates = [sp.Integer(1), sp.Integer(2), sp.Integer(3), R(1, 2)]
    if v == 0:
        l1, l2 = rng.sample(rates, 2)
        d = sp.log(l1 / l2) + l2 / l1 - 1
        rev = sp.log(l2 / l1) + l1 / l2 - 1
        return Problem(
            prompt=("Derive $D_{\\mathrm{KL}}(\\mathrm{Exp}(\\lambda_1)\\,\\|\\,\\mathrm{Exp}(\\lambda_2))$ in nats for exponential "
                    f"densities $\\lambda e^{{-\\lambda x}}$ on $x \\ge 0$, then evaluate it for $\\lambda_1 = {L(l1)}$, $\\lambda_2 = {L(l2)}$."),
            answer=fmt(d), kind="num", tol=TOL, answer_hint=NATS_HINT,
            rubric=["ln(p/q) = ln(λ₁/λ₂) - (λ₁ - λ₂)x", "Take the expectation under Exp(λ₁): E[x] = 1/λ₁",
                    "D = ln(λ₁/λ₂) + λ₂/λ₁ - 1", "Substitute"],
            solution=(f"$D = \\ln\\frac{{\\lambda_1}}{{\\lambda_2}} + \\frac{{\\lambda_2}}{{\\lambda_1}} - 1 = {L(d)} \\approx {d4(d)}$ nats."),
            misconceptions=[(fmt(rev), "kl-argument-order"), (fmt(tidy(d / LN2)), "reported-bits-not-nats")],
        )
    a, b = rng.sample([1, 2, 3, 4], 2)
    d = a * sp.log(R(a, b)) + b - a
    rev = b * sp.log(R(b, a)) + a - b
    return Problem(
        prompt=("Derive $D_{\\mathrm{KL}}(\\mathrm{Poisson}(a)\\,\\|\\,\\mathrm{Poisson}(b))$ in nats (pmf $e^{-\\lambda}\\lambda^k/k!$), "
                f"then evaluate it for $a = {a}$, $b = {b}$."),
        answer=fmt(d), kind="num", tol=TOL, answer_hint=NATS_HINT,
        rubric=["ln(p/q) = k ln(a/b) - (a - b) (the k! cancels)", "Take E under Poisson(a): E[k] = a",
                "D = a ln(a/b) + b - a", "Substitute"],
        solution=(f"$D = a\\ln\\frac{{a}}{{b}} + b - a = {a if a > 1 else ''}\\ln\\frac{{{a}}}{{{b}}} "
                  f"{'+' if b > a else '-'} {abs(b - a)} = {L(d)} \\approx {d4(d)}$ nats."),
        misconceptions=[(fmt(rev), "kl-argument-order"), (fmt(a * sp.log(R(a, b))), "forgot-normaliser-difference")],
    )


@gen("info.mutual_information")
def mutual_information_scenarios(rng, level):
    v = rng.randrange(2)
    if level == 1:
        if v == 0:
            hy = rng.choice([R(1), R(3, 2), R(7, 4), R(2)])
            b = rng.choice([t for t in (R(1, 4), R(1, 2), R(3, 4), R(1)) if t < hy])
            return Problem(
                prompt=(f"Before seeing $X$, the uncertainty about $Y$ is $H(Y) = {L(hy)}$ bits; after seeing $X$ it drops, on "
                        f"average, to $H(Y\\mid X) = {L(b)}$ bits. How much information does $X$ carry about $Y$, i.e. $I(X;Y)$?"),
                answer=fmt(hy - b), kind="num", tol=TOL, answer_hint=BITS_HINT,
                rubric=["I(X;Y) = H(Y) - H(Y|X): the reduction in uncertainty", "Subtract"],
                solution=f"$I(X;Y) = H(Y) - H(Y\\mid X) = {L(hy)} - {L(b)} = {L(hy - b)}$ bits.",
                misconceptions=[(fmt(hy + b), "added-instead-of-subtracted"), (fmt(b), "reported-remaining-uncertainty")],
            )
        k = rng.randint(3, 5)
        ps = dyadic(rng, k)
        h = H(ps)
        return Problem(
            prompt=(f"A noiseless channel delivers its input unchanged, $Y = X$, where $X$ takes {k} symbols with probabilities "
                    f"${tup(ps)}$. Compute $I(X;Y)$ in bits."),
            answer=fmt(h), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Y = X ⇒ H(X|Y) = 0", "I(X;Y) = H(X) - H(X|Y) = H(X)", "Compute H(X)"],
            solution=f"$I(X;X) = H(X) = {L(h)}$ bits — a variable shares all its information with itself.",
            misconceptions=[(fmt(lg(k)), "assumed-uniform"), ("0", "thought-identical-means-zero")],
        )
    if level == 2:
        if v == 0:
            q = rng.choice([R(1, 4), R(1, 3), R(1, 5), R(3, 4)])
            e = rng.choice([R(1, 4), R(1, 3), R(1, 2), R(1, 5)])
            i = tidy((1 - e) * Hb(q))
            return Problem(
                prompt=(f"Bits $X \\sim \\text{{Bernoulli}}({L(q)})$ are sent over a binary erasure channel that replaces each bit "
                        f"by '?' with probability $\\varepsilon = {L(e)}$ (otherwise it arrives intact). Compute $I(X;Y)$ in bits."),
                answer=fmt(i), kind="num", tol=TOL, answer_hint=BITS_HINT,
                rubric=["I(X;Y) = H(X) - H(X|Y)", "Y = ? (prob ε) leaves X as uncertain as before: H(X|Y=?) = H(X)",
                        "Y = 0 or 1 reveals X: conditional entropy 0", "I = (1-ε) H_b(q)"],
                solution=(f"$H(X\\mid Y) = \\varepsilon H_b(q)$, so $I = (1-\\varepsilon)H_b({L(q)}) = {L(1 - e)}\\cdot H_b({L(q)}) "
                          f"= {L(i)} \\approx {d4(i)}$ bits."),
                misconceptions=[(fmt(1 - e), "assumed-uniform-input"), (fmt(Hb(q)), "ignored-erasures")],
            )
        while True:
            a, b = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(2, 3)]), rng.choice([R(1, 2), R(1, 3), R(1, 4), R(2, 3)])
            op = rng.choice(["AND", "OR"])
            pz = a * b if op == "AND" else 1 - (1 - a) * (1 - b)
            hzx = a * Hb(b) if op == "AND" else (1 - a) * Hb(b)
            i = tidy(Hb(pz) - hzx)
            if float(i) > 0.02:
                break
        sym = "\\wedge" if op == "AND" else "\\vee"
        return Problem(
            prompt=(f"$X$ and $Y$ are independent bits with $P(X=1) = {L(a)}$ and $P(Y=1) = {L(b)}$. A logic gate outputs "
                    f"$Z = X {sym} Y$ (logical {op}). Compute $I(X;Z)$ in bits."),
            answer=fmt(i), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=[f"P(Z=1) = {pz}", "I(X;Z) = H(Z) - H(Z|X)",
                    f"Given X = {0 if op == 'AND' else 1} the output is fixed; otherwise Z = Y", "Evaluate"],
            solution=(f"$H(Z) = H_b({L(pz)})$, $H(Z\\mid X) = {L(a if op == 'AND' else 1 - a)}\\,H_b({L(b)})$; "
                      f"$I(X;Z) = {L(i)} \\approx {d4(i)}$ bits."),
            misconceptions=[(fmt(Hb(pz)), "forgot-subtract-conditional"), (fmt(tidy(hzx)), "reported-conditional-entropy")],
        )
    if v == 0:  # data-processing inequality
        p1 = rng.choice([R(1, 10), R(1, 8), R(1, 5), R(1, 4)])
        p2 = rng.choice([R(1, 10), R(1, 5), R(1, 4), R(1, 3)])
        pc = p1 * (1 - p2) + p2 * (1 - p1)
        loss = tidy(Hb(pc) - Hb(p1))
        return Problem(
            prompt=(f"Markov chain $X \\to Y \\to Z$: a uniform bit $X$ passes through a BSC with crossover ${L(p1)}$ to give $Y$, "
                    f"and $Y$ through a second BSC with crossover ${L(p2)}$ to give $Z$. Compute $I(X;Y) - I(X;Z)$ in bits — "
                    "the information destroyed by the second stage (the data-processing inequality says it is $\\ge 0$)."),
            answer=fmt(loss), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Uniform input ⇒ I(X;Y) = 1 - H_b(p₁)", "X → Z is a BSC with p = p₁(1-p₂) + p₂(1-p₁)",
                    "I(X;Z) = 1 - H_b(p)", "Subtract"],
            solution=(f"$p = {L(pc)}$; $I(X;Y) - I(X;Z) = H_b({L(pc)}) - H_b({L(p1)}) = {L(loss)} \\approx {d4(loss)}$ bits."),
            misconceptions=[(fmt(Hb(p2)), "subtracted-second-channel-entropy"), (fmt(tidy(1 - Hb(pc))), "reported-IXZ")],
        )
    a, b = rng.choice([(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4)])
    counts = [sum(1 for x in range(a) for t in range(b) if x + t == y) for y in range(a + b - 1)]
    hy = H([R(c, a * b) for c in counts])
    i = tidy(hy - lg(b))
    return Problem(
        prompt=(f"$X$ is uniform on $\\{{{', '.join(map(str, range(a)))}\\}}$ and independent noise $N$, uniform on "
                f"$\\{{{', '.join(map(str, range(b)))}\\}}$, "
                "is added: $Y = X + N$. Derive $I(X;Y)$ in bits."),
        answer=fmt(i), kind="num", tol=TOL, answer_hint=BITS_HINT,
        rubric=["I(X;Y) = H(Y) - H(Y|X)", f"Given X, Y is a shifted copy of N ⇒ H(Y|X) = log₂{b}",
                "Distribution of Y by counting pairs (x, n)", "Subtract"],
        solution=(f"Counts of $Y = 0..{a + b - 2}$: ${counts}$ out of ${a * b}$, so $H(Y) \\approx {d4(hy)}$; "
                  f"$I = H(Y) - \\log_2 {b} = {L(i)} \\approx {d4(i)}$ bits."),
        misconceptions=[(fmt(lg(a)), "mi-equals-HX"), (fmt(hy), "forgot-subtract-noise-entropy")],
    )


@gen("info.cross_entropy")
def cross_entropy_scenarios(rng, level):
    if level == 1:  # average log-loss over a few examples, in bits
        n = 4
        ls = [rng.randint(0, 4) for _ in range(n)]
        if sum(ls) == 0:
            ls[0] = 2
        ps = [R(1, 2 ** l) for l in ls]
        avg = R(sum(ls), n)
        return Problem(
            prompt=(f"On {n} test examples a classifier assigns probabilities ${tup(ps)}$ to the **correct** class. "
                    "Compute its average cross-entropy (log-loss) in **bits** per example."),
            answer=fmt(avg), kind="num", answer_hint="an exact fraction, e.g. 7/4",
            rubric=["Per-example loss = log₂(1/p_correct)", "Each term is an integer here", f"Average over {n} examples"],
            solution=(f"Losses ${', '.join(map(str, ls))}$ bits; average $= {sum(ls)}/{n} = {L(avg)}$ bits."),
            misconceptions=[(fmt(sum(ls)), "summed-not-averaged"), (fmt(avg * LN2), "used-natural-log")],
        )
    if level == 2:  # label smoothing
        k = rng.choice([3, 4])
        eps = rng.choice([R(1, 10), R(1, 5)])
        q, _, _ = count_dist(rng, k, 1, 7)
        c = rng.randrange(k)
        t = [1 - eps if i == c else eps / (k - 1) for i in range(k)]
        ce = sum(-ti * sp.log(qi) for ti, qi in zip(t, q))
        return Problem(
            prompt=(f"Training with **label smoothing** $\\varepsilon = {L(eps)}$: for a {k}-class example whose true class is "
                    f"{c + 1}, the target puts $1-\\varepsilon$ on class {c + 1} and spreads $\\varepsilon$ equally over the other "
                    f"classes. The model predicts $\\hat q = {tup(q)}$. Compute the cross-entropy $-\\sum_i t_i \\ln \\hat q_i$ in nats."),
            answer=fmt(ce), kind="num", tol=TOL, answer_hint=NATS_HINT,
            rubric=["Build the smoothed target t", f"t(true) = 1 - ε, t(other) = ε/{k - 1}", "Sum -t ln q̂ over all classes"],
            solution=(f"$t = {tup(t)}$; $H(t, \\hat q) = " + " + ".join(f"{L(ti)}\\ln{L(1 / qi)}" for ti, qi in zip(t, q)) +
                      f" \\approx {d4(ce)}$ nats."),
            misconceptions=[(fmt(-sp.log(q[c])), "ignored-label-smoothing"), (fmt(tidy(ce / LN2)), "used-log2")],
        )
    while True:
        a = [rng.randint(1, 6) for _ in range(3)]
        if len(set(a)) > 1:
            break
    y = rng.randrange(3)
    S = sum(a)
    loss = sp.log(R(S, a[y]))
    return Problem(
        prompt=(f"Show that the cross-entropy of $p = \\mathrm{{softmax}}(z)$ against a one-hot target for class $y$ equals "
                f"$\\log\\sum_j e^{{z_j}} - z_y$ (the 'log-sum-exp' form). Then evaluate it in nats for "
                f"$z = (\\ln {a[0]},\\ \\ln {a[1]},\\ \\ln {a[2]})$ and $y = {y + 1}$."),
        answer=fmt(loss), kind="num", tol=TOL, answer_hint="exact like log(9/2), or a decimal to 4 places",
        rubric=["Loss = -ln p_y = -ln(e^{z_y} / Σ e^{z_j})", "= log Σ e^{z_j} - z_y", f"e^{{ln a}} = a ⇒ Σ = {S}",
                f"Loss = ln({S}/{a[y]})"],
        solution=(f"$-\\ln\\frac{{e^{{z_y}}}}{{\\sum_j e^{{z_j}}}} = \\ln\\sum_j e^{{z_j}} - z_y = \\ln {S} - \\ln {a[y]} = {L(loss)} "
                  f"\\approx {d4(loss)}$ nats."),
        misconceptions=[(fmt(sp.log(S)), "forgot-subtract-true-logit"), (fmt(tidy(loss / LN2)), "used-log2")],
    )


CODES = [{"a": "0", "b": "10", "c": "110", "d": "111"},
         {"a": "00", "b": "01", "c": "10", "d": "110", "e": "111"},
         {"a": "1", "b": "01", "c": "001", "d": "000"},
         {"a": "10", "b": "11", "c": "00", "d": "010", "e": "011"}]


@gen("info.source_coding")
def source_coding_scenarios(rng, level):
    if level == 1:  # decode with a prefix code
        code = rng.choice(CODES)
        syms = sorted(code)
        msg = "".join(rng.choice(syms) for _ in range(rng.randint(5, 7)))
        bits = "".join(code[s] for s in msg)
        table = ", ".join(f"${s} \\to \\texttt{{{code[s]}}}$" for s in syms)
        return Problem(
            prompt=(f"A prefix-free code is {table}. Decode the bit string $\\texttt{{{bits}}}$."),
            answer=msg, kind="text", answer_hint="the decoded letters with no spaces, e.g. abca",
            rubric=["Read bits from the left", "Prefix-free ⇒ the first codeword that matches is the right one",
                    "Restart after each decoded symbol"],
            solution=("Splitting: " + " | ".join(f"$\\texttt{{{code[s]}}}$" for s in msg) + f" → **{msg}**."),
        )
    if level == 2:  # Shannon code expected length
        k = rng.choice([3, 4, 5])
        ps, c, N = count_dist(rng, k, 1, 9)
        ls = []
        for p in ps:
            l = 0
            while 2 ** l * p < 1:
                l += 1
            ls.append(l)
        Lbar = sum(p * l for p, l in zip(ps, ls))
        h = H(ps)
        lh, _ = huffman(ps)
        mis = [(fmt(h), "entropy-not-achievable")]
        if lh != Lbar:
            mis.append((fmt(lh), "built-huffman-instead"))
        return Problem(
            prompt=(f"A **Shannon code** gives symbol $x$ a codeword of length $\\ell(x) = \\lceil \\log_2 1/p(x) \\rceil$. "
                    f"For probabilities ${tup(ps)}$ (symbols ${', '.join(SYMS[:k])}$), compute the expected codeword length."),
            answer=fmt(Lbar), kind="num", answer_hint="exact fraction, e.g. 37/16",
            rubric=["For each p find the smallest ℓ with 2^ℓ ≥ 1/p", "Expected length Σ p ℓ",
                    "Check H(X) ≤ L̄ < H(X) + 1"],
            solution=(f"Lengths ${ls}$; $\\bar L = " + " + ".join(f"{L(p)}\\cdot{l}" for p, l in zip(ps, ls)) +
                      f" = {L(Lbar)} \\approx {float(Lbar):.4f}$ (entropy $\\approx {d4(h)}$)."),
            misconceptions=mis,
        )
    p = rng.choice([R(1, 3), R(1, 4), R(1, 5), R(2, 3), R(3, 4)])
    red = tidy((1 - Hb(p)) / p)
    return Problem(
        prompt=(f"$X \\sim \\mathrm{{Geometric}}({L(p)})$ on $\\{{1, 2, \\dots\\}}$, i.e. $P(X=k) = (1-p)^{{k-1}}p$, is encoded with "
                "the **unary** code ($k-1$ ones followed by a zero, length $k$). Derive the expected codeword length and the "
                "redundancy $\\mathbb E[\\ell] - H(X)$; give the redundancy in bits/symbol."),
        answer=fmt(red), kind="num", tol=TOL, answer_hint=BITS_HINT,
        rubric=["E[ℓ] = E[X] = 1/p", "H(X) = H_b(p)/p (geometric entropy)", "Redundancy = (1 - H_b(p))/p",
                "Evaluate (zero only for p = 1/2)"],
        solution=(f"$\\mathbb E[\\ell] = 1/p = {L(1 / p)}$, $H(X) = H_b(p)/p \\approx {d4(Hb(p) / p)}$; redundancy "
                  f"$= \\frac{{1 - H_b(p)}}{{p}} = {L(red)} \\approx {d4(red)}$ bits."),
        misconceptions=[(fmt(1 / p), "forgot-subtract-entropy"), (fmt(tidy(1 - Hb(p))), "forgot-divide-by-p")],
    )


@gen("info.channel_capacity")
def channel_capacity_scenarios(rng, level):
    v = rng.randrange(2)
    if level == 1:
        if v == 0:  # Shannon-Hartley
            B = rng.choice([1, 2, 3, 4, 5, 8, 10])
            k = rng.randint(2, 8)
            snr = 2 ** k - 1
            return Problem(
                prompt=(f"A line has bandwidth $B = {B}$ kHz and signal-to-noise ratio $S/N = {snr}$ (linear, not dB). "
                        "Using the Shannon–Hartley formula $C = B\\log_2(1 + S/N)$, give its capacity in kbit/s."),
                answer=str(B * k), kind="num", tol=TOL, answer_hint="a number in kbit/s, e.g. 12",
                rubric=["1 + S/N", "log₂ of it (a power of two here)", "Multiply by B"],
                solution=f"$C = {B}\\log_2(1 + {snr}) = {B}\\log_2 {snr + 1} = {B}\\cdot {k} = {B * k}$ kbit/s.",
                misconceptions=[(fmt(B * lg(snr)), "forgot-plus-one"), (fmt(B * sp.log(snr + 1)), "used-natural-log")],
            )
        m = rng.choice([2, 3, 4, 5, 8])
        n = m * rng.choice([2, 3, 4])
        return Problem(
            prompt=(f"A deterministic channel maps each input $x \\in \\{{0, 1, \\dots, {n - 1}\\}}$ to the output "
                    f"$y = x \\bmod {m}$ (no noise at all). What is its capacity in bits per use?"),
            answer=fmt(lg(m)), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["Deterministic ⇒ H(Y|X) = 0, so I(X;Y) = H(Y)", f"Y takes {m} values ⇒ H(Y) ≤ log₂{m}",
                    "Achieved by inputs that make Y uniform"],
            solution=f"$C = \\max H(Y) = \\log_2 {m} = {L(lg(m))}$ bits/use — only distinct *outputs* count.",
            misconceptions=[(fmt(lg(n)), "counted-inputs-not-outputs")],
        )
    if level == 2:
        if v == 0:  # required SNR
            B = rng.choice([2, 3, 4, 5, 10])
            k = rng.randint(2, 8)
            return Problem(
                prompt=(f"You must send ${B * k}$ kbit/s over a channel of bandwidth ${B}$ kHz. By the Shannon–Hartley theorem, "
                        "what is the minimum signal-to-noise ratio $S/N$ (linear, not dB) that makes this possible?"),
                answer=str(2 ** k - 1), kind="num", answer_hint="a number, e.g. 63",
                rubric=["Require R ≤ B log₂(1 + S/N)", f"log₂(1 + S/N) ≥ R/B = {k}", f"S/N ≥ 2^{k} - 1"],
                solution=f"$\\log_2(1 + S/N) \\ge {B * k}/{B} = {k} \\Rightarrow S/N \\ge 2^{{{k}}} - 1 = {2 ** k - 1}$.",
                misconceptions=[(str(2 ** k), "forgot-minus-one"), (str(k), "forgot-exponentiate")],
            )
        n = rng.choice([4, 6, 8, 10, 12, 26])
        c = lg(R(n, 2))
        return Problem(
            prompt=(f"**Noisy typewriter**: the channel has {n} input letters arranged in a circle. Each typed letter is received "
                    "either as itself or as the next letter (cyclically), each with probability $\\tfrac12$. "
                    "Compute the capacity in bits per use."),
            answer=fmt(c), kind="num", tol=TOL, answer_hint=BITS_HINT,
            rubric=["H(Y|X) = 1 bit for every input", "Uniform input ⇒ uniform output: H(Y) = log₂ n",
                    "C = log₂ n - 1", f"(Achievable with zero error by using every other letter: {n // 2} letters)"],
            solution=f"$C = \\max H(Y) - H(Y\\mid X) = \\log_2 {n} - 1 = {L(c)} \\approx {d4(c)}$ bits/use.",
            misconceptions=[(fmt(lg(n)), "ignored-noise"), (fmt(lg(n) / 2), "halved-capacity")],
        )
    if v == 0:  # Z-channel
        p = rng.choice([R(1, 2), R(1, 3), R(1, 4)])
        c = sp.log(1 + (1 - p) * p ** (p / (1 - p))) / LN2
        q = R(1, 2)
        iu = tidy(Hb(q * (1 - p)) - q * Hb(p))
        return Problem(
            prompt=(f"**Z-channel**: input 0 is always received as 0; input 1 is received as 1 with probability ${L(1 - p)}$ and "
                    f"as 0 with probability $p = {L(p)}$. With $q = P(X=1)$, $I(X;Y) = H_b(q(1-p)) - q\\,H_b(p)$. Maximise over $q$ "
                    "to find the capacity (bits/use)."),
            answer=fmt(c), kind="num", tol=TOL, answer_hint="exact like log(5/4, 2), or a decimal to 4 places",
            rubric=["d/dq: (1-p) log₂((1 - q(1-p))/(q(1-p))) = H_b(p)", "Solve: q(1-p) = 1/(1 + 2^{H_b(p)/(1-p)})",
                    "Substitute back: C = log₂(1 + 2^{-H_b(p)/(1-p)}) = log₂(1 + (1-p)p^{p/(1-p)})", "Evaluate"],
            solution=(f"$C = \\log_2\\!\\left(1 + (1-p)\\,p^{{p/(1-p)}}\\right) = {L(c)} \\approx {d4(c)}$ bits/use "
                      f"(uniform input would give only $\\approx {d4(iu)}$)."),
            misconceptions=[(fmt(tidy(1 - Hb(p))), "used-bsc-formula"), (fmt(iu), "assumed-uniform-input")],
        )
    p = rng.choice([R(1, 10), R(1, 5), R(1, 4), R(1, 3)])
    n = rng.choice([3, 5])
    pe = sum(sp.binomial(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))
    return Problem(
        prompt=(f"Over a BSC with crossover $p = {L(p)}$ each data bit is sent **{n} times** and decoded by majority vote. "
                "Derive the probability that a decoded bit is wrong and evaluate it exactly. "
                f"(On paper: the rate is $1/{n}$ — compare with the capacity $1 - H_b(p)$.)"),
        answer=fmt(pe), kind="num", answer_hint="an exact fraction, e.g. 7/125",
        rubric=[f"Error iff at least {n // 2 + 1} of the {n} copies flip", "Binomial: Σ C(n,k) p^k (1-p)^{n-k}",
                "Evaluate exactly", "Smaller than p, but at rate only 1/n"],
        solution=(f"$P_e = \\sum_{{k={n // 2 + 1}}}^{{{n}}} \\binom{{{n}}}{{k}} p^k(1-p)^{{{n}-k}} = {L(pe)} \\approx {d4(pe)}$ "
                  f"(vs $p = {L(p)}$ uncoded)."),
        misconceptions=[(fmt(p ** n), "only-all-copies-flipped"), (fmt(p), "no-coding-gain")],
    )


VOCAB = ["the", "cat", "sat", "on", "mat", "dog"]


@gen("info.perplexity")
def perplexity_scenarios(rng, level):
    v = rng.randrange(2)
    if level == 1:
        if v == 0:
            h = R(rng.randint(4, 20), 2)
            ppl = sp.Integer(2) ** h
            return Problem(
                prompt=(f"A language model's cross-entropy on held-out text is ${L(h)}$ **bits** per token. "
                        "What is its perplexity?"),
                answer=fmt(ppl), kind="num", tol=TOL, answer_hint="exact like 2^(11/2), or a decimal to 3 places",
                rubric=["Perplexity = 2^{cross-entropy in bits}", "Evaluate"],
                solution=f"$\\mathrm{{PPL}} = 2^{{{L(h)}}} = {L(ppl)} \\approx {float(ppl):.3f}$.",
                misconceptions=[(fmt(sp.exp(h)), "used-e-with-bits"), (fmt(h), "reported-cross-entropy")],
            )
        ppl = rng.choice([8, 16, 20, 32, 50, 100])
        return Problem(
            prompt=(f"A model reports a test perplexity of {ppl}. What is its average cross-entropy loss in **nats** per token?"),
            answer=fmt(sp.log(ppl)), kind="num", tol=TOL, answer_hint="exact like log(20), or a decimal to 4 places",
            rubric=["PPL = e^{loss in nats}", "Invert: loss = ln PPL"],
            solution=f"loss $= \\ln {ppl} \\approx {d4(sp.log(ppl))}$ nats (equivalently $\\log_2 {ppl} \\approx {d4(lg(ppl))}$ bits).",
            misconceptions=[(fmt(lg(ppl)), "gave-bits-not-nats"), (fmt(R(1, ppl)), "inverted-perplexity")],
        )
    if level == 2:
        if v == 0:  # compare two models on one sentence
            n = 4
            while True:
                la = [rng.randint(1, 5) for _ in range(n)]
                lb = [rng.randint(1, 5) for _ in range(n)]
                if sum(la) != sum(lb):
                    break
            pa, pb = sp.Integer(2) ** R(sum(la), n), sp.Integer(2) ** R(sum(lb), n)
            return Problem(
                prompt=(f"Two language models score the same {n}-token test sentence. Model A gives the tokens probabilities "
                        f"${tup([R(1, 2 ** l) for l in la])}$, model B gives ${tup([R(1, 2 ** l) for l in lb])}$. "
                        "Compute both perplexities."),
                answer=[fmt(pa), fmt(pb)], kind="list", tol=TOL, answer_hint="PPL_A, PPL_B   e.g. 4, 2^(5/4)",
                rubric=["Bits per token: log₂(1/p) for each token", "Average over the sentence", "PPL = 2^{average}",
                        "Lower perplexity = better model"],
                solution=(f"A: average ${L(R(sum(la), n))}$ bits ⇒ ${L(pa)} \\approx {float(pa):.3f}$; "
                          f"B: average ${L(R(sum(lb), n))}$ bits ⇒ ${L(pb)} \\approx {float(pb):.3f}$; "
                          f"model {'A' if pa < pb else 'B'} is better."),
                misconceptions=[([fmt(pb), fmt(pa)], "swapped-models"),
                                ([fmt(R(sum(la), n)), fmt(R(sum(lb), n))], "reported-cross-entropy-not-perplexity")],
            )
        N = 16
        k = rng.choice([3, 4, 5])
        words = rng.sample(VOCAB, k)
        while True:  # a uniform model would make the counts pointless
            ps = dyadic(rng, k)
            if len(set(ps)) > 1:
                break
        cnt = [int(p * N) for p in ps]
        sent = [rng.randrange(k) for _ in range(4)]
        bits = R(sum(-lg(ps[i]) for i in sent), 4)
        ppl = sp.Integer(2) ** bits
        return Problem(
            prompt=(f"A unigram model is estimated by counting a corpus of {N} tokens: " +
                    ", ".join(f"*{w}*: {c}" for w, c in zip(words, cnt)) +
                    " (maximum-likelihood probabilities = count / total). Compute its perplexity on the test sentence "
                    "“" + " ".join(words[i] for i in sent) + "”."),
            answer=fmt(ppl), kind="num", tol=TOL, answer_hint="exact like 2^(9/4), or a decimal to 3 places",
            rubric=["p(w) = count/N", "log₂(1/p) for each test token", "Average over the 4 tokens", "PPL = 2^{average}"],
            solution=(f"Token probabilities ${tup([ps[i] for i in sent])}$, average ${L(bits)}$ bits/token, "
                      f"$\\mathrm{{PPL}} = 2^{{{L(bits)}}} \\approx {float(ppl):.3f}$."),
            misconceptions=[(fmt(bits), "reported-cross-entropy-not-perplexity"), (str(k), "used-vocab-size")],
        )
    if v == 0:  # mixture of easy and hard tokens
        f = rng.choice([R(1, 4), R(1, 2), R(3, 4), R(1, 3), R(2, 3)])
        a, b = rng.sample(range(1, 7), 2)
        e = f * a + (1 - f) * b
        ppl = sp.Integer(2) ** e
        return Problem(
            prompt=(f"On a long text a model gives the correct next token probability $2^{{-{a}}}$ on a fraction ${L(f)}$ of "
                    f"positions and $2^{{-{b}}}$ on all others. Derive the long-run perplexity and evaluate it. "
                    "(On paper: why is it **not** the weighted average of the two 'local' perplexities?)"),
            answer=fmt(ppl), kind="num", tol=TOL, answer_hint="exact like 2^(7/2), or a decimal to 3 places",
            rubric=["PPL = 2^{average bits per token}", f"Average bits = {f}·{a} + (1-{f})·{b}", "Exponentiate",
                    "Geometric, not arithmetic, mean of 1/p"],
            solution=(f"Average bits $= {L(f)}\\cdot{a} + {L(1 - f)}\\cdot{b} = {L(e)}$, so $\\mathrm{{PPL}} = 2^{{{L(e)}}} "
                      f"\\approx {float(ppl):.3f}$ (the arithmetic mix would give ${L(f * 2 ** a + (1 - f) * 2 ** b)}$)."),
            misconceptions=[(fmt(f * 2 ** a + (1 - f) * 2 ** b), "averaged-perplexities")],
        )
    na = rng.choice([6, 8, 10, 12])
    nb = rng.choice([na * 3 // 2, 2 * na]) if na % 2 == 0 else 2 * na
    T = (na * nb // sp.gcd(na, nb)) * rng.choice([1, 2])
    pa, pb = sp.Integer(2) ** R(T, na), sp.Integer(2) ** R(T, nb)
    return Problem(
        prompt=(f"Two models assign the same sentence the same total probability $2^{{-{T}}}$, i.e. ${T}$ bits of log-loss. "
                f"Model A's tokenizer splits the sentence into {na} tokens, model B's into {nb} tokens. Derive the per-token "
                "perplexity each model reports. (On paper: why should such models be compared in bits per *character*?)"),
        answer=[fmt(pa), fmt(pb)], kind="list", tol=TOL, answer_hint="PPL_A, PPL_B   e.g. 8, 4",
        rubric=["Per-token cross-entropy = total bits / #tokens", f"A: {T}/{na}, B: {T}/{nb}", "PPL = 2^{bits per token}",
                "Same model quality, different perplexities: normalise by characters instead"],
        solution=(f"A: $2^{{{T}/{na}}} = {L(pa)}$; B: $2^{{{T}/{nb}}} = {L(pb)}$. The finer tokenizer looks 'better' "
                  "although the sentence probability is identical."),
        misconceptions=[([fmt(pb), fmt(pa)], "swapped-models"), ([fmt(R(T, na)), fmt(R(T, nb))], "reported-bits-not-perplexity")],
    )


@gen("info.data_eda")
def data_eda_scenarios(rng, level):
    v = rng.randrange(2)
    if level == 1:
        if v == 0:
            n = rng.choice([6, 7])
            while True:
                data = [rng.randint(1, 20) for _ in range(n)]
                m, med = R(sum(data), n), _median(data)
                if m != med:
                    break
            return Problem(
                prompt=f"Data (unsorted): ${data}$. Give the mean and the median.",
                answer=[fmt(m), fmt(med)], kind="list", answer_hint="mean, median   e.g. 23/3, 8",
                rubric=["Mean = sum / n", "Sort the data", "Median = middle value (average of the two middle ones if n is even)"],
                solution=f"Sorted ${sorted(data)}$; mean $= {sum(data)}/{n} = {L(m)}$, median $= {L(med)}$.",
                misconceptions=[([fmt(med), fmt(m)], "swapped-mean-median"),
                                ([fmt(m), fmt(data[n // 2] if n % 2 else R(data[n // 2 - 1] + data[n // 2], 2))],
                                 "median-of-unsorted-data")],
            )
        mu = rng.choice([50, 60, 70, 100, 500])
        s = rng.choice([4, 5, 8, 10, 12, 15, 20])
        x = mu + s * rng.choice([-3, -2, -1, 1, 2, 3]) + rng.choice([0, s // 2])
        z = R(x - mu, s)
        return Problem(
            prompt=(f"Exam scores have mean ${mu}$ and standard deviation ${s}$. A student scores ${x}$. "
                    "Compute the student's $z$-score (how many standard deviations from the mean)."),
            answer=fmt(z), kind="num", answer_hint="exact fraction or decimal, e.g. -3/2",
            rubric=["z = (x - mean) / sd", "Keep the sign"],
            solution=f"$z = ({x} - {mu})/{s} = {L(z)}$.",
            misconceptions=[(fmt(R(x - mu, s * s)), "divided-by-variance"), (fmt(-z), "sign-error")],
        )
    if level == 2:
        if v == 0:  # frequency table
            vals = sorted(rng.sample(range(0, 10), 4))
            cnt = [rng.randint(1, 6) for _ in vals]
            N = sum(cnt)
            m = R(sum(a * c for a, c in zip(vals, cnt)), N)
            var = R(sum(c * a * a for a, c in zip(vals, cnt)), N) - m ** 2
            tab = ("$$\\begin{array}{c|" + "c" * 4 + "} \\text{value} & " + " & ".join(map(str, vals)) +
                   " \\\\ \\hline \\text{count} & " + " & ".join(map(str, cnt)) + "\\end{array}$$")
            return Problem(
                prompt=("A survey's answers are summarised in a frequency table:\n" + tab +
                        f"\nCompute the mean and the population variance (divide by the total count $N = {N}$)."),
                answer=[fmt(m), fmt(var)], kind="list", answer_hint="mean, variance   e.g. 7/2, 5/4",
                rubric=["N = Σ counts", "Mean = Σ count·value / N", "Variance = Σ count·value² / N - mean²",
                        "(or Σ count·(value - mean)² / N)"],
                solution=(f"$\\bar x = {sum(a * c for a, c in zip(vals, cnt))}/{N} = {L(m)}$; "
                          f"$\\overline{{x^2}} = {sum(c * a * a for a, c in zip(vals, cnt))}/{N}$, variance $= {L(var)}$."),
                misconceptions=[([fmt(R(sum(vals), 4)), fmt(R(sum(a * a for a in vals), 4) - R(sum(vals), 4) ** 2)],
                                 "ignored-frequencies"),
                                ([fmt(m), fmt(var * N / (N - 1))], "divided-by-n-minus-1")],
            )
        m = rng.randint(-5, 30)
        s = rng.choice([2, 3, 4, 5, 10])
        mf, varf = R(9, 5) * m + 32, R(81, 25) * s ** 2
        return Problem(
            prompt=(f"Daily temperatures in °C have mean ${m}$ and standard deviation ${s}$. After converting to Fahrenheit, "
                    "$F = \\tfrac95 C + 32$, what are the mean and the **variance** of the Fahrenheit temperatures?"),
            answer=[fmt(mf), fmt(varf)], kind="list", answer_hint="mean, variance   e.g. 77, 81",
            rubric=["E[aX + b] = aE[X] + b", "Var(aX + b) = a² Var(X): shifts don't change spread", "Var(C) = sd²"],
            solution=(f"Mean $= \\tfrac95\\cdot{m} + 32 = {L(mf)}$; variance $= \\left(\\tfrac95\\right)^2\\cdot{s}^2 = {L(varf)}$."),
            misconceptions=[([fmt(mf), fmt(R(9, 5) * s ** 2)], "variance-scaled-by-a-not-a-squared"),
                            ([fmt(mf), fmt(varf + 32)], "shift-changed-spread")],
        )
    if v == 0:  # combine two groups
        while True:
            n1, n2 = rng.choice([2, 3, 4, 5]), rng.choice([2, 3, 4, 5])
            m1, m2 = rng.randint(0, 10), rng.randint(0, 10)
            v1, v2 = rng.randint(1, 9), rng.randint(1, 9)
            if m1 != m2:
                break
        N = n1 + n2
        M = R(n1 * m1 + n2 * m2, N)
        var = R(n1 * (v1 + m1 ** 2) + n2 * (v2 + m2 ** 2), N) - M ** 2
        return Problem(
            prompt=(f"Group A has {n1}0 values with mean ${m1}$ and (population) variance ${v1}$; group B has {n2}0 values with "
                    f"mean ${m2}$ and variance ${v2}$. Derive the mean and the population variance of the combined data, and "
                    "give the combined variance."),
            answer=fmt(var), kind="num", answer_hint="an exact fraction, e.g. 74/9",
            rubric=["Combined mean = weighted mean of group means", "E[X²] per group = variance + mean²",
                    "Combined E[X²] = weighted average", "Variance = E[X²] - mean² (= within + between)"],
            solution=(f"$\\bar x = {L(M)}$; within-group part ${L(R(n1 * v1 + n2 * v2, N))}$, between-group part "
                      f"${L(R(n1, N) * (m1 - M) ** 2 + R(n2, N) * (m2 - M) ** 2)}$; "
                      f"total variance $= {L(var)}$."),
            misconceptions=[(fmt(R(n1 * v1 + n2 * v2, N)), "ignored-between-group-variance"),
                            (fmt(R(v1 + v2, 2)), "unweighted-average-of-variances")],
        )
    n = rng.choice([5, 7])
    while True:
        data = sorted(rng.randint(0, 15) for _ in range(n))
        m, med = R(sum(data), n), data[n // 2]
        if m != med:
            break
    sse = sum((x - m) ** 2 for x in data)
    sae = sum(abs(x - med) for x in data)
    sae_mean = sum(abs(x - m) for x in data)
    return Problem(
        prompt=(f"For the data ${data}$, find the minimum over $c$ of $\\sum_i (x_i - c)^2$ and the minimum over $c$ of "
                "$\\sum_i |x_i - c|$. (Derive which $c$ achieves each.)"),
        answer=[fmt(sse), fmt(sae)], kind="list", answer_hint="min squared loss, min absolute loss   e.g. 46/5, 12",
        rubric=["d/dc Σ(x-c)² = 0 ⇒ c = mean", "Σ|x-c| has slope (#below - #above) ⇒ c = median",
                "Evaluate both losses at their minimisers"],
        solution=(f"Squared loss: $c = \\bar x = {L(m)}$, minimum $= {L(sse)}$. Absolute loss: $c = \\text{{median}} = {med}$, "
                  f"minimum $= {sae}$."),
        misconceptions=[([fmt(sse), fmt(sae_mean)], "used-mean-for-absolute-loss"),
                        ([fmt(sum((x - med) ** 2 for x in data)), fmt(sae)], "used-median-for-squared-loss")],
    )


CI_COMBOS = [(R(1, 5), 100), (R(1, 2), 100), (R(1, 10), 900), (R(1, 5), 400), (R(1, 2), 400), (R(1, 10), 100),
             (R(1, 2), 2500)]
MDE_COMBOS = [(R(1, 2), 200), (R(1, 2), 800), (R(1, 5), 800), (R(1, 5), 200), (R(1, 10), 200), (R(1, 10), 1800),
              (R(1, 2), 50)]


@gen("info.ab_testing")
def ab_testing_scenarios(rng, level):
    v = rng.randrange(2)
    if level == 1:
        if v == 0:  # relative lift
            n = rng.choice([200, 400, 500, 1000])
            while True:
                xa, xb = rng.randint(n // 20, n // 5), rng.randint(n // 20, n // 4)
                if xa != xb:
                    break
            pa, pb = R(xa, n), R(xb, n)
            lift = (pb - pa) / pa
            return Problem(
                prompt=(f"Variant A converts {xa} of {n} visitors and variant B converts {xb} of {n}. What is B's **relative "
                        "lift** over A, $(\\hat p_B - \\hat p_A)/\\hat p_A$?"),
                answer=fmt(lift), kind="num", answer_hint="exact fraction or decimal, e.g. 1/4 for +25%",
                rubric=["Conversion rates p̂ = conversions / visitors", "Absolute difference p̂_B - p̂_A",
                        "Divide by the baseline p̂_A"],
                solution=f"$\\hat p_A = {L(pa)}$, $\\hat p_B = {L(pb)}$, lift $= {L(pb - pa)}/{L(pa)} = {L(lift)}$ "
                         f"($\\approx {float(lift) * 100:.1f}\\%$).",
                misconceptions=[(fmt(pb - pa), "reported-absolute-difference"), (fmt((pb - pa) / pb), "divided-by-treatment-rate")],
            )
        p, n = rng.choice(CI_COMBOS)
        h = R(196, 100) * sp.sqrt(p * (1 - p) / n)
        se = sp.sqrt(p * (1 - p) / n)
        return Problem(
            prompt=(f"Of {n} users, {p * n} converted. Give the 95% normal-approximation (Wald) confidence interval "
                    "$\\hat p \\pm 1.96\\sqrt{\\hat p(1-\\hat p)/n}$ for the conversion rate."),
            answer=[fmt(p - h), fmt(p + h)], kind="list", answer_hint="lower, upper   e.g. 0.1216, 0.2784",
            rubric=["p̂ = conversions / n", "SE = sqrt(p̂(1-p̂)/n)", "Half-width = 1.96·SE", "p̂ ∓ half-width"],
            solution=(f"$\\hat p = {L(p)}$, $\\mathrm{{SE}} = \\sqrt{{{L(p * (1 - p) / n)}}} = {L(se)}$, half-width "
                      f"$= {float(h):.4f}$, CI $= [{float(p - h):.4f},\\ {float(p + h):.4f}]$."),
            misconceptions=[([fmt(p - se), fmt(p + se)], "forgot-1.96"),
                            ([fmt(p - R(196, 100) * p * (1 - p) / n), fmt(p + R(196, 100) * p * (1 - p) / n)],
                             "forgot-square-root")],
        )
    if level == 2:
        if v == 0:  # multiple testing
            m = rng.choice([3, 4, 5, 6, 10, 20])
            fwer = 1 - R(95, 100) ** m
            return Problem(
                prompt=(f"A team tests {m} independent variants against control, each at significance level $\\alpha = 0.05$, "
                        "and none of them truly differs. (a) What is the probability of at least one false positive? "
                        "(b) What per-test level does the Bonferroni correction prescribe to keep that below 0.05?"),
                answer=[fmt(fwer), fmt(R(5, 100 * m))], kind="list", tol=TOL,
                answer_hint="FWER, per-test alpha   e.g. 0.2262, 0.01",
                rubric=["P(no false positive) = 0.95^m (independence)", "FWER = 1 - 0.95^m", "Bonferroni: α/m"],
                solution=f"(a) $1 - 0.95^{{{m}}} \\approx {float(fwer):.4f}$; (b) $0.05/{m} = {L(R(5, 100 * m))}$.",
                misconceptions=[([fmt(R(5 * m, 100)), fmt(R(5, 100 * m))], "added-error-rates"),
                                ([fmt(fwer), fmt(R(5, 100))], "no-correction")],
            )
        N = rng.choice([1000, 2000, 4000, 5000, 10000])
        d = rng.randint(N // 200, N // 40)
        na, nb = N // 2 + d, N // 2 - d
        chi = R((na - nb) ** 2, N)
        return Problem(
            prompt=(f"An experiment should split traffic 50/50, but variant A received {na} users and B received {nb}. "
                    "Compute the chi-square statistic $\\sum (O - E)^2/E$ for this **sample-ratio mismatch** check "
                    "(on paper: compare with the 5% critical value 3.84 for 1 degree of freedom)."),
            answer=fmt(chi), kind="num", tol=TOL, answer_hint="exact or a decimal to 2 places, e.g. 6.4",
            rubric=[f"Expected counts E = {N // 2} each", "(O - E)²/E for both groups", "Sum",
                    "> 3.84 ⇒ the split is broken: don't trust the test"],
            solution=(f"$\\chi^2 = 2\\cdot\\frac{{{d}^2}}{{{N // 2}}} = {L(chi)} \\approx {float(chi):.2f}$ → "
                      f"{'mismatch (investigate!)' if chi > R(384, 100) else 'consistent with 50/50'}."),
            misconceptions=[(fmt(R(d * d, N // 2)), "only-one-group"), (fmt(R(na - nb, N // 2)), "forgot-to-square")],
        )
    if v == 0:  # minimum detectable effect
        p, n = rng.choice(MDE_COMBOS)
        delta = R(28, 10) * sp.sqrt(2 * p * (1 - p) / n)
        return Problem(
            prompt=(f"An A/B test will have $n = {n}$ users per group and a baseline conversion rate $p = {L(p)}$. Starting from "
                    "$n = (z_{0.025} + z_{0.2})^2\\cdot 2p(1-p)/\\delta^2$ (two-sided 5% test, 80% power, both groups with "
                    "variance $p(1-p)$, $z_{0.025} = 1.96$, $z_{0.2} = 0.84$), derive the **minimum detectable effect** $\\delta$."),
            answer=fmt(delta), kind="num", answer_hint="exact decimal or fraction, e.g. 0.056",
            rubric=["Solve for δ: δ = (z_{0.025} + z_{0.2})·sqrt(2p(1-p)/n)", "1.96 + 0.84 = 2.8",
                    "Evaluate the square root (a perfect square here)", "Multiply"],
            solution=(f"$\\delta = 2.8\\sqrt{{2\\cdot{L(p)}\\cdot{L(1 - p)}/{n}}} = 2.8\\cdot{L(sp.sqrt(2 * p * (1 - p) / n))} = {L(delta)}$ "
                      f"$= {float(delta):g}$ (absolute)."),
            misconceptions=[(fmt(R(28, 10) * sp.sqrt(p * (1 - p) / n)), "forgot-two-groups"),
                            (fmt(R(196, 100) * sp.sqrt(2 * p * (1 - p) / n)), "ignored-power")],
        )
    while True:
        na, nb = rng.randint(10, 60), rng.randint(10, 60)
        xa, xb = rng.randint(1, na - 1), rng.randint(1, nb - 1)
        diff = R(xb + 1, nb + 2) - R(xa + 1, na + 2)
        mle = R(xb, nb) - R(xa, na)
        if diff != 0 and diff != mle:
            break
    return Problem(
        prompt=(f"Bayesian A/B test with uniform $\\mathrm{{Beta}}(1,1)$ priors on both conversion rates. A converted {xa} of "
                f"{na} users, B converted {xb} of {nb}. Derive the posterior of each rate and give the difference of posterior "
                "means $\\mathbb E[p_B\\mid\\text{data}] - \\mathbb E[p_A\\mid\\text{data}]$."),
        answer=fmt(diff), kind="num", answer_hint="exact fraction, e.g. 5/84",
        rubric=["Beta prior × binomial likelihood ⇒ Beta(1 + x, 1 + n - x) posterior", "Posterior mean = (x + 1)/(n + 2)",
                "Subtract", "Compare with the raw difference x_B/n_B - x_A/n_A"],
        solution=(f"$p_A\\mid\\text{{data}} \\sim \\mathrm{{Beta}}({xa + 1}, {na - xa + 1})$, $p_B\\mid\\text{{data}} \\sim "
                  f"\\mathrm{{Beta}}({xb + 1}, {nb - xb + 1})$; means ${L(R(xa + 1, na + 2))}$ and ${L(R(xb + 1, nb + 2))}$, "
                  f"difference $= {L(diff)} \\approx {float(diff):.4f}$."),
        misconceptions=[(fmt(mle), "ignored-prior"), (fmt(R(xb + 1, nb + 1) - R(xa + 1, na + 1)), "posterior-mean-off-by-one")],
    )
