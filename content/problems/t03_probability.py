"""Track 3 — Probability & statistics (Harvard Stat 110 / MIT 18.05 style, original problems).

Conventions follow content/problems/t00_foundations.py:
  * (rng, level) -> Problem, deterministic given rng; answers computed with sympy (exact Rationals).
  * Levels: 1 = mechanical, 2 = multi-step, 3 = derivation / reasoning with a checkable final result.
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

# Standard normal CDF values quoted in prompts (4 d.p., as in a printed table).
PHI = {R(1, 2): R("0.6915"), 1: R("0.8413"), R(3, 2): R("0.9332"), 2: R("0.9772"), R(5, 2): R("0.9938")}
PHI_TABLE = ("$\\Phi(0.5)=0.6915,\\ \\Phi(1)=0.8413,\\ \\Phi(1.5)=0.9332,\\ "
             "\\Phi(2)=0.9772,\\ \\Phi(2.5)=0.9938$")
FRAC_HINT = "an exact fraction, e.g. 7/24"


def pct(r) -> str:
    """Rational probability -> percentage text, e.g. 19/20 -> '95%'."""
    return f"{float(r) * 100:g}%"


def dec(r) -> str:
    return f"{float(r):g}"


def nz(rng: random.Random, lo: int, hi: int) -> int:
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def lst(xs) -> str:
    return ", ".join(str(v) for v in xs)


# --------------------------------------------------------------------------- counting
@gen("prob.counting")
def counting(rng, level):
    if level == 1:
        n = rng.randint(7, 15)
        if rng.random() < 0.5:
            k = rng.randint(2, 5)
            ans = sp.binomial(n, k)
            return Problem(
                prompt=f"A club has {n} members. How many different {k}-person committees can be formed?",
                answer=str(ans), kind="num", answer_hint="an integer",
                rubric=["Recognise that order does not matter", f"Use C({n},{k}) = {n}!/({k}!·{n - k}!)", "Evaluate"],
                solution=f"$\\binom{{{n}}}{{{k}}} = {ans}$.",
                misconceptions=[(str(sp.ff(n, k)), "order-matters-overcount")],
            )
        ans = sp.ff(n, 3)
        return Problem(
            prompt=(f"A club has {n} members. In how many ways can it choose a president, a vice-president and a "
                    "treasurer (three different people)?"),
            answer=str(ans), kind="num", answer_hint="an integer",
            rubric=["Recognise that the roles make order matter", f"Multiplication rule {n}·{n - 1}·{n - 2}", "Evaluate"],
            solution=f"${n}\\cdot{n - 1}\\cdot{n - 2} = {ans}$.",
            misconceptions=[(str(sp.binomial(n, 3)), "ignored-order")],
        )
    if level == 2:
        if rng.random() < 0.5:
            w, m, k = rng.randint(3, 7), rng.randint(4, 8), rng.randint(3, 5)
            ans = sp.binomial(w + m, k) - sp.binomial(m, k)
            return Problem(
                prompt=(f"A department has {w} women and {m} men. How many {k}-person committees contain "
                        "**at least one** woman?"),
                answer=str(ans), kind="num", answer_hint="an integer",
                rubric=["Count all committees", "Count committees with no women (complement)", "Subtract"],
                solution=(f"All: $\\binom{{{w + m}}}{{{k}}} = {sp.binomial(w + m, k)}$; no women: "
                          f"$\\binom{{{m}}}{{{k}}} = {sp.binomial(m, k)}$. Answer ${ans}$."),
                misconceptions=[(str(w * sp.binomial(w + m - 1, k - 1)), "overcount-at-least-one"),
                                (str(sp.binomial(m, k)), "forgot-complement")],
            )
        word = rng.choice(["STATISTICS", "BANANA", "MISSISSIPPI", "PROBABILITY", "BOOKKEEPER", "COMMITTEE", "ASSESS"])
        cnt = Counter(word)
        ans = sp.factorial(len(word))
        for c in cnt.values():
            ans /= sp.factorial(c)
        reps = ", ".join(f"{ch}×{c}" for ch, c in sorted(cnt.items()) if c > 1)
        return Problem(
            prompt=f"How many distinct arrangements (anagrams) of the letters of **{word}** are there?",
            answer=str(ans), kind="num", answer_hint="an integer",
            rubric=["Count the letters and the repeated ones", "Divide n! by the factorial of each multiplicity", "Evaluate"],
            solution=f"{len(word)} letters, repeats {reps}: $\\dfrac{{{len(word)}!}}{{\\prod m_i!}} = {ans}$.",
            misconceptions=[(str(sp.factorial(len(word))), "ignored-repeats")],
        )
    # level 3: stars and bars with lower bounds
    while True:
        kk, a = rng.randint(3, 5), rng.randint(2, 4)
        n = rng.randint(8, 15)
        if n - a - (kk - 1) >= 1:
            break
    ans = sp.binomial(n - a, kk - 1)
    return Problem(
        prompt=(f"In how many ways can {n} identical cookies be given to {kk} children so that every child gets at "
                f"least one cookie and the first child gets at least {a}? (Children are distinguishable, cookies are not.)"),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=[f"Pre-allocate {a} to child 1 and 1 to each other child", "Reduce to non-negative solutions of y_1+…+y_k = remaining",
                "Stars and bars: C(remaining + k - 1, k - 1)", "Evaluate"],
        solution=(f"After pre-allocating, ${n - a - (kk - 1)}$ cookies remain for ${kk}$ children freely: "
                  f"$\\binom{{{n - a - (kk - 1)} + {kk - 1}}}{{{kk - 1}}} = \\binom{{{n - a}}}{{{kk - 1}}} = {ans}$."),
        misconceptions=[(str(sp.binomial(n + kk - 1, kk - 1)), "ignored-constraints"),
                        (str(sp.binomial(n - 1, kk - 1)), "ignored-lower-bound-a")],
    )


# --------------------------------------------------------------------------- axioms / inclusion-exclusion
@gen("prob.axioms")
def axioms(rng, level):
    if level == 1:
        while True:
            pa, pb, pab = R(rng.randint(4, 12), 20), R(rng.randint(4, 12), 20), R(rng.randint(1, 5), 20)
            if pab < min(pa, pb) and pa + pb - pab < 1:
                break
        ans = 1 - (pa + pb - pab)
        return Problem(
            prompt=(f"$P(A) = {dec(pa)}$, $P(B) = {dec(pb)}$ and $P(A\\cap B) = {dec(pab)}$. "
                    "Find the probability that **neither** $A$ nor $B$ occurs."),
            answer=fmt(ans), kind="num", answer_hint="a number, e.g. 0.35 or 7/20",
            rubric=["Inclusion-exclusion for P(A ∪ B)", "Neither = complement of the union", "Evaluate"],
            solution=(f"$P(A\\cup B) = {dec(pa)}+{dec(pb)}-{dec(pab)} = {dec(pa + pb - pab)}$, "
                      f"so $P(A^c\\cap B^c) = {dec(ans)}$."),
            misconceptions=[(fmt(1 - pa - pb), "inclusion-exclusion-missing-intersection"),
                            (fmt(pa + pb - pab), "forgot-complement")],
        )
    if level == 2:
        n, what = rng.choice([(12, "birth month"), (7, "day of the week they were born on"), (10, "last digit of their phone number")])
        kk = rng.randint(3, 5)
        p_dist = sp.prod([R(n - j, n) for j in range(kk)])
        ans = 1 - p_dist
        return Problem(
            prompt=(f"{kk} people each have a {what}, all {n} values equally likely and independent. "
                    "What is the probability that at least two of them share the same value?"),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["Use the complement: all values distinct", f"P(all distinct) = {n}·{n - 1}·…/{n}^{kk}", "Subtract from 1"],
            solution=(f"$P(\\text{{all distinct}}) = \\dfrac{{{n}\\cdot{n - 1}\\cdots{n - kk + 1}}}{{{n}^{{{kk}}}}} = {L(p_dist)}$, "
                      f"so the answer is $1 - {L(p_dist)} = {L(ans)}$."),
            misconceptions=[(fmt(p_dist), "forgot-complement"), (fmt(R(kk * (kk - 1), 2 * n)), "union-bound-as-equality")],
        )
    if rng.random() < 0.5:
        n = rng.randint(4, 6)
        p_none = sum(R((-1) ** j, math.factorial(j)) for j in range(n + 1))
        ans = 1 - p_none
        return Problem(
            prompt=(f"{n} people put their hats in a box and each draws one at random (a random permutation). "
                    "Using inclusion-exclusion, find the exact probability that **at least one** person gets their own hat."),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["Let A_i = person i gets own hat; P(A_i) = 1/n", "P of any j-fold intersection = (n-j)!/n!",
                    "Inclusion-exclusion: Σ (-1)^(j+1) C(n,j)(n-j)!/n! = Σ (-1)^(j+1)/j!", "Evaluate exactly"],
            solution=(f"$P\\left(\\bigcup A_i\\right) = \\sum_{{j=1}}^{{{n}}} \\frac{{(-1)^{{j+1}}}}{{j!}} = {L(ans)}$ "
                      f"$\\approx {float(ans):.4f}$ (close to $1 - 1/e$)."),
            misconceptions=[(fmt(1 - R(n - 1, n) ** n), "assumed-independence"), (fmt(R(1)), "union-bound-as-equality")],
        )
    N = rng.choice([60, 90, 120, 150, 210, 300])
    a, b, c = 2, 3, 5
    cnt = sum(1 for m in range(1, N + 1) if m % a == 0 or m % b == 0 or m % c == 0)
    ans = R(cnt, N)
    return Problem(
        prompt=(f"An integer is chosen uniformly from $\\{{1, 2, \\dots, {N}\\}}$. What is the probability it is "
                "divisible by 2, 3 **or** 5?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Count multiples of 2, 3, 5 individually", "Subtract multiples of 6, 10, 15", "Add back multiples of 30",
                "Divide by N"],
        solution=(f"$\\frac{{{N // 2}+{N // 3}+{N // 5}-{N // 6}-{N // 10}-{N // 15}+{N // 30}}}{{{N}}} = {L(ans)}$."),
        misconceptions=[(fmt(R(N // 2 + N // 3 + N // 5, N)), "forgot-intersections")],
    )


# --------------------------------------------------------------------------- conditional probability & Bayes
PREV = [R(1, 100), R(1, 50), R(1, 200), R(1, 20), R(1, 1000)]
SENS = [R(9, 10), R(95, 100), R(99, 100), R(98, 100)]
SPEC = [R(9, 10), R(95, 100), R(98, 100), R(99, 100)]


@gen("prob.conditional_bayes", name="medical_test")
def medical_test(rng, level):
    p, se, sp_ = rng.choice(PREV), rng.choice(SENS), rng.choice(SPEC)
    setup = (f"A disease affects {pct(p)} of a population. A test has sensitivity "
             f"$P(+\\mid D) = {pct(se)}$ and specificity $P(-\\mid D^c) = {pct(sp_)}$.").replace("%$", "\\%$")
    ppos = p * se + (1 - p) * (1 - sp_)
    if level == 1:
        return Problem(
            prompt=setup + " A random person is tested. What is $P(+)$, the probability the test is positive?",
            answer=fmt(ppos), kind="num", answer_hint="an exact fraction or decimal, e.g. 0.0585",
            rubric=["Split on D vs not D (law of total probability)", "True positives p·sens",
                    "False positives (1-p)(1-spec)", "Add"],
            solution=(f"$P(+) = {L(p)}\\cdot{L(se)} + {L(1 - p)}\\cdot{L(1 - sp_)} = {L(ppos)} \\approx {float(ppos):.5f}$."),
            misconceptions=[(fmt(p * se), "forgot-false-positives"), (fmt(se), "confused-sensitivity-with-p-positive")],
        )
    if level == 2:
        post = p * se / ppos
        return Problem(
            prompt=setup + " A random person tests **positive**. What is the probability they have the disease?",
            answer=fmt(post), kind="num", answer_hint=FRAC_HINT,
            rubric=["Write Bayes' rule P(D|+) = P(+|D)P(D)/P(+)", "Compute P(+) by total probability", "Divide and simplify"],
            solution=(f"$P(D\\mid +) = \\dfrac{{{L(se)}\\cdot{L(p)}}}{{{L(ppos)}}} = {L(post)} \\approx {float(post):.4f}$. "
                      "The base rate keeps this far below the sensitivity."),
            misconceptions=[(fmt(se), "base-rate-neglect"), (fmt(sp_), "confused-specificity-with-ppv")],
        )
    if rng.random() < 0.5:
        num, alt = p * se ** 2, (1 - p) * (1 - sp_) ** 2
        txt = "Two **independent** tests (conditionally independent given disease status) are both **positive**."
        one = p * se / ppos
        mis = [(fmt(one), "used-only-one-test"), (fmt(se ** 2), "base-rate-neglect")]
    else:
        num, alt = p * se * (1 - se), (1 - p) * (1 - sp_) * sp_
        txt = ("Two tests, conditionally independent given disease status, come back **first positive, then negative**.")
        one = p * se / ppos
        mis = [(fmt(one), "ignored-second-test"), (fmt(p), "assumed-results-cancel")]
    post = num / (num + alt)
    return Problem(
        prompt=setup + " " + txt + " What is the probability the person has the disease?",
        answer=fmt(post), kind="num", answer_hint=FRAC_HINT,
        rubric=["Use conditional independence to write P(results | D) and P(results | D^c)",
                "Apply Bayes with prior p (or update sequentially: posterior after test 1 becomes prior for test 2)",
                "Simplify"],
        solution=(f"$P(D\\mid\\text{{data}}) = \\dfrac{{{L(num)}}}{{{L(num)} + {L(alt)}}} = {L(post)} \\approx {float(post):.4f}$."),
        misconceptions=mis,
    )


@gen("prob.conditional_bayes", name="coins_urns")
def coins_urns(rng, level):
    if level == 1:
        a, b, c, d = rng.randint(1, 6), rng.randint(1, 6), rng.randint(1, 6), rng.randint(1, 6)
        faces = rng.choice([1, 2, 4, 5])
        q = R(faces, 6)
        shows = "1" if faces == 1 else f"1–{faces}"
        ans = q * R(a, a + b) + (1 - q) * R(c, c + d)
        return Problem(
            prompt=(f"Urn I has {a} red and {b} blue balls; urn II has {c} red and {d} blue. Roll a fair die: if it shows "
                    f"{shows} pick urn I, otherwise urn II. Then draw one ball. What is $P(\\text{{red}})$?"),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["Condition on which urn was chosen", "P(red | urn) for each urn", "Law of total probability"],
            solution=f"$P(\\text{{red}}) = {L(q)}\\cdot{L(R(a, a + b))} + {L(1 - q)}\\cdot{L(R(c, c + d))} = {L(ans)}$.",
            misconceptions=[(fmt(R(a + c, a + b + c + d)), "pooled-urns")],
        )
    m, kk = rng.randint(2, 9), rng.randint(2, 5)
    like_f = R(1, 2 ** kk)
    post_f = m * like_f / (m * like_f + 1)
    setup = (f"A jar has {m} fair coins and 1 double-headed coin. You pick a coin at random and flip it {kk} times: "
             f"all {kk} flips are heads.")
    if level == 2:
        return Problem(
            prompt=setup + " What is the probability that the coin is **fair**?",
            answer=fmt(post_f), kind="num", answer_hint=FRAC_HINT,
            rubric=[f"Prior P(fair) = {m}/{m + 1}", f"Likelihoods (1/2)^{kk} and 1", "Bayes: normalise", "Simplify"],
            solution=(f"$P(F\\mid H^{{{kk}}}) = \\dfrac{{\\frac{{{m}}}{{{m + 1}}}\\cdot 2^{{-{kk}}}}}"
                      f"{{\\frac{{{m}}}{{{m + 1}}}\\cdot 2^{{-{kk}}} + \\frac{{1}}{{{m + 1}}}}} = {L(post_f)}$."),
            misconceptions=[(fmt(R(m, m + 1)), "ignored-evidence"), (fmt(like_f), "likelihood-as-posterior")],
        )
    ans = post_f * R(1, 2) + (1 - post_f)
    return Problem(
        prompt=setup + " What is the probability that the **next** flip of the same coin is heads?",
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Compute the posterior P(fair | data) with Bayes", "Condition the next flip on the coin type",
                "Combine: P(fair|data)·1/2 + P(double|data)·1"],
        solution=(f"$P(F\\mid\\text{{data}}) = {L(post_f)}$, so $P(H_{{next}}) = {L(post_f)}\\cdot\\frac12 + {L(1 - post_f)} = {L(ans)}$."),
        misconceptions=[(fmt(R(1, 2)), "ignored-posterior-update"),
                        (fmt(R(m, m + 1) * R(1, 2) + R(1, m + 1)), "used-prior-not-posterior")],
    )


# --------------------------------------------------------------------------- discrete random variables
@gen("prob.discrete_rv")
def discrete_rv(rng, level):
    if level == 1:
        n = rng.randint(4, 8)
        p = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(2, 3), R(1, 5)])
        kk = rng.randint(1, n - 1)
        ans = sp.binomial(n, kk) * p ** kk * (1 - p) ** (n - kk)
        return Problem(
            prompt=f"$X \\sim \\mathrm{{Bin}}({n}, {L(p)})$. Compute $P(X = {kk})$ exactly.",
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["Write the Binomial PMF C(n,k) p^k (1-p)^(n-k)", "Evaluate the binomial coefficient", "Simplify"],
            solution=f"$\\binom{{{n}}}{{{kk}}}\\left({L(p)}\\right)^{{{kk}}}\\left({L(1 - p)}\\right)^{{{n - kk}}} = {L(ans)}$.",
            misconceptions=[(fmt(p ** kk * (1 - p) ** (n - kk)), "missing-binomial-coefficient")],
        )
    if level == 2:
        lam, m = rng.randint(1, 4), rng.randint(1, 3)
        lt = sp.Integer(lam)
        cdf = sp.exp(-lt) * sum(lt ** j / sp.factorial(j) for j in range(m + 1))
        if rng.random() < 0.5:
            ans, q = cdf, f"P(X \\le {m})"
            mis = [(str(sp.exp(-lt) * sum(lt ** j / sp.factorial(j) for j in range(m))), "off-by-one-boundary"),
                   (str(1 - cdf), "forgot-complement")]
        else:
            ans, q = 1 - cdf, f"P(X > {m})"
            mis = [(str(cdf), "forgot-complement"),
                   (str(1 - sp.exp(-lt) * sum(lt ** j / sp.factorial(j) for j in range(m))), "off-by-one-boundary")]
        return Problem(
            prompt=f"Calls arrive at a help desk as $X \\sim \\mathrm{{Pois}}({lam})$ per minute. Compute ${q}$.",
            answer=str(ans), kind="num", tol=1e-4,
            answer_hint="exact, e.g. 5*exp(-2), or a decimal to 4 places",
            rubric=["Poisson PMF e^{-λ} λ^k / k!", "Sum the needed terms (use the complement if helpful)", "Simplify / evaluate"],
            solution=f"${q} = {L(ans)} \\approx {float(ans):.4f}$.",
            misconceptions=mis,
        )
    p = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(1, 5), R(1, 6), R(2, 5)])
    ans = p / (1 - (1 - p) ** 2)
    return Problem(
        prompt=(f"Alice and Bob take turns flipping a coin with $P(\\text{{heads}}) = {L(p)}$, Alice first. The first to "
                "flip heads wins. Find $P(\\text{Alice wins})$. (Sum a geometric series or use first-step analysis.)"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Alice wins on her j-th flip with prob ((1-p)^2)^{j-1} p", "Sum the geometric series (or: x = p + (1-p)^2 x)",
                "Simplify to 1/(2-p)"],
        solution=f"$P = \\sum_{{j\\ge 0}} (1-p)^{{2j}}p = \\dfrac{{p}}{{1-(1-p)^2}} = \\dfrac{{1}}{{2-p}} = {L(ans)}$.",
        misconceptions=[(fmt(R(1, 2)), "ignored-first-mover-advantage"), (fmt(p), "only-first-flip")],
    )


# --------------------------------------------------------------------------- expectation & variance
def _pmf(rng, nvals):
    den = rng.choice([10, 12, 20])
    while True:
        cuts = sorted(rng.sample(range(1, den), nvals - 1))
        ws = [b - a for a, b in zip([0] + cuts, cuts + [den])]
        if all(w > 0 for w in ws):
            break
    vals = sorted(rng.sample(range(-3, 8), nvals))
    return vals, [R(w, den) for w in ws]


def _pmf_table(vals, probs):
    return ("$$\\begin{array}{c|" + "c" * len(vals) + "} x & " + " & ".join(map(str, vals)) +
            " \\\\ \\hline P(X=x) & " + " & ".join(L(p) for p in probs) + "\\end{array}$$")


@gen("prob.expectation_variance", name="pmf_moments")
def pmf_moments(rng, level):
    if level in (1, 2):
        vals, probs = _pmf(rng, rng.randint(3, 4))
        ex = sum(v * p for v, p in zip(vals, probs))
        ex2 = sum(v * v * p for v, p in zip(vals, probs))
        var = ex2 - ex ** 2
        if level == 1:
            return Problem(
                prompt="$X$ has the PMF below. Compute $E[X]$.\n\n" + _pmf_table(vals, probs),
                answer=fmt(ex), kind="num", answer_hint=FRAC_HINT,
                rubric=["E[X] = Σ x P(X=x)", "Multiply and add", "Simplify"],
                solution=f"$E[X] = {' + '.join(f'({v})({L(p)})' for v, p in zip(vals, probs))} = {L(ex)}$.",
                misconceptions=[(fmt(R(sum(vals), len(vals))), "unweighted-mean")],
            )
        a, b = nz(rng, -4, 4), nz(rng, -5, 5)
        while a in (1, -1):
            a = nz(rng, -4, 4)
        ans = a ** 2 * var
        return Problem(
            prompt=f"$X$ has the PMF below. Let $Y = {L(a * sp.Symbol('X') + b)}$. Compute $\\mathrm{{Var}}(Y)$.\n\n" + _pmf_table(vals, probs),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["Compute E[X] and E[X^2]", "Var(X) = E[X^2] - (E[X])^2", "Var(aX+b) = a^2 Var(X)"],
            solution=(f"$E[X] = {L(ex)}$, $E[X^2] = {L(ex2)}$, $\\mathrm{{Var}}(X) = {L(var)}$, "
                      f"$\\mathrm{{Var}}(Y) = {a ** 2}\\cdot{L(var)} = {L(ans)}$."),
            misconceptions=[(fmt(a * var), "forgot-square-scale"), (fmt(a ** 2 * var + b), "shift-changes-variance"),
                            (fmt(ex2 - ex), "variance-formula-no-square")],
        )
    n = rng.choice([4, 6, 8, 10])
    kind = rng.choice(["max", "min"])
    if kind == "max":
        ans = sum(1 - R((kk - 1) ** 2, n ** 2) for kk in range(1, n + 1))
        tail = "P(M \\ge k) = 1 - \\left(\\frac{k-1}{n}\\right)^2"
    else:
        ans = sum(R((n - kk + 1) ** 2, n ** 2) for kk in range(1, n + 1))
        tail = "P(M \\ge k) = \\left(\\frac{n-k+1}{n}\\right)^2"
    return Problem(
        prompt=(f"Roll two fair {n}-sided dice (faces $1..{n}$). Let $M$ be the **{kind}imum** of the two. "
                "Find $E[M]$ exactly, e.g. via the tail-sum formula $E[M] = \\sum_{k\\ge1} P(M\\ge k)$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Find P(M ≥ k) using independence of the two dice", "Apply the tail-sum formula (or get the PMF first)",
                "Sum and simplify"],
        solution=f"${tail}$ with $n={n}$; summing $k=1..{n}$ gives $E[M] = {L(ans)} \\approx {float(ans):.3f}$.",
        misconceptions=[(fmt(R(n + 1, 2)), f"ignored-{kind}")],
    )


@gen("prob.expectation_variance", name="indicators")
def indicators(rng, level):
    if level == 1:
        r, b, kk = rng.randint(3, 10), rng.randint(3, 10), rng.randint(2, 5)
        ans = R(kk * r, r + b)
        return Problem(
            prompt=(f"An urn has {r} red and {b} blue balls. You draw {kk} balls **without** replacement. "
                    "What is the expected number of red balls drawn? (Use indicators — no PMF needed.)"),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["Let I_j = 1 if draw j is red", f"By symmetry P(I_j = 1) = {r}/{r + b} for every j",
                    "Linearity of expectation holds despite dependence"],
            solution=f"$E[X] = \\sum_{{j=1}}^{{{kk}}} P(I_j=1) = {kk}\\cdot\\frac{{{r}}}{{{r + b}}} = {L(ans)}$.",
            misconceptions=[(fmt(R(r, r + b)), "single-draw-only")],
        )
    if level == 2:
        n = rng.randint(8, 20)
        p = rng.choice([R(1, 2), R(1, 3), R(2, 3), R(1, 4), R(3, 4)])
        if rng.random() < 0.5:
            pat, pp = "HH", p ** 2
        else:
            pat, pp = "HT", p * (1 - p)
        ans = (n - 1) * pp
        return Problem(
            prompt=(f"A coin with $P(H) = {L(p)}$ is flipped {n} times. Find the expected number of times the "
                    f"pattern **{pat}** appears in consecutive positions (overlaps count)."),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=[f"Define indicators I_j for positions j, j+1 (j = 1..{n - 1})", f"P(I_j = 1) for pattern {pat}",
                    "Linearity: multiply by the number of positions"],
            solution=f"${n - 1}$ starting positions, each with probability ${L(pp)}$: $E = {L(ans)}$.",
            misconceptions=[(fmt(n * pp), "off-by-one-positions"), (fmt(R(n, 2) * pp), "non-overlapping-count")],
        )
    n, kk = rng.randint(4, 8), rng.randint(3, 8)
    if rng.random() < 0.5:
        ans = n * R(n - 1, n) ** kk
        return Problem(
            prompt=(f"{kk} balls are thrown independently and uniformly into {n} boxes. Find the expected number of "
                    "**empty** boxes."),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["Indicator I_j that box j is empty", f"P(I_j = 1) = (1 - 1/{n})^{kk}", "Linearity: multiply by n"],
            solution=f"$E = {n}\\left(\\frac{{{n - 1}}}{{{n}}}\\right)^{{{kk}}} = {L(ans)}$.",
            misconceptions=[(fmt(max(n - kk, 0)), "assumed-all-distinct"), (fmt(R(n - 1, n) ** kk), "forgot-sum-over-boxes")],
        )
    ans = n * (1 - R(n - 1, n) ** kk)
    return Problem(
        prompt=(f"A fair {n}-sided die is rolled {kk} times. Find the expected number of **distinct** faces that appear "
                "(a birthday-type problem)."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Indicator I_j that face j appears at least once", f"P(I_j = 1) = 1 - (1 - 1/{n})^{kk} (complement)",
                "Linearity: multiply by n"],
        solution=f"$E = {n}\\left(1 - \\left(\\frac{{{n - 1}}}{{{n}}}\\right)^{{{kk}}}\\right) = {L(ans)}$.",
        misconceptions=[(fmt(min(n, kk)), "assumed-all-distinct"), (fmt(n * R(n - 1, n) ** kk), "forgot-complement")],
    )


# --------------------------------------------------------------------------- continuous RVs
@gen("prob.continuous_rv")
def continuous_rv(rng, level):
    if level == 1:
        if rng.random() < 0.5:
            lam = rng.choice([R(1, 2), R(1, 3), 2, 3, R(1, 4)])
            t = rng.randint(1, 3)
            ans = sp.exp(-lam * t)
            mis = [(str(1 - ans), "used-cdf-not-tail")]
            if lam != 1:
                mis.append((str(sp.exp(-t / sp.nsimplify(lam))), "rate-vs-mean-confusion"))
            return Problem(
                prompt=(f"A light bulb's lifetime (years) is $X \\sim \\mathrm{{Expo}}(\\lambda = {L(sp.nsimplify(lam))})$ "
                        f"(rate $\\lambda$, mean $1/\\lambda$). Find $P(X > {t})$."),
                answer=str(ans), kind="num", tol=1e-4,
                answer_hint="exact, e.g. exp(-3/2), or a decimal to 4 places",
                rubric=["CDF F(x) = 1 - e^{-λx}", "Tail = 1 - F(t)", "Evaluate"],
                solution=f"$P(X>{t}) = e^{{-\\lambda t}} = {L(ans)} \\approx {float(ans):.4f}$.",
                misconceptions=mis,
            )
        mu, sigma = rng.choice([(100, 15), (50, 10), (170, 8), (20, 4), (500, 100)])
        a, b = rng.choice([1, 2]), rng.choice([1, 2])
        ans = PHI[b] - (1 - PHI[a])
        return Problem(
            prompt=(f"$X \\sim \\mathcal N({mu}, {sigma}^2)$. Using {PHI_TABLE}, find "
                    f"$P({mu - a * sigma} < X < {mu + b * sigma})$."),
            answer=fmt(ans), kind="num", tol=1e-3, answer_hint="a decimal, e.g. 0.8186",
            rubric=["Standardise: Z = (X - μ)/σ", "Write as Φ(b) - Φ(-a)", "Use symmetry Φ(-a) = 1 - Φ(a)"],
            solution=(f"$P(-{a} < Z < {b}) = \\Phi({b}) - (1-\\Phi({a})) = {dec(PHI[b])} - {dec(1 - PHI[a])} = {dec(ans)}$."),
            misconceptions=[(fmt(PHI[b] - PHI[a]), "forgot-symmetry")],
        )
    if level == 2:
        m, a = rng.randint(1, 4), rng.randint(1, 4)
        c = R(m + 1, a ** (m + 1))
        ans = R((m + 1) * a, m + 2)
        xx = sp.Symbol("x")
        return Problem(
            prompt=(f"$X$ has density $f(x) = c\\,x^{{{m}}}$ for $0 \\le x \\le {a}$ (and 0 otherwise). "
                    "Find the constant $c$ on paper, then type $E[X]$."),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["Normalise: ∫ c x^m dx over [0,a] = 1", f"c = {L(c)}", "E[X] = ∫ x f(x) dx", "Simplify"],
            solution=(f"$c = {L(c)}$, $E[X] = \\int_0^{{{a}}} x\\cdot {L(c * xx ** m)}\\,dx = {L(ans)}$."),
            misconceptions=[(fmt(R(a, 2)), "assumed-uniform"), (fmt(R(a ** (m + 2), m + 2)), "forgot-normalising-constant")],
        )
    l1, l2 = rng.sample([1, 2, 3, 4, 5], 2)
    ans = R(1, l1) + R(1, l2) - R(1, l1 + l2)
    return Problem(
        prompt=(f"Two independent servers fail after $X \\sim \\mathrm{{Expo}}({l1})$ and $Y \\sim \\mathrm{{Expo}}({l2})$ "
                "hours (rates). Find the expected time until **both** have failed, $E[\\max(X,Y)]$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["min(X,Y) ~ Expo(λ1+λ2)", "max + min = X + Y", "Linearity: E[max] = 1/λ1 + 1/λ2 - 1/(λ1+λ2)",
                "(Alternative: memorylessness after the first failure)"],
        solution=(f"$E[\\max] = E[X]+E[Y]-E[\\min] = \\frac1{{{l1}}}+\\frac1{{{l2}}}-\\frac1{{{l1 + l2}}} = {L(ans)}$."),
        misconceptions=[(fmt(max(R(1, l1), R(1, l2))), "max-of-means"), (fmt(R(1, l1) + R(1, l2)), "forgot-overlap"),
                        (fmt(R(1, l1 + l2)), "computed-min-not-max")],
    )


# --------------------------------------------------------------------------- joint distributions & covariance
@gen("prob.joint_covariance")
def joint_covariance(rng, level):
    if level == 1:
        while True:
            vx, vy = rng.randint(1, 9), rng.randint(1, 9)
            cov = rng.randint(-4, 4)
            if cov != 0 and cov * cov < vx * vy:
                break
        a, b = nz(rng, -3, 3), nz(rng, -3, 3)
        ans = a * a * vx + b * b * vy + 2 * a * b * cov
        X, Y = sp.symbols("X Y")
        return Problem(
            prompt=(f"$\\mathrm{{Var}}(X) = {vx}$, $\\mathrm{{Var}}(Y) = {vy}$, $\\mathrm{{Cov}}(X,Y) = {cov}$. "
                    f"Compute $\\mathrm{{Var}}({L(a * X + b * Y)})$."),
            answer=str(ans), kind="num",
            rubric=["Var(aX+bY) = a²Var X + b²Var Y + 2ab Cov(X,Y)", "Substitute", "Evaluate (watch the signs)"],
            solution=f"$= {a * a}\\cdot{vx} + {b * b}\\cdot{vy} + 2({a})({b})({cov}) = {ans}$.",
            misconceptions=[(str(a * a * vx + b * b * vy), "forgot-covariance"), (str(a * vx + b * vy), "forgot-square-scale")],
        )
    if level == 2:
        while True:
            den = rng.choice([10, 12, 20])
            cuts = sorted(rng.sample(range(1, den), 3))
            w = [bb - aa for aa, bb in zip([0] + cuts, cuts + [den])]
            p00, p01, p10, p11 = (R(v, den) for v in w)
            px, py = p10 + p11, p01 + p11
            cov = p11 - px * py
            if cov != 0:
                break
        return Problem(
            prompt=("$X, Y \\in \\{0,1\\}$ have joint PMF $P(0,0) = " + L(p00) + "$, $P(0,1) = " + L(p01) +
                    "$, $P(1,0) = " + L(p10) + "$, $P(1,1) = " + L(p11) + "$ (written $P(x,y)$). Compute $\\mathrm{Cov}(X,Y)$."),
            answer=fmt(cov), kind="num", answer_hint=FRAC_HINT,
            rubric=["Marginals P(X=1), P(Y=1)", "E[XY] = P(X=1,Y=1)", "Cov = E[XY] - E[X]E[Y]"],
            solution=f"$E[X] = {L(px)}$, $E[Y] = {L(py)}$, $E[XY] = {L(p11)}$, $\\mathrm{{Cov}} = {L(p11)} - {L(px)}\\cdot{L(py)} = {L(cov)}$.",
            misconceptions=[(fmt(p11), "cov-is-joint-prob"), (fmt(R(0)), "assumed-independent")],
        )
    n, kk = rng.randint(6, 30), rng.choice([3, 4, 6, 8])
    if rng.random() < 0.5:
        ans = R(-n, kk * kk)
        return Problem(
            prompt=(f"A fair {kk}-sided die is rolled {n} times. $X$ = number of 1s, $Y$ = number of 2s. "
                    "Compute $\\mathrm{Cov}(X, Y)$ (hint: write X and Y as sums of indicators)."),
            answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
            rubric=["X = Σ I_j, Y = Σ J_j over rolls", "Cov(I_j, J_l) = 0 for j ≠ l (different rolls independent)",
                    "Cov(I_j, J_j) = 0 - (1/k)^2 (a roll can't be both)", "Sum over n rolls"],
            solution=f"$\\mathrm{{Cov}}(X,Y) = n\\left(0 - \\frac{{1}}{{{kk}^2}}\\right) = {L(ans)}$.",
            misconceptions=[(fmt(R(0)), "assumed-independent"), (fmt(-ans), "sign-error")],
        )
    ans = R(-1, kk - 1)
    return Problem(
        prompt=(f"A fair {kk}-sided die is rolled {n} times. $X$ = number of 1s, $Y$ = number of 2s. "
                "Compute the correlation $\\mathrm{Corr}(X, Y)$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Cov(X,Y) = -n/k² via indicators", "Var(X) = Var(Y) = n(1/k)(1-1/k)", "Corr = Cov / sqrt(Var X Var Y)"],
        solution=(f"$\\mathrm{{Cov}} = -\\frac{{{n}}}{{{kk * kk}}}$, $\\mathrm{{Var}} = {n}\\cdot\\frac{{{kk - 1}}}{{{kk * kk}}}$, so "
                  f"$\\mathrm{{Corr}} = -\\frac{{1}}{{{kk - 1}}}$ (independent of $n$)."),
        misconceptions=[(fmt(R(0)), "assumed-independent"), (fmt(R(-n, kk * kk)), "covariance-not-correlation")],
    )


# --------------------------------------------------------------------------- LLN & CLT
@gen("prob.lln_clt")
def lln_clt(rng, level):
    if level == 1:
        sigma, n = rng.randint(2, 30), rng.choice([4, 9, 16, 25, 36, 49, 64, 100])
        ans = R(sigma, int(math.isqrt(n)))
        return Problem(
            prompt=(f"$X_1,\\dots,X_{{{n}}}$ are i.i.d. with standard deviation $\\sigma = {sigma}$. What is the "
                    "standard deviation of the sample mean $\\bar X$ (the standard error)?"),
            answer=fmt(ans), kind="num", answer_hint="exact, e.g. 5/2",
            rubric=["Var(X̄) = σ²/n by independence", "Take the square root: σ/√n"],
            solution=f"$\\mathrm{{SD}}(\\bar X) = {sigma}/\\sqrt{{{n}}} = {L(ans)}$.",
            misconceptions=[(fmt(R(sigma, n)), "divided-by-n-not-root-n"), (fmt(R(sigma ** 2, n)), "variance-not-sd")],
        )
    if level == 2:
        mu, sigma = rng.randint(2, 20), rng.randint(1, 6)
        n = rng.choice([16, 25, 36, 49, 64, 100])
        z = rng.choice([R(1, 2), 1, R(3, 2), 2])
        rn = math.isqrt(n)
        ans = 1 - PHI[z]
        if rng.random() < 0.5:
            thr = n * mu + z * sigma * rn
            q = f"the total $S = X_1 + \\dots + X_{{{n}}}$ exceeds ${L(thr)}$"
            sol = f"$E[S] = {n * mu}$, $\\mathrm{{SD}}(S) = {sigma}\\sqrt{{{n}}} = {sigma * rn}$, $z = {L(sp.nsimplify(z))}$"
        else:
            thr = mu + z * R(sigma, rn)
            q = f"the sample mean $\\bar X$ exceeds ${L(thr)}$"
            sol = f"$E[\\bar X] = {mu}$, $\\mathrm{{SD}}(\\bar X) = {sigma}/\\sqrt{{{n}}} = {L(R(sigma, rn))}$, $z = {L(sp.nsimplify(z))}$"
        return Problem(
            prompt=(f"$X_1,\\dots,X_{{{n}}}$ are i.i.d. with mean ${mu}$ and standard deviation ${sigma}$. Use the CLT and "
                    f"{PHI_TABLE} to approximate the probability that {q}."),
            answer=fmt(ans), kind="num", tol=1e-3, answer_hint="a decimal, e.g. 0.0668",
            rubric=["Mean and SD of the sum / mean (SD scales with √n)", "Standardise to z", "Upper tail: 1 - Φ(z)"],
            solution=sol + f"; $P \\approx 1 - \\Phi({L(sp.nsimplify(z))}) = {dec(ans)}$.",
            misconceptions=[(fmt(PHI[z]), "wrong-tail")],
        )
    eps = rng.choice([R(1, 10), R(1, 20), R(1, 50), R(1, 5)])
    delta = rng.choice([R(1, 10), R(1, 20), R(1, 5), R(1, 4)])
    ans = sp.ceiling(1 / (4 * eps ** 2 * delta))
    return Problem(
        prompt=(f"You estimate an unknown coin bias $p$ by the fraction of heads $\\bar X$ in $n$ flips. Using **Chebyshev's "
                f"inequality** and the worst case $p(1-p) \\le 1/4$, find the smallest $n$ guaranteeing "
                f"$P(|\\bar X - p| \\ge {dec(eps)}) \\le {dec(delta)}$."),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=["Var(X̄) = p(1-p)/n ≤ 1/(4n)", "Chebyshev: P(|X̄-p| ≥ ε) ≤ Var/ε²", "Solve 1/(4nε²) ≤ δ for n", "Round up"],
        solution=f"$n \\ge \\dfrac{{1}}{{4\\varepsilon^2\\delta}} = {L(1 / (4 * eps ** 2 * delta))}$, so $n = {ans}$.",
        misconceptions=[(str(sp.ceiling(1 / (4 * eps * delta))), "forgot-square-epsilon"),
                        (str(sp.ceiling(1 / (eps ** 2 * delta))), "ignored-worst-case-quarter")],
    )


# --------------------------------------------------------------------------- MLE
@gen("prob.mle", name="mle_counts")
def mle_counts(rng, level):
    if level == 1:
        n = rng.randint(8, 15)
        while True:
            data = [rng.randint(0, 1) for _ in range(n)]
            k = sum(data)
            if 0 < k < n and 2 * k != n:
                break
        return Problem(
            prompt=(f"Observations from $\\mathrm{{Bern}}(p)$: {lst(data)}. Write the log-likelihood and find the MLE $\\hat p$."),
            answer=fmt(R(k, n)), kind="num", answer_hint=FRAC_HINT,
            rubric=["ℓ(p) = k log p + (n-k) log(1-p)", "Set ℓ'(p) = 0", "p̂ = k/n"],
            solution=f"$k = {k}$ successes in $n = {n}$: $\\hat p = {L(R(k, n))}$.",
            misconceptions=[(fmt(R(n - k, n)), "counted-failures")],
        )
    if level == 2:
        n = rng.randint(4, 7)
        data = [rng.randint(1, 8) for _ in range(n)]
        s = sum(data)
        return Problem(
            prompt=(f"The number of attempts until the first success is modelled as Geometric on $\\{{1,2,\\dots\\}}$: "
                    f"$P(X = x) = (1-p)^{{x-1}}p$. Observations: {lst(data)}. Find the MLE $\\hat p$."),
            answer=fmt(R(n, s)), kind="num", answer_hint=FRAC_HINT,
            rubric=["ℓ(p) = n log p + (Σx - n) log(1-p)", "ℓ'(p) = n/p - (Σx - n)/(1-p) = 0", "p̂ = n / Σx"],
            solution=f"$\\hat p = n/\\sum x_i = {n}/{s} = {L(R(n, s))}$.",
            misconceptions=[(fmt(R(s, n)), "inverted-estimator"), (fmt(R(n, s - n) if s != n else R(2)), "used-failures-support")],
        )
    n = rng.randint(4, 6)
    data = sorted(rng.sample(range(1, 30), n))
    rng.shuffle(data)
    mx = max(data)
    return Problem(
        prompt=(f"Observations from $\\mathrm{{Unif}}(0, \\theta)$: {lst(data)}. (a) Find the MLE $\\hat\\theta$ (sketch the "
                "likelihood — calculus alone won't do it). (b) Using $E[\\max_i X_i] = \\frac{n}{n+1}\\theta$, give the "
                "unbiased rescaling of $\\hat\\theta$. Type both."),
        answer=[fmt(mx), fmt(R((n + 1) * mx, n))], kind="list", answer_hint="MLE, unbiased   e.g. 20, 24",
        rubric=["L(θ) = θ^{-n} if θ ≥ max x_i, else 0", "L is decreasing on its support, so θ̂ = max x_i",
                "Bias: E[θ̂] = nθ/(n+1)", "Rescale by (n+1)/n"],
        solution=(f"$\\hat\\theta = \\max = {mx}$; unbiased: $\\frac{{{n + 1}}}{{{n}}}\\cdot {mx} = {L(R((n + 1) * mx, n))}$."),
        misconceptions=[([fmt(R(2 * sum(data), n)), fmt(R((n + 1) * mx, n))], "method-of-moments")],
    )


@gen("prob.mle", name="mle_continuous")
def mle_continuous(rng, level):
    if level == 1:
        n = rng.randint(3, 6)
        data = [R(rng.randint(1, 40), 10) for _ in range(n)]
        s = sum(data)
        return Problem(
            prompt=(f"Waiting times (minutes) modelled as $\\mathrm{{Expo}}(\\lambda)$ with density $\\lambda e^{{-\\lambda x}}$: "
                    f"{lst(dec(d) for d in data)}. Find the MLE $\\hat\\lambda$ (a **rate**)."),
            answer=fmt(n / s), kind="num", answer_hint=FRAC_HINT,
            rubric=["ℓ(λ) = n log λ - λ Σx", "ℓ'(λ) = n/λ - Σx = 0", "λ̂ = n/Σx = 1/x̄"],
            solution=f"$\\sum x = {dec(s)}$, $\\hat\\lambda = {n}/{dec(s)} = {L(n / s)}$.",
            misconceptions=[(fmt(s / n), "inverted-estimator")],
        )
    if level == 2:
        n = rng.randint(5, 8)
        while True:
            data = [rng.randint(0, 5) for _ in range(n)]
            if sum(data) > 0:
                break
        xbar = R(sum(data), n)
        ans = sp.exp(-xbar)
        mis = [(fmt(xbar), "returned-lambda-not-target")]
        z = R(data.count(0), n)
        mis.append((fmt(z), "empirical-frequency-not-mle"))
        return Problem(
            prompt=(f"Daily counts modelled as $\\mathrm{{Pois}}(\\lambda)$: {lst(data)}. Find the MLE of "
                    "$\\theta = P(X = 0) = e^{-\\lambda}$."),
            answer=str(ans), kind="num", tol=1e-4, answer_hint="exact, e.g. exp(-7/3), or a decimal to 4 places",
            rubric=["ℓ(λ) = Σx log λ - nλ + const", "λ̂ = x̄", "Invariance: θ̂ = e^{-λ̂}"],
            solution=f"$\\hat\\lambda = \\bar x = {L(xbar)}$, so $\\hat\\theta = e^{{-{L(xbar)}}} \\approx {float(ans):.4f}$.",
            misconceptions=mis,
        )
    n = rng.randint(3, 5)
    es = [rng.randint(1, 4) for _ in range(n)]
    S = sum(es)
    ans = n / (S * sp.log(2))
    return Problem(
        prompt=("$X$ has density $f(x;\\theta) = \\theta x^{\\theta-1}$ on $(0,1)$, $\\theta > 0$. Observations: "
                + ", ".join(f"$2^{{-{e}}}$" for e in es) + ". Derive the MLE $\\hat\\theta$ and evaluate it."),
        answer=str(ans), kind="num", tol=1e-3, answer_hint="exact, e.g. 4/(7*log(2)), or a decimal to 3 places",
        rubric=["ℓ(θ) = n log θ + (θ-1) Σ log x_i", "ℓ'(θ) = n/θ + Σ log x_i = 0", "θ̂ = -n / Σ log x_i",
                f"Σ log x_i = -{S} log 2"],
        solution=f"$\\hat\\theta = \\dfrac{{-n}}{{\\sum\\ln x_i}} = \\dfrac{{{n}}}{{{S}\\ln 2}} \\approx {float(ans):.3f}$.",
        misconceptions=[(str(S * sp.log(2) / n), "inverted-estimator"), (fmt(R(n, S)), "dropped-log")],
    )


# --------------------------------------------------------------------------- Bayesian inference
@gen("prob.bayesian_inference")
def bayesian_inference(rng, level):
    if level in (1, 2):
        lo = 1 if level == 1 else 2
        a, b = rng.randint(lo, 6), rng.randint(lo, 6)
        n = rng.randint(5, 20)
        k = rng.randint(1, n - 1)
        mean = R(a + k, a + b + n)
        mode = R(a + k - 1, a + b + n - 2)
        intro = (f"A click-through rate $p$ has prior $\\mathrm{{Beta}}({a}, {b})$. Out of {n} visitors, {k} click "
                 "(Binomial likelihood).")
        if level == 1:
            return Problem(
                prompt=intro + " What is the **posterior mean** of $p$?",
                answer=fmt(mean), kind="num", answer_hint=FRAC_HINT,
                rubric=["Beta is conjugate to the Binomial", f"Posterior Beta(a+k, b+n-k) = Beta({a + k}, {b + n - k})",
                        "Mean of Beta(α,β) = α/(α+β)"],
                solution=f"Posterior $\\mathrm{{Beta}}({a + k}, {b + n - k})$, mean ${L(mean)}$.",
                misconceptions=[(fmt(R(k, n)), "ignored-prior"), (fmt(R(a, a + b)), "ignored-data")],
            )
        return Problem(
            prompt=intro + " What is the **MAP** estimate (posterior mode) of $p$?",
            answer=fmt(mode), kind="num", answer_hint=FRAC_HINT,
            rubric=[f"Posterior Beta({a + k}, {b + n - k})", "Maximise log posterior: (α-1)log p + (β-1)log(1-p)",
                    "Mode = (α-1)/(α+β-2)"],
            solution=f"Posterior $\\mathrm{{Beta}}({a + k}, {b + n - k})$; mode $= \\frac{{{a + k - 1}}}{{{a + b + n - 2}}} = {L(mode)}$.",
            misconceptions=[(fmt(mean), "posterior-mean-not-mode"), (fmt(R(k, n)), "ignored-prior")],
        )
    al, be = rng.randint(1, 6), rng.randint(1, 4)
    n = rng.randint(3, 6)
    data = [rng.randint(0, 7) for _ in range(n)]
    s = sum(data)
    pa, pb = al + s, be + n
    return Problem(
        prompt=(f"Goals per match $X_i \\mid \\lambda \\sim \\mathrm{{Pois}}(\\lambda)$ with prior "
                f"$\\lambda \\sim \\mathrm{{Gamma}}(\\alpha={al}, \\beta={be})$ (density $\\propto \\lambda^{{\\alpha-1}}e^{{-\\beta\\lambda}}$). "
                f"Observed: {lst(data)}. Derive the posterior and give its **mean and variance**."),
        answer=[fmt(R(pa, pb)), fmt(R(pa, pb ** 2))], kind="list", answer_hint="mean, variance   e.g. 7/3, 7/9",
        rubric=["Likelihood ∝ λ^{Σx} e^{-nλ}", "Multiply by prior → Gamma(α+Σx, β+n)", "Gamma mean α/β, variance α/β²"],
        solution=(f"Posterior $\\mathrm{{Gamma}}({al}+{s}, {be}+{n}) = \\mathrm{{Gamma}}({pa}, {pb})$: mean ${L(R(pa, pb))}$, "
                  f"variance ${L(R(pa, pb ** 2))}$."),
        misconceptions=[([fmt(R(s, n)), fmt(R(s, n * n))], "ignored-prior"),
                        ([fmt(R(pa, pb)), fmt(R(pa, pb))], "variance-forgot-square-rate")],
    )


# --------------------------------------------------------------------------- CIs & hypothesis tests
@gen("prob.hypothesis_ci")
def hypothesis_ci(rng, level):
    if level == 1:
        n = rng.choice([16, 25, 36, 64, 100])
        rn = math.isqrt(n)
        se = R(rng.randint(1, 4), 1) if rng.random() < 0.5 else R(rng.randint(1, 9), 2)
        sigma = se * rn
        xbar = rng.randint(20, 200)
        me = R(196, 100) * se
        lo, hi = xbar - me, xbar + me
        return Problem(
            prompt=(f"A sample of $n = {n}$ has mean $\\bar x = {xbar}$; the population SD is known, $\\sigma = {dec(sigma)}$. "
                    "Give the 95% confidence interval for $\\mu$ (use $z = 1.96$)."),
            answer=[fmt(lo), fmt(hi)], kind="list", answer_hint="lower, upper   e.g. 46.08, 53.92",
            rubric=["Standard error σ/√n", "Margin = 1.96 · SE", "x̄ ± margin"],
            solution=f"$SE = {dec(sigma)}/{rn} = {dec(se)}$, margin $= {dec(me)}$, CI $= [{dec(lo)}, {dec(hi)}]$.",
            misconceptions=[([fmt(xbar - R(196, 100) * sigma), fmt(xbar + R(196, 100) * sigma)], "forgot-sqrt-n")],
        )
    if level == 2:
        n = rng.choice([16, 25, 36, 49, 64, 100])
        rn = math.isqrt(n)
        sigma = rn * rng.randint(1, 4)
        se = sigma // rn
        mu0 = rng.randint(50, 200)
        z = rng.choice([1, R(3, 2), 2, R(5, 2)])
        sgn = rng.choice([1, -1])
        xbar = mu0 + sgn * z * se
        p2 = 2 * (1 - PHI[z])
        return Problem(
            prompt=(f"Test $H_0: \\mu = {mu0}$ vs $H_1: \\mu \\ne {mu0}$ with known $\\sigma = {sigma}$. A sample of "
                    f"$n = {n}$ gives $\\bar x = {dec(xbar)}$. Using {PHI_TABLE}, compute the **two-sided p-value**."),
            answer=fmt(p2), kind="num", tol=1e-3, answer_hint="a decimal, e.g. 0.0456",
            rubric=["z = (x̄ - μ0)/(σ/√n)", "Two-sided: p = 2(1 - Φ(|z|))", "Compare with α = 0.05 (on paper)"],
            solution=(f"$z = \\frac{{{dec(xbar)} - {mu0}}}{{{sigma}/{rn}}} = {dec(sgn * z)}$, "
                      f"$p = 2(1-\\Phi({dec(z)})) = {dec(p2)}$."),
            misconceptions=[(fmt(1 - PHI[z]), "one-sided-p-value"), (fmt(PHI[z]), "wrong-tail")],
        )
    if rng.random() < 0.5:
        sigma, E = rng.randint(5, 30), rng.randint(1, 5)
        ans = sp.ceiling((R(196, 100) * sigma / E) ** 2)
        return Problem(
            prompt=(f"Measurements have known SD $\\sigma = {sigma}$. How many observations are needed so that the 95% "
                    f"confidence interval for the mean has margin of error at most ${E}$? (Use $z = 1.96$.)"),
            answer=str(ans), kind="num", answer_hint="an integer",
            rubric=["Margin = 1.96 σ/√n", "Solve 1.96 σ/√n ≤ E for n", "n ≥ (1.96σ/E)²", "Round **up**"],
            solution=f"$n \\ge (1.96\\cdot{sigma}/{E})^2 = {dec((R(196, 100) * sigma / E) ** 2)}$, so $n = {ans}$.",
            misconceptions=[(str(sp.ceiling(R(196, 100) * sigma / E)), "forgot-square"),
                            (str(sp.ceiling((R(sigma, E)) ** 2)), "forgot-z-value")],
        )
    E = rng.choice([R(1, 100), R(2, 100), R(3, 100), R(4, 100), R(5, 100)])
    ans = sp.ceiling((R(196, 100) / (2 * E)) ** 2)
    return Problem(
        prompt=(f"A poll estimates a proportion $p$. How many respondents guarantee a 95% margin of error of at most "
                f"{dec(E)} **whatever** $p$ is? (Use $z = 1.96$ and the worst case for $p(1-p)$.)"),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=["SE = sqrt(p(1-p)/n)", "Worst case p = 1/2 gives p(1-p) = 1/4", "Solve 1.96·(1/2)/√n ≤ E", "Round up"],
        solution=f"$n \\ge \\left(\\frac{{1.96}}{{2\\cdot {dec(E)}}}\\right)^2 = {dec((R(196, 100) / (2 * E)) ** 2)}$, so $n = {ans}$.",
        misconceptions=[(str(sp.ceiling((R(196, 100) / E) ** 2)), "ignored-worst-case-quarter"),
                        (str(sp.ceiling(R(196, 100) / (2 * E))), "forgot-square")],
    )


# --------------------------------------------------------------------------- multivariate Gaussians
def _spd2(rng):
    while True:
        s11, s22 = rng.randint(1, 9), rng.randint(1, 9)
        s12 = nz(rng, -4, 4)
        if s12 * s12 < s11 * s22:
            return sp.Matrix([[s11, s12], [s12, s22]])


@gen("prob.multivariate_gaussian")
def multivariate_gaussian(rng, level):
    S = _spd2(rng)
    mu = sp.Matrix([rng.randint(-5, 5), rng.randint(-5, 5)])
    setup = f"$\\mathbf X = (X_1, X_2)^\\top \\sim \\mathcal N(\\boldsymbol\\mu, \\Sigma)$ with $\\boldsymbol\\mu = {L(mu)}$, $\\Sigma = {L(S)}$."
    if level == 1:
        a = sp.Matrix([nz(rng, -3, 3), nz(rng, -3, 3)])
        var = (a.T * S * a)[0]
        X1, X2 = sp.symbols("X_1 X_2")
        return Problem(
            prompt=setup + f" Compute $\\mathrm{{Var}}({L(a[0] * X1 + a[1] * X2)})$.",
            answer=fmt(var), kind="num",
            rubric=["Var(aᵀX) = aᵀΣa", "Expand: a1²Σ11 + a2²Σ22 + 2a1a2Σ12", "Evaluate"],
            solution=f"$a^\\top\\Sigma a = {var}$.",
            misconceptions=[(fmt(a[0] ** 2 * S[0, 0] + a[1] ** 2 * S[1, 1]), "forgot-covariance")],
        )
    if level == 2:
        while True:
            A = sp.Matrix(2, 2, [rng.randint(-2, 2) for _ in range(4)])
            if A.det() != 0:
                break
        C = A * S * A.T
        return Problem(
            prompt=setup + f" Let $\\mathbf Y = A\\mathbf X$ with $A = {L(A)}$. Give the covariance matrix of $\\mathbf Y$.",
            answer=fmt(C), kind="matrix",
            rubric=["Cov(AX) = A Σ Aᵀ", "Multiply A Σ", "Multiply by Aᵀ (result is symmetric)"],
            solution=f"$A\\Sigma A^\\top = {L(C)}$ (and $E[\\mathbf Y] = A\\boldsymbol\\mu = {L(A * mu)}$).",
            misconceptions=[(fmt(A * S), "forgot-transpose"), (fmt(A.T * S * A), "transpose-wrong-side")],
        )
    x2 = mu[1] + nz(rng, -4, 4)
    cm = mu[0] + R(S[0, 1], S[1, 1]) * (x2 - mu[1])
    cv = S[0, 0] - R(S[0, 1] ** 2, S[1, 1])
    wrong_m = mu[0] + R(S[0, 1], S[0, 0]) * (x2 - mu[1])
    return Problem(
        prompt=setup + f" Find $E[X_1 \\mid X_2 = {x2}]$ and $\\mathrm{{Var}}(X_1 \\mid X_2 = {x2})$.",
        answer=[fmt(cm), fmt(cv)], kind="list", answer_hint="mean, variance   e.g. 7/3, 5/3",
        rubric=["Conditional mean μ1 + Σ12/Σ22 (x2 - μ2)", "Conditional variance Σ11 - Σ12²/Σ22 (Schur complement)",
                "Note the variance does not depend on x2"],
        solution=(f"$E = {mu[0]} + \\frac{{{S[0, 1]}}}{{{S[1, 1]}}}({x2} - ({mu[1]})) = {L(cm)}$, "
                  f"$\\mathrm{{Var}} = {S[0, 0]} - \\frac{{{S[0, 1] ** 2}}}{{{S[1, 1]}}} = {L(cv)}$."),
        misconceptions=[([fmt(mu[0]), fmt(S[0, 0])], "ignored-conditioning"),
                        ([fmt(wrong_m), fmt(cv)], "regressed-wrong-direction")],
    )


# --------------------------------------------------------------------------- Markov chains
@gen("prob.markov_chains")
def markov_chains(rng, level):
    if level in (1, 2):
        den = rng.choice([4, 5, 10])
        a, b = R(rng.randint(1, den - 1), den), R(rng.randint(1, den - 1), den)
        while a == b and level == 2:
            b = R(rng.randint(1, den - 1), den)
        P = sp.Matrix([[1 - a, a], [b, 1 - b]])
        setup = (f"A two-state Markov chain (states 1 = sunny, 2 = rainy) has transition matrix $P = {L(P)}$ "
                 "(row = current state).")
        if level == 1:
            i, j = rng.randint(0, 1), rng.randint(0, 1)
            ans = (P ** 2)[i, j]
            return Problem(
                prompt=setup + f" Given state {i + 1} today, what is the probability of state {j + 1} in **two** days?",
                answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
                rubric=["Two-step probability = (P²)_{ij}", "Sum over the intermediate state", "Simplify"],
                solution=f"$(P^2)_{{{i + 1}{j + 1}}} = \\sum_k P_{{{i + 1}k}}P_{{k{j + 1}}} = {L(ans)}$.",
                misconceptions=[(fmt(P[i, j]), "one-step-only"), (fmt(P[i, j] ** 2), "squared-entry")],
            )
        pi = [b / (a + b), a / (a + b)]
        return Problem(
            prompt=setup + " Find the stationary distribution $\\pi = (\\pi_1, \\pi_2)$.",
            answer=[fmt(v) for v in pi], kind="vector", answer_hint="pi1, pi2   e.g. 2/5, 3/5",
            rubric=["Solve πP = π", "Balance: π1 a = π2 b", "Normalise π1 + π2 = 1"],
            solution=f"$\\pi_1 = \\frac{{b}}{{a+b}} = {L(pi[0])}$, $\\pi_2 = {L(pi[1])}$.",
            misconceptions=[([fmt(pi[1]), fmt(pi[0])], "swapped-rates")],
        )
    den = rng.choice([4, 5, 6])
    while True:
        rows = []
        for _ in range(3):
            c1, c2 = sorted(rng.sample(range(1, den), 2))
            rows.append([R(c1, den), R(c2 - c1, den), R(den - c2, den)])
        P = sp.Matrix(rows)
        pis = sp.symbols("p1:4")
        sol = sp.solve(list(sp.Matrix([pis]) * P - sp.Matrix([pis])) + [sum(pis) - 1], pis, dict=True)
        if sol:
            pi = [sol[0][s] for s in pis]
            if len(set(pi)) > 1:
                break
    return Problem(
        prompt=(f"A Markov chain on states $\\{{1,2,3\\}}$ has transition matrix $P = {L(P)}$. Find its stationary "
                "distribution $\\pi$ (solve $\\pi P = \\pi$, $\\sum \\pi_i = 1$)."),
        answer=[fmt(v) for v in pi], kind="vector", answer_hint="pi1, pi2, pi3   e.g. 1/4, 1/2, 1/4",
        rubric=["Write πP = π as linear equations (one is redundant)", "Add the normalisation Σπ = 1",
                "Solve the 3×3 system", "Check: π is a probability vector"],
        solution=f"$\\pi = \\left({', '.join(L(v) for v in pi)}\\right)$.",
        misconceptions=[([fmt(R(1, 3))] * 3, "assumed-uniform")],
    )


# =========================================================================== extra problem types
# Each skill-level must offer several genuinely different tasks (tests/test_variety.py); the generators
# below add new stories alongside the originals. Every builder takes rng and returns a Problem.
X_, Y_, U_ = sp.symbols("X Y U")
PHI_Z = [R(1, 2), 1, R(3, 2), 2]


# ---------------------------------------------------------------- counting
def _cnt_lock_code(rng):
    k = rng.randint(3, 6)
    ans = 9 * sp.Integer(10) ** (k - 1)
    return Problem(
        prompt=(f"A door code is a string of {k} digits (0–9). Digits may repeat, but the code may **not** begin "
                "with 0. How many different codes are possible?"),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=["First position: 9 choices (1–9)", f"Each of the other {k - 1} positions: 10 choices",
                "Multiplication rule"],
        solution=f"$9\\cdot 10^{{{k - 1}}} = {ans}$.",
        misconceptions=[(str(sp.Integer(10) ** k), "ignored-leading-restriction"),
                        (str(9 * sp.ff(9, k - 1)), "assumed-no-repetition")],
    )


def _cnt_books_together(rng):
    n = rng.randint(5, 9)
    ans = 2 * sp.factorial(n - 1)
    return Problem(
        prompt=(f"{n} different books are placed in a row on a shelf. In how many orders can they stand if two "
                "particular books must be **next to each other**?"),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=["Glue the two books into one block", f"Arrange {n - 1} objects: ({n - 1})!",
                "Multiply by 2 for the order inside the block"],
        solution=f"$2\\cdot {n - 1}! = 2\\cdot {sp.factorial(n - 1)} = {ans}$.",
        misconceptions=[(str(sp.factorial(n)), "ignored-adjacency"), (str(sp.factorial(n - 1)), "forgot-internal-order")],
    )


def _cnt_card_hand(rng):
    k, j = rng.choice([4, 5]), rng.randint(1, 3)
    ans = sp.binomial(13, j) * sp.binomial(39, k - j)
    return Problem(
        prompt=(f"A {k}-card hand is dealt from a standard 52-card deck (13 cards in each of the 4 suits). How many "
                f"different hands contain **exactly** {j} heart{'s' if j > 1 else ''}?"),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=[f"Choose the {j} heart(s) from 13", f"Choose the other {k - j} card(s) from the 39 non-hearts",
                "Multiply"],
        solution=(f"$\\binom{{13}}{{{j}}}\\binom{{39}}{{{k - j}}} = {sp.binomial(13, j)}\\cdot{sp.binomial(39, k - j)} "
                  f"= {ans}$."),
        misconceptions=[(str(sp.binomial(13, j) * sp.binomial(52 - j, k - j)), "rest-may-include-hearts"),
                        (str(sp.binomial(13, j)), "forgot-other-cards")],
    )


def _cnt_grid_paths(rng):
    a, b = rng.randint(3, 6), rng.randint(3, 6)
    c, d = rng.randint(1, a - 1), rng.randint(1, b - 1)
    first, second = sp.binomial(c + d, c), sp.binomial(a - c + b - d, a - c)
    ans = first * second
    return Problem(
        prompt=(f"A robot walks on a grid from $(0,0)$ to $({a},{b})$; every step is one unit right or one unit up. "
                f"How many such routes pass through the point $({c},{d})$?"),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=["A route = an arrangement of R and U steps", f"Routes to ({c},{d}): C({c + d},{c})",
                f"Routes from ({c},{d}) to ({a},{b}): C({a - c + b - d},{a - c})", "Multiply (multiplication rule)"],
        solution=(f"$\\binom{{{c + d}}}{{{c}}}\\binom{{{a - c + b - d}}}{{{a - c}}} = {first}\\cdot{second} = {ans}$."),
        misconceptions=[(str(sp.binomial(a + b, a)), "ignored-waypoint"), (str(first + second), "added-instead-of-multiplied")],
    )


def _cnt_round_table(rng):
    n = rng.randint(5, 9)
    ans = sp.factorial(n - 1) - 2 * sp.factorial(n - 2)
    return Problem(
        prompt=(f"{n} people sit around a round table; seatings that differ only by a rotation count as the same. Two "
                "of them, Ann and Bo, refuse to sit next to each other. How many seatings are possible?"),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=[f"All circular seatings: ({n}-1)! = {sp.factorial(n - 1)}",
                f"Seatings with Ann and Bo together: glue them, giving ({n}-2)!·2", "Subtract (complement)"],
        solution=(f"${n - 1}! - 2\\cdot{n - 2}! = {sp.factorial(n - 1)} - {2 * sp.factorial(n - 2)} = {ans}$."),
        misconceptions=[(str(sp.factorial(n) - 2 * sp.factorial(n - 1)), "linear-not-circular"),
                        (str(sp.factorial(n - 1)), "ignored-restriction"),
                        (str(sp.factorial(n - 1) - sp.factorial(n - 2)), "forgot-internal-order")],
    )


def _cnt_bounded_solutions(rng):
    u = rng.randint(3, 6)
    n = rng.randint(u + 2, 3 * u - 1)
    terms = [(j, sp.binomial(3, j) * sp.binomial(n - j * (u + 1) + 2, 2)) for j in range(4) if n - j * (u + 1) >= 0]
    ans = sum((-1) ** j * t for j, t in terms)
    sol_terms = " ".join(f"{'+' if j % 2 == 0 else '-'} \\binom{{3}}{{{j}}}\\binom{{{n - j * (u + 1) + 2}}}{{2}}"
                         for j, _ in terms).lstrip("+ ")
    return Problem(
        prompt=(f"How many integer solutions does $x_1 + x_2 + x_3 = {n}$ have with $0 \\le x_i \\le {u}$ for every $i$? "
                "(Stars and bars plus inclusion-exclusion on the violated upper bounds.)"),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=[f"Without upper bounds: C({n}+2, 2)",
                f"If x_i ≥ {u + 1}, substitute y_i = x_i - {u + 1}: C({n - u - 1}+2, 2)",
                "Inclusion-exclusion over which bounds are violated", "Evaluate"],
        solution=f"${sol_terms} = {ans}$.",
        misconceptions=[(str(sp.binomial(n + 2, 2)), "ignored-upper-bound"),
                        (str(sp.binomial(n + 2, 2) - 3 * sp.binomial(n - u + 1, 2)), "forgot-add-back")],
    )


def _cnt_no_consecutive(rng):
    n = rng.randint(6, 12)
    ans = sp.fibonacci(n + 2)
    return Problem(
        prompt=(f"How many subsets of $\\{{1, 2, \\dots, {n}\\}}$ (including the empty set) contain **no two consecutive** "
                f"integers? (Find a recursion by conditioning on whether ${n}$ is in the subset.)"),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=["Let a_m = number of good subsets of {1..m}",
                "If m is not used: a_{m-1}; if m is used, m-1 is not: a_{m-2}",
                "a_m = a_{m-1} + a_{m-2} with a_1 = 2, a_2 = 3", f"Iterate up to m = {n}"],
        solution=(f"$a_m = a_{{m-1}} + a_{{m-2}}$, $a_1 = 2$, $a_2 = 3$ (Fibonacci numbers): "
                  f"${', '.join(str(sp.fibonacci(m + 2)) for m in range(1, n + 1))}$, so $a_{{{n}}} = {ans}$."),
        misconceptions=[(str(2 ** (n - 1)), "guessed-half-of-subsets"), (str(sp.fibonacci(n + 1)), "off-by-one-recursion")],
    )


@gen("prob.counting", name="counting_stories")
def counting_stories(rng, level):
    if level == 1:
        return rng.choice([_cnt_lock_code, _cnt_books_together])(rng)
    if level == 2:
        return rng.choice([_cnt_card_hand, _cnt_grid_paths])(rng)
    return rng.choice([_cnt_round_table, _cnt_bounded_solutions, _cnt_no_consecutive])(rng)


# ---------------------------------------------------------------- axioms
def _ax_both_languages(rng):
    while True:
        pf, ps = rng.randint(20, 60), rng.randint(15, 50)
        both = rng.randint(5, min(pf, ps) - 2)
        pu = pf + ps - both
        if pu <= 95:
            break
    ans = R(both, 100)
    return Problem(
        prompt=(f"At a school, {pf}% of students study French, {ps}% study Spanish and {pu}% study at least one of the "
                "two. What is the probability that a randomly chosen student studies **both**?"),
        answer=fmt(ans), kind="num", answer_hint="a number, e.g. 0.15 or 3/20",
        rubric=["Write P(F ∪ S) = P(F) + P(S) - P(F ∩ S)", "Solve for P(F ∩ S)", "Evaluate"],
        solution=f"$P(F\\cap S) = {dec(R(pf, 100))} + {dec(R(ps, 100))} - {dec(R(pu, 100))} = {dec(ans)}$.",
        misconceptions=[(fmt(R(pf * ps, 10000)), "assumed-independence"), (fmt(R(pu, 100)), "union-not-intersection")],
    )


def _ax_independent_union(rng):
    a, b = R(rng.randint(1, 9), 10), R(rng.randint(1, 9), 10)
    ans = a + b - a * b
    return Problem(
        prompt=(f"Two smoke alarms work **independently**: in a fire, alarm A sounds with probability {dec(a)} and alarm B "
                f"with probability {dec(b)}. What is the probability that at least one alarm sounds?"),
        answer=fmt(ans), kind="num", answer_hint="a number, e.g. 0.94 or 47/50",
        rubric=["P(A ∪ B) = P(A) + P(B) - P(A ∩ B)", "Independence: P(A ∩ B) = P(A)P(B)",
                "(or: 1 - P(neither) = 1 - (1-a)(1-b))"],
        solution=f"$1 - ({dec(1 - a)})({dec(1 - b)}) = {dec(ans)}$.",
        misconceptions=[(fmt(a + b), "forgot-intersection"), (fmt(a * b), "computed-intersection-not-union")],
    )


def _ax_dice_both_faces(rng):
    k = rng.randint(3, 6)
    ans = 1 - 2 * R(5, 6) ** k + R(4, 6) ** k
    return Problem(
        prompt=(f"A fair die is rolled {k} times. What is the probability that **both** a 1 and a 6 appear "
                "(each at least once)?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Complement: no 1 OR no 6", f"P(no 1) = P(no 6) = (5/6)^{k}, P(no 1 and no 6) = (4/6)^{k}",
                "Inclusion-exclusion for the union", "Subtract from 1"],
        solution=(f"$1 - \\left[2\\left(\\tfrac56\\right)^{{{k}}} - \\left(\\tfrac46\\right)^{{{k}}}\\right] = {L(ans)} "
                  f"\\approx {float(ans):.4f}$."),
        misconceptions=[(fmt((1 - R(5, 6) ** k) ** 2), "assumed-independence"),
                        (fmt(1 - 2 * R(5, 6) ** k), "forgot-add-back")],
    )


def _ax_mixed_committee(rng):
    w, m = rng.randint(3, 7), rng.randint(3, 7)
    k = rng.randint(3, min(w, m))
    N = w + m
    ans = 1 - R(sp.binomial(w, k) + sp.binomial(m, k), sp.binomial(N, k))
    return Problem(
        prompt=(f"A {k}-person committee is chosen at random from {w} women and {m} men. What is the probability that "
                "the committee includes at least one woman **and** at least one man?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Complement: all women or all men (disjoint events)", f"P(all women) = C({w},{k})/C({N},{k})",
                f"P(all men) = C({m},{k})/C({N},{k})", "Subtract the sum from 1"],
        solution=(f"$1 - \\dfrac{{\\binom{{{w}}}{{{k}}} + \\binom{{{m}}}{{{k}}}}}{{\\binom{{{N}}}{{{k}}}}} = "
                  f"1 - \\dfrac{{{sp.binomial(w, k) + sp.binomial(m, k)}}}{{{sp.binomial(N, k)}}} = {L(ans)}$."),
        misconceptions=[(fmt(1 - R(sp.binomial(m, k), sp.binomial(N, k))), "only-excluded-one-case")],
    )


def _ax_overlap_bounds(rng):
    while True:
        a, b = R(rng.randint(3, 9), 10), R(rng.randint(3, 9), 10)
        if a + b > 1 and a != b:
            break
    lo, hi = a + b - 1, min(a, b)
    return Problem(
        prompt=(f"All you know about two events is that $P(A) = {dec(a)}$ and $P(B) = {dec(b)}$. What are the smallest "
                "and the largest values that $P(A\\cap B)$ can possibly take? (Justify both bounds from the axioms.)"),
        answer=[fmt(lo), fmt(hi)], kind="list", answer_hint="smallest, largest   e.g. 0.1, 0.4",
        rubric=["Upper bound: A ∩ B ⊆ A and ⊆ B, so P(A ∩ B) ≤ min(P(A), P(B))",
                "Lower bound: P(A ∪ B) ≤ 1 gives P(A ∩ B) ≥ P(A) + P(B) - 1",
                "Both bounds are attained (nested sets / union = whole space)"],
        solution=(f"Max: $\\min({dec(a)}, {dec(b)}) = {dec(hi)}$ (one event inside the other). "
                  f"Min: $P(A)+P(B)-1 = {dec(lo)}$ (union is everything)."),
        misconceptions=[([fmt(R(0)), fmt(hi)], "ignored-forced-overlap"), ([fmt(a * b), fmt(hi)], "assumed-independence")],
    )


@gen("prob.axioms", name="axioms_stories")
def axioms_stories(rng, level):
    if level == 1:
        return rng.choice([_ax_both_languages, _ax_independent_union])(rng)
    if level == 2:
        return rng.choice([_ax_dice_both_faces, _ax_mixed_committee])(rng)
    return _ax_overlap_bounds(rng)


# ---------------------------------------------------------------- conditional probability & Bayes
def _factory(rng):
    while True:
        a, b = rng.randint(2, 5) * 10, rng.randint(2, 5) * 10
        c = 100 - a - b
        if c >= 10:
            break
    shares = [R(a, 100), R(b, 100), R(c, 100)]
    rates = [R(rng.randint(1, 6), 100) for _ in range(3)]
    while len(set(rates)) == 1:
        rates[2] = R(rng.randint(1, 6), 100)
    txt = (f"Machines A, B and C make {a}%, {b}% and {c}% of a factory's output; {pct(rates[0])}, {pct(rates[1])} and "
           f"{pct(rates[2])} of their items, respectively, are defective.")
    return shares, rates, txt


def _cb_factory_total(rng):
    shares, rates, txt = _factory(rng)
    ans = sum(s * r for s, r in zip(shares, rates))
    return Problem(
        prompt=txt + " What is the probability that a randomly chosen item is defective?",
        answer=fmt(ans), kind="num", answer_hint="a number, e.g. 0.031 or 31/1000",
        rubric=["Partition by machine", "P(D) = Σ P(machine) P(D | machine)", "Evaluate"],
        solution=("$P(D) = " + " + ".join(f"{dec(s)}\\cdot{dec(r)}" for s, r in zip(shares, rates)) + f" = {dec(ans)}$."),
        misconceptions=[(fmt(sum(rates) / 3), "unweighted-average"), (fmt(sum(rates)), "added-rates")],
    )


def _cb_factory_reverse(rng):
    shares, rates, txt = _factory(rng)
    pd = sum(s * r for s, r in zip(shares, rates))
    i = rng.randint(0, 2)
    m = "ABC"[i]
    ans = shares[i] * rates[i] / pd
    return Problem(
        prompt=txt + f" An item is found to be defective. What is the probability that it was made by machine {m}?",
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["P(D) by the law of total probability", f"Bayes: P({m} | D) = P(D | {m})P({m}) / P(D)", "Simplify"],
        solution=(f"$P(D) = {dec(pd)}$, so $P({m}\\mid D) = \\dfrac{{{dec(shares[i])}\\cdot{dec(rates[i])}}}{{{dec(pd)}}}"
                  f" = {L(ans)}$."),
        misconceptions=[(fmt(shares[i]), "ignored-evidence"), (fmt(rates[i]), "likelihood-as-posterior")],
    )


def _cb_second_draw(rng):
    r, b = rng.randint(2, 9), rng.randint(2, 9)
    ans = R(r, r + b)
    return Problem(
        prompt=(f"A bag contains {r} red and {b} blue marbles. Two marbles are drawn one after the other **without** "
                "replacement, and nobody looks at the first one. What is the probability that the second marble is red?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Condition on the colour of the first marble", "P(R2) = P(R1)P(R2|R1) + P(B1)P(R2|B1)",
                "Simplify (symmetry: every draw is equally likely to be red)"],
        solution=(f"$\\frac{{{r}}}{{{r + b}}}\\cdot\\frac{{{r - 1}}}{{{r + b - 1}}} + \\frac{{{b}}}{{{r + b}}}\\cdot"
                  f"\\frac{{{r}}}{{{r + b - 1}}} = {L(ans)}$ — the same as for the first draw."),
        misconceptions=[(fmt(R(r - 1, r + b - 1)), "assumed-first-was-red"), (fmt(R(r, r + b - 1)), "assumed-first-was-blue")],
    )


def _cb_spam(rng):
    s = R(rng.choice([20, 30, 40, 50, 60]), 100)
    w1 = rng.choice([R(1, 2), R(2, 5), R(3, 10), R(1, 4), R(1, 5)])
    w2 = rng.choice([R(1, 50), R(1, 20), R(1, 25), R(1, 100), R(1, 10)])
    word = rng.choice(["prize", "winner", "free", "urgent"])
    ans = s * w1 / (s * w1 + (1 - s) * w2)
    return Problem(
        prompt=(f"{pct(s)} of incoming email is spam. The word *{word}* appears in {pct(w1)} of spam emails and in "
                f"{pct(w2)} of legitimate emails. An email contains *{word}*. What is the probability that it is spam?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Prior P(spam) and likelihoods P(word | spam), P(word | ham)", "P(word) by total probability",
                "Bayes: P(spam | word) = P(word | spam)P(spam)/P(word)"],
        solution=(f"$\\dfrac{{{dec(w1)}\\cdot{dec(s)}}}{{{dec(w1)}\\cdot{dec(s)} + {dec(w2)}\\cdot{dec(1 - s)}}} = {L(ans)} "
                  f"\\approx {float(ans):.4f}$."),
        misconceptions=[(fmt(w1), "likelihood-as-posterior"), (fmt(s), "ignored-evidence")],
    )


def _cb_monty(rng):
    n = rng.randint(4, 7)
    k = rng.randint(1, n - 2)
    ans = R(n - 1, n) * R(1, n - k - 1)
    return Problem(
        prompt=(f"A game show has {n} doors: one hides a car, the others are empty. You pick door 1. The host, who "
                f"**knows** where the car is, opens {k} of the other doors, always choosing empty ones. You then switch to "
                "one of the other unopened doors, chosen at random. What is your probability of winning the car?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=[f"Condition on whether door 1 hides the car (prob 1/{n})", "If it does, switching always loses",
                f"If not (prob {n - 1}/{n}), the car is behind one of the {n - k - 1} other closed doors",
                "Multiply and simplify"],
        solution=(f"$P(\\text{{win}}) = \\frac{{{n - 1}}}{{{n}}}\\cdot\\frac{{1}}{{{n - k - 1}}} = {L(ans)}$, "
                  f"better than staying (${L(R(1, n))}$)."),
        misconceptions=[(fmt(R(1, n - k)), "ignored-host-knowledge"), (fmt(R(1, n)), "switching-irrelevant")],
    )


@gen("prob.conditional_bayes", name="bayes_stories")
def bayes_stories(rng, level):
    if level == 1:
        return rng.choice([_cb_factory_total, _cb_second_draw])(rng)
    if level == 2:
        return rng.choice([_cb_spam, _cb_factory_reverse])(rng)
    return _cb_monty(rng)


# ---------------------------------------------------------------- discrete random variables
P_CHOICES = [R(1, 2), R(1, 3), R(2, 3), R(3, 4), R(1, 4), R(2, 5)]


def _dr_first_success(rng):
    p, k = rng.choice(P_CHOICES), rng.randint(2, 5)
    ans = (1 - p) ** (k - 1) * p
    return Problem(
        prompt=(f"A basketball player makes each free throw with probability ${L(p)}$, independently. What is the "
                f"probability that her **first** successful throw is throw number {k}?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=[f"First {k - 1} throws miss: (1-p)^{k - 1}", "Then a make: × p", "Geometric PMF (1-p)^(k-1) p"],
        solution=f"$\\left({L(1 - p)}\\right)^{{{k - 1}}}\\cdot {L(p)} = {L(ans)}$.",
        misconceptions=[(fmt((1 - p) ** k * p), "off-by-one-failures"), (fmt(p ** (k - 1) * (1 - p)), "swapped-success-failure")],
    )


def _dr_hypergeometric(rng):
    g, d, m = rng.randint(6, 12), rng.randint(2, 5), rng.randint(3, 5)
    j = rng.randint(1, min(d, m))
    ans = R(sp.binomial(d, j) * sp.binomial(g, m - j), sp.binomial(g + d, m))
    q = R(d, g + d)
    return Problem(
        prompt=(f"A box holds {g} good and {d} defective fuses. An inspector takes {m} fuses at random **without** "
                f"replacement. What is the probability that exactly {j} of them {'is' if j == 1 else 'are'} defective?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=[f"Total samples: C({g + d},{m})", f"Favourable: C({d},{j})·C({g},{m - j})",
                "Divide (Hypergeometric PMF)"],
        solution=(f"$\\dfrac{{\\binom{{{d}}}{{{j}}}\\binom{{{g}}}{{{m - j}}}}}{{\\binom{{{g + d}}}{{{m}}}}} = "
                  f"\\dfrac{{{sp.binomial(d, j) * sp.binomial(g, m - j)}}}{{{sp.binomial(g + d, m)}}} = {L(ans)}$."),
        misconceptions=[(fmt(sp.binomial(m, j) * q ** j * (1 - q) ** (m - j)), "used-binomial-with-replacement")],
    )


def _dr_linear_pmf(rng):
    n = rng.randint(4, 8)
    m = rng.randint(2, n - 1)
    c = R(2, n * (n + 1))
    ans = sum(c * k for k in range(m, n + 1))
    return Problem(
        prompt=(f"A random variable takes the values $1, 2, \\dots, {n}$ with $P(X = k) = c\\,k$. Find the constant $c$ "
                f"on paper, then compute $P(X \\ge {m})$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=[f"Normalise: c(1 + 2 + … + {n}) = 1", f"c = 2/({n}·{n + 1}) = {c}", f"Sum c·k for k = {m}..{n}",
                "Simplify"],
        solution=(f"$c = \\frac{{2}}{{{n}\\cdot{n + 1}}} = {L(c)}$; $P(X\\ge {m}) = {L(c)}({' + '.join(map(str, range(m, n + 1)))}) "
                  f"= {L(ans)}$."),
        misconceptions=[(fmt(sum(c * k for k in range(m + 1, n + 1))), "off-by-one-boundary"),
                        (fmt(R(n - m + 1, n)), "assumed-uniform")],
    )


def _dr_at_least_two(rng):
    n, p = rng.randint(4, 7), rng.choice([R(1, 2), R(1, 3), R(1, 4), R(1, 5)])
    p0, p1 = (1 - p) ** n, n * p * (1 - p) ** (n - 1)
    ans = 1 - p0 - p1
    return Problem(
        prompt=(f"Each component fails a stress test with probability ${L(p)}$, independently of the others. {n} "
                "components are tested. What is the probability that **at least two** fail?"),
        answer=fmt(ans), kind="num", tol=1e-4, answer_hint="an exact fraction, or a decimal to 4 places",
        rubric=[f"Number of failures X ~ Bin({n}, {p})", "Complement: P(X ≥ 2) = 1 - P(X = 0) - P(X = 1)",
                "Evaluate both terms", "Subtract"],
        solution=(f"$P(X=0) = {L(p0)}$, $P(X=1) = {n}\\cdot{L(p)}\\cdot\\left({L(1 - p)}\\right)^{{{n - 1}}} = {L(p1)}$, "
                  f"so $P(X\\ge2) = {L(ans)} \\approx {float(ans):.4f}$."),
        misconceptions=[(fmt(1 - p0), "complement-dropped-one-term"), (fmt(1 - p1), "complement-dropped-zero-term")],
    )


def _dr_thinning(rng):
    lam, q, k = rng.randint(2, 6), rng.choice([R(1, 2), R(1, 3), R(1, 4), R(2, 3)]), rng.randint(0, 4)
    mu = lam * (1 - q)
    ans = sp.exp(-mu) * mu ** k / sp.factorial(k)
    return Problem(
        prompt=(f"The number of emails you receive in an hour is $N \\sim \\mathrm{{Pois}}({lam})$. Each email is spam "
                f"with probability ${L(q)}$, independently. Let $X$ be the number of **non-spam** emails. Show that $X$ is "
                f"Poisson (condition on $N$ and sum) and compute $P(X = {k})$."),
        answer=str(ans), kind="num", tol=1e-4, answer_hint="exact, e.g. 2*exp(-3), or a decimal to 4 places",
        rubric=["P(X = k) = Σ_n P(N = n) C(n,k) (1-q)^k q^(n-k)",
                "Pull out e^{-λ}(λ(1-q))^k/k! and recognise Σ (λq)^m/m! = e^{λq}",
                f"So X ~ Pois(λ(1-q)) = Pois({L(mu)})", "Evaluate the PMF"],
        solution=(f"$X \\sim \\mathrm{{Pois}}(\\lambda(1-q)) = \\mathrm{{Pois}}({L(mu)})$, so "
                  f"$P(X={k}) = {L(ans)} \\approx {float(ans):.4f}$."),
        misconceptions=[(str(sp.exp(-lam) * sp.Integer(lam) ** k / sp.factorial(k)), "ignored-thinning"),
                        (str(sp.exp(-lam * q) * (lam * q) ** k / sp.factorial(k)), "used-spam-rate")],
    )


def _dr_poisson_reciprocal(rng):
    lam = rng.randint(1, 5)
    ans = (1 - sp.exp(-lam)) / lam
    return Problem(
        prompt=(f"$X \\sim \\mathrm{{Pois}}({lam})$. Use LOTUS to find $E\\left[\\dfrac{{1}}{{X+1}}\\right]$ exactly "
                "(rewrite the sum as a Poisson series with one term missing)."),
        answer=str(ans), kind="num", tol=1e-4, answer_hint="exact, e.g. (1-exp(-3))/3, or a decimal to 4 places",
        rubric=["LOTUS: Σ_k 1/(k+1) · e^{-λ} λ^k / k!", "Note 1/((k+1)·k!) = 1/(k+1)!",
                "Shift index: (e^{-λ}/λ) Σ_{m≥1} λ^m/m! = (e^{-λ}/λ)(e^λ - 1)", "Simplify to (1 - e^{-λ})/λ"],
        solution=(f"$E\\left[\\frac1{{X+1}}\\right] = \\frac{{e^{{-\\lambda}}}}{{\\lambda}}\\sum_{{m\\ge1}}\\frac{{\\lambda^m}}{{m!}} "
                  f"= \\frac{{1-e^{{-\\lambda}}}}{{\\lambda}} = {L(ans)} \\approx {float(ans):.4f}$."),
        misconceptions=[(fmt(R(1, lam + 1)), "jensen-swap-expectation-inside"), (fmt(R(1, lam)), "dropped-missing-term")],
    )


@gen("prob.discrete_rv", name="discrete_stories")
def discrete_stories(rng, level):
    if level == 1:
        return rng.choice([_dr_first_success, _dr_hypergeometric])(rng)
    if level == 2:
        return rng.choice([_dr_linear_pmf, _dr_at_least_two])(rng)
    return rng.choice([_dr_thinning, _dr_poisson_reciprocal])(rng)


# ---------------------------------------------------------------- expectation & variance
def _ev_game(rng):
    f = rng.randint(1, 3)
    w, l = rng.randint(4, 20), rng.randint(1, 8)
    p = R(f, 6)
    ans = p * w - (1 - p) * l
    faces = "a 6" if f == 1 else ("a 5 or 6" if f == 2 else "4, 5 or 6")
    return Problem(
        prompt=(f"A fair die is rolled. If it shows {faces} you win {w} points; otherwise you lose {l} points. "
                "What are your expected winnings per game?"),
        answer=fmt(ans), kind="num", answer_hint="an exact value, e.g. -1/3",
        rubric=["Winnings W takes two values", f"P(win) = {f}/6", "E[W] = Σ w P(W = w)", "Simplify (a negative answer is allowed)"],
        solution=f"$E[W] = {L(p)}\\cdot{w} - {L(1 - p)}\\cdot{l} = {L(ans)}$.",
        misconceptions=[(fmt(R(w - l, 2)), "unweighted-mean"), (fmt(p * w), "ignored-losses")],
    )


def _ev_die_variance(rng):
    n = rng.choice([4, 6, 8, 10, 12, 20])
    ex = R(n + 1, 2)
    ex2 = R(sum(k * k for k in range(1, n + 1)), n)
    var = ex2 - ex ** 2
    return Problem(
        prompt=f"A fair {n}-sided die has faces $1, 2, \\dots, {n}$. Compute the variance of a single roll.",
        answer=fmt(var), kind="num", answer_hint=FRAC_HINT,
        rubric=["E[X] = (n+1)/2", "E[X²] = (1² + … + n²)/n = (n+1)(2n+1)/6", "Var = E[X²] - (E[X])²"],
        solution=f"$E[X] = {L(ex)}$, $E[X^2] = {L(ex2)}$, $\\mathrm{{Var}}(X) = {L(ex2)} - \\left({L(ex)}\\right)^2 = {L(var)}$.",
        misconceptions=[(fmt(ex2), "forgot-subtract-mean-square"), (fmt(ex2 - ex), "variance-formula-no-square")],
    )


def _ev_independent_combo(rng):
    n, p, lam = rng.randint(4, 20), rng.choice([R(1, 2), R(1, 3), R(1, 4), R(2, 5)]), rng.randint(1, 6)
    a, b = rng.randint(2, 4), rng.randint(2, 4)
    vx, vy = n * p * (1 - p), sp.Integer(lam)
    ans = a ** 2 * vx + b ** 2 * vy
    return Problem(
        prompt=(f"$X \\sim \\mathrm{{Bin}}({n}, {L(p)})$ and $Y \\sim \\mathrm{{Pois}}({lam})$ are independent. "
                f"Find $\\mathrm{{Var}}({L(a * X_ - b * Y_)})$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Var(Bin(n,p)) = np(1-p); Var(Pois(λ)) = λ", "Independence: no covariance term",
                "Var(aX - bY) = a²Var X + b²Var Y (variances ADD)"],
        solution=(f"$\\mathrm{{Var}}(X) = {L(vx)}$, $\\mathrm{{Var}}(Y) = {lam}$, so "
                  f"${a ** 2}\\cdot{L(vx)} + {b ** 2}\\cdot{lam} = {L(ans)}$."),
        misconceptions=[(fmt(a ** 2 * vx - b ** 2 * vy), "subtracted-variances"),
                        (fmt(a * vx + b * vy), "forgot-square-scale")],
    )


def _ev_play_until_win(rng):
    p = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(1, 5), R(1, 6), R(2, 5)])
    c, prize = rng.randint(2, 5), rng.randint(10, 40)
    ans = prize - c / p
    return Problem(
        prompt=(f"Each play of a carnival game costs {c} points and wins with probability ${L(p)}$, independently. You "
                f"keep playing until your first win, which pays {prize} points, and then stop. What is your expected net "
                "gain (prize minus total cost)?"),
        answer=fmt(ans), kind="num", answer_hint="an exact value, e.g. -5/2",
        rubric=["Number of plays N ~ Geometric on {1,2,…} with E[N] = 1/p", "Net gain = prize - c·N",
                "Linearity: prize - c·E[N]"],
        solution=f"$E[N] = 1/p = {L(1 / p)}$, so $E[\\text{{gain}}] = {prize} - {c}\\cdot{L(1 / p)} = {L(ans)}$.",
        misconceptions=[(fmt(prize - c * (1 / p - 1)), "forgot-winning-play-cost"), (fmt(sp.Integer(prize - c)), "single-play")],
    )


def _ev_trials_variance(rng):
    p = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(1, 5), R(2, 3), R(2, 5)])
    var = (1 - p) / p ** 2
    return Problem(
        prompt=(f"You roll a weighted die until the first 6; each roll shows 6 with probability ${L(p)}$. Let $N$ be the "
                "number of rolls. Using first-step analysis (condition on the first roll), find $E[N]$ and $E[N^2]$, "
                "then type $\\mathrm{Var}(N)$."),
        answer=fmt(var), kind="num", answer_hint=FRAC_HINT,
        rubric=["E[N] = 1 + (1-p)E[N], so E[N] = 1/p", "E[N²] = p·1 + (1-p)E[(1+N)²]",
                "Solve: E[N²] = (2-p)/p²", "Var = E[N²] - (E[N])² = (1-p)/p²"],
        solution=(f"$E[N] = {L(1 / p)}$, $E[N^2] = \\frac{{2-p}}{{p^2}} = {L((2 - p) / p ** 2)}$, "
                  f"$\\mathrm{{Var}}(N) = \\frac{{1-p}}{{p^2}} = {L(var)}$."),
        misconceptions=[(fmt((1 - p) / p), "used-failure-count-mean"), (fmt((2 - p) / p ** 2), "forgot-subtract-mean-square")],
    )


@gen("prob.expectation_variance", name="ev_stories")
def ev_stories(rng, level):
    if level == 1:
        return rng.choice([_ev_game, _ev_die_variance])(rng)
    if level == 2:
        return rng.choice([_ev_independent_combo, _ev_play_until_win])(rng)
    return _ev_trials_variance(rng)


# ---------------------------------------------------------------- continuous random variables
def _cr_bus_uniform(rng):
    T = rng.choice([10, 12, 15, 20, 30])
    a, b = sorted(rng.sample(range(1, T), 2))
    ans = R(b - a, T)
    return Problem(
        prompt=(f"A bus arrives at a uniformly random time between 0 and {T} minutes after you reach the stop. "
                f"What is the probability that you wait between {a} and {b} minutes?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=[f"Waiting time X ~ Unif(0, {T}), density 1/{T}", "P(a < X < b) = (b - a)/length", "Simplify"],
        solution=f"$P({a} < X < {b}) = \\frac{{{b} - {a}}}{{{T}}} = {L(ans)}$.",
        misconceptions=[(fmt(R(b, T)), "used-cdf-at-upper-only"), (fmt(R(b - a, b)), "wrong-interval-length")],
    )


def _cr_pareto_median(rng):
    c, k = rng.choice([1, 2, 3, 5, 10]), rng.choice([2, 3, 4])
    ans = c * sp.Integer(2) ** R(1, k)
    return Problem(
        prompt=(f"Insurance claim sizes (in thousands) follow a Pareto law with CDF $F(x) = 1 - "
                f"\\left(\\frac{{{c}}}{{x}}\\right)^{{{k}}}$ for $x \\ge {c}$. Find the **median** claim size."),
        answer=str(ans), kind="num", tol=1e-4, answer_hint="exact, e.g. 3*2^(1/3), or a decimal to 4 places",
        rubric=["Median m solves F(m) = 1/2", f"(c/m)^{k} = 1/2", f"m = c·2^(1/{k})"],
        solution=f"$\\left(\\frac{{{c}}}{{m}}\\right)^{{{k}}} = \\frac12 \\Rightarrow m = {c}\\cdot 2^{{1/{k}}} \\approx {float(ans):.4f}$.",
        misconceptions=[(str(2 * c), "forgot-root"), (fmt(R(k * c, k - 1)), "mean-not-median")],
    )


def _cr_triangular(rng):
    a = rng.choice([2, 3, 4, 5, 6])
    b = rng.randint(1, a - 1)
    xx = sp.Symbol("x")
    c = 1 / sp.integrate(a - xx, (xx, 0, a))
    ans = sp.integrate(c * (a - xx), (xx, 0, b))
    return Problem(
        prompt=(f"A random distance $X$ (km) has a density proportional to ${a} - x$ on $[0, {a}]$, i.e. "
                f"$f(x) = c\\,({a} - x)$ there and 0 elsewhere. Find $c$ on paper, then compute $P(X < {b})$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Normalise: ∫_0^a c(a - x) dx = c a²/2 = 1", f"c = {L(c)}", f"P(X < {b}) = ∫_0^{b} c({a} - x) dx",
                "Simplify"],
        solution=(f"$c = \\frac{{2}}{{{a}^2}} = {L(c)}$, $P(X<{b}) = {L(c)}\\left[{a}x - \\tfrac{{x^2}}{{2}}\\right]_0^{{{b}}} "
                  f"= {L(ans)}$."),
        misconceptions=[(fmt(R(b, a)), "assumed-uniform"), (fmt(sp.integrate(a - xx, (xx, 0, b))), "forgot-normalising-constant")],
    )


def _cr_broken_stick(rng):
    Lg, k = rng.choice([1, 2, 4, 6, 10, 12]), rng.randint(2, 5)
    e_long, p = R(3 * Lg, 4), R(2, k + 1)
    return Problem(
        prompt=(f"A stick of length {Lg} is broken at a uniformly random point. (a) Find the expected length of the "
                f"**longer** piece. (b) Find the probability that the longer piece is at least {k} times as long as the "
                "shorter piece."),
        answer=[fmt(e_long), fmt(p)], kind="list", answer_hint="E[longer], probability   e.g. 3/2, 1/3",
        rubric=["Break point U ~ Unif(0, L); longer = max(U, L - U)", "By symmetry, longer ~ Unif(L/2, L): mean 3L/4",
                f"Longer ≥ {k}·shorter ⇔ shorter ≤ L/{k + 1}", f"⇔ U within L/{k + 1} of an end: probability 2/{k + 1}"],
        solution=(f"(a) $E[\\max] = \\tfrac34\\cdot{Lg} = {L(e_long)}$. (b) $\\min(U, {Lg}-U) \\le \\frac{{{Lg}}}{{{k + 1}}}$ "
                  f"has probability $\\frac{{2}}{{k+1}} = {L(p)}$."),
        misconceptions=[([fmt(R(Lg, 2)), fmt(p)], "longer-piece-mean-is-half"), ([fmt(e_long), fmt(R(1, k + 1))], "one-end-only")],
    )


def _cr_meeting(rng):
    w = rng.choice([10, 12, 15, 20, 30])
    q = R(w, 60)
    ans = 1 - (1 - q) ** 2
    return Problem(
        prompt=(f"Two friends each arrive at a café at a uniformly random time between 12:00 and 13:00, independently. "
                f"Each waits {w} minutes for the other and then leaves. What is the probability that they meet?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Model (X, Y) uniform on the 60 × 60 square", f"They meet iff |X - Y| ≤ {w}",
                f"Complement = two corner triangles, total area (60 - {w})²", "Divide by 60² and subtract from 1"],
        solution=f"$1 - \\left(\\frac{{60-{w}}}{{60}}\\right)^2 = 1 - \\left({L(1 - q)}\\right)^2 = {L(ans)}$.",
        misconceptions=[(fmt(q), "one-sided-wait"), (fmt(2 * q), "ignored-corner-overlap")],
    )


@gen("prob.continuous_rv", name="continuous_stories")
def continuous_stories(rng, level):
    if level == 1:
        return _cr_bus_uniform(rng)
    if level == 2:
        return rng.choice([_cr_pareto_median, _cr_triangular])(rng)
    return rng.choice([_cr_broken_stick, _cr_meeting])(rng)


# ---------------------------------------------------------------- joint distributions & covariance
def _jc_bilinear_shift(rng):
    c = nz(rng, -6, 6)
    a, d = nz(rng, -4, 4), nz(rng, -4, 4)
    b, e = nz(rng, -9, 9), nz(rng, -9, 9)
    ans = a * d * c
    return Problem(
        prompt=(f"Given only that $\\mathrm{{Cov}}(X, Y) = {c}$, use bilinearity to compute "
                f"$\\mathrm{{Cov}}({L(a * X_ + b)},\\ {L(d * Y_ + e)})$."),
        answer=str(ans), kind="num",
        rubric=["Adding constants does not change covariance", "Cov(aX, dY) = a·d·Cov(X, Y)", "Evaluate (watch signs)"],
        solution=f"$({a})({d})\\cdot({c}) = {ans}$.",
        misconceptions=[(str(ans + b * e), "shift-changes-covariance"), (str(a * a * d * d * c), "squared-scale-factors")],
    )


def _jc_correlation(rng):
    sx, sy = rng.sample([1, 2, 3, 4, 5], 2)
    cov = nz(rng, -(sx * sy - 1), sx * sy - 1)
    ans = R(cov, sx * sy)
    return Problem(
        prompt=(f"The daily returns $X$ and $Y$ of two stocks have variances ${sx * sx}$ and ${sy * sy}$ and covariance "
                f"${cov}$. What is their correlation?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Corr = Cov / (SD_X · SD_Y)", "SD = square root of the variance", "Simplify"],
        solution=f"$\\rho = \\dfrac{{{cov}}}{{\\sqrt{{{sx * sx}}}\\sqrt{{{sy * sy}}}}} = {L(ans)}$.",
        misconceptions=[(fmt(R(cov, sx * sx * sy * sy)), "divided-by-variances")],
    )


def _jc_die_and_sum(rng):
    n = rng.choice([4, 6, 8, 10, 12])
    var = R(n * n - 1, 12)
    return Problem(
        prompt=(f"A fair {n}-sided die is rolled twice. Let $X$ be the first roll and $S$ the **sum** of both rolls. "
                "Compute $\\mathrm{Cov}(X, S)$."),
        answer=fmt(var), kind="num", answer_hint=FRAC_HINT,
        rubric=["S = X + Y with X, Y independent", "Cov(X, X + Y) = Var X + Cov(X, Y) = Var X",
                "Var of one roll = (n² - 1)/12 (or E[X²] - E[X]²)"],
        solution=f"$\\mathrm{{Cov}}(X, S) = \\mathrm{{Var}}(X) = \\frac{{{n}^2-1}}{{12}} = {L(var)}$.",
        misconceptions=[(fmt(R(0)), "assumed-independent"), (fmt(2 * var), "used-var-of-sum")],
    )


def _jc_bilinear_combo(rng):
    while True:
        vx, vy, cxy = rng.randint(1, 9), rng.randint(1, 9), nz(rng, -4, 4)
        if cxy * cxy < vx * vy:
            break
    a, b, c, d = (nz(rng, -3, 3) for _ in range(4))
    ans = a * c * vx + b * d * vy + (a * d + b * c) * cxy
    return Problem(
        prompt=(f"Expand by bilinearity. We know $\\mathrm{{Var}}(X) = {vx}$, $\\mathrm{{Var}}(Y) = {vy}$ and "
                f"$\\mathrm{{Cov}}(X, Y) = {cxy}$. Find $\\mathrm{{Cov}}({L(a * X_ + b * Y_)},\\ {L(c * X_ + d * Y_)})$."),
        answer=str(ans), kind="num",
        rubric=["Expand into four covariance terms", "Cov(X, X) = Var X, Cov(Y, Y) = Var Y",
                "Collect: ac·VarX + bd·VarY + (ad + bc)·Cov(X,Y)", "Evaluate"],
        solution=(f"$({a})({c})\\cdot{vx} + ({b})({d})\\cdot{vy} + [({a})({d}) + ({b})({c})]\\cdot({cxy}) = {ans}$."),
        misconceptions=[(str(a * c * vx + b * d * vy), "forgot-cross-terms"),
                        (str(a * c * vx + b * d * vy + a * d * cxy), "dropped-one-cross-term")],
    )


def _jc_density_cov(rng):
    k = rng.randint(1, 4)
    xx, yy = sp.symbols("x y")
    f0 = xx + k * yy
    c = 1 / sp.integrate(f0, (xx, 0, 1), (yy, 0, 1))
    f = c * f0

    def ev(g):
        return sp.integrate(g * f, (xx, 0, 1), (yy, 0, 1))

    ex, ey, exy = ev(xx), ev(yy), ev(xx * yy)
    cov = sp.nsimplify(exy - ex * ey)
    return Problem(
        prompt=(f"$(X, Y)$ has joint density $f(x, y) = c\\,({L(f0)})$ on the unit square $0 \\le x, y \\le 1$. Find $c$, "
                "$E[X]$, $E[Y]$ and $E[XY]$ on paper, and type $\\mathrm{Cov}(X, Y)$."),
        answer=fmt(cov), kind="num", answer_hint=FRAC_HINT,
        rubric=["Normalise: ∬ f = 1 gives c", "E[X] = ∬ x f, E[Y] = ∬ y f", "E[XY] = ∬ xy f",
                "Cov = E[XY] - E[X]E[Y] (expect a small negative number)"],
        solution=(f"$c = {L(c)}$, $E[X] = {L(ex)}$, $E[Y] = {L(ey)}$, $E[XY] = {L(exy)}$, so "
                  f"$\\mathrm{{Cov}} = {L(exy)} - {L(ex)}\\cdot{L(ey)} = {L(cov)}$."),
        misconceptions=[(fmt(R(0)), "assumed-independent"), (fmt(exy), "cov-is-E-XY")],
    )


def _jc_hypergeom_var(rng):
    r, b = rng.randint(3, 8), rng.randint(3, 8)
    N = r + b
    k = rng.randint(2, N - 2)
    p = R(r, N)
    ans = k * p * (1 - p) * R(N - k, N - 1)
    cij = R(r * (r - 1), N * (N - 1)) - p ** 2
    return Problem(
        prompt=(f"An urn holds {r} red and {b} blue balls; {k} are drawn **without** replacement. Let $X$ be the number "
                "of red balls drawn. Writing $X = \\sum_j I_j$ ($I_j$ = draw $j$ is red), use $\\mathrm{Var}(I_j)$ and "
                "$\\mathrm{Cov}(I_i, I_j)$ to find $\\mathrm{Var}(X)$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=[f"P(I_j = 1) = {r}/{N}, so Var(I_j) = p(1-p)", f"P(I_i = I_j = 1) = {r}·{r - 1}/({N}·{N - 1})",
                "Var X = k Var(I_1) + k(k-1) Cov(I_1, I_2)", "Simplify (= kp(1-p)(N-k)/(N-1))"],
        solution=(f"$\\mathrm{{Cov}}(I_1, I_2) = {L(cij)}$, so $\\mathrm{{Var}}(X) = {k}\\cdot{L(p * (1 - p))} + "
                  f"{k * (k - 1)}\\cdot\\left({L(cij)}\\right) = {L(ans)}$."),
        misconceptions=[(fmt(k * p * (1 - p)), "used-binomial-variance")],
    )


@gen("prob.joint_covariance", name="joint_stories")
def joint_stories(rng, level):
    if level == 1:
        return rng.choice([_jc_bilinear_shift, _jc_correlation])(rng)
    if level == 2:
        return rng.choice([_jc_die_and_sum, _jc_bilinear_combo])(rng)
    return rng.choice([_jc_density_cov, _jc_hypergeom_var])(rng)


# ---------------------------------------------------------------- LLN & CLT
def _cl_markov(rng):
    mu = rng.randint(3, 20)
    a = mu * rng.randint(2, 6) + rng.randint(0, mu - 1)
    ans = R(mu, a)
    return Problem(
        prompt=(f"The waiting time $T$ at a clinic is non-negative with mean {mu} minutes; nothing else is known about its "
                f"distribution. Use **Markov's inequality** to bound $P(T \\ge {a})$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Markov: P(T ≥ a) ≤ E[T]/a for T ≥ 0", "Substitute", "Simplify"],
        solution=f"$P(T\\ge {a}) \\le \\frac{{{mu}}}{{{a}}} = {L(ans)}$.",
        misconceptions=[(fmt(1 - ans), "complement-confusion"), (fmt(R(mu * mu, a * a)), "used-chebyshev-form")],
    )


def _cl_chebyshev(rng):
    sigma = rng.randint(2, 8)
    c = sigma * rng.randint(2, 4) + rng.randint(0, sigma - 1)
    mu = rng.randint(100, 500)
    ans = R(sigma ** 2, c ** 2)
    return Problem(
        prompt=(f"A machine fills bags with mean weight {mu} g and standard deviation {sigma} g (distribution unknown). "
                f"Use **Chebyshev's inequality** to bound the probability that a bag is off by at least {c} g."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Chebyshev: P(|X - μ| ≥ c) ≤ σ²/c²", "Use the variance σ², not σ", "Simplify"],
        solution=f"$P(|X-{mu}|\\ge {c}) \\le \\frac{{{sigma}^2}}{{{c}^2}} = {L(ans)}$.",
        misconceptions=[(fmt(R(sigma, c)), "forgot-square"), (fmt(1 - R(sigma ** 2, c ** 2)), "complement-confusion")],
    )


def _cl_coin_normal(rng):
    n = rng.choice([100, 400, 900, 1600, 2500])
    sd = R(math.isqrt(n), 2)
    z = rng.choice(PHI_Z)
    thr = R(n, 2) + z * sd
    ans = 1 - PHI[z]
    return Problem(
        prompt=(f"A fair coin is flipped {n} times. Using the normal approximation to the Binomial (no continuity "
                f"correction) and {PHI_TABLE}, approximate the probability of getting more than {dec(thr)} heads."),
        answer=fmt(ans), kind="num", tol=1e-3, answer_hint="a decimal, e.g. 0.0668",
        rubric=["Heads ~ Bin(n, 1/2): mean n/2, SD √n/2", "Standardise the threshold", "Upper tail 1 - Φ(z)"],
        solution=(f"Mean ${n // 2}$, SD $\\sqrt{{{n}}}/2 = {L(sd)}$, $z = \\frac{{{dec(thr)} - {n // 2}}}{{{L(sd)}}} = "
                  f"{L(sp.nsimplify(z))}$, so $P \\approx 1 - \\Phi({L(sp.nsimplify(z))}) = {dec(ans)}$."),
        misconceptions=[(fmt(PHI[z]), "wrong-tail"), (fmt(2 * (1 - PHI[z])), "two-tailed-for-one-tailed")],
    )


def _cl_poisson_lower(rng):
    lam, n = rng.choice([1, 4]), rng.choice([25, 36, 64, 100])
    sd = math.isqrt(n * lam)
    z = rng.choice(PHI_Z)
    thr = n * lam - z * sd
    ans = 1 - PHI[z]
    return Problem(
        prompt=(f"A shop's number of customers per day is Poisson with mean {lam}, independently from day to day. Using "
                f"the CLT and {PHI_TABLE}, approximate the probability that the total over {n} days is **below** "
                f"{dec(thr)}."),
        answer=fmt(ans), kind="num", tol=1e-3, answer_hint="a decimal, e.g. 0.0668",
        rubric=["Total ~ approx Normal with mean nλ and variance nλ (Poisson: variance = mean)", "Standardise",
                "Lower tail: Φ(-z) = 1 - Φ(z)"],
        solution=(f"Mean ${n * lam}$, SD $\\sqrt{{{n * lam}}} = {sd}$, $z = -{L(sp.nsimplify(z))}$, so "
                  f"$P \\approx \\Phi(-{L(sp.nsimplify(z))}) = {dec(ans)}$."),
        misconceptions=[(fmt(PHI[z]), "wrong-tail")],
    )


def _cl_monte_carlo(rng):
    m = rng.randint(1, 3)
    eps = rng.choice([R(1, 100), R(1, 50), R(1, 20)])
    var = R(1, 2 * m + 1) - R(1, (m + 1) ** 2)
    ans = sp.ceiling(var / eps ** 2)
    g = L(U_ ** m)
    return Problem(
        prompt=(f"To estimate $\\int_0^1 {L(sp.Symbol('x') ** m)}\\,dx$ by Monte Carlo you average ${g}$ over $n$ i.i.d. "
                f"draws $U \\sim \\mathrm{{Unif}}(0,1)$. Compute $\\mathrm{{Var}}({g})$ and find the smallest $n$ for which "
                f"the standard deviation of the estimate is at most ${dec(eps)}$."),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=[f"E[U^m] = 1/(m+1), E[U^(2m)] = 1/(2m+1)", f"Var = {L(var)}",
                "SD of the average = √(Var/n)", "Solve √(Var/n) ≤ ε and round n up"],
        solution=(f"$\\mathrm{{Var}}({g}) = \\frac{{1}}{{{2 * m + 1}}} - \\frac{{1}}{{{(m + 1) ** 2}}} = {L(var)}$; "
                  f"$n \\ge {L(var)}/{L(eps ** 2)} = {L(var / eps ** 2)}$, so $n = {ans}$."),
        misconceptions=[(str(sp.ceiling(R(1, 2 * m + 1) / eps ** 2)), "forgot-subtract-mean-square"),
                        (str(sp.ceiling(var / eps)), "forgot-square-epsilon")],
    )


def _cl_rounding(rng):
    n = rng.choice([12, 48, 108, 192, 300])
    sd = math.isqrt(n // 12)
    z = rng.choice(PHI_Z)
    t = z * sd
    ans = 2 * (1 - PHI[z])
    return Problem(
        prompt=(f"{n} amounts are each rounded to the nearest whole number before being added; model the rounding "
                f"errors as i.i.d. $\\mathrm{{Unif}}(-\\tfrac12, \\tfrac12)$. Use the CLT and {PHI_TABLE} to approximate "
                f"the probability that the total rounding error exceeds ${dec(t)}$ in absolute value."),
        answer=fmt(ans), kind="num", tol=1e-3, answer_hint="a decimal, e.g. 0.1336",
        rubric=["One error: mean 0, variance 1/12", f"Total: variance {n}/12, SD {sd}", "Standardise",
                "Two tails: 2(1 - Φ(z))"],
        solution=(f"$\\mathrm{{SD}} = \\sqrt{{{n}/12}} = {sd}$, $z = {dec(t)}/{sd} = {L(sp.nsimplify(z))}$, "
                  f"$P \\approx 2(1-\\Phi({L(sp.nsimplify(z))})) = {dec(ans)}$."),
        misconceptions=[(fmt(1 - PHI[z]), "one-sided-tail"), (fmt(PHI[z]), "wrong-tail")],
    )


@gen("prob.lln_clt", name="clt_stories")
def clt_stories(rng, level):
    if level == 1:
        return rng.choice([_cl_markov, _cl_chebyshev])(rng)
    if level == 2:
        return rng.choice([_cl_coin_normal, _cl_poisson_lower])(rng)
    return rng.choice([_cl_monte_carlo, _cl_rounding])(rng)


# ---------------------------------------------------------------- Bayesian inference
def _bi_three_coins(rng):
    n = rng.randint(2, 4)
    k = rng.randint(0, n)
    ps = [R(1, 4), R(1, 2), R(3, 4)]
    target = rng.choice(ps)
    like = [p ** k * (1 - p) ** (n - k) for p in ps]
    ans = like[ps.index(target)] / sum(like)
    return Problem(
        prompt=(f"A coin's heads-probability $p$ is equally likely to be $\\tfrac14$, $\\tfrac12$ or $\\tfrac34$ (discrete "
                f"prior). You flip it {n} times and see {k} head{'' if k == 1 else 's'}. What is the posterior probability that "
                f"$p = {L(target)}$?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Likelihood p^k (1-p)^(n-k) for each of the 3 values", "Prior is uniform, so posterior ∝ likelihood",
                "Normalise over the three values"],
        solution=("Likelihoods " + ", ".join(f"${L(v)}$" for v in like) + f"; posterior $= {L(like[ps.index(target)])}"
                  f"/{L(sum(like))} = {L(ans)}$."),
        misconceptions=[(fmt(R(1, 3)), "ignored-data"), (fmt(like[ps.index(target)]), "likelihood-as-posterior")],
    )


def _bi_normal_one(rng):
    m0, x = rng.randint(-5, 10), rng.randint(-5, 15)
    while x == m0:
        x = rng.randint(-5, 15)
    tau2, sig2 = rng.sample([1, 2, 3, 4, 9], 2)
    ans = (R(m0, tau2) + R(x, sig2)) / (R(1, tau2) + R(1, sig2))
    return Problem(
        prompt=(f"A sensor reading satisfies $X \\mid \\mu \\sim \\mathcal N(\\mu, {sig2})$ and the prior is "
                f"$\\mu \\sim \\mathcal N({m0}, {tau2})$ (second parameters are variances). You observe $X = {x}$. "
                "What is the posterior mean of $\\mu$?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Normal-Normal conjugacy: precisions add", "Posterior mean = precision-weighted average of prior mean and data",
                f"(m0/τ² + x/σ²)/(1/τ² + 1/σ²)"],
        solution=(f"$\\dfrac{{{m0}/{tau2} + {x}/{sig2}}}{{1/{tau2} + 1/{sig2}}} = {L(ans)}$."),
        misconceptions=[(fmt(sp.Integer(x)), "ignored-prior"), (fmt(R(m0 + x, 2)), "unweighted-average"),
                        ((fmt((R(m0, sig2) + R(x, tau2)) / (R(1, tau2) + R(1, sig2)))), "swapped-weights")],
    )


def _bi_normal_n(rng):
    m0, tau2, sig2 = rng.randint(0, 20), rng.choice([1, 4, 9]), rng.choice([4, 9, 16, 25])
    n = rng.randint(2, 9)
    xbar = m0 + nz(rng, -6, 6)
    prec = R(1, tau2) + R(n, sig2)
    pv = 1 / prec
    pm = (R(m0, tau2) + R(n, sig2) * xbar) * pv
    return Problem(
        prompt=(f"Batch weights satisfy $X_i \\mid \\mu \\sim \\mathcal N(\\mu, {sig2})$ i.i.d., with prior "
                f"$\\mu \\sim \\mathcal N({m0}, {tau2})$ (variances). A sample of $n = {n}$ has mean $\\bar x = {xbar}$. "
                "Find the posterior mean and variance of $\\mu$."),
        answer=[fmt(pm), fmt(pv)], kind="list", answer_hint="mean, variance   e.g. 12/5, 4/5",
        rubric=["Sufficient statistic: x̄ ~ N(μ, σ²/n)", "Posterior precision = 1/τ² + n/σ²",
                "Posterior mean = (m0/τ² + n x̄/σ²) / precision"],
        solution=(f"Precision ${L(R(1, tau2))} + {L(R(n, sig2))} = {L(prec)}$, variance ${L(pv)}$, mean "
                  f"$\\left({L(R(m0, tau2))} + {L(R(n, sig2))}\\cdot{xbar}\\right)\\cdot{L(pv)} = {L(pm)}$."),
        misconceptions=[([fmt(sp.Integer(xbar)), fmt(R(sig2, n))], "ignored-prior"),
                        ([fmt((R(m0, tau2) + R(xbar, sig2)) / (R(1, tau2) + R(1, sig2))), fmt(1 / (R(1, tau2) + R(1, sig2)))],
                         "treated-sample-as-one-observation")],
    )


def _bi_which_die(rng):
    sides = [4, 6, 8]
    while True:
        r1, r2 = rng.randint(1, 8), rng.randint(1, 8)
        possible = [s for s in sides if max(r1, r2) <= s]
        if len(possible) >= 2:
            break
    s = rng.choice(possible)
    like = {d: (R(1, d) ** 2 if d in possible else R(0)) for d in sides}
    ans = like[s] / sum(like.values())
    return Problem(
        prompt=(f"A bag holds a fair 4-sided, a fair 6-sided and a fair 8-sided die. You draw one at random and roll it "
                f"twice, getting {r1} and {r2}. What is the posterior probability that you drew the {s}-sided die?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Prior 1/3 each", "Likelihood (1/s)² if both rolls ≤ s, otherwise 0", "Posterior ∝ prior × likelihood",
                "Normalise"],
        solution=("Likelihoods: " + ", ".join(f"$d{d}: {L(like[d])}$" for d in sides) +
                  f"; posterior $= {L(like[s])}/{L(sum(like.values()))} = {L(ans)}$."),
        misconceptions=[(fmt(R(1, len(possible))), "ignored-likelihood"),
                        (fmt(R(1, s) / sum(R(1, d) for d in possible)), "used-one-roll-only")],
    )


def _bi_predictive_two(rng):
    a, b = rng.randint(1, 4), rng.randint(1, 4)
    n = rng.randint(4, 12)
    k = rng.randint(1, n - 1)
    al, be = a + k, b + n - k
    ans = R(al * (al + 1), (al + be) * (al + be + 1))
    return Problem(
        prompt=(f"A new drug's cure rate $p$ has prior $\\mathrm{{Beta}}({a}, {b})$. In a trial, {k} of {n} patients were "
                "cured. Treating future patients as conditionally i.i.d. given $p$, find the posterior predictive "
                "probability that the **next two** patients are both cured."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=[f"Posterior Beta({al}, {be})", "P(both cured | data) = E[p² | data] (integrate out p)",
                "E[p²] for Beta(α, β) = α(α+1)/((α+β)(α+β+1))", "Simplify"],
        solution=(f"Posterior $\\mathrm{{Beta}}({al}, {be})$; $E[p^2] = \\frac{{{al}\\cdot{al + 1}}}{{{al + be}\\cdot{al + be + 1}}}"
                  f" = {L(ans)}$ (slightly more than $(E[p])^2 = {L(R(al, al + be) ** 2)}$)."),
        misconceptions=[(fmt(R(al, al + be) ** 2), "assumed-independent-future"), (fmt(R(k, n) ** 2), "ignored-prior")],
    )


def _bi_sample_size(rng):
    while True:
        tau2, sig2 = rng.choice([1, 4, 9]), rng.choice([4, 9, 16, 25, 36])
        s = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(1, 5)])
        need = sig2 * (1 / s ** 2 - R(1, tau2))
        if need > 1:
            break
    ans = sp.ceiling(need)
    return Problem(
        prompt=(f"For a Normal mean with prior $\\mu \\sim \\mathcal N(0, {tau2})$ and observations i.i.d. "
                f"$\\mathcal N(\\mu, {sig2})$ (variances), how many observations are needed so that the posterior standard "
                f"deviation of $\\mu$ is at most ${L(s)}$, whatever values are observed?"),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=["Posterior variance = 1/(1/τ² + n/σ²) — it does not depend on the data", "Require 1/(1/τ² + n/σ²) ≤ s²",
                "Solve n ≥ σ²(1/s² - 1/τ²)", "Round up"],
        solution=f"$n \\ge {sig2}\\left({L(1 / s ** 2)} - {L(R(1, tau2))}\\right) = {L(need)}$, so $n = {ans}$.",
        misconceptions=[(str(sp.ceiling(sig2 / s ** 2)), "ignored-prior"), (str(sp.ceiling(sig2 / s - sig2 / sp.Integer(tau2))), "forgot-square")],
    )


@gen("prob.bayesian_inference", name="bayes_inference_stories")
def bayes_inference_stories(rng, level):
    if level == 1:
        return rng.choice([_bi_three_coins, _bi_normal_one])(rng)
    if level == 2:
        return rng.choice([_bi_normal_n, _bi_which_die])(rng)
    return rng.choice([_bi_predictive_two, _bi_sample_size])(rng)


# ---------------------------------------------------------------- CIs & hypothesis tests
def _ci_proportion(rng):
    n = rng.choice([100, 400, 2500])
    phat = rng.choice([R(1, 10), R(1, 5), R(1, 2), R(4, 5), R(9, 10)])
    sd = sp.sqrt(phat * (1 - phat))
    se = sd / math.isqrt(n)
    me = R(196, 100) * se
    lo, hi = phat - me, phat + me
    return Problem(
        prompt=(f"In a survey of {n} people, {phat * n} say they would buy a new product. Give the approximate 95% "
                "confidence interval for the true proportion, $\\hat p \\pm 1.96\\sqrt{\\hat p(1-\\hat p)/n}$."),
        answer=[fmt(lo), fmt(hi)], kind="list", answer_hint="lower, upper   e.g. 0.4412, 0.5588",
        rubric=["p̂ = count / n", "SE = sqrt(p̂(1-p̂)/n)", "Margin 1.96·SE", "p̂ ± margin"],
        solution=f"$\\hat p = {dec(phat)}$, $SE = {dec(sd)}/{math.isqrt(n)} = {dec(se)}$, margin ${dec(me)}$: $[{dec(lo)}, {dec(hi)}]$.",
        misconceptions=[([fmt(phat - R(196, 100) * sd), fmt(phat + R(196, 100) * sd)], "forgot-sqrt-n")],
    )


def _ci_z_stat(rng):
    n = rng.choice([16, 25, 36, 49, 64, 100])
    rn = math.isqrt(n)
    sigma = rn * rng.randint(1, 3)
    mu0 = rng.randint(100, 500)
    z = nz(rng, -6, 6) * R(1, 2)
    xbar = mu0 + z * R(sigma, rn)
    return Problem(
        prompt=(f"A bottling line should average $\\mu_0 = {mu0}$ ml with known $\\sigma = {sigma}$ ml. A sample of "
                f"$n = {n}$ bottles has mean $\\bar x = {dec(xbar)}$ ml. Compute the $z$ test statistic."),
        answer=fmt(z), kind="num", answer_hint="a number (sign matters), e.g. -1.5",
        rubric=["Standard error σ/√n", "z = (x̄ - μ0)/SE", "Keep the sign"],
        solution=f"$SE = {sigma}/{rn} = {sigma // rn}$, $z = \\frac{{{dec(xbar)} - {mu0}}}{{{sigma // rn}}} = {dec(z)}$.",
        misconceptions=[(fmt((xbar - mu0) / sigma), "forgot-sqrt-n"), (fmt(-z), "sign-error")],
    )


def _ci_false_alarm(rng):
    n = rng.choice([16, 25, 36, 64, 100])
    rn = math.isqrt(n)
    sigma = rn * rng.randint(1, 4)
    se = sigma // rn
    mu0, z = rng.randint(20, 200), rng.choice([1, R(3, 2), 2, R(5, 2)])
    c = mu0 + z * se
    ans = 1 - PHI[z]
    return Problem(
        prompt=(f"An inspector weighs $n = {n}$ items (known $\\sigma = {sigma}$) and raises an alarm when "
                f"$\\bar x > {dec(c)}$. When the process is working correctly, $\\mu = {mu0}$. Using {PHI_TABLE}, what "
                "is the probability of a false alarm (the significance level $\\alpha$)?"),
        answer=fmt(ans), kind="num", tol=1e-3, answer_hint="a decimal, e.g. 0.0228",
        rubric=["False alarm = reject when H0 is true", "Under H0, x̄ ~ N(μ0, σ²/n)", "Standardise the cutoff",
                "One-sided upper tail 1 - Φ(z)"],
        solution=f"$z = \\frac{{{dec(c)} - {mu0}}}{{{sigma}/{rn}}} = {dec(z)}$, $\\alpha = 1 - \\Phi({dec(z)}) = {dec(ans)}$.",
        misconceptions=[(fmt(2 * (1 - PHI[z])), "two-sided-for-one-sided"), (fmt(PHI[z]), "wrong-tail")],
    )


def _ci_power(rng):
    n = rng.choice([16, 25, 36, 64, 100])
    rn = math.isqrt(n)
    sigma = rn * rng.randint(1, 4)
    se = sigma // rn
    mu1, z = rng.randint(30, 200), rng.choice(PHI_Z)
    sgn = rng.choice([1, -1])
    c = mu1 + sgn * z * se
    ans = 1 - PHI[z] if sgn == 1 else PHI[z]
    return Problem(
        prompt=(f"A new fertiliser is declared effective if the mean yield of $n = {n}$ test plots exceeds {dec(c)} "
                f"(known $\\sigma = {sigma}$). If the true mean yield is actually {mu1}, what is the probability that the "
                f"test declares it effective (the **power**)? Use {PHI_TABLE}."),
        answer=fmt(ans), kind="num", tol=1e-3, answer_hint="a decimal, e.g. 0.8413",
        rubric=["Power = P(reject | true μ = μ1)", "x̄ ~ N(μ1, σ²/n) under the alternative", "Standardise the cutoff",
                "Upper tail, using Φ(-z) = 1 - Φ(z) if needed"],
        solution=(f"$z = \\frac{{{dec(c)} - {mu1}}}{{{sigma}/{rn}}} = {dec(sgn * z)}$, power $= 1 - \\Phi({dec(sgn * z)}) "
                  f"= {dec(ans)}$."),
        misconceptions=[(fmt(1 - ans), "wrong-tail")],
    )


def _ci_power_sample_size(rng):
    sigma, delta = rng.randint(4, 30), rng.randint(1, 6)
    need = (R(2645, 1000) * sigma / delta) ** 2
    ans = sp.ceiling(need)
    return Problem(
        prompt=(f"Planning a study: a one-sided $z$-test of $H_0: \\mu = \\mu_0$ vs $H_1: \\mu > \\mu_0$ at level "
                f"$\\alpha = 0.05$ ($z_{{0.05}} = 1.645$) uses known $\\sigma = {sigma}$. Find the smallest $n$ such that "
                f"the power against $\\mu = \\mu_0 + {delta}$ is at least $0.8413 = \\Phi(1)$."),
        answer=str(ans), kind="num", answer_hint="an integer",
        rubric=["Reject when x̄ > μ0 + 1.645 σ/√n", "Power = P(Z > 1.645 - δ√n/σ) ≥ Φ(1)",
                "Need δ√n/σ ≥ 1.645 + 1", "n ≥ ((1.645 + 1)σ/δ)², round up"],
        solution=f"$n \\ge \\left(\\frac{{2.645\\cdot{sigma}}}{{{delta}}}\\right)^2 = {dec(need)}$, so $n = {ans}$.",
        misconceptions=[(str(sp.ceiling((R(1645, 1000) * sigma / delta) ** 2)), "ignored-power-requirement"),
                        (str(sp.ceiling(R(2645, 1000) * sigma / delta)), "forgot-square")],
    )


@gen("prob.hypothesis_ci", name="ci_stories")
def ci_stories(rng, level):
    if level == 1:
        return rng.choice([_ci_proportion, _ci_z_stat])(rng)
    if level == 2:
        return rng.choice([_ci_false_alarm, _ci_power])(rng)
    return _ci_power_sample_size(rng)


# ---------------------------------------------------------------- multivariate Gaussians
def _mv_correlation(rng):
    s1, s2 = rng.choice([1, 2, 3, 4, 5]), rng.choice([1, 2, 3, 4, 5])
    c = nz(rng, -(s1 * s2 - 1), s1 * s2 - 1) if s1 * s2 > 1 else None
    while c is None:
        s1, s2 = rng.choice([2, 3, 4, 5]), rng.choice([1, 2, 3, 4, 5])
        c = nz(rng, -(s1 * s2 - 1), s1 * s2 - 1)
    S = sp.Matrix([[s1 * s1, c], [c, s2 * s2]])
    ans = R(c, s1 * s2)
    return Problem(
        prompt=f"The random vector $(X_1, X_2)$ has covariance matrix $\\Sigma = {L(S)}$. What is the correlation between $X_1$ and $X_2$?",
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Read off Var X1 = Σ11, Var X2 = Σ22, Cov = Σ12", "ρ = Σ12 / sqrt(Σ11 Σ22)", "Simplify"],
        solution=f"$\\rho = \\dfrac{{{c}}}{{\\sqrt{{{s1 * s1}\\cdot{s2 * s2}}}}} = {L(ans)}$.",
        misconceptions=[(fmt(R(c, s1 * s1 * s2 * s2)), "divided-by-variances"), (fmt(sp.Integer(c)), "covariance-not-correlation")],
    )


def _mv_mahalanobis(rng):
    S = _spd2(rng)
    mu = sp.Matrix([rng.randint(-5, 5), rng.randint(-5, 5)])
    d = sp.Matrix([nz(rng, -3, 3), nz(rng, -3, 3)])
    x = mu + d
    ans = (d.T * S.inv() * d)[0]
    return Problem(
        prompt=(f"How unusual is a point? For $\\mathbf X \\sim \\mathcal N(\\boldsymbol\\mu, \\Sigma)$ with "
                f"$\\boldsymbol\\mu = {L(mu)}$ and $\\Sigma = {L(S)}$, compute the squared Mahalanobis distance "
                f"$(\\mathbf x-\\boldsymbol\\mu)^\\top\\Sigma^{{-1}}(\\mathbf x-\\boldsymbol\\mu)$ of $\\mathbf x = {L(x)}$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["d = x - μ", "Σ⁻¹ = (1/det) [[Σ22, -Σ12], [-Σ12, Σ11]]", "Compute dᵀ Σ⁻¹ d"],
        solution=f"$\\mathbf d = {L(d)}$, $\\Sigma^{{-1}} = {L(S.inv())}$, $\\mathbf d^\\top\\Sigma^{{-1}}\\mathbf d = {L(ans)}$.",
        misconceptions=[(fmt((d.T * d)[0]), "ignored-covariance"), (fmt((d.T * S * d)[0]), "used-sigma-not-inverse")],
    )


def _mv_decorrelate(rng):
    S = _spd2(rng)
    ans = R(S[0, 1], S[1, 1])
    return Problem(
        prompt=(f"Let $(X_1, X_2)$ be bivariate normal with covariance matrix $\\Sigma = {L(S)}$. For which constant $c$ "
                "is $X_1 - c\\,X_2$ **independent** of $X_2$?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["(X1 - cX2, X2) is jointly Gaussian, so uncorrelated ⇔ independent",
                "Cov(X1 - cX2, X2) = Σ12 - c Σ22", "Set to 0: c = Σ12/Σ22"],
        solution=f"$\\Sigma_{{12}} - c\\,\\Sigma_{{22}} = 0 \\Rightarrow c = \\Sigma_{{12}}/\\Sigma_{{22}} = {S[0, 1]}/{S[1, 1]} = {L(ans)}$.",
        misconceptions=[(fmt(R(S[0, 1], S[0, 0])), "regressed-wrong-direction"), (fmt(R(0)), "assumed-already-independent")],
    )


def _mv_prob_greater(rng):
    while True:
        s11, s22, s12 = rng.randint(1, 9), rng.randint(1, 9), nz(rng, -4, 4)
        v = s11 + s22 - 2 * s12
        if s12 * s12 < s11 * s22 and v > 0 and math.isqrt(v) ** 2 == v:
            break
    d = math.isqrt(v)
    z = rng.choice([1, 2] + ([R(1, 2), R(3, 2)] if d % 2 == 0 else []))
    mu1 = rng.randint(50, 80)
    sgn = rng.choice([1, -1])
    mu2 = mu1 + sgn * z * d
    ans = 1 - PHI[z] if sgn == 1 else PHI[z]
    S = sp.Matrix([[s11, s12], [s12, s22]])
    return Problem(
        prompt=(f"A student's scores $(X_1, X_2)$ on two exam papers are bivariate normal with means ${mu1}$ and "
                f"${L(mu2)}$ and covariance matrix $\\Sigma = {L(S)}$. Using {PHI_TABLE}, find $P(X_1 > X_2)$."),
        answer=fmt(ans), kind="num", tol=1e-3, answer_hint="a decimal, e.g. 0.3085",
        rubric=["D = X1 - X2 is normal", "E[D] = μ1 - μ2, Var D = Σ11 + Σ22 - 2Σ12", "P(D > 0): standardise",
                "Use the table (and symmetry if needed)"],
        solution=(f"$D \\sim \\mathcal N({L(mu1 - mu2)}, {s11} + {s22} - 2({s12}) = {v})$, so "
                  f"$P(D>0) = P\\left(Z > \\frac{{0 - ({L(mu1 - mu2)})}}{{{d}}}\\right) = P(Z > {L(sp.nsimplify(R(mu2 - mu1, 1) / d))}) "
                  f"= {dec(ans)}$."),
        misconceptions=[(fmt(1 - ans), "wrong-tail")],
    )


def _mv_given_sum(rng):
    while True:
        S = _spd2(rng)
        vs = S[0, 0] + S[1, 1] + 2 * S[0, 1]
        if vs > 0:
            break
    mu = sp.Matrix([rng.randint(-5, 5), rng.randint(-5, 5)])
    s = mu[0] + mu[1] + nz(rng, -5, 5)
    c1s = S[0, 0] + S[0, 1]
    cm = mu[0] + R(c1s, vs) * (s - mu[0] - mu[1])
    cv = S[0, 0] - R(c1s ** 2, vs)
    return Problem(
        prompt=(f"Two noisy measurements $(X_1, X_2)$ are bivariate normal with mean ${L(mu)}$ and covariance "
                f"${L(S)}$, but you only learn their **sum**, $X_1 + X_2 = {s}$. Find "
                f"$E[X_1 \\mid X_1 + X_2 = {s}]$ and $\\mathrm{{Var}}(X_1 \\mid X_1 + X_2 = {s})$."),
        answer=[fmt(cm), fmt(cv)], kind="list", answer_hint="mean, variance   e.g. 7/3, 5/3",
        rubric=["(X1, S) with S = X1 + X2 is jointly normal", "Cov(X1, S) = Σ11 + Σ12, Var S = Σ11 + Σ22 + 2Σ12",
                "Conditional mean μ1 + Cov/VarS·(s - E S)", "Conditional variance Σ11 - Cov²/VarS"],
        solution=(f"$\\mathrm{{Cov}}(X_1,S) = {c1s}$, $\\mathrm{{Var}}(S) = {vs}$; mean $= {mu[0]} + \\frac{{{c1s}}}{{{vs}}}"
                  f"\\left({s} - ({mu[0] + mu[1]})\\right) = {L(cm)}$, variance $= {S[0, 0]} - \\frac{{{c1s ** 2}}}{{{vs}}} = {L(cv)}$."),
        misconceptions=[([fmt(mu[0]), fmt(S[0, 0])], "ignored-conditioning"),
                        ([fmt(R(s, 2)), fmt(cv)], "split-sum-equally")],
    )


def _mv_pca(rng):
    t, l2 = rng.randint(1, 2), rng.randint(1, 6)
    a, b = rng.choice([(9, 16), (16, 9)])
    S = sp.Matrix([[l2 + a * t, 12 * t], [12 * t, l2 + b * t]])
    if rng.random() < 0.5:
        S = sp.Matrix([[l2 + a * t, -12 * t], [-12 * t, l2 + b * t]])
    l1 = l2 + 25 * t
    assert sorted(S.eigenvals()) == sorted([l1, l2])
    frac = R(l1, l1 + l2)
    return Problem(
        prompt=(f"Data are modelled as $\\mathcal N(\\mathbf 0, \\Sigma)$ with $\\Sigma = {L(S)}$. Find the variance along "
                "the first principal direction (the largest eigenvalue of $\\Sigma$) and the fraction of the total "
                "variance it explains."),
        answer=[fmt(l1), fmt(frac)], kind="list", answer_hint="largest eigenvalue, fraction   e.g. 9, 9/10",
        rubric=["Characteristic polynomial λ² - tr(Σ)λ + det Σ = 0", "Solve for the larger root",
                "Total variance = trace = λ1 + λ2", "Fraction = λ1 / trace"],
        solution=(f"$\\operatorname{{tr}} = {S.trace()}$, $\\det = {S.det()}$, eigenvalues ${l1}, {l2}$; fraction "
                  f"$\\lambda_1/\\operatorname{{tr}} = {l1}/{S.trace()} = {L(frac)}$."),
        misconceptions=[([fmt(max(S[0, 0], S[1, 1])), fmt(R(max(S[0, 0], S[1, 1]), S.trace()))], "took-largest-diagonal")],
    )


@gen("prob.multivariate_gaussian", name="mvn_stories")
def mvn_stories(rng, level):
    if level == 1:
        return rng.choice([_mv_correlation, _mv_mahalanobis])(rng)
    if level == 2:
        return rng.choice([_mv_decorrelate, _mv_prob_greater])(rng)
    return rng.choice([_mv_given_sum, _mv_pca])(rng)


# ---------------------------------------------------------------- Markov chains
def _two_state(rng):
    den = rng.choice([4, 5, 10])
    a, b = R(rng.randint(1, den - 1), den), R(rng.randint(1, den - 1), den)
    return a, b, sp.Matrix([[1 - a, a], [b, 1 - b]])


def _mk_one_step(rng):
    a, b, P = _two_state(rng)
    p0 = rng.choice([R(1, 2), R(1, 4), R(3, 4), R(1, 5), R(2, 5), R(3, 5)])
    v = sp.Matrix([[p0, 1 - p0]])
    ans = v * P
    return Problem(
        prompt=(f"A phone's battery saver is either off (state 1) or on (state 2); each hour it moves according to "
                f"$P = {L(P)}$ (row = current state). Right now it is off with probability ${L(p0)}$. Find the "
                "distribution of its state one hour from now."),
        answer=[fmt(ans[0]), fmt(ans[1])], kind="vector", answer_hint="P(off), P(on)   e.g. 3/8, 5/8",
        rubric=["Row vector π0 = (p, 1-p)", "π1 = π0 P (row vector times matrix)", "Check the entries sum to 1"],
        solution=f"$\\pi_1 = {L(v)}{L(P)} = {L(ans)}$.",
        misconceptions=[([fmt((P * v.T)[0]), fmt((P * v.T)[1])], "multiplied-wrong-side"),
                        ([fmt(P[0, 0]), fmt(P[0, 1])], "used-single-row")],
    )


def _mk_holding_time(rng):
    a = rng.choice([R(1, 2), R(1, 3), R(1, 4), R(1, 5), R(1, 10), R(2, 5), R(3, 10)])
    ans = 1 / a
    return Problem(
        prompt=(f"A machine that is working on a given day breaks down by the next day with probability ${L(a)}$, "
                "independently each day (a two-state Markov chain). Starting from a working day, what is the expected "
                "number of consecutive working days, **counting today**, before the first breakdown?"),
        answer=fmt(ans), kind="num", answer_hint="an exact value, e.g. 10/3",
        rubric=["Run length is Geometric on {1, 2, …} with 'success' = breakdown", "E = 1/a"],
        solution=f"$E[\\text{{run}}] = \\dfrac{{1}}{{{L(a)}}} = {L(ans)}$.",
        misconceptions=[(fmt(1 / a - 1), "off-by-one-geometric"), (fmt(1 / (1 - a)), "swapped-stay-leave")],
    )


def _mk_gamblers_ruin(rng):
    p = rng.choice([R(1, 3), R(2, 3), R(2, 5), R(3, 5), R(1, 4), R(3, 4)])
    N = rng.randint(3, 5)
    k = rng.randint(1, N - 1)
    r = (1 - p) / p
    ans = (r ** k - 1) / (r ** N - 1)
    return Problem(
        prompt=(f"A gambler with {k} chips bets 1 chip per round and wins each round with probability ${L(p)}$. She stops "
                f"when she reaches {N} chips or goes broke. What is the probability that she reaches {N} chips?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["First-step analysis: h_i = p h_{i+1} + q h_{i-1}, h_0 = 0, h_N = 1",
                "Solution h_i = (r^i - 1)/(r^N - 1) with r = q/p", "Evaluate"],
        solution=f"$r = q/p = {L(r)}$, $h_{{{k}}} = \\dfrac{{r^{{{k}}} - 1}}{{r^{{{N}}} - 1}} = {L(ans)}$.",
        misconceptions=[(fmt(R(k, N)), "used-fair-game-formula"), (fmt((p / (1 - p)) ** k * 0 + (((p / (1 - p)) ** k - 1) / ((p / (1 - p)) ** N - 1))), "inverted-ratio")],
    )


def _mk_return_time(rng):
    a, b, P = _two_state(rng)
    piA = b / (a + b)
    ans = 1 / piA
    return Problem(
        prompt=(f"A web crawler moves between page A (state 1) and page B (state 2) with transition matrix "
                f"$P = {L(P)}$ (row = current page). Starting from A, what is the expected number of steps until it "
                "first **returns** to A?"),
        answer=fmt(ans), kind="num", answer_hint="an exact value, e.g. 7/3",
        rubric=["Stationary distribution π_A = b/(a + b)", "Mean return time = 1/π_A (or first-step analysis)", "Simplify"],
        solution=f"$\\pi_A = {L(piA)}$, so $E[\\text{{return time}}] = 1/\\pi_A = {L(ans)}$.",
        misconceptions=[(fmt(piA), "stationary-prob-not-time"), (fmt(1 / a), "time-to-leave-not-return")],
    )


def _mk_hitting_time(rng):
    den = rng.choice([3, 4, 5, 6])
    rows = []
    for _ in range(2):
        c1, c2 = sorted(rng.sample(range(1, den), 2)) if den > 2 else (1, 1)
        rows.append([R(c1, den), R(c2 - c1, den), R(den - c2, den)])
    P = sp.Matrix(rows + [[0, 0, 1]])
    h1, h2 = sp.symbols("h1 h2")
    sol = sp.solve([sp.Eq(h1, 1 + P[0, 0] * h1 + P[0, 1] * h2), sp.Eq(h2, 1 + P[1, 0] * h1 + P[1, 1] * h2)], [h1, h2])
    ans = sol[h1]
    return Problem(
        prompt=(f"A frog hops between lily pads 1, 2 and 3; pad 3 is the bank, where it stays. Its moves follow the "
                f"transition matrix $P = {L(P)}$ (row = current pad). Starting on pad 1, what is the expected number of "
                "hops until it reaches the bank?"),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["Let h_i = expected hops to reach 3 from i; h_3 = 0", "First-step analysis: h_i = 1 + Σ_j P_ij h_j",
                "Solve the 2×2 linear system", "Report h_1"],
        solution=(f"$h_1 = 1 + {L(P[0, 0])}h_1 + {L(P[0, 1])}h_2$, $h_2 = 1 + {L(P[1, 0])}h_1 + {L(P[1, 1])}h_2$ "
                  f"give $h_1 = {L(ans)}$, $h_2 = {L(sol[h2])}$."),
        misconceptions=[(fmt(1 / P[0, 2]), "ignored-detour-via-2"), (fmt(sol[h2]), "reported-wrong-start")],
    )


def _mk_graph_walk(rng):
    n = rng.choice([4, 5])
    allE = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    while True:
        E = sorted(rng.sample(allE, rng.randint(n, n + 2)))
        adj = {v: set() for v in range(1, n + 1)}
        for i, j in E:
            adj[i].add(j)
            adj[j].add(i)
        seen, stack = {1}, [1]
        while stack:
            for w in adj[stack.pop()] - seen:
                seen.add(w)
                stack.append(w)
        degs = [len(adj[v]) for v in range(1, n + 1)]
        if len(seen) == n and len(set(degs)) > 1:
            break
    pi = [R(d, 2 * len(E)) for d in degs]
    edges = ", ".join(f"\\{{{i},{j}\\}}" for i, j in E)
    return Problem(
        prompt=(f"A random walk on an undirected graph with vertices $1, \\dots, {n}$ and edges ${edges}$ moves at each "
                "step to a uniformly random neighbour of its current vertex. Find its stationary distribution "
                "(check detailed balance)."),
        answer=[fmt(v) for v in pi], kind="vector", answer_hint=f"pi1, ..., pi{n} in vertex order, e.g. " + ", ".join(["1/10"] * (n - 1) + [f"{10 - (n - 1)}/10"]),
        rubric=["Guess π_v ∝ deg(v)", "Detailed balance: π_u/deg(u) = π_v/deg(v) along every edge",
                f"Normalise by Σ deg = 2|E| = {2 * len(E)}"],
        solution=f"Degrees ${', '.join(map(str, degs))}$, so $\\pi = \\left({', '.join(L(v) for v in pi)}\\right)$.",
        misconceptions=[([fmt(R(1, n))] * n, "assumed-uniform"), ([fmt(R(d, len(E))) for d in degs], "forgot-double-count-edges")],
    )


@gen("prob.markov_chains", name="markov_stories")
def markov_stories(rng, level):
    if level == 1:
        return rng.choice([_mk_one_step, _mk_holding_time])(rng)
    if level == 2:
        return rng.choice([_mk_gamblers_ruin, _mk_return_time])(rng)
    return rng.choice([_mk_hitting_time, _mk_graph_walk])(rng)


# ---------------------------------------------------------------- MLE (third family of models)
def _ml_poisson(rng):
    n = rng.randint(5, 9)
    while True:
        data = [rng.randint(0, 6) for _ in range(n)]
        if sum(data) > 0:
            break
    lam = R(sum(data), n)
    return Problem(
        prompt=(f"Goals scored in {n} matches, modelled as i.i.d. $\\mathrm{{Pois}}(\\lambda)$: {lst(data)}. Write the "
                "log-likelihood and find the MLE $\\hat\\lambda$."),
        answer=fmt(lam), kind="num", answer_hint=FRAC_HINT,
        rubric=["ℓ(λ) = (Σx) log λ - nλ + const", "ℓ'(λ) = Σx/λ - n = 0", "λ̂ = x̄"],
        solution=f"$\\hat\\lambda = \\bar x = {sum(data)}/{n} = {L(lam)}$.",
        misconceptions=[(fmt(1 / lam), "inverted-estimator")],
    )


def _ml_normal_var(rng):
    m = rng.randint(0, 20)
    n = rng.randint(4, 6)
    data = [m + nz(rng, -4, 4) for _ in range(n)]
    ss = sum((v - m) ** 2 for v in data)
    ans = R(ss, n)
    return Problem(
        prompt=(f"Measurements are i.i.d. $\\mathcal N({m}, \\sigma^2)$ — the mean ${m}$ is **known**, the variance is not. "
                f"Observations: {lst(data)}. Derive and evaluate the MLE $\\hat\\sigma^2$."),
        answer=fmt(ans), kind="num", answer_hint=FRAC_HINT,
        rubric=["ℓ(σ²) = -(n/2) log σ² - Σ(x_i - μ)²/(2σ²)", "Differentiate in v = σ² and set to 0",
                "σ̂² = (1/n) Σ (x_i - μ)² with the known μ"],
        solution=f"$\\sum(x_i - {m})^2 = {ss}$, so $\\hat\\sigma^2 = {ss}/{n} = {L(ans)}$.",
        misconceptions=[(fmt(R(ss, n - 1)), "divided-by-n-minus-1"),
                        (fmt(sp.sqrt(ans)), "returned-sigma-not-variance")],
    )


def _ml_regression_origin(rng):
    n = rng.randint(3, 5)
    while True:
        xs = [nz(rng, -4, 5) for _ in range(n)]
        ys = [rng.randint(-6, 12) for _ in range(n)]
        sxy, sxx = sum(a * b for a, b in zip(xs, ys)), sum(a * a for a in xs)
        if sxy != 0 and sum(xs) != 0:
            break
    beta = R(sxy, sxx)
    pts = ", ".join(f"$({a}, {b})$" for a, b in zip(xs, ys))
    return Problem(
        prompt=(f"In the model $Y_i = \\beta x_i + \\varepsilon_i$ (no intercept) with $\\varepsilon_i$ i.i.d. "
                f"$\\mathcal N(0, \\sigma^2)$, derive the MLE of $\\beta$ and evaluate it for the data $(x_i, y_i)$: {pts}."),
        answer=fmt(beta), kind="num", answer_hint=FRAC_HINT,
        rubric=["Log-likelihood ∝ -Σ (y_i - βx_i)²: maximising = least squares", "d/dβ: -2Σ x_i(y_i - βx_i) = 0",
                "β̂ = Σx_i y_i / Σx_i²", "Evaluate"],
        solution=f"$\\hat\\beta = \\dfrac{{\\sum x_i y_i}}{{\\sum x_i^2}} = \\dfrac{{{sxy}}}{{{sxx}}} = {L(beta)}$.",
        misconceptions=[(fmt(R(sum(ys), sum(xs))), "ratio-of-means")],
    )


@gen("prob.mle", name="mle_more")
def mle_more(rng, level):
    if level == 1:
        return _ml_poisson(rng)
    if level == 2:
        return _ml_normal_var(rng)
    return _ml_regression_origin(rng)
