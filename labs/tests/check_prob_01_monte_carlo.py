"""Hidden checks for lab prob_01_monte_carlo. Each check_*(ns) gets the notebook globals()."""
import math

import numpy as np


def check_mc_prob(ns):
    mc_prob = ns["mc_prob"]
    rng = np.random.default_rng(1)
    est = mc_prob(lambda r: r.random() < 0.3, 20000, rng)
    assert abs(est - 0.3) < 0.02, f"P(U < 0.3) estimated as {est}"
    rng = np.random.default_rng(2)
    est = mc_prob(lambda r: r.integers(1, 7) + r.integers(1, 7) == 7, 20000, rng)
    assert abs(est - 1 / 6) < 0.02, f"P(two dice sum to 7) estimated as {est}, expected about 1/6"


def check_birthday_exact(ns):
    f = ns["birthday_exact"]
    assert abs(f(23) - 0.507297) < 1e-4, f"birthday_exact(23) = {f(23)}, expected 0.5073"
    assert abs(f(1)) < 1e-12, "one person cannot share a birthday"
    assert abs(f(3, days=7) - (1 - 7 * 6 * 5 / 7 ** 3)) < 1e-12, "wrong for days=7, k=3"
    assert abs(f(400) - 1.0) < 1e-12, "k > days must give probability 1 (pigeonhole)"


def check_birthday_mc(ns):
    f = ns["birthday_mc"]
    rng = np.random.default_rng(3)
    est = f(23, 20000, rng)
    assert abs(est - 0.5073) < 0.02, f"MC estimate {est} too far from 0.5073"
    est = f(4, 20000, np.random.default_rng(4), days=12)
    exact = 1 - 12 * 11 * 10 * 9 / 12 ** 4
    assert abs(est - exact) < 0.02, f"MC estimate for 4 people / 12 months: {est}, expected about {exact:.4f}"


def check_sample_means_clt(ns):
    sm, stdz = ns["sample_means"], ns["standardize_means"]
    rng = np.random.default_rng(5)
    means = np.asarray(sm(lambda r, size: r.exponential(2.0, size), 50, 4000, rng))
    assert means.shape == (4000,), f"expected shape (4000,), got {means.shape}"
    assert abs(means.mean() - 2.0) < 0.05, f"average of sample means {means.mean():.3f}, expected about 2 (LLN)"
    se = 2.0 / math.sqrt(50)
    assert abs(means.std() - se) / se < 0.1, f"SD of sample means {means.std():.3f}, expected about {se:.3f}"
    z = np.asarray(stdz(means, 2.0, 2.0, 50))
    frac = np.mean(np.abs(z) < 1.96)
    assert 0.93 < frac < 0.97, f"{frac:.3f} of z-scores within ±1.96, expected about 0.95 (CLT)"


def check_mle(ns):
    assert abs(ns["bernoulli_mle"]([1, 0, 1, 1]) - 0.75) < 1e-12
    x = [2, 0, 3, 1, 4]
    lam_hat = ns["poisson_mle"](x)
    assert abs(lam_hat - 2.0) < 1e-12, f"poisson_mle({x}) = {lam_hat}"
    ll = ns["poisson_loglik"]
    expected = sum(v * math.log(1.5) - 1.5 - math.lgamma(v + 1) for v in x)
    assert abs(ll(1.5, x) - expected) < 1e-9, "poisson_loglik value is wrong (did you include -log x!?)"
    grid = np.linspace(0.5, 4.0, 351)
    best = grid[int(np.argmax([ll(g, x) for g in grid]))]
    assert abs(best - 2.0) < 0.011, f"log-likelihood maximised at {best}, expected at the MLE 2.0"


def check_beta_binomial(ns):
    post = tuple(ns["beta_binomial_posterior"](2, 3, 7, 10))
    assert post == (9, 6), f"Beta(2,3) prior + 7/10 successes -> {post}, expected (9, 6)"
    assert abs(ns["beta_mean"](9, 6) - 0.6) < 1e-12
