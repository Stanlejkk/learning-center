"""Reference solution for lab prob_01_monte_carlo (Monte Carlo, LLN/CLT, birthday problem, MLE, Beta-Binomial)."""
from __future__ import annotations

import math

import numpy as np


def mc_prob(trial, n, rng):
    """Monte-Carlo estimate of P(event): run `trial(rng)` (returns bool) n times, return the fraction of True."""
    return float(np.mean([bool(trial(rng)) for _ in range(n)]))


def sample_means(sampler, n, reps, rng):
    """Return an array of `reps` sample means, each of n draws from `sampler(rng, size)`."""
    draws = np.asarray(sampler(rng, (reps, n)), dtype=float)
    return draws.mean(axis=1)


def standardize_means(means, mu, sigma, n):
    """CLT z-scores: (mean - mu) / (sigma / sqrt(n))."""
    return (np.asarray(means, dtype=float) - mu) / (sigma / math.sqrt(n))


def birthday_exact(k, days=365):
    """P(at least two of k people share a birthday), days equally likely."""
    if k > days:
        return 1.0
    p_distinct = 1.0
    for j in range(k):
        p_distinct *= (days - j) / days
    return 1.0 - p_distinct


def birthday_mc(k, n_trials, rng, days=365):
    """Monte-Carlo estimate of the birthday probability (vectorised)."""
    b = rng.integers(0, days, size=(n_trials, k))
    b.sort(axis=1)
    shared = (np.diff(b, axis=1) == 0).any(axis=1)
    return float(shared.mean())


def bernoulli_mle(x):
    return float(np.mean(x))


def poisson_mle(x):
    return float(np.mean(x))


def poisson_loglik(lam, x):
    """Sum_i [x_i log(lam) - lam - log(x_i!)]."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * math.log(lam) - lam - np.array([math.lgamma(v + 1) for v in x])))


def beta_binomial_posterior(a, b, k, n):
    """Beta(a, b) prior + k successes in n trials -> posterior Beta parameters."""
    return (a + k, b + n - k)


def beta_mean(a, b):
    return a / (a + b)
