"""Hidden checks for ml_02_unsupervised. Each check_*(ns) raises AssertionError on failure."""
from __future__ import annotations

import numpy as np

CENTERS = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])


def _blobs(seed=0, n=60):
    rng = np.random.default_rng(seed)
    return np.vstack([c + 0.5 * rng.normal(size=(n, 2)) for c in CENTERS])


def _obj(X, C, labels):
    return float(((X - C[labels]) ** 2).sum())


def check_kmeans_objective(ns):
    X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 4.0]])
    C = np.array([[0.5, 0.5], [5.0, 5.0]])
    labels = np.array([0, 0, 1, 1])
    got = float(ns["kmeans_objective"](X, C, labels))
    assert abs(got - 3.0) < 1e-9, f"kmeans_objective: expected 3.0, got {got}"


def check_kmeans_recovers_blobs(ns):
    X = _blobs()
    best = None
    for seed in range(8):  # k-means has local optima: take the best of several restarts
        C, labels = ns["kmeans"](X, 3, n_iter=100, seed=seed)
        C, labels = np.asarray(C, float), np.asarray(labels)
        assert C.shape == (3, 2) and labels.shape == (len(X),), "kmeans must return (k, d) centroids and (n,) labels"
        # consistency: every point is assigned to its nearest returned centroid
        d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        assert np.array_equal(d2.argmin(1), labels), "labels are not the nearest-centroid assignments"
        J = _obj(X, C, labels)
        if best is None or J < best[0]:
            best = (J, C)
    C = best[1]
    for c in CENTERS:
        dist = np.min(np.linalg.norm(C - c, axis=1))
        assert dist < 0.3, f"no centroid near true centre {c} (closest at distance {dist:.2f})"


def check_pca_matches_eig(ns):
    rng = np.random.default_rng(5)
    A = np.array([[3.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, -0.3, 0.2]])
    X = rng.normal(size=(300, 3)) @ A + np.array([5.0, -2.0, 1.0])
    comps, ev, Z = ns["pca_svd"](X, 2)
    comps, ev, Z = np.asarray(comps, float), np.asarray(ev, float), np.asarray(Z, float)
    assert comps.shape == (2, 3) and ev.shape == (2,) and Z.shape == (300, 2), "wrong output shapes"
    vals, vecs = np.linalg.eigh(np.cov(X.T))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    assert np.allclose(ev, vals[:2], rtol=1e-8), f"explained variances {ev} should equal top covariance eigenvalues {vals[:2]}"
    for i in range(2):
        assert abs(abs(comps[i] @ vecs[:, i]) - 1) < 1e-6, f"component {i} is not the {i + 1}-th eigenvector (up to sign)"
    Xc = X - X.mean(0)
    assert np.allclose(Z, Xc @ comps.T, atol=1e-8), "Z must be the centred data projected onto the components"


def check_e_step(ns):
    x = np.array([-2.0, 0.0, 0.5, 3.0])
    pi, mu, var = np.array([0.3, 0.7]), np.array([-1.0, 2.0]), np.array([1.0, 0.5])
    dens = np.exp(-(x[:, None] - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)
    ref = pi * dens / (pi * dens).sum(1, keepdims=True)
    R = np.asarray(ns["e_step"](x, pi, mu, var), float)
    assert R.shape == (4, 2), f"responsibilities should have shape (4, 2), got {R.shape}"
    assert np.allclose(R, ref, atol=1e-10), f"e_step mismatch:\n{R}\nvs\n{ref}"


def check_em_recovers_mixture(ns):
    rng = np.random.default_rng(11)
    n = 600
    z = rng.random(n) < 0.4
    x = np.where(z, rng.normal(-3.0, 1.0, n), rng.normal(3.0, 0.5, n))
    best = None
    for seed in range(4):
        pi, mu, var, hist = ns["gmm_em_1d"](x, 2, n_iter=300, seed=seed)
        hist = np.asarray(hist, float)
        assert np.all(np.diff(hist) > -1e-6), "EM log-likelihood must never decrease"
        if best is None or hist[-1] > best[0]:
            best = (hist[-1], np.asarray(pi, float), np.asarray(mu, float), np.asarray(var, float))
    _, pi, mu, var = best
    o = np.argsort(mu)
    pi, mu, var = pi[o], mu[o], var[o]
    assert np.allclose(mu, [-3.0, 3.0], atol=0.25), f"means {mu} not close to [-3, 3]"
    assert np.allclose(pi, [0.4, 0.6], atol=0.07), f"weights {pi} not close to [0.4, 0.6]"
    assert np.allclose(np.sqrt(var), [1.0, 0.5], atol=0.15), f"std devs {np.sqrt(var)} not close to [1, 0.5]"
