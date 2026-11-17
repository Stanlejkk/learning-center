"""Reference solution for labs/ml_02_unsupervised.ipynb (keep in sync with the notebook's TODO cells)."""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------- k-means
def kmeans_objective(X, C, labels):
    """Sum of squared distances of each point to its assigned centroid."""
    X, C = np.asarray(X, float), np.asarray(C, float)
    return float(((X - C[np.asarray(labels)]) ** 2).sum())


def kmeans(X, k, n_iter=100, seed=0):
    """Lloyd's algorithm. Init: k distinct data points chosen with np.random.default_rng(seed).
    Returns (centroids (k, d), labels (n,))."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = None
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = d2.argmin(1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if np.any(labels == j):  # keep the old centroid if a cluster empties
                C[j] = X[labels == j].mean(0)
    return C, labels


# ----------------------------------------------------------------- PCA
def pca_svd(X, k):
    """PCA via SVD of the centred data.
    Returns (components (k, d) with unit-norm rows, explained_variance (k,) = S²/(n-1), Z = Xc @ components.T)."""
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]
    ev = S[:k] ** 2 / (len(X) - 1)
    return comps, ev, Xc @ comps.T


# ----------------------------------------------------------------- 1-D Gaussian mixture via EM
def normal_pdf(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


def e_step(x, pi, mu, var):
    """Responsibilities R[i, k] = P(z_i = k | x_i)."""
    x = np.asarray(x, float)[:, None]
    w = np.asarray(pi)[None, :] * normal_pdf(x, np.asarray(mu)[None, :], np.asarray(var)[None, :])
    return w / w.sum(1, keepdims=True)


def m_step(x, R):
    x = np.asarray(x, float)
    Nk = R.sum(0)
    pi = Nk / len(x)
    mu = (R * x[:, None]).sum(0) / Nk
    var = (R * (x[:, None] - mu[None, :]) ** 2).sum(0) / Nk
    return pi, mu, np.maximum(var, 1e-6)


def log_likelihood(x, pi, mu, var):
    x = np.asarray(x, float)[:, None]
    return float(np.log((np.asarray(pi)[None, :] * normal_pdf(x, np.asarray(mu)[None, :], np.asarray(var)[None, :])).sum(1)).sum())


def gmm_em_1d(x, K, n_iter=200, seed=0):
    """EM for a K-component 1-D Gaussian mixture.
    Init: means = K distinct data points (np.random.default_rng(seed)), variances = var(x), weights uniform.
    Returns (pi, mu, var, loglik_history) with loglik_history[t] = log-likelihood before iteration t's E-step."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    mu = x[rng.choice(len(x), size=K, replace=False)].copy()
    var = np.full(K, x.var())
    pi = np.full(K, 1 / K)
    hist = []
    for _ in range(n_iter):
        hist.append(log_likelihood(x, pi, mu, var))
        R = e_step(x, pi, mu, var)
        pi, mu, var = m_step(x, R)
    hist.append(log_likelihood(x, pi, mu, var))
    return pi, mu, var, hist
