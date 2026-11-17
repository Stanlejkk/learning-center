"""Reference solution for labs/ml_01_regression_from_scratch.ipynb (keep in sync with the notebook's TODO cells)."""
from __future__ import annotations

import numpy as np


def add_bias(X):
    """Append a column of ones so the last weight is the intercept."""
    X = np.asarray(X, dtype=float)
    return np.hstack([X, np.ones((X.shape[0], 1))])


# ----------------------------------------------------------------- linear regression
def fit_normal_equations(X, y):
    """Least-squares weights solving (XᵀX) w = Xᵀy (X already contains a bias column if wanted)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    return np.linalg.solve(X.T @ X, X.T @ y)


def mse_loss(w, X, y):
    r = X @ w - y
    return float(np.mean(r ** 2))


def mse_grad(w, X, y):
    return 2 * X.T @ (X @ w - y) / len(y)


def fit_gd(X, y, lr=0.1, steps=2000):
    """Batch gradient descent on the MSE, starting from w = 0. Returns the final w."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w = w - lr * mse_grad(w, X, y)
    return w


# ----------------------------------------------------------------- logistic regression
def sigmoid(z):
    """Numerically stable logistic function (no overflow warnings for |z| ~ 1000)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def bce_loss(w, X, y):
    """Mean binary cross-entropy (nats) of sigmoid(Xw) against labels y in {0,1}."""
    z = X @ w
    # -[y log σ(z) + (1-y) log(1-σ(z))] = softplus(z) - y z   (stable form)
    return float(np.mean(np.logaddexp(0, z) - y * z))


def bce_grad(w, X, y):
    """Hand-derived gradient: Xᵀ(σ(Xw) - y) / n."""
    return X.T @ (sigmoid(X @ w) - y) / len(y)


def numerical_grad(f, w, eps=1e-6):
    """Central finite differences of a scalar function f at w."""
    w = np.asarray(w, dtype=float)
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = eps
        g[i] = (f(w + e) - f(w - e)) / (2 * eps)
    return g


def fit_logistic(X, y, lr=0.5, steps=3000):
    """Gradient descent on the mean BCE from w = 0. Returns (w, loss_history)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    w = np.zeros(X.shape[1])
    hist = []
    for _ in range(steps):
        hist.append(bce_loss(w, X, y))
        w = w - lr * bce_grad(w, X, y)
    return w, hist


def accuracy(w, X, y):
    return float(np.mean((sigmoid(X @ w) >= 0.5) == (y == 1)))
