"""Reference solution for lab opt_01_optimizers (GD, momentum, Adam, Newton on quadratics and Rosenbrock)."""
from __future__ import annotations

import numpy as np


def quadratic(A, b):
    """f(x) = 1/2 x^T A x - b^T x. Returns (f, grad, hess) as functions of x."""
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    f = lambda x: 0.5 * x @ A @ x - b @ x  # noqa: E731
    grad = lambda x: A @ x - b  # noqa: E731
    hess = lambda x: A  # noqa: E731
    return f, grad, hess


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])


def rosenbrock_hess(x):
    return np.array([[2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]], [-400 * x[0], 200.0]])


def gd(grad, x0, lr, n_steps):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(n_steps):
        x = x - lr * grad(x)
        path.append(x.copy())
    return np.array(path)


def momentum(grad, x0, lr, beta, n_steps):
    """Heavy ball: v <- beta v + grad(x); x <- x - lr v (v0 = 0)."""
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(n_steps):
        v = beta * v + grad(x)
        x = x - lr * v
        path.append(x.copy())
    return np.array(path)


def adam(grad, x0, lr, n_steps, beta1=0.9, beta2=0.999, eps=1e-8):
    x = np.array(x0, dtype=float)
    m, v = np.zeros_like(x), np.zeros_like(x)
    path = [x.copy()]
    for t in range(1, n_steps + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mh, vh = m / (1 - beta1 ** t), v / (1 - beta2 ** t)
        x = x - lr * mh / (np.sqrt(vh) + eps)
        path.append(x.copy())
    return np.array(path)


def newton(grad, hess, x0, n_steps):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(n_steps):
        x = x - np.linalg.solve(hess(x), grad(x))
        path.append(x.copy())
    return np.array(path)


def iters_to_converge(path, x_star, tol):
    """First k with ||path[k] - x_star|| < tol, or -1 if never."""
    d = np.linalg.norm(np.asarray(path) - np.asarray(x_star), axis=1)
    hit = np.nonzero(d < tol)[0]
    return int(hit[0]) if hit.size else -1
