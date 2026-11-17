"""Hidden checks for ml_01_regression_from_scratch. Each check_*(ns) raises AssertionError on failure."""
from __future__ import annotations

import warnings

import numpy as np


def _lin_data(seed=0, n=60):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(n, 2))
    X = np.hstack([x, np.ones((n, 1))])
    w_true = np.array([1.5, -0.7, 0.3])
    y = X @ w_true + 0.1 * rng.normal(size=n)
    return X, y


def _blobs(seed=1, n=100):
    rng = np.random.default_rng(seed)
    a = rng.normal([-1.5, -1.0], 0.8, size=(n, 2))
    b = rng.normal([1.5, 1.0], 0.8, size=(n, 2))
    X = np.vstack([a, b])
    X = np.hstack([X, np.ones((2 * n, 1))])
    y = np.r_[np.zeros(n), np.ones(n)]
    return X, y


def _sig(z):
    return 0.5 * (1 + np.tanh(z / 2))


def check_normal_equations(ns):
    X, y = _lin_data()
    w = np.asarray(ns["fit_normal_equations"](X, y), float)
    ref = np.linalg.lstsq(X, y, rcond=None)[0]
    assert w.shape == ref.shape, f"expected shape {ref.shape}, got {w.shape}"
    assert np.allclose(w, ref, atol=1e-8), f"normal-equation weights {w} differ from least squares {ref}"


def check_gradient_descent_matches(ns):
    X, y = _lin_data(seed=3)
    w = np.asarray(ns["fit_gd"](X, y, lr=0.1, steps=2000), float)
    ref = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(w, ref, atol=1e-3), f"GD did not converge to the least-squares solution: {w} vs {ref}"


def check_sigmoid_stable(ns):
    s = ns["sigmoid"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # overflow warnings count as failures
        v = np.asarray(s(np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])), float)
    assert np.all(np.isfinite(v)), "sigmoid returned nan/inf for large |z|"
    assert np.allclose(v, _sig(np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])), atol=1e-12), f"wrong sigmoid values {v}"


def check_bce_loss_and_gradient(ns):
    rng = np.random.default_rng(7)
    X = np.hstack([rng.normal(size=(20, 3)), np.ones((20, 1))])
    y = (rng.random(20) < 0.5).astype(float)
    w = rng.normal(size=4)
    p = _sig(X @ w)
    ref_loss = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    got = float(ns["bce_loss"](w, X, y))
    assert abs(got - ref_loss) < 1e-8, f"bce_loss: expected {ref_loss:.8f}, got {got}"
    g = np.asarray(ns["bce_grad"](w, X, y), float)
    eps = 1e-6
    num = np.array([(ns["bce_loss"](w + eps * e, X, y) - ns["bce_loss"](w - eps * e, X, y)) / (2 * eps)
                    for e in np.eye(4)])
    assert np.allclose(g, num, atol=1e-6), f"analytic gradient {g} disagrees with finite differences {num}"
    assert np.allclose(g, X.T @ (p - y) / len(y), atol=1e-10), "gradient should be Xᵀ(σ(Xw) - y)/n"


def check_numerical_grad(ns):
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    f = lambda w: float(w @ A @ w + np.sin(w[0]))  # noqa: E731
    w = np.array([0.3, -0.8])
    g = np.asarray(ns["numerical_grad"](f, w), float)
    ref = 2 * A @ w + np.array([np.cos(w[0]), 0.0])
    assert np.allclose(g, ref, atol=1e-5), f"numerical_grad gives {g}, expected {ref}"


def check_logistic_fit(ns):
    X, y = _blobs()
    w, hist = ns["fit_logistic"](X, y, lr=0.5, steps=3000)
    w = np.asarray(w, float)
    acc = float(np.mean((_sig(X @ w) >= 0.5) == (y == 1)))
    assert acc > 0.9, f"training accuracy {acc:.3f} too low"
    hist = list(hist)
    assert len(hist) > 10 and hist[-1] < hist[0], "loss history should decrease"
    p = _sig(X @ w)
    final = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    assert final < 0.25, f"final mean BCE {final:.3f} is higher than expected (<0.25)"
