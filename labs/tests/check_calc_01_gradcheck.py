"""Hidden checks for calc_01_gradcheck. Each check_*(ns) pulls the learner's functions from ns."""
import numpy as np


def _ref_num_grad(f, x, h=1e-6):
    x = np.array(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros(x.size)
        e[i] = h
        e = e.reshape(x.shape)
        g.flat[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def check_numerical_derivative(ns):
    nd = ns["numerical_derivative"]
    for x0 in (-1.3, 0.2, 2.0):
        assert abs(nd(np.sin, x0) - np.cos(x0)) < 1e-8, "wrong derivative of sin"
    # central (not forward) difference: error O(h^2). For x^3 at x=2 with h=1e-3 a forward
    # difference is off by ~6e-3, a central one by ~1e-6.
    val = nd(lambda t: t ** 3, 2.0, h=1e-3)
    assert abs(val - 12.0) < 1e-5, f"got {val}; use the central difference (f(x+h) - f(x-h)) / (2h)"


def check_numerical_gradient(ns):
    ng = ns["numerical_gradient"]
    rng = np.random.default_rng(7)
    X = rng.normal(size=(3, 4))
    X0 = X.copy()
    f = lambda M: float(np.sum(M ** 3) + np.sum(M[:, 0] * M[:, 1]))  # noqa: E731
    g = ng(f, X)
    assert np.shape(g) == X.shape, f"gradient shape {np.shape(g)} != input shape {X.shape}"
    true = 3 * X0 ** 2
    true[:, 0] += X0[:, 1]
    true[:, 1] += X0[:, 0]
    assert np.allclose(g, true, atol=1e-6), "numerical gradient is wrong"
    assert np.array_equal(X, X0), "numerical_gradient must not modify its input"


def check_grad_check_detects_bugs(ns):
    gc, rel = ns["grad_check"], ns["relative_error"]
    assert rel(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
    assert abs(rel(np.array([1.0]), np.array([3.0])) - 0.5) < 1e-12, "relative_error should be |a-b|/(|a|+|b|)"
    f = lambda v: float(np.sum(np.exp(v)) + v[0] * v[1])  # noqa: E731
    good = lambda v: np.exp(v) + np.array([v[1], v[0], 0.0])  # noqa: E731
    bad = lambda v: np.exp(v) + np.array([v[0], v[1], 0.0])  # noqa: E731
    x = np.array([0.3, -0.7, 1.1])
    ok, err = gc(f, good, x)
    assert ok and err < 1e-6, f"correct gradient rejected (rel err {err})"
    ok, err = gc(f, bad, x)
    assert not ok, "grad_check accepted a wrong gradient"


def check_grad_quadratic(ns):
    gq = ns["grad_quadratic"]
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4))  # NOT symmetric
    b = rng.normal(size=4)
    x = rng.normal(size=4)
    f = lambda v: 0.5 * v @ A @ v - b @ v  # noqa: E731
    assert np.allclose(gq(A, b, x), _ref_num_grad(f, x), atol=1e-6), \
        "grad of 0.5 x^T A x - b^T x is 0.5 (A + A^T) x - b when A is not symmetric"


def check_logistic_grad(ns):
    gl = ns["grad_logistic_loss"]
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 3))
    y = (rng.random(6) > 0.5).astype(float)
    w = rng.normal(size=3)

    def loss(v):
        p = 1 / (1 + np.exp(-(X @ v)))
        return float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))
    assert np.allclose(gl(w, X, y), _ref_num_grad(loss, w), atol=1e-6), "logistic-loss gradient is wrong (mean over n?)"


def check_mlp_grads(ns):
    mg = ns["mlp_grads"]
    rng = np.random.default_rng(11)
    params = {"W1": rng.normal(size=(4, 3)), "b1": rng.normal(size=4),
              "W2": rng.normal(size=4), "b2": np.array(0.3)}
    x, y = rng.normal(size=3), 0.7

    def loss_of(key):
        def f(v):
            p = dict(params)
            p[key] = v
            h = np.tanh(p["W1"] @ x + p["b1"])
            return float(0.5 * (p["W2"] @ h + p["b2"] - y) ** 2)
        return f
    grads = mg(params, x, y)
    for key in ("W1", "b1", "W2"):
        assert key in grads, f"missing gradient for {key}"
        num = _ref_num_grad(loss_of(key), params[key])
        assert np.shape(grads[key]) == np.shape(params[key]), f"{key}: shape {np.shape(grads[key])}"
        assert np.allclose(grads[key], num, atol=1e-6), f"{key}: gradient mismatch"
