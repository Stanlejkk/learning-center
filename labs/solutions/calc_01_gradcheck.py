"""Reference solution for calc_01_gradcheck (numerical derivatives + gradient checking)."""
import numpy as np


def numerical_derivative(f, x, h=1e-5):
    """Central difference f'(x) ≈ (f(x+h) - f(x-h)) / (2h) for scalar f and scalar x."""
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x, h=1e-5):
    """Central-difference gradient of a scalar function f at array x (any shape). Must not modify x."""
    x = np.array(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + h
        fp = f(x)
        x[idx] = old - h
        fm = f(x)
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * h)
    return grad


def relative_error(a, b, eps=1e-12):
    """max_i |a_i - b_i| / max(|a_i| + |b_i|, eps)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), eps)))


def grad_check(f, grad_f, x, h=1e-5, tol=1e-6):
    """Compare analytic grad_f(x) with numerical_gradient(f, x). Return (ok, rel_err)."""
    num = numerical_gradient(f, x, h)
    ana = grad_f(np.array(x, dtype=float))
    err = relative_error(num, ana)
    return err < tol, err


def grad_quadratic(A, b, x):
    """Gradient of f(x) = 0.5 x^T A x - b^T x for a (not necessarily symmetric) square A."""
    return 0.5 * (A + A.T) @ x - b


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(w, X, y):
    p = sigmoid(X @ w)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))


def grad_logistic_loss(w, X, y):
    """Gradient of the mean binary cross-entropy of logistic regression: X^T (sigmoid(Xw) - y) / n."""
    return X.T @ (sigmoid(X @ w) - y) / X.shape[0]


def mlp_loss(params, x, y):
    """Tiny MLP: h = tanh(W1 x + b1); yhat = W2 h + b2 (scalar); L = 0.5 (yhat - y)^2."""
    h = np.tanh(params["W1"] @ x + params["b1"])
    yhat = params["W2"] @ h + params["b2"]
    return float(0.5 * (yhat - y) ** 2)


def mlp_grads(params, x, y):
    """Backprop for mlp_loss. Returns dict with the same keys/shapes as params."""
    z = params["W1"] @ x + params["b1"]
    h = np.tanh(z)
    yhat = params["W2"] @ h + params["b2"]
    d = yhat - y                       # dL/dyhat (scalar)
    dW2 = d * h
    db2 = np.array(d, dtype=float).reshape(np.shape(params["b2"]))
    dh = d * params["W2"]
    dz = dh * (1 - h ** 2)
    dW1 = np.outer(dz, x)
    db1 = dz
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
