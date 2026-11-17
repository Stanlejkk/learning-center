"""Hidden checks for lab opt_01_optimizers. Each check_*(ns) gets the notebook globals()."""
import numpy as np

A = np.array([[3.0, 1.0], [1.0, 2.0]])
B = np.array([1.0, -1.0])
X_STAR = np.linalg.solve(A, B)


def _grad(x):
    return A @ x - B


def check_quadratic(ns):
    f, grad, hess = ns["quadratic"](A, B)
    x = np.array([0.5, -2.0])
    assert abs(f(x) - (0.5 * x @ A @ x - B @ x)) < 1e-12, "quadratic f is wrong"
    assert np.allclose(grad(x), A @ x - B), "quadratic gradient is wrong"
    assert np.allclose(hess(x), A), "quadratic Hessian is wrong"


def check_gd(ns):
    gd = ns["gd"]
    path = np.asarray(gd(_grad, [0.0, 0.0], 0.1, 3))
    assert path.shape == (4, 2), f"gd must return all iterates incl. x0: expected shape (4, 2), got {path.shape}"
    x1 = 0.1 * B
    assert np.allclose(path[1], x1), f"first GD step {path[1]}, expected {x1}"
    lmax = np.linalg.eigvalsh(A).max()
    ok = np.asarray(gd(_grad, [1.0, 1.0], 1.9 / lmax, 400))
    assert np.linalg.norm(ok[-1] - X_STAR) < 1e-6, "GD with lr = 1.9/lambda_max should converge"
    bad = np.asarray(gd(_grad, [1.0, 1.0], 2.1 / lmax, 200))
    assert np.linalg.norm(bad[-1] - X_STAR) > 1e3, "GD with lr = 2.1/lambda_max should diverge"


def check_momentum(ns):
    mom = ns["momentum"]
    # f(x) = x^2, beta = 1/2, lr = 1/4, x0 = 2  ->  x1 = 1, x2 = 0, x3 = -1/2 (heavy-ball convention)
    path = np.asarray(mom(lambda x: 2 * x, [2.0], 0.25, 0.5, 3))
    assert np.allclose(path[:, 0], [2.0, 1.0, 0.0, -0.5]), f"momentum iterates {path[:, 0]}, expected [2, 1, 0, -0.5]"
    long = np.asarray(mom(_grad, [1.0, 1.0], 0.1, 0.9, 600))
    assert np.linalg.norm(long[-1] - X_STAR) < 1e-6, "momentum should converge on the quadratic"


def _ref_adam(grad, x0, lr, n, b1=0.9, b2=0.999, eps=1e-8):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, n + 1):
        g = grad(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return x


def check_adam(ns):
    adam = ns["adam"]
    path = np.asarray(adam(_grad, [2.0, -3.0], 0.05, 1))
    g0 = _grad(np.array([2.0, -3.0]))
    assert np.allclose(path[1], np.array([2.0, -3.0]) - 0.05 * np.sign(g0), atol=1e-6), \
        "first bias-corrected Adam step should move each coordinate by lr * sign(g)"
    path = np.asarray(adam(_grad, [2.0, -3.0], 0.05, 25))
    assert path.shape == (26, 2), f"expected shape (26, 2), got {path.shape}"
    assert np.allclose(path[-1], _ref_adam(_grad, [2.0, -3.0], 0.05, 25), atol=1e-8), "Adam iterates differ from the reference"


def check_newton(ns):
    newton = ns["newton"]
    path = np.asarray(newton(_grad, lambda x: A, [5.0, 5.0], 1))
    assert np.allclose(path[1], X_STAR), "Newton must solve a quadratic in one step"
    rp = np.asarray(newton(ns["rosenbrock_grad"], ns["rosenbrock_hess"], [-1.2, 1.0], 20))
    assert np.linalg.norm(rp[-1] - np.array([1.0, 1.0])) < 1e-8, f"Newton on Rosenbrock ended at {rp[-1]}"


def check_rosenbrock(ns):
    f, g, h = ns["rosenbrock"], ns["rosenbrock_grad"], ns["rosenbrock_hess"]
    rng = np.random.default_rng(0)
    assert abs(f(np.array([1.0, 1.0]))) < 1e-12, "Rosenbrock minimum is f(1, 1) = 0"
    for _ in range(3):
        x = rng.uniform(-1.5, 1.5, 2)
        e = 1e-6
        num_g = np.array([(f(x + e * d) - f(x - e * d)) / (2 * e) for d in np.eye(2)])
        assert np.allclose(g(x), num_g, rtol=1e-4, atol=1e-4), f"gradient mismatch at {x}"
        num_h = np.array([(g(x + e * d) - g(x - e * d)) / (2 * e) for d in np.eye(2)]).T
        assert np.allclose(h(x), num_h, rtol=1e-4, atol=1e-3), f"Hessian mismatch at {x}"


def check_iters_to_converge(ns):
    it = ns["iters_to_converge"]
    path = np.array([[4.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
    assert it(path, [0.0, 0.0], 1.5) == 2
    assert it(path, [0.0, 0.0], 0.1) == -1, "return -1 if the tolerance is never reached"
