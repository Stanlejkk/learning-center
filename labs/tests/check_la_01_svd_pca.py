"""Hidden checks for la_01_svd_pca. Each check_*(ns) pulls the learner's functions from ns."""
import numpy as np


def _spd(rng, eigs):
    Q, _ = np.linalg.qr(rng.normal(size=(len(eigs), len(eigs))))
    return Q @ np.diag(eigs) @ Q.T, Q


def check_power_iteration(ns):
    pi = ns["power_iteration"]
    rng = np.random.default_rng(5)
    eigs = np.array([5.0, 2.0, 1.0, 0.5, 0.1])
    A, Q = _spd(rng, eigs)
    lam, v = pi(A)
    v = np.asarray(v, dtype=float)
    assert abs(lam - 5.0) < 1e-6, f"top eigenvalue {lam}, expected 5"
    assert abs(np.linalg.norm(v) - 1) < 1e-6, "eigenvector must have unit length"
    assert abs(abs(v @ Q[:, 0]) - 1) < 1e-5, "eigenvector does not match the top eigenvector (up to sign)"
    assert np.allclose(A @ v, lam * v, atol=1e-5), "A v != lambda v"


def check_low_rank_approx(ns):
    lra, eye = ns["low_rank_approx"], ns["eckart_young_error"]
    rng = np.random.default_rng(9)
    M = rng.normal(size=(8, 6))
    s = np.linalg.svd(M, compute_uv=False)
    for k in (1, 2, 4):
        Mk = np.asarray(lra(M, k))
        assert Mk.shape == M.shape, f"rank-{k} approximation has shape {Mk.shape}"
        assert np.linalg.matrix_rank(Mk, tol=1e-8) == k, f"result is not rank {k}"
        err = np.linalg.norm(M - Mk)
        assert abs(err - np.sqrt(np.sum(s[k:] ** 2))) < 1e-8, f"rank-{k} error {err} is not optimal (Eckart–Young)"
        assert abs(eye(s, k) - err) < 1e-8, "eckart_young_error should be sqrt(sum of the discarded sigma^2)"
    assert np.allclose(lra(M, 6), M), "full-rank approximation should reproduce M"


def _data(rng):
    n = 200
    z = rng.normal(size=(n, 2)) * np.array([3.0, 1.0])
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    X = np.c_[z, 0.1 * rng.normal(size=n)] @ R.T + np.array([2.0, -1.0, 5.0])
    return X


def check_pca_fit(ns):
    fit = ns["pca_fit"]
    rng = np.random.default_rng(21)
    X = _data(rng)
    mean, comps, var = fit(X, 2)
    comps, var = np.asarray(comps), np.asarray(var)
    assert np.allclose(mean, X.mean(axis=0)), "mean is wrong"
    assert comps.shape == (2, 3), f"components should have shape (k, d) = (2, 3), got {comps.shape}"
    assert np.allclose(comps @ comps.T, np.eye(2), atol=1e-8), "components must be orthonormal rows"
    C = np.cov(X, rowvar=False)  # 1/(n-1)
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    assert np.allclose(var, w[:2], rtol=1e-6), "explained_variance should be the top eigenvalues of the 1/(n-1) covariance"
    for j in range(2):
        assert abs(abs(comps[j] @ V[:, j]) - 1) < 1e-6, f"component {j} is not the j-th eigenvector of the covariance"


def check_pca_roundtrip(ns):
    fit, tr, rec, evr = ns["pca_fit"], ns["pca_transform"], ns["pca_reconstruct"], ns["explained_variance_ratio"]
    rng = np.random.default_rng(4)
    X = _data(rng)
    mean, comps, _ = fit(X, 3)
    Z = tr(X, mean, comps)
    assert np.shape(Z) == (X.shape[0], 3)
    assert np.allclose(rec(Z, mean, comps), X), "full-rank PCA reconstruction should give X back"
    mean, comps, _ = fit(X, 1)
    Xh = rec(tr(X, mean, comps), mean, comps)
    lra_err = np.linalg.norm(X - Xh) ** 2
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    assert abs(lra_err - np.sum(s[1:] ** 2)) < 1e-6, "rank-1 reconstruction error should equal the discarded variance"
    assert abs(evr(X, 1) - s[0] ** 2 / np.sum(s ** 2)) < 1e-9, "explained_variance_ratio is wrong"
