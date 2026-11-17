"""Reference solution for la_01_svd_pca (power iteration, low-rank approximation, PCA from scratch)."""
import numpy as np


def power_iteration(A, num_iters=1000, tol=1e-10, seed=0):
    """Top eigenpair of a symmetric PSD matrix A. Returns (eigenvalue, unit eigenvector)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=A.shape[0])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(num_iters):
        w = A @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            return 0.0, v
        v_new = w / nw
        lam_new = float(v_new @ A @ v_new)
        if abs(lam_new - lam) < tol and np.linalg.norm(v_new - v) < 1e-8:
            v = v_new
            lam = lam_new
            break
        v, lam = v_new, lam_new
    return lam, v


def low_rank_approx(M, k):
    """Best rank-k approximation of M (Eckart–Young) via np.linalg.svd."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def eckart_young_error(s, k):
    """Frobenius error of the best rank-k approximation, from the singular values s."""
    s = np.asarray(s, dtype=float)
    return float(np.sqrt(np.sum(s[k:] ** 2)))


def pca_fit(X, k):
    """PCA on rows of X (n samples × d features).

    Returns (mean (d,), components (k, d) with orthonormal rows sorted by decreasing variance,
    explained_variance (k,) = eigenvalues of the sample covariance with 1/(n-1)).
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2 / (X.shape[0] - 1)
    return mean, Vt[:k], var[:k]


def pca_transform(X, mean, components):
    """Coordinates of the rows of X in the principal subspace: (X - mean) @ components.T."""
    return (X - mean) @ components.T


def pca_reconstruct(Z, mean, components):
    """Map PCA coordinates back to the original space."""
    return Z @ components + mean


def explained_variance_ratio(X, k):
    """Fraction of the total variance captured by the first k principal components."""
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return float(np.sum(s[:k] ** 2) / np.sum(s ** 2))
