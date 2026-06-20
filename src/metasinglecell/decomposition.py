"""PCA on Apple Silicon — three solvers, one interface.

Mirrors sklearn/scanpy ``pca``. The heavy matmuls run fp32 on the Metal GPU
(MLX); the stability-critical SVD core runs fp64 on Accelerate/LAPACK (NumPy/
SciPy) — MLX is fp32-only and has no GPU SVD, so this split is both necessary
and the architectural point.

Solvers
-------
* ``"randomized"`` — random projection + power iterations (GPU matmuls) then a
  small fp64 SVD of the projected matrix. The one that actually exploits the GPU
  and scales to large cohorts.
* ``"full"``       — exact ``np.linalg.svd`` (LAPACK gesdd) on the centered matrix.
* ``"arpack"``     — truncated Lanczos via ``scipy.sparse.linalg.svds`` (what the
  reference oracle uses).

All return ``(X_pca, components, variance_ratio)`` with components as (n_comps x
n_features), signs fixed by ``svd_flip`` (sklearn convention).
"""

from __future__ import annotations

import numpy as np


def _svd_flip(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sklearn svd_flip (u_based_decision=True): deterministic sign per component."""
    signs = np.sign(u[np.argmax(np.abs(u), axis=0), range(u.shape[1])])
    return u * signs, v * signs[:, None]


def _finalize(U, S, Vt, n_samples: int, n_comps: int, total_var: float):
    U, Vt = _svd_flip(U, Vt)
    components = Vt[:n_comps]
    x_pca = U[:, :n_comps] * S[:n_comps]
    explained_variance = (S[:n_comps] ** 2) / (n_samples - 1)
    variance_ratio = explained_variance / total_var
    return x_pca.astype(np.float32), components.astype(np.float32), variance_ratio.astype(np.float64)


def _center_gpu(X: np.ndarray):
    """Mean-center on the GPU (fp32). Returns the MLX centered array."""
    import mlx.core as mx

    Xg = mx.array(np.asarray(X, dtype=np.float32))
    Xc = Xg - mx.mean(Xg, axis=0)
    mx.eval(Xc)
    return Xc


def pca(
    X: np.ndarray,
    n_comps: int = 50,
    solver: str = "randomized",
    random_state: int = 0,
    n_oversamples: int = 10,
    n_iter: int = 7,
):
    """Principal component analysis. See module docstring for solver semantics."""
    n_samples = X.shape[0]
    Xc = _center_gpu(X)

    if solver == "randomized":
        return _pca_randomized(Xc, n_samples, n_comps, random_state, n_oversamples, n_iter)

    # full / arpack: hand the centered matrix to LAPACK in fp64.
    Xc64 = np.asarray(Xc).astype(np.float64)
    total_var = Xc64.var(axis=0, ddof=1).sum()

    if solver == "full":
        U, S, Vt = np.linalg.svd(Xc64, full_matrices=False)
    elif solver == "arpack":
        from scipy.sparse.linalg import svds

        U, S, Vt = svds(Xc64, k=n_comps, random_state=random_state)
        order = np.argsort(-S)  # svds returns ascending; reorder descending
        U, S, Vt = U[:, order], S[order], Vt[order]
    else:
        raise ValueError(f"unknown solver {solver!r} (full|arpack|randomized)")

    return _finalize(U, S, Vt, n_samples, n_comps, total_var)


def _pca_randomized(Xc, n_samples: int, n_comps: int, random_state: int,
                    n_oversamples: int, n_iter: int):
    """Randomized SVD: GPU matmuls (fp32) + fp64 core SVD of the projection."""
    import mlx.core as mx

    rng = np.random.RandomState(random_state)
    size = n_comps + n_oversamples
    n_features = Xc.shape[1]
    Q = mx.array(rng.normal(size=(n_features, size)).astype(np.float32))

    # Range finder with QR normalizer (power iterations). Matmuls on the GPU;
    # the (small, tall) QRs on CPU since MLX QR is CPU-only.
    for _ in range(n_iter):
        Q, _ = mx.linalg.qr(Xc @ Q, stream=mx.cpu)        # n x size
        Q, _ = mx.linalg.qr(Xc.T @ Q, stream=mx.cpu)      # features x size
    Q, _ = mx.linalg.qr(Xc @ Q, stream=mx.cpu)            # n x size

    B = Q.T @ Xc                                          # size x features (GPU)
    mx.eval(B, Q)

    # fp64 SVD of the small projected matrix (Accelerate/LAPACK).
    Uhat, S, Vt = np.linalg.svd(np.asarray(B).astype(np.float64), full_matrices=False)
    U = np.asarray(Q).astype(np.float64) @ Uhat
    total_var = np.asarray(Xc).astype(np.float64).var(axis=0, ddof=1).sum()
    return _finalize(U, S, Vt, n_samples, n_comps, total_var)
