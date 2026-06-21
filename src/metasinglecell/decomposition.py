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
    X,
    n_comps: int = 50,
    solver: str = "randomized",
    random_state: int = 0,
    n_oversamples: int = 10,
    n_iter: int = 7,
):
    """Principal component analysis. See module docstring for solver semantics.

    ``X`` may be a dense array OR a ``sparse.CSR``. For a CSR the randomized solver
    runs **sparse-aware**: implicit mean-centering (zero_center=True) with a Metal
    SpMM range-finder, never densifying — this is the scalable atlas path (matches
    ``rapids-singlecell``/scanpy PCA on sparse lognorm). Scaling (z-scoring) is the
    dense path on purpose, since z-scoring destroys sparsity.
    """
    from .sparse import CSR

    if isinstance(X, CSR):
        if solver != "randomized":
            raise ValueError("CSR input supports solver='randomized' only (sparse path)")
        return _pca_randomized_sparse(X, n_comps, random_state, n_oversamples, n_iter)

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


def _ortho(Z):
    """Gram-matrix orthonormalization ``Z(ZᵀZ)^{-1/2}`` (GPU matmuls + tiny host eigh).

    Replaces a CPU QR (MLX has no GPU QR): the tall n×size work stays on the GPU and
    only a size×size eigendecomposition touches the host — the range-finder's hot path.
    """
    import mlx.core as mx

    G = Z.T @ Z                                       # size × size (GPU)
    w, V = mx.linalg.eigh(G, stream=mx.cpu)           # tiny eigendecomposition
    inv_sqrt = V @ (mx.diag(1.0 / mx.sqrt(mx.maximum(w, 1e-12))) @ V.T)
    return Z @ inv_sqrt                               # orthonormal columns (GPU)


def _pca_randomized(Xc, n_samples: int, n_comps: int, random_state: int,
                    n_oversamples: int, n_iter: int):
    """Randomized SVD: GPU matmuls (fp32) + fp64 core SVD of the projection."""
    import mlx.core as mx

    rng = np.random.RandomState(random_state)
    size = n_comps + n_oversamples
    n_features = Xc.shape[1]
    Q = mx.array(rng.normal(size=(n_features, size)).astype(np.float32))

    for _ in range(n_iter):
        Q = _ortho(Xc @ Q)                                # n × size
        Q = _ortho(Xc.T @ Q)                              # features × size
    Q = _ortho(Xc @ Q)                                    # n × size

    B = Q.T @ Xc                                          # size × features (GPU)
    mx.eval(B, Q)

    # fp64 SVD of the small projected matrix (Accelerate/LAPACK).
    Uhat, S, Vt = np.linalg.svd(np.asarray(B).astype(np.float64), full_matrices=False)
    U = np.asarray(Q).astype(np.float64) @ Uhat
    total_var = np.asarray(Xc).astype(np.float64).var(axis=0, ddof=1).sum()
    return _finalize(U, S, Vt, n_samples, n_comps, total_var)


def _pca_randomized_sparse(csr, n_comps: int, random_state: int,
                           n_oversamples: int, n_iter: int):
    """Sparse-aware randomized PCA with implicit mean-centering (no densify).

    The centered matrix is ``Xc = X - 1·μᵀ`` (μ = column means). Every product with
    ``Xc`` is the sparse product with ``X`` (GPU SpMM kernel) plus a rank-1 correction,
    so X is never densified and the lognorm sparsity is preserved end-to-end:
        ``Xc·Q  = X·Q  − 1·(μᵀQ)``           (subtract μ·Q from every row)
        ``Xcᵀ·Q = Xᵀ·Q − μ·(1ᵀQ)``           (subtract outer(μ, colsum Q))
    """
    import mlx.core as mx

    n, f = csr.shape
    size = n_comps + n_oversamples
    mean, var = csr.col_moments()                         # μ and per-gene variance (GPU)
    mu = mean                                             # (f,)

    rng = np.random.RandomState(random_state)
    Q = mx.array(rng.normal(size=(f, size)).astype(np.float32))

    def AQ(Qf):                                           # Xc · Qf : (f,size) -> (n,size)
        return csr.spmm(Qf) - (mu @ Qf)[None, :]

    def ATQ(Qn):                                          # Xcᵀ · Qn : (n,size) -> (f,size)
        return csr.spmm_t(Qn) - mu[:, None] * mx.sum(Qn, axis=0)[None, :]

    for _ in range(n_iter):
        Q = _ortho(AQ(Q))                                 # n × size
        Q = _ortho(ATQ(Q))                                # f × size
    Q = _ortho(AQ(Q))                                     # n × size (range basis)

    B = ATQ(Q).T                                          # (size × f) = Qᵀ Xc
    mx.eval(B, Q)

    Uhat, S, Vt = np.linalg.svd(np.asarray(B).astype(np.float64), full_matrices=False)
    U = np.asarray(Q).astype(np.float64) @ Uhat
    total_var = float(np.asarray(mx.sum(var)))
    return _finalize(U, S, Vt, n, n_comps, total_var)
