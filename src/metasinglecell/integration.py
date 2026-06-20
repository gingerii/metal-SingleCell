"""Batch integration on Metal/MLX — Harmony (rapids-singlecell ``pp.harmony_integrate``).

Port of the Harmony algorithm (Korsunsky et al. 2019), following the harmony-pytorch
implementation (https://github.com/lilab-bcb/harmony-pytorch) that rapids-singlecell
is based on. Harmony adjusts a PCA embedding so cells mix across batches while
preserving biological structure, by alternating:

1. **Soft cosine k-means** with a batch-diversity penalty (clusters that are over-
   represented by one batch are down-weighted via the observed/expected ratio).
2. **Per-cluster ridge correction** that removes the batch-specific shift.

Heavy matmuls run on the GPU (MLX); the small per-cluster ridge solves run in
NumPy (fp64). Operates on the embedding ``Z`` (cells × n_pcs); returns corrected Z.
"""

from __future__ import annotations

import numpy as np


def _l2_normalize(M):
    import mlx.core as mx
    return M / (mx.linalg.norm(M, axis=1, keepdims=True, stream=mx.cpu) + 1e-12)


def harmonize(Z, batch, n_clusters: int | None = None, sigma: float = 0.1,
              theta: float = 2.0, ridge_lambda: float = 1.0,
              max_iter_harmony: int = 10, max_iter_clustering: int = 200,
              tol_harmony: float = 1e-4, tol_clustering: float = 1e-5,
              random_state: int = 0) -> np.ndarray:
    """Harmony batch correction of an embedding. Returns the corrected ``Z``.

    ``Z`` is (cells × n_pcs); ``batch`` is a length-cells array of batch labels.
    """
    import mlx.core as mx
    from sklearn.cluster import KMeans

    Z = np.asarray(Z, dtype=np.float32)
    N, d = Z.shape
    batch = np.asarray(batch)
    cats = np.unique(batch)
    B = len(cats)
    code = np.searchsorted(cats, batch)
    Phi = np.zeros((N, B), dtype=np.float32)
    Phi[np.arange(N), code] = 1.0
    Pr_b = Phi.mean(axis=0)                                   # batch proportions (B,)
    if n_clusters is None:
        n_clusters = int(min(100, max(2, N // 30)))
    K = n_clusters

    Zg = mx.array(Z)
    Z_norm = _l2_normalize(Zg)
    Phi_g = mx.array(Phi)
    Pr_g = mx.array(Pr_b)

    # --- initialize centroids (cosine KMeans on the normalized embedding) ---
    km = KMeans(n_clusters=K, n_init=10, max_iter=25, random_state=random_state)
    km.fit(np.asarray(Z_norm))
    Y = _l2_normalize(mx.array(km.cluster_centers_.astype(np.float32)))

    # soft assignment R (K × N) from cosine distance
    def assign(Y, omega_n=None):
        dist = 2.0 * (1.0 - Y @ Z_norm.T)                    # K × N
        R = mx.exp(-dist / sigma)
        if omega_n is not None:
            R = R * omega_n
        return R / (mx.sum(R, axis=0, keepdims=True) + 1e-12)

    R = assign(Y)
    O = R @ Phi_g                                            # observed K × B
    E = mx.sum(R, axis=1, keepdims=True) * Pr_g[None, :]     # expected K × B

    def objective():
        dist = 2.0 * (1.0 - Y @ Z_norm.T)
        kmeans_err = mx.sum(R * dist)
        entropy = sigma * mx.sum(R * mx.log(R + 1e-12))
        diversity = sigma * theta * mx.sum(R * (mx.log((O + 1e-12) / (E + 1e-12)) @ Phi_g.T))
        return float((kmeans_err + entropy + diversity).item())

    prev_obj = objective()
    for _ in range(max_iter_harmony):
        # ---- clustering: alternate centroid update and penalized assignment ----
        for _ in range(max_iter_clustering):
            Y = _l2_normalize(R @ Z_norm)                    # K × d centroids
            omega = mx.power((E + 1e-12) / (O + 1e-12), theta)   # K × B penalty
            omega_n = omega @ Phi_g.T                        # K × N (each cell's batch)
            R_new = assign(Y, omega_n)
            O = R_new @ Phi_g
            E = mx.sum(R_new, axis=1, keepdims=True) * Pr_g[None, :]
            shift = float(mx.mean(mx.abs(R_new - R)).item())
            R = R_new
            mx.eval(R, O, E)
            if shift < tol_clustering:
                break

        # ---- correction: per-cluster ridge regression removing batch shift ----
        phi1 = np.concatenate([np.ones((N, 1), np.float32), Phi], axis=1)   # N × (B+1)
        Rn = np.asarray(R)                                   # K × N
        Zc = np.asarray(Zg)                                  # corrected embedding (host)
        pen = np.diag([0.0] + [ridge_lambda] * B)            # don't penalize intercept
        for k in range(K):
            phi_rk = phi1 * Rn[k][:, None]                   # N × (B+1)
            A = phi_rk.T @ phi1 + pen
            W = np.linalg.solve(A, phi_rk.T @ Zc)            # (B+1) × d
            W[0] = 0.0                                       # keep the global mean
            Zc = Zc - phi_rk @ W
        Zg = mx.array(Zc.astype(np.float32))
        Z_norm = _l2_normalize(Zg)
        R = assign(Y)
        O = R @ Phi_g
        E = mx.sum(R, axis=1, keepdims=True) * Pr_g[None, :]

        obj = objective()
        if abs(obj - prev_obj) < tol_harmony * abs(prev_obj):
            break
        prev_obj = obj

    return np.asarray(Zg)
