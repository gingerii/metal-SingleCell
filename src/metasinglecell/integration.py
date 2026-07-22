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
    # Row L2-norm on the GPU. (Previously forced onto stream=mx.cpu, which put a
    # device→host dependency on the hottest path — every assign + centroid update.)
    return M / (mx.sqrt(mx.sum(M * M, axis=1, keepdims=True)) + 1e-12)


def _correction_gpu(Zg, R, Phi_g, O, ridge_lambda: float, B: int):
    """On-GPU Harmony ridge correction — analytic block-inverse, no linear solver.

    Port of harmony-pytorch's ``correction`` (lilab-bcb/harmony-pytorch), the acknowledged
    upstream for scanpy's/rapids' Harmony. Each cluster's ridge system is the tiny
    ``(B+1)×(B+1)`` matrix ``Φ₁ᵀ diag(R_k) Φ₁ + λ·pen``; harmony-pytorch inverts it in closed
    form from its block structure (a diagonal build + two matmuls) instead of calling a solver —
    which sidesteps MLX's CPU-backed ``linalg.solve`` entirely and keeps every op on the GPU.

    This is a **Jacobi** correction (all K clusters use the fixed ``Zg`` passed in, accumulating
    into ``Z``), matching harmony-pytorch — unlike our previous running-``Zc`` (Gauss-Seidel)
    host loop. Conventions here: ``R`` is (K×N), ``O`` is (K×B), ``Phi_g`` is (N×B). fp32 on-device
    (the upstream reference runs fp32 on-device too).
    """
    import mlx.core as mx
    N, d = Zg.shape
    K = R.shape[0]
    Phi_1 = mx.concatenate([mx.ones((N, 1)), Phi_g], axis=1)   # N × (B+1)
    Z = Zg
    for k in range(K):
        O_k = O[k]                                             # (B,)
        N_k = mx.sum(O_k)
        factor = 1.0 / (O_k + ridge_lambda)                   # (B,)
        c = N_k + mx.sum(-factor * O_k * O_k)
        c_inv = 1.0 / c
        top = -factor * O_k                                   # P[0, 1:]  (B,)
        P = mx.eye(B + 1)
        P = P.at[0, 1:].add(top)
        diag_vec = mx.concatenate([c_inv[None], factor])      # (B+1,)
        P_t_B_inv = mx.diag(diag_vec)
        P_t_B_inv = P_t_B_inv.at[1:, 0].add(top * c_inv)
        inv_mat = P_t_B_inv @ P                               # (B+1)×(B+1)
        Phi_t_diag_R = Phi_1.T * R[k][None, :]                # (B+1) × N
        W = inv_mat @ (Phi_t_diag_R @ Zg)                     # (B+1) × d  (uses fixed Zg)
        W = mx.concatenate([mx.zeros((1, d)), W[1:]], axis=0)  # keep global mean (W[0]=0)
        Z = Z - Phi_t_diag_R.T @ W
    mx.eval(Z)
    return Z


def _correction_host(Zg, R, Phi, ridge_lambda: float, B: int, K: int):
    """fp64 host ridge correction (running-Zc / Gauss-Seidel) — retained as a validation
    oracle / fp64 numerical anchor. K small (B+1)×(B+1) ``np.linalg.solve`` on the CPU.
    """
    import mlx.core as mx
    N = Zg.shape[0]
    phi1 = np.concatenate([np.ones((N, 1)), np.asarray(Phi, np.float64)], axis=1)  # N × (B+1)
    Rn = np.asarray(R, np.float64)                       # K × N
    Zc = np.asarray(Zg, np.float64)
    pen = np.diag([0.0] + [ridge_lambda] * B)
    for k in range(K):
        phi_rk = phi1 * Rn[k][:, None]
        A = phi_rk.T @ phi1 + pen
        W = np.linalg.solve(A, phi_rk.T @ Zc)
        W[0] = 0.0
        Zc = Zc - phi_rk @ W
    return mx.array(Zc.astype(np.float32))


def harmonize(Z, batch, n_clusters: int | None = None, sigma: float = 0.1,
              theta: float = 2.0, ridge_lambda: float = 1.0,
              max_iter_harmony: int = 10, max_iter_clustering: int = 20,
              tol_harmony: float = 1e-4, tol_clustering: float = 1e-5,
              random_state: int = 0, correction: str = "gpu") -> np.ndarray:
    """Harmony batch correction of an embedding. Returns the corrected ``Z``.

    ``Z`` is (cells × n_pcs); ``batch`` is a length-cells array of batch labels.
    """
    import mlx.core as mx

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
    # GPU Lloyd's (our tools.kmeans) on the L2-normalized embedding — for unit vectors
    # Euclidean k-means ≡ cosine k-means, the same normalized-space init sklearn/harmony-pytorch
    # use. Returns labels; derive normalized centroids with one one-hot matmul. Replaces the
    # host sklearn KMeans (n_init=10) — a CPU excursion at every run's start. Init only needs to
    # be reasonable (Harmony refines it).
    from .tools import kmeans as _kmeans
    labels = _kmeans(np.asarray(Z_norm), n_clusters=K, max_iter=25, random_state=random_state)
    onehot = (mx.array(labels)[:, None] == mx.arange(K)[None, :]).astype(mx.float32)  # N × K
    Y = _l2_normalize((onehot.T @ Z_norm) / mx.maximum(mx.sum(onehot, axis=0, keepdims=True).T, 1.0))

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

    rng = np.random.default_rng(random_state)
    block_size = max(1, int(0.05 * N))
    prev_obj = objective()
    for _ in range(max_iter_harmony):
        # ---- clustering: BLOCK-STOCHASTIC penalized assignment (harmony) ----
        # Cells are updated in random blocks; each block's contribution is removed
        # from O/E before re-assigning, so the diversity penalty (E/O) reflects the
        # rest of the data — this is what actually drives batch mixing.
        for _ in range(max_iter_clustering):
            Y = _l2_normalize(R @ Z_norm)                    # K × d centroids
            R_prev = R
            perm = rng.permutation(N)                         # random blocks
            for start in range(0, N, block_size):
                blk = perm[start:start + block_size]
                bg = mx.array(blk.astype(np.int32))
                Phi_b = Phi_g[bg]
                Rb_old = R[:, bg]
                O = O - Rb_old @ Phi_b                       # remove block
                E = E - mx.sum(Rb_old, axis=1, keepdims=True) * Pr_g[None, :]
                omega = mx.power((E + 1e-12) / (O + 1e-12), theta)
                dist = 2.0 * (1.0 - Y @ Z_norm[bg].T)        # K × |blk|
                Rb = mx.exp(-dist / sigma) * (omega @ Phi_b.T)
                Rb = Rb / (mx.sum(Rb, axis=0, keepdims=True) + 1e-12)
                R = R.at[:, bg].add(Rb - Rb_old)             # write block
                O = O + Rb @ Phi_b                           # add block back
                E = E + mx.sum(Rb, axis=1, keepdims=True) * Pr_g[None, :]
            mx.eval(R, O, E)
            if float(mx.mean(mx.abs(R - R_prev)).item()) < tol_clustering:
                break

        # ---- correction: per-cluster ridge regression removing batch shift ----
        # "gpu" (default): on-GPU analytic block-inverse (ported from harmony-pytorch) — no
        # linear solver, no host round-trip. "host": fp64 CPU oracle (running-Zc). See the
        # two _correction_* helpers. O here is (K×B); it pairs with R (K×N).
        if correction == "gpu":
            Zg = _correction_gpu(Zg, R, Phi_g, O, ridge_lambda, B)
        else:
            Zg = _correction_host(Zg, R, Phi, ridge_lambda, B, K)
        Z_norm = _l2_normalize(Zg)
        R = assign(Y)
        O = R @ Phi_g
        E = mx.sum(R, axis=1, keepdims=True) * Pr_g[None, :]

        obj = objective()
        if abs(obj - prev_obj) < tol_harmony * abs(prev_obj):
            break
        prev_obj = obj

    return np.asarray(Zg)
