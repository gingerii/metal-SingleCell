---
name: project
description: metaSingleCell goal (Metal/MLX port of rapids-singlecell), the 3-stage roadmap, and repo layout. Load when planning work, prioritizing, or navigating the codebase.
---

# Project — metaSingleCell

## Goal
Bring GPU-accelerated single-cell analysis to **Apple Silicon (Metal)** — a drop-in
re-implementation of [rapids-singlecell](https://rapids-singlecell.readthedocs.io), which is
**CUDA/CuPy-only**. Target: scanpy-compatible functions running on the M-series GPU.

## The core blocker
Single-cell count matrices are large and ~90–95% sparse (cells × genes). The scanpy front-end
(QC → normalize → HVG → PCA) is built on sparse linear algebra. Apple's `Accelerate` has sparse
support on **CPU only**; MPS/PyTorch-MPS sparse is thin. **Apple GPUs have no float64** (fp32, with
fp16/bf16). So "build the missing GPU sparse substrate" precedes "port scanpy", and numerical
parity means "how well does an fp32 GPU pipeline preserve fp64 CPU-scanpy results".

## Key architectural decision (2026-06-20)
**MLX-primary, custom Metal kernels for sparse, Accelerate/LAPACK as the numerical anchor.**
- MLX = array layer + dense post-PCA pipeline (KNN/UMAP/Leiden run on the dense embedding).
- Custom Metal kernels = the sparse front-end primitives (CSR SpMM, segmented row/col reductions) —
  the novel IP; no Apple framework provides them well.
- Accelerate/LAPACK anchors the stability-critical math: SVD for PCA (randomized SVD w/ LAPACK core),
  variance via two-pass/Welford (never naive `E[x²]−E[x]²` in fp32).
- Rejected: pure raw-Metal (must re-validate SVD in fp32 — stability risk); PyTorch-MPS (MPS
  correctness bugs, weak sparse); jax-metal (experimental/unstable).

## Insight that bounds the work
Most of the pipeline goes **dense after PCA**. Sparsity is load-bearing only for QC→normalize→HVG→
PCA-input. So the sparse kernel set is small and finite.

## 3-stage roadmap (from CLAUDE.md)
1. **Sparse + numerical substrate** in Metal (the primitives scanpy/rapids-singlecell/scipy need).
2. **Drop-in scanpy replacements** built on it, using rapids-singlecell as the API guide.
3. **Validation + benchmarking** — fp64 CPU-scanpy parity + speedup vs CPU.

## Repo layout
- `src/metasinglecell/` — installable library (`pip install -e .`); `config.py` resolves paths via `DATA_ROOT`.
- `validation_notebooks/` — lightweight drivers + the user-facing scanpy-workflow example.
- `results/<analysis>/` — csv/png/pdf outputs (gitignored except `.gitkeep`).
- `data/{raw,processed,external}/` — gitignored.
- `envs/` — dedicated conda env(s). `resources/` + `RESOURCES.md` — Apple-GPU numerical-computing links.
