# metal-SingleCell

**GPU-accelerated single-cell analysis on Apple Silicon.**

A Metal/[MLX](https://github.com/ml-explore/mlx) re-implementation of
[rapids-singlecell](https://rapids-singlecell.readthedocs.io) — drop-in replacements for the core
[scanpy](https://scanpy.readthedocs.io) functions that run on the M-series GPU. rapids-singlecell is
CUDA/CuPy-only; there is no Apple-silicon path because the M-series GPU has no native sparse-matrix
support. This project builds that missing substrate and the scanpy front-end on top of it.

> ⚠️ Early-stage / exploratory. The scaffold and the CPU reference oracle exist; the Metal kernels do not yet.

## Why it's hard

Single-cell count matrices are large and ~90–95% sparse (cells × genes), and the scanpy front-end
(QC → normalize → highly-variable genes → PCA) is built on sparse linear algebra. Two Apple-specific
constraints shape everything:

- **No GPU sparse.** Apple's `Accelerate` framework has sparse support on the **CPU** only; MPS/
  PyTorch-MPS sparse is thin. Building GPU sparse primitives (CSR SpMM, segmented reductions) is the
  core contribution.
- **No float64 on the GPU.** Metal is fp32 (plus fp16/bf16). So "correct" means *reproducing the fp64
  CPU-scanpy result within a justified tolerance* — which makes a frozen reference oracle essential.

**Bounding insight:** most of the pipeline goes *dense* after PCA (KNN, UMAP, Leiden all run on the
~50-dim embedding), so the sparse-critical kernel set is small and finite.

## Architecture

**MLX-primary · custom Metal kernels for sparse · Accelerate/LAPACK as the numerical anchor.**

| Layer | Used for |
|-------|----------|
| **MLX** | Array layer + dense post-PCA pipeline (KNN/UMAP/Leiden). |
| **Custom Metal kernels** | The sparse front-end primitives MLX/MPS lack (CSR SpMM, row/col reductions). |
| **Accelerate / LAPACK** | Stability-critical math — randomized SVD for PCA, two-pass/Welford variance. |

## Roadmap

1. **Sparse + numerical substrate** — Metal primitives the scanpy/rapids-singlecell front-end needs.
2. **scanpy drop-ins** — `normalize_total`, `log1p`, `highly_variable_genes`, `scale`, `pca`,
   `neighbors`, `leiden`, `rank_genes_groups`, mirroring rapids-singlecell signatures.
3. **Validation + benchmarking** — fp64 CPU-scanpy numerical parity + speedup vs CPU.

## Getting started

```bash
# dedicated environment (Apple Silicon)
conda env create -f envs/metasinglecell.yml
conda activate metasinglecell
pip install -e .

# build the fp64 CPU reference oracle (PBMC3k) — the parity ground truth
python validation_notebooks/00_cpu_reference_oracle.py
```

## Repository layout

```
src/metasinglecell/      installable library (config.py resolves paths via DATA_ROOT)
  reference.py           fp64 CPU scanpy oracle (snapshots every intermediate)
validation_notebooks/    lightweight drivers + the user-facing example workflow
results/<analysis>/      reportable outputs (csv/png/pdf) — gitignored
data/{raw,processed,external}/   inputs & derived objects — gitignored
envs/                    dedicated conda env(s)
resources/, RESOURCES.md Apple-GPU numerical-computing references
.claude/skills/          durable project knowledge (see SKILLS.md)
```

## License

BSD-3-Clause.
