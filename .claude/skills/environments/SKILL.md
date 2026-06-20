---
name: environments
description: Dedicated conda env for metaSingleCell (envs/metasinglecell.yml), install steps, the mlx/Metal backend, and the fp32-only-GPU fact. Load when setting up envs or running code.
---

# Environments — metaSingleCell

## Rule: dedicated envs only
Never reuse another project's env (e.g. the Xenium `spatial`/`morphometrics` envs, which also have
scanpy 1.11.5). This project has its own env so its deps (esp. `mlx`) stay isolated.

## The env
Defined in `envs/metasinglecell.yml` (Python 3.11, conda-forge). Two-step setup:

```bash
conda env create -f envs/metasinglecell.yml
conda activate metasinglecell
pip install -e .          # installs the metasinglecell package
```

Contents: numpy/scipy/pandas/anndata/h5py/scikit-learn, **scanpy + leidenalg + igraph + umap-learn**
(the CPU fp64 reference oracle), matplotlib/jupyter, and **mlx** (the Metal GPU backend, via pip).

The `-e ..` pip line is intentionally **not** in the yml (its cwd is unreliable during `conda env
create`) — run `pip install -e .` as the documented second step.

## Backend facts
- **mlx** = Apple's Metal array framework; the GPU compute layer. Custom `.metal` kernels interop with
  mlx arrays for the sparse primitives.
- **Apple GPU is fp32-only** (no float64). The CPU oracle runs fp64; parity tolerances must account for
  fp32 accumulation. Stability-critical decompositions (SVD) lean on Accelerate/LAPACK.
- Local machine is Apple M3 (Metal/MPS), no CUDA.

## Conda on this machine
`source /opt/anaconda3/etc/profile.d/conda.sh` to enable `conda activate` in a fresh shell. Base is
anaconda3 (Python 3.12); a benign "anaconda-anon-usage" warning prints on `conda env list`.
