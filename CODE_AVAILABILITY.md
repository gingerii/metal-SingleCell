# Code & software availability

**metal-SingleCell** (this repository) — a Metal/MLX GPU implementation of the scanpy/squidpy
single-cell analysis stack for Apple Silicon. Version `0.0.1`. Source:
`https://github.com/gingerii/metal-SingleCell`.

All benchmarks and validation were run on Apple **M3 Max** (40-core GPU, Metal 3, ~400 GB/s; 16-core
CPU; 48 GB unified memory), **macOS 14.4**, with the single `metasinglecell` conda environment below.
GPU compute uses Apple's Metal backend through MLX; CPU references use the standard scanpy/squidpy
stack on the same machine.

## metasinglecell environment (Python 3.11.15)

**GPU backend:** mlx 0.31.2.

**Numerics:** numpy 2.4.6, scipy 1.17.1, pandas 2.3.3.

**Single-cell / spatial:** anndata 0.12.17, scanpy 1.11.5, squidpy 1.8.2.

**Statistics & ML (CPU references / helpers):** scikit-learn 1.9.0, scikit-misc 0.5.2,
statsmodels 0.14.6.

**Graph clustering (CPU reference):** igraph 1.0.0, leidenalg 0.12.0.

**Embedding & nearest-neighbors:** umap-learn 0.5.12, pynndescent 0.5.13, numba 0.65.1.

**Integration & doublet detection (CPU references):** harmonypy 2.0.0, scrublet 0.2.3.

**I/O & notebooks:** h5py 3.16.0, matplotlib 3.11.0, nbformat 5.10.4, nbconvert 7.17.1.

## Notes

- The package itself depends only on **numpy**, **scipy**, and **mlx** (the GPU substrate); all other
  packages are the CPU reference implementations used for accuracy/speed validation and for the
  tutorial notebooks (plotting, dataset loaders).
- Heavy backends are lazy-imported, so `metasinglecell` installs and imports in any environment;
  install with `pip install -e .`.
- Environment specifications: [`envs/`](envs/). Full benchmark methodology and per-function results:
  [`results/validation/RESULTS_v_benchmark.md`](results/validation/RESULTS_v_benchmark.md).
