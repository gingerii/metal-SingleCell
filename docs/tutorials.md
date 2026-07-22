# Tutorials

Executable notebooks mirroring the rapids-singlecell tutorials. Notebooks 1, 2, and 4 are self-contained
(they auto-download their datasets); `brain_1M` needs the 10x 1.3M-neuron `.h5` placed at
`data/external/1M_neurons.h5`.

```{toctree}
:maxdepth: 1

notebooks/01_basic_workflow
notebooks/02_pearson_residuals
notebooks/04_squidpy
notebooks/brain_1M
```

| Notebook | Workflow |
|---|---|
| {doc}`notebooks/01_basic_workflow` | QC → normalize → HVG → scale → PCA → neighbors → UMAP → Leiden → markers → Harmony → diffmap (PBMC 3k) |
| {doc}`notebooks/02_pearson_residuals` | Analytic Pearson-residual normalization → PCA → clustering (PBMC 3k) |
| {doc}`notebooks/04_squidpy` | Spatial graph, Moran's I / Geary's C, co-occurrence (squidpy IMC) |
| {doc}`notebooks/brain_1M` | Full 1,000,000-cell workflow on a laptop GPU |
