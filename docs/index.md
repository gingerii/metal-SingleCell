# metalsinglecell

**GPU-accelerated single-cell analysis on Apple Silicon.**

A Metal/[MLX](https://github.com/ml-explore/mlx) re-implementation of
[rapids-singlecell](https://rapids-singlecell.readthedocs.io) — drop-in replacements for the core
[scanpy](https://scanpy.readthedocs.io) and [squidpy](https://squidpy.readthedocs.io) functions that
run on the M-series GPU. Swap the import prefix and your existing pipeline runs on the GPU.

```python
import scanpy as sc
import metalsinglecell as msc

adata = sc.datasets.pbmc3k()
msc.pp.normalize_total(adata, target_sum=1e4)
msc.pp.log1p(adata)
msc.pp.highly_variable_genes(adata, n_top_genes=2000)
msc.pp.pca(adata)
msc.pp.neighbors(adata)
msc.tl.leiden(adata, backend="gpu")
msc.tl.umap(adata)
sc.pl.umap(adata, color="leiden")
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`download` Installation
:link: installation
:link-type: doc

Get started — `pip install metalsinglecell` and environment setup.
:::

:::{grid-item-card} {octicon}`book` Tutorials
:link: tutorials
:link-type: doc

Executable notebooks mirroring the rapids-singlecell workflows.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: api/index
:link-type: doc

Every `pp` / `tl` / `gr` function, documented.
:::

:::{grid-item-card} {octicon}`mark-github` GitHub
:link: https://github.com/gingerii/metal-SingleCell

Source, issues, and releases.
:::

::::

```{toctree}
:hidden: true
:maxdepth: 1

installation
tutorials
api/index
GitHub <https://github.com/gingerii/metal-SingleCell>
```
