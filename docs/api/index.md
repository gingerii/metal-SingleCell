# API reference

`metalsinglecell` mirrors the scanpy / squidpy namespaces: `pp` (preprocessing), `tl` (tools),
`gr` (spatial graph). Import as:

```python
import metalsinglecell as msc
```

## Preprocessing — `msc.pp`

```{eval-rst}
.. currentmodule:: metalsinglecell.pp
.. autosummary::
   :toctree: generated
   :nosignatures:

   calculate_qc_metrics
   filter_cells
   filter_genes
   normalize_total
   log1p
   normalize_pearson_residuals
   highly_variable_genes
   scale
   regress_out
   pca
   neighbors
   bbknn
   harmony_integrate
   scrublet
```

## Tools — `msc.tl`

```{eval-rst}
.. currentmodule:: metalsinglecell.tl
.. autosummary::
   :toctree: generated
   :nosignatures:

   leiden
   louvain
   kmeans
   umap
   tsne
   diffmap
   draw_graph
   embedding_density
   rank_genes_groups
   score_genes
   score_genes_cell_cycle
```

## Spatial graph — `msc.gr`

```{eval-rst}
.. currentmodule:: metalsinglecell.gr
.. autosummary::
   :toctree: generated
   :nosignatures:

   spatial_neighbors
   spatial_autocorr
   co_occurrence
   ligrec
   calculate_niche
```
