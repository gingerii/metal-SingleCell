"""Generate the four metal-SingleCell tutorial notebooks (nbformat).

Mirrors the rapids-singlecell tutorials (01_basic_workflow, 02_pearson_residuals,
04_squidpy, brain_1M) but with the metal-SingleCell (`msc.pp/tl/gr`) API. Run:
    python notebooks/_build_tutorials.py
then execute with: jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

OUT = Path(__file__).parent


def nb(cells):
    n = new_notebook()
    n.cells = [new_markdown_cell(c[1]) if c[0] == "md" else new_code_cell(c[1]) for c in cells]
    n.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
    return n


# ───────────────────────── 01 basic workflow (PBMC3k) ─────────────────────────
basic = [
("md", "# 01 · Basic single-cell workflow on the Apple-silicon GPU\n\n"
 "This mirrors the [rapids-singlecell basic workflow](https://rapids-singlecell.readthedocs.io/en/latest/notebooks/01_basic_workflow.html), "
 "but every accelerated step runs on the **Apple GPU via MLX/Metal** through "
 "**metal-SingleCell** (`msc`). The API is a drop-in for `scanpy` / `rapids_singlecell`: "
 "functions take an `AnnData`, compute on the GPU, and write results back to the standard "
 "slots — so plotting still uses `scanpy.pl`. We use the classic **PBMC 3k** dataset."),
("code", "import warnings; warnings.filterwarnings('ignore')\n"
 "import numpy as np, scanpy as sc\n"
 "import metalsinglecell as msc\n"
 "sc.settings.verbosity = 1\n"
 "print('metalsinglecell', msc.__version__)"),
("md", "## Load data"),
("code", "adata = sc.datasets.pbmc3k()\nadata.var_names_make_unique()\nadata"),
("md", "## Quality control\nFlag mitochondrial genes and compute QC metrics (scanpy), then filter."),
("code", "adata.var['mt'] = adata.var_names.str.startswith('MT-')\n"
 "sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)\n"
 "sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],\n"
 "             jitter=0.4, multi_panel=True)"),
("code", "msc.pp.filter_cells(adata, min_genes=200)\n"
 "msc.pp.filter_genes(adata, min_cells=3)\n"
 "adata = adata[adata.obs.pct_counts_mt < 20].copy()\n"
 "adata.shape"),
("md", "## Normalize, log1p, and highly variable genes\nAll on the GPU."),
("code", "adata.layers['counts'] = adata.X.copy()\n"
 "msc.pp.normalize_total(adata, target_sum=1e4)\n"
 "msc.pp.log1p(adata)\n"
 "msc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')\n"
 "adata.var.highly_variable.sum()"),
("code", "adata.raw = adata\nadata = adata[:, adata.var.highly_variable].copy()"),
("md", "## Scale and PCA"),
("code", "msc.pp.scale(adata, max_value=10)\n"
 "msc.pp.pca(adata, n_comps=50, use_highly_variable=False)\n"
 "sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50)"),
("md", "## Neighbors, UMAP, and Leiden clustering\n`backend='gpu'` runs the Metal parallel Leiden."),
("code", "msc.pp.neighbors(adata, n_neighbors=15)\n"
 "msc.tl.umap(adata)\n"
 "msc.tl.leiden(adata, resolution=1.0, backend='gpu')\n"
 "adata.obs.leiden.value_counts()"),
("code", "sc.pl.umap(adata, color=['leiden'], legend_loc='on data')"),
("md", "## Marker genes per cluster"),
("code", "msc.tl.rank_genes_groups(adata, 'leiden', method='t-test')\n"
 "sc.pl.rank_genes_groups(adata, n_genes=15, sharey=False)"),
("md", "## Batch integration with Harmony (illustrative)\nPBMC3k has no real batch, so we add a "
 "**synthetic** one purely to demonstrate the `harmony_integrate` API; the corrected embedding "
 "is written to `obsm['X_pca_harmony']`."),
("code", "adata.obs['batch'] = np.where(np.arange(adata.n_obs) % 2 == 0, 'A', 'B').astype(str)\n"
 "msc.pp.harmony_integrate(adata, key='batch')\n"
 "adata.obsm['X_pca_harmony'].shape"),
("md", "## Diffusion map"),
("code", "msc.tl.diffmap(adata, n_comps=15)\nadata.obsm['X_diffmap'].shape"),
("md", "Every accelerated step (`normalize_total`, `log1p`, `highly_variable_genes`, `scale`, "
 "`pca`, `neighbors`, `umap`, `leiden`, `rank_genes_groups`, `harmony_integrate`, `diffmap`) "
 "ran on the Apple GPU; the AnnData object is identical in structure to a scanpy run."),
]

# ───────────────────────── 02 Pearson residuals (PBMC3k) ─────────────────────────
pearson = [
("md", "# 02 · Analytic Pearson residuals\n\n"
 "Mirrors the [rapids-singlecell Pearson-residuals tutorial](https://rapids-singlecell.readthedocs.io/en/latest/notebooks/02_pearson_residuals.html). "
 "Pearson residuals (Lause et al. 2021) are an alternative to log-normalization that models "
 "counts with a regularized negative binomial; the residuals feed directly into PCA. All on "
 "the Metal GPU via **metal-SingleCell**."),
("code", "import warnings; warnings.filterwarnings('ignore')\n"
 "import numpy as np, scanpy as sc\n"
 "import metalsinglecell as msc"),
("md", "## Load and QC"),
("code", "adata = sc.datasets.pbmc3k(); adata.var_names_make_unique()\n"
 "adata.var['mt'] = adata.var_names.str.startswith('MT-')\n"
 "sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)\n"
 "msc.pp.filter_cells(adata, min_genes=200)\n"
 "msc.pp.filter_genes(adata, min_cells=3)\n"
 "adata = adata[adata.obs.pct_counts_mt < 20].copy()\n"
 "adata.layers['counts'] = adata.X.copy()\n"
 "adata.shape"),
("md", "## Highly variable genes from raw counts\nThe rapids tutorial uses `flavor='pearson_residuals'`; "
 "metal-SingleCell currently offers `seurat_v3` (the closest count-based selector), which also "
 "operates directly on the counts layer."),
("code", "msc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', layer='counts')\n"
 "adata = adata[:, adata.var.highly_variable].copy()\n"
 "adata.X = adata.layers['counts'].copy()   # residuals are computed from raw counts\n"
 "adata.shape"),
("md", "## Pearson residuals → PCA\n`normalize_pearson_residuals` writes the residuals into `adata.X`; "
 "PCA then runs on them."),
("code", "msc.pp.normalize_pearson_residuals(adata, theta=100.0)\n"
 "msc.pp.pca(adata, n_comps=50, use_highly_variable=False)\n"
 "sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50)"),
("md", "## Neighbors, UMAP, Leiden on the residual PCA"),
("code", "msc.pp.neighbors(adata, n_neighbors=15)\n"
 "msc.tl.umap(adata)\n"
 "msc.tl.leiden(adata, resolution=1.0, backend='gpu')\n"
 "sc.pl.umap(adata, color=['leiden'], legend_loc='on data')"),
("md", "Pearson-residual normalization (no log1p, no scaling) produces a clustered embedding "
 "directly from counts — useful when sequencing depth varies strongly across cells."),
]

# ───────────────────────── 04 squidpy spatial (IMC) ─────────────────────────
squidpy = [
("md", "# 04 · Spatial analysis (squidpy-GPU)\n\n"
 "Mirrors the [rapids-singlecell squidpy tutorial](https://rapids-singlecell.readthedocs.io/en/latest/notebooks/04_squidpy.html). "
 "metal-SingleCell's `msc.gr` namespace is a drop-in for `squidpy.gr`, GPU-accelerated on Apple "
 "silicon. We use squidpy's **IMC** (imaging mass cytometry) dataset — it has cell-type labels "
 "and channel intensities, so we can show spatial autocorrelation **and** the fused-kernel "
 "co-occurrence analysis."),
("code", "import warnings; warnings.filterwarnings('ignore')\n"
 "import numpy as np, scanpy as sc, squidpy as sq\n"
 "import metalsinglecell as msc"),
("code", "adata = sq.datasets.imc()\nadata"),
("md", "## Spatial neighbors graph\nBuilds `obsp['spatial_connectivities']` from `obsm['spatial']`."),
("code", "msc.gr.spatial_neighbors(adata, n_neighs=6)\n"
 "adata.obsp['spatial_connectivities'].shape"),
("md", "## Moran's I — global spatial autocorrelation\nWhich channels vary smoothly across the tissue?"),
("code", "msc.gr.spatial_autocorr(adata, mode='moran', genes=adata.var_names.tolist(), n_perms=100)\n"
 "adata.uns['moranI'].head(8)"),
("md", "## Geary's C — a complementary autocorrelation statistic"),
("code", "msc.gr.spatial_autocorr(adata, mode='geary', genes=adata.var_names.tolist(), n_perms=100)\n"
 "adata.uns['gearyC'].head(8)"),
("md", "## Co-occurrence — how cell types cluster vs distance\n"
 "`co_occurrence` uses a fused Metal atomic-histogram kernel (tiled, scales to large sections). "
 "It measures the enrichment of each cell type around a target type as a function of radius."),
("code", "msc.gr.co_occurrence(adata, cluster_key='cell type', interval=50)\n"
 "occ = adata.uns['cell type_co_occurrence']['occ']\n"
 "print('co-occurrence tensor (types × types × radii):', occ.shape)"),
("code", "try:\n"
 "    sq.pl.co_occurrence(adata, cluster_key='cell type',\n"
 "                        clusters=list(adata.obs['cell type'].cat.categories[:3]))\n"
 "except Exception as e:\n"
 "    print('squidpy plot skipped:', e)"),
("md", "## Visualize the tissue\nTop spatially-autocorrelated channel and the cell-type map."),
("code", "top_gene = adata.uns['moranI'].index[0]\n"
 "sq.pl.spatial_scatter(adata, color=[top_gene, 'cell type'], shape=None, size=8)"),
("md", "All spatial graph statistics (`spatial_neighbors`, `spatial_autocorr` for Moran/Geary, "
 "`co_occurrence`) ran on the Apple GPU and match squidpy's CPU results exactly."),
]

# ───────────────────────── brain_1M (1.3M neurons → 1M) ─────────────────────────
brain = [
("md", "# 1.3 Million brain cells on a laptop GPU\n\n"
 "Mirrors the [rapids-singlecell 1M-brain tutorial](https://rapids-singlecell.readthedocs.io/en/latest/notebooks/brain_1M.html) "
 "— the same workflow that needs a datacenter GPU there, run here on an **Apple M-series laptop** "
 "via metal-SingleCell. We process **1,000,000** cells end-to-end. (Expect a few minutes; the "
 "neighbor graph is the slow step.)\n\n"
 "> Requires the 10x 1.3M-neuron file at `data/external/1M_neurons.h5`."),
("code", "import warnings; warnings.filterwarnings('ignore')\n"
 "import time, gc, numpy as np, scanpy as sc\n"
 "import metalsinglecell as msc\n"
 "t0 = time.time()"),
("md", "## Load and subset to 1M cells"),
("code", "import os\n"
 "def _find(rel):              # resolve path from notebook dir up to the repo root\n"
 "    d = os.getcwd()\n"
 "    for _ in range(6):\n"
 "        if os.path.exists(os.path.join(d, rel)): return os.path.join(d, rel)\n"
 "        d = os.path.dirname(d)\n"
 "    raise FileNotFoundError(rel + ' (place the 10x 1.3M-neuron .h5 at data/external/)')\n"
 "adata = sc.read_10x_h5(_find('data/external/1M_neurons.h5'))\n"
 "adata.var_names_make_unique()\n"
 "adata = adata[:1_000_000].copy()\n"
 "adata.shape"),
("md", "## QC and filtering"),
("code", "adata.var['mt'] = adata.var_names.str.lower().str.startswith('mt-')\n"
 "sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, percent_top=None)\n"
 "msc.pp.filter_genes(adata, min_cells=3)\n"
 "adata = adata[(adata.obs.n_genes_by_counts > 500) & (adata.obs.pct_counts_mt < 20)].copy()\n"
 "gc.collect(); adata.shape"),
("md", "## Normalize, log1p, highly variable genes (Seurat v3 on counts)"),
("code", "adata.layers['counts'] = adata.X.copy()\n"
 "msc.pp.normalize_total(adata, target_sum=1e4)\n"
 "msc.pp.log1p(adata)\n"
 "msc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor='seurat_v3', layer='counts')\n"
 "adata = adata[:, adata.var.highly_variable].copy()\n"
 "del adata.layers['counts']; gc.collect(); adata.shape"),
("md", "## Sparse PCA, neighbors, UMAP, Leiden\nWe run PCA directly on the sparse log-normalized "
 "HVG matrix (no densifying `scale`/`regress_out`, which would need ~20 GB at this scale)."),
("code", "msc.pp.pca(adata, n_comps=50, use_highly_variable=False)\n"
 "adata.obsm['X_pca'].shape"),
("code", "msc.pp.neighbors(adata, n_neighbors=15)\n"
 "msc.tl.leiden(adata, resolution=1.0, backend='gpu')\n"
 "adata.obs.leiden.nunique()"),
("code", "msc.tl.umap(adata, min_dist=0.3)\n"
 "adata.obsm['X_umap'].shape"),
("code", "print(f'{adata.n_obs:,} cells · {adata.obs.leiden.nunique()} Leiden clusters · "
 "total wall time {time.time() - t0:.0f}s')"),
("code", "sc.pl.umap(adata, color=['leiden'], legend_loc=None, size=2, frameon=False)"),
("md", "A full 1M-cell clustering workflow — normalize → HVG → PCA → kNN graph → Leiden → UMAP — "
 "on a laptop GPU, with the Metal parallel Leiden doing the clustering."),
]

for name, cells in [("01_basic_workflow", basic), ("02_pearson_residuals", pearson),
                    ("04_squidpy", squidpy), ("brain_1M", brain)]:
    p = OUT / f"{name}.ipynb"
    nbf.write(nb(cells), p)
    print("wrote", p)
