# Larger-real-data validation — remaining functions at scale (100k real neurons)

Closes the PBMC-only gap for the tools/integration/pp/neighbors functions. Scalable functions
re-confirmed on **100k cells subsampled from the 10x 1.3M-neuron atlas** (HVG-restricted,
canonical), vs their references. Driver: `validation_notebooks/v_remaining_scale.py`.

| function | scale | result |
|----------|-------|--------|
| kmeans | 100k | ARI 0.726 vs sklearn, 0.1s |
| score_genes | 100k | score corr **1.000** vs scanpy |
| rank_genes_groups | 100k | top-25 marker overlap **1.000**, 4.0s |
| normalize_pearson_residuals | 100k | residual corr **1.0000** |
| diffmap | 100k | eigval corr 0.993, comp subspace 0.792 |
| harmonize | 100k | batch mixing 0.01→**0.50**, 49s |
| bbknn | 100k | batch-balanced graph (opp-batch 0.56), 7.2s |
| scrublet | 25k (combined ~75k) | injected-doublet AUC **0.931**, 13.4s |

Inherently O(n²) by design (subsample-only, NOT run at 100k): exact t-SNE, gaussian-KDE
embedding_density.

## Two real scalability bugs FOUND and FIXED (surfaced only by larger-data validation)
1. **bbknn GPU OOM**: built the full `n × |batch|` distance matrix on the GPU per batch — at
   100k with a 50k batch that's ~20 GB → Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
   Fixed by tiling the query rows (~256M-entry cap). Now 100k in 7.2s, no OOM.
2. **scrublet GPU OOM**: densified the combined real+simulated matrix (`.toarray()` on ~3n cells
   × all ~20k genes ≈ 24 GB). Fixed by HVG-restricting before a sparse-aware PCA (also the
   canonical scrublet approach). Verified to combined ~91k cells.

## Known remaining scale limit (future work)
scrublet's brute-force `_knn_gpu` becomes memory-heavy past ~combined-100k cells (large distance
tiles on a ~3n-cell set). Validated to combined ~91k; for very large doublet-simulation sets it
should route through the tiled/pynndescent `neighbors()` path. Not a correctness issue.

## Note on the dev-machine crash during this work
An earlier full 100k run coincided with a hard reboot. Reproduction showed this harness drove
memory to exhaustion and triggered GPU OOMs at the bbknn/scrublet stage (the two bugs above, plus
the harness not freeing ~8 GB dense intermediates between functions). Those are the most likely
crash contributors; all are now fixed/mitigated (tiling, HVG-restriction, `gc` cleanup, safe test
sizes). No kernel-panic/jetsam signature was recoverable post-reboot.
