"""Stage 1/2 validation: Metal PCA (3 solvers) vs the fp64 CPU oracle.

The oracle used arpack on the scaled HVG matrix (snapshot ``05_scaled``), saving
``06_X_pca`` (cells x 50), ``06_pca_components`` (genes x 50) and
``06_pca_variance_ratio``. PCs are defined up to sign, so we compare per-component
absolute correlation (sign-invariant) plus variance_ratio and subspace overlap.

    conda activate metalsinglecell
    python validation_notebooks/05_pca_parity.py
"""

import logging

import numpy as np

from metalsinglecell import config, validation
from metalsinglecell.decomposition import pca
from metalsinglecell.reference import N_PCS, SEED


def main() -> None:
    res_dir = config.results_dir("pca")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(res_dir / "pca.log", mode="w"),
                  logging.StreamHandler()],
    )
    log = logging.getLogger("pca")

    X = validation.load_snapshot("05_scaled")
    x_pca_ref = validation.load_snapshot("06_X_pca")
    comps_ref = validation.load_snapshot("06_pca_components")        # genes x 50
    vr_ref = validation.load_snapshot("06_pca_variance_ratio")
    log.info("PCA input %s; oracle X_pca %s, components %s", X.shape, x_pca_ref.shape, comps_ref.shape)

    records = []

    # --- exact solvers: validate against the arpack oracle (06_*) -----------
    for solver in ("arpack", "full"):
        x_pca, comps, vr = pca(X, n_comps=N_PCS, solver=solver, random_state=SEED)
        r_emb = validation.compare_signed_columns(f"{solver}:X_pca", x_pca, x_pca_ref, 0.999)
        r_cmp = validation.compare_signed_columns(f"{solver}:components", comps.T, comps_ref, 0.999)
        r_vr = validation.compare(f"{solver}:variance_ratio", vr, vr_ref, rtol=1e-3, atol=1e-3)
        overlap = validation.subspace_overlap(comps.T, comps_ref)
        for r in (r_emb, r_cmp):
            r["subspace_overlap_vs_exact"] = round(overlap, 6)
        records += [r_emb, r_cmp, r_vr]
        log.info("%-11s X_pca min|r|=%.5f comps min|r|=%.5f vr max_rel=%.3g subspace=%.5f [%s]",
                 solver, r_emb["min_abs_corr"], r_cmp["min_abs_corr"], r_vr["max_rel_err"], overlap,
                 "PASS" if all(x["passed"] for x in (r_emb, r_cmp, r_vr)) else "FAIL")

    # --- randomized: approximate + seeded, so validate against sklearn's own
    #     randomized_svd (its true ground truth). Subspace overlap vs the EXACT
    #     oracle is reported as an informational approximation-quality metric. ---
    from sklearn.utils.extmath import randomized_svd

    Xc = X.astype(np.float64) - X.astype(np.float64).mean(0)
    total_var = Xc.var(axis=0, ddof=1).sum()
    Ur, Sr, Vtr = randomized_svd(Xc, n_components=N_PCS, n_oversamples=10, n_iter=7,
                                 random_state=SEED, power_iteration_normalizer="QR")
    ref_emb, ref_comps, ref_vr = Ur * Sr, Vtr.T, (Sr ** 2 / (X.shape[0] - 1)) / total_var

    x_pca, comps, vr = pca(X, n_comps=N_PCS, solver="randomized", random_state=SEED)
    r_emb = validation.compare_signed_columns("randomized:X_pca(vs sklearn)", x_pca, ref_emb, 0.999)
    r_cmp = validation.compare_signed_columns("randomized:components(vs sklearn)", comps.T, ref_comps, 0.999)
    r_vr = validation.compare("randomized:variance_ratio(vs sklearn)", vr, ref_vr, rtol=1e-3, atol=1e-3)
    overlap_exact = validation.subspace_overlap(comps.T, comps_ref)
    for r in (r_emb, r_cmp):
        r["subspace_overlap_vs_exact"] = round(overlap_exact, 6)
    records += [r_emb, r_cmp, r_vr]
    log.info("randomized  vs sklearn: X_pca min|r|=%.5f comps min|r|=%.5f vr max_rel=%.3g [%s]",
             r_emb["min_abs_corr"], r_cmp["min_abs_corr"], r_vr["max_rel_err"],
             "PASS" if all(x["passed"] for x in (r_emb, r_cmp, r_vr)) else "FAIL")
    log.info("randomized  approximation quality: subspace overlap vs EXACT oracle = %.5f "
             "(slowly-decaying spectrum; trailing PCs under-resolved)", overlap_exact)

    validation.write_report(records, "pca")
    print(f"\nPCA parity: {'PASS' if all(r['passed'] for r in records) else 'FAIL'}")


if __name__ == "__main__":
    main()
