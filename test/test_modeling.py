from __future__ import annotations

import numpy as np
import pandas as pd

from spatiold import (
    fit_ols_for_celltype,
    fit_ols_for_radius,
    prepare_radius_design,
)


def test_modeling_helpers_run_on_synthetic_data() -> None:
    rng = np.random.default_rng(7)

    n = 24
    genes = [f"g{i}" for i in range(5)]
    cell_ids = [f"c{i}" for i in range(n)]

    X_gene_df = pd.DataFrame(rng.normal(size=(n, len(genes))), index=cell_ids, columns=genes)
    radii = [20.0, 40.0]

    base_signal = 0.5 * X_gene_df["g0"] - 0.3 * X_gene_df["g1"]
    y20 = base_signal + rng.normal(scale=0.1, size=n)
    y40 = base_signal + 0.2 * X_gene_df["g2"] + rng.normal(scale=0.1, size=n)
    Y_ld_df = pd.DataFrame({20.0: y20, 40.0: y40}, index=cell_ids)

    out_ct = fit_ols_for_celltype(
        ct_label="synthetic",
        X_gene_df=X_gene_df,
        Y_ld_df=Y_ld_df,
        ld_radii_list=radii,
        min_cells=5,
        n_splits=3,
        return_cv_metrics=False,
    )

    assert out_ct["skipped"] is False
    assert out_ct["coef_s"].shape[0] == len(genes) * 2 + 1

    cell_type_s = pd.Series(["A"] * (n // 2) + ["B"] * (n - n // 2), index=cell_ids)
    design = prepare_radius_design(X_gene_df, cell_type_s)

    out_r = fit_ols_for_radius(
        radius_um=20.0,
        G=design["G"],
        C=design["C"],
        GCT=design["GCT"],
        gene_names=design["gene_names"],
        ct_names=design["ct_names"],
        gct_names=design["gct_names"],
        y=Y_ld_df[20.0].to_numpy(),
        min_cells=5,
        n_splits=3,
        return_cv_metrics=False,
    )

    assert out_r["skipped"] is False
    assert out_r["coef_s"].shape[0] == (
        len(design["gene_names"]) + len(design["ct_names"]) + len(design["gct_names"])
    )
